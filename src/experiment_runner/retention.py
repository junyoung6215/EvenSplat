"""Small sequential launcher, deliberately independent of the legacy runner.

Planning needs only PyYAML and the standard library, never imports the model,
and never writes files. Training/evaluation run in separate subprocesses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "config/experiment_runs/retention_suite.yaml"
SCORES = ("random", "base_opacity", "opacity", "reliability", "reliability_filtered")
FULL_TEST_INDEX = "assets/evaluation_index_re10k.json"


def read_settings(path):
    defaults = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    supplied = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    def merge(base, extra, prefix=""):
        if not isinstance(extra, dict):
            raise ValueError(f"{prefix or 'settings'} must be a mapping")
        for key, value in extra.items():
            if key not in base:
                raise ValueError(f"Unknown setting: {prefix}{key}")
            if isinstance(base[key], dict):
                merge(base[key], value, f"{prefix}{key}.")
            else:
                base[key] = value
    merge(defaults, supplied)
    return defaults


def local_path(value):
    if not isinstance(value, str) or not value.strip() or value == "???":
        raise ValueError("A non-empty local path is required")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def override(key, value):
    # JSON strings/lists are valid Hydra values; quoting is inside each argv
    # element, not shell quoting. This also protects spaces and backslashes.
    encoded = json.dumps(value, ensure_ascii=False)
    if "${" in encoded:
        raise ValueError("Runner settings must be literal, not OmegaConf interpolations")
    return f"{key}={encoded}"


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def ratio_value(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Ratios must be finite and in (0, 1]")
    return value


def build_plan(cfg, args):
    """Build JSON-serializable jobs without touching disk or loading a checkpoint."""
    train = args.command == "train"
    arm = args.arm if train else "control"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(cfg["gpu"])):
        raise ValueError("gpu must select exactly one device, e.g. '0' or a GPU UUID")
    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_id):
        raise ValueError("run-id must be a short alphanumeric/underscore/hyphen name")
    output = local_path(cfg["output_root"]) / run_id
    checkpoint = local_path(args.checkpoint or cfg["initial_checkpoint"])
    if not train and args.split == "full":
        # Official RE10K test index, as reported by NoPoSplat and others.
        index = local_path(FULL_TEST_INDEX)
    else:
        index = local_path(cfg["validation_index" if train or args.split == "validation" else "evaluation_index"])
    t = cfg["train"]
    for name, value in (("olp_enabled", cfg["olp_enabled"]),
                        ("reliability_enabled", cfg["reliability_enabled"]),
                        ("train.validation_videos", t["validation_videos"]),
                        ("evaluation.align_pose", cfg["evaluation"]["align_pose"]),
                        ("evaluation.save_image", cfg["evaluation"]["save_image"])):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a YAML boolean, not a string")
    for name, value in (("lr", t["lr"]), ("backbone_lr_multiplier", t["backbone_lr_multiplier"]),
                        ("legacy_weight", t["legacy_weight"]), ("prior_weight", t["prior_weight"]),
                        ("detail_strength", cfg["local"]["detail_strength"])):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not isinstance(cfg["reliability_floor"], (float, int)) or not 0 <= cfg["reliability_floor"] <= 1:
        raise ValueError("reliability_floor must be in [0, 1]")
    if not 0 <= cfg["olp_mix"] <= 1 or not 0 <= cfg["reliability_r_min"] < 1:
        raise ValueError("Invalid OLP mix or reliability r_min")
    positive_int(cfg["local"]["cell_size"], "local.cell_size")
    if cfg["local"]["min_per_cell"] not in (0, 1):
        raise ValueError("local.min_per_cell must be 0 or 1")
    if cfg["local"]["demand_source"] not in ("image_detail", "uniform"):
        raise ValueError("local.demand_source must be image_detail or uniform")
    for key in ("stratified", "merge"):
        if type(cfg["local"][key]) is not bool:
            raise ValueError(f"local.{key} must be a YAML boolean")
    for key in ("steps", "batch_size", "accumulate", "validation_interval", "checkpoint_interval"):
        positive_int(t[key], f"train.{key}")
    if type(t["workers"]) is not int or t["workers"] < 0:
        raise ValueError("train.workers must be a nonnegative integer")
    if not 0 <= t["loss_warmup_start"] <= t["loss_warmup_end"]:
        raise ValueError("Invalid loss warm-up range")
    base = {
        "mode": "train" if train else "test",
        "wandb.mode": "disabled", "seed": cfg["seed"],
        "dataset.re10k.roots": [local_path(cfg["data_root"]).as_posix()],
        "dataset.re10k.view_sampler.evaluation_index_path": index.as_posix(),
        # EncoderNoPoSplatCfg.pretrained_weights is a str; null fails typed config load.
        "model.encoder.pretrained_weights": checkpoint.as_posix() if train else "",
        "checkpointing.load": None if train else checkpoint.as_posix(),
        "checkpointing.save_weights_only": False,
        "model.anchor_selection.enabled": False,
        "model.encoder.orthogonal_preconditioner.enabled": cfg["olp_enabled"],
        "model.encoder.orthogonal_preconditioner.seed": cfg["olp_seed"],
        "model.encoder.orthogonal_preconditioner.mix": cfg["olp_mix"],
        "model.encoder.orthogonal_preconditioner.target_mix": cfg["olp_mix"],
        "model.encoder.orthogonal_preconditioner.warmup_steps": 0,
        "model.encoder.gaussian_reliability.enabled": cfg["reliability_enabled"],
        "model.encoder.gaussian_reliability.r_min": cfg["reliability_r_min"],
        "model.encoder.gaussian_reliability.detach_gate": False,
        "test.align_pose": cfg["evaluation"]["align_pose"],
        "test.pose_align_steps": cfg["evaluation"]["pose_align_steps"],
        "test.save_image": cfg["evaluation"]["save_image"],
        "test.save_video": False, "test.save_compare": False,
        "test.save_olp_stats": False, "test.save_gaussian_cap_stats": False,
        "model.gaussian_retention.mode": "fixed_ratio",
        "model.gaussian_retention.random_seed": cfg["seed"],
        "model.gaussian_retention.reliability_floor": cfg["reliability_floor"],
        "model.gaussian_retention.start_step": 0,
        "model.gaussian_retention.diagnostics_every_n_steps": 1,
        "model.gaussian_retention.local_cells.demand_source": cfg["local"]["demand_source"],
        "model.gaussian_retention.local_cells.stratified": cfg["local"]["stratified"],
        "model.gaussian_retention.local_cells.merge": cfg["local"]["merge"],
        "model.gaussian_retention.local_cells.cell_size": cfg["local"]["cell_size"],
        "model.gaussian_retention.local_cells.detail_strength": cfg["local"]["detail_strength"],
        "model.gaussian_retention.local_cells.min_per_cell": cfg["local"]["min_per_cell"],
        "data_loader.test.num_workers": t["workers"],
        "data_loader.test.persistent_workers": False,
    }
    sampler = t.get("sampler", "fixed")
    if sampler not in ("fixed", "random"):
        raise ValueError("train.sampler must be fixed or random")
    if train and arm == "prior" and sampler != "fixed":
        raise ValueError("The prior arm needs train.sampler=fixed (cache is keyed by fixed pairs)")
    if train and sampler == "random":
        # Original NoPoSplat bounded random pairs over the whole train split.
        # Warm-up 0: fine-tuning starts from a converged checkpoint, so use the
        # final 45-90 context gap immediately instead of re-ramping from 25.
        base["dataset.re10k.view_sampler.warm_up_steps"] = 0
    else:
        base["dataset.re10k.view_sampler.index_path"] = local_path(cfg["train_pairs"]).as_posix()
    if train:
        scope = args.train_scope if args.train_scope is not None else t["finetune_scope"]
        if scope not in ("full", "heads"):
            raise ValueError("train.finetune_scope must be full or heads")
        accumulate = args.accumulate if args.accumulate is not None else t["accumulate"]
        positive_int(accumulate, "accumulate")
        if args.local and not args.prune:
            raise ValueError("--local requires --prune; otherwise allocation has no effect")
        if arm != "control" and not cfg["reliability_enabled"]:
            raise ValueError("Auxiliary reliability losses require the reliability head")
        if not math.isfinite(args.budget_weight) or args.budget_weight < 0:
            raise ValueError("--budget-weight must be finite and nonnegative")
        if type(args.soft_steps) is not int or args.soft_steps < 0:
            raise ValueError("--soft-steps must be a nonnegative integer")
        if not math.isfinite(args.budget_hard_scale) or args.budget_hard_scale < 0:
            raise ValueError("--budget-hard-scale must be finite and nonnegative")
        if args.budget_warmup and args.soft_steps == 0:
            raise ValueError("--budget-warmup ramps over --soft-steps, which must be positive")
        if (args.budget_weight > 0 or args.soft_steps > 0) and not (
                args.local and args.score == "reliability"):
            raise ValueError("--budget-weight/--soft-steps need --prune --local --score reliability")
        base.update({
            "trainer.max_steps": args.steps if args.steps is not None else t["steps"],
            "trainer.accumulate_grad_batches": accumulate,
            "train.finetune_scope": scope,
            "trainer.val_check_interval": t["validation_interval"],
            "train.validation_videos": t["validation_videos"],
            "checkpointing.every_n_train_steps": t["checkpoint_interval"],
            "data_loader.train.batch_size": t["batch_size"],
            "data_loader.train.num_workers": t["workers"],
            "data_loader.train.persistent_workers": t["workers"] > 0,
            "optimizer.lr": t["lr"],
            "optimizer.backbone_lr_multiplier": t["backbone_lr_multiplier"],
            "model.gaussian_retention.diagnostics_every_n_steps": 100,
            "model.gaussian_retention.budget_weight": args.budget_weight,
            "model.gaussian_retention.start_step": args.soft_steps,
            "model.gaussian_retention.budget_warmup": args.budget_warmup,
            "model.gaussian_retention.budget_hard_scale": args.budget_hard_scale,
        })
        positive_int(base["trainer.max_steps"], "steps")
        if arm in ("legacy", "prior"):
            name = "reliability" if arm == "legacy" else "reliability_prior"
            base.update({f"loss.{name}.enabled": True,
                         f"loss.{name}.weight": t[f"{arm}_weight"],
                         f"loss.{name}.warmup_start": t["loss_warmup_start"],
                         f"loss.{name}.warmup_end": t["loss_warmup_end"]})
        if arm == "prior":
            prior = cfg["prior"]
            for key in ("root", "teacher_id", "teacher_settings_id"):
                if not prior[key]:
                    raise ValueError(f"prior.{key} must be set before planning a real-prior run")
                base[f"dataset.re10k.foundation_prior.{key}"] = (
                    local_path(prior[key]).as_posix() if key == "root" else prior[key])
        variants = [(arm + ("-pruned" if args.prune else "-full"), args.score,
                     ratio_value(args.ratio) if args.prune else 1.0, args.local, args.prune)]
    else:
        scores = cfg["evaluation"]["scores"] if args.scores is None else args.scores
        ratios = [ratio_value(r) for r in (args.ratios or cfg["evaluation"]["ratios"])]
        if set(scores) - set(SCORES) or len(set(scores)) != len(scores):
            raise ValueError("scores must be distinct supported scores")
        if not scores and args.baselines == "none":
            raise ValueError("eval-grid needs at least one baseline or score")
        if len(set(ratios)) != len(ratios):
            raise ValueError("ratios must be distinct")
        # Both baselines are intentional: OFF vs identity is a deployment gate,
        # not two independent scientific baselines.
        variants = [("baseline-off", "opacity", 1.0, False, False),
                    ("baseline-identity", "opacity", 1.0, False, True)]
        variants = {"both": variants, "off": variants[:1], "none": []}[args.baselines]
        for score in scores:
            for ratio in ratios:
                for local in ([False, True] if args.local == "both" else [args.local == "on"]):
                    variants.append((f"{score}-r{ratio:g}-local{int(local)}", score, ratio, local, True))
    jobs = []
    for name, score, ratio, local, enabled in variants:
        if enabled and score.startswith("reliability") and not cfg["reliability_enabled"]:
            raise ValueError("Reliability ranking requires reliability_enabled=true")
        directory = output / name
        settings = dict(base, **{
            "wandb.name": name, "hydra.run.dir": directory.as_posix(),
            "test.output_path": (directory / "renders").as_posix(),
            "model.gaussian_retention.enabled": enabled,
            "model.gaussian_retention.score_source": score,
            "model.gaussian_retention.keep_ratio": ratio,
            "model.gaussian_retention.local_cells.enabled": local,
        })
        preset = "retention_prior" if train and arm == "prior" else "retention_control"
        argv = [sys.executable, "-u", "-m", "src.main", f"+experiment={preset}"]
        if train and sampler == "random":
            argv.append("dataset@dataset.re10k=re10k")
        if train and arm == "legacy":
            argv.append("loss=[mse,lpips,reliability]")
            # The inherited experiment still contains a disabled prior override;
            # remove it so this arm has only the three selected loss wrappers.
            argv.append("~loss.reliability_prior")
        argv.extend(override(k, v) for k, v in settings.items())
        jobs.append({"name": name, "directory": str(directory), "argv": argv,
                     "score": score, "ratio": ratio, "local": local, "retention": enabled,
                     "status": "pending"})
    return {"schema_version": 1, "command": args.command, "arm": arm if train else None,
            "training": ({"finetune_scope": scope, "accumulate": accumulate, "sampler": sampler,
                          "effective_batch_size": t["batch_size"] * accumulate,
                          "optimizer_steps": base["trainer.max_steps"]} if train else None),
            "split": "train" if train else args.split, "checkpoint": str(checkpoint),
            "index": str(index), "settings": cfg, "output": str(output), "jobs": jobs}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(plan):
    """Local inputs only. No model imports, downloads, or GPU probing."""
    cfg = plan["settings"]
    for path in (Path(plan["checkpoint"]), Path(plan["index"])):
        if not path.is_file():
            raise ValueError(f"Missing file: {path}")
    if not local_path(cfg["data_root"]).is_dir():
        raise ValueError("data_root is not an existing directory")
    def read_index(path):
        index = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(index, dict) or not any(v is not None for v in index.values()):
            raise ValueError("Evaluation/validation index must contain valid scene entries")
        for scene, row in index.items():
            if row is None:
                continue
            if not isinstance(row, dict) or not isinstance(row.get("context"), list) or len(row["context"]) != 2:
                raise ValueError(f"{scene}: evaluation requires exactly two context indices")
            if not isinstance(row.get("target"), list) or not row["target"]:
                raise ValueError(f"{scene}: missing evaluation target indices")
            if any(type(i) is not int or i < 0 for i in row["context"] + row["target"]):
                raise ValueError(f"{scene}: invalid frame indices")
        return index
    validation = read_index(local_path(cfg["validation_index"]))
    held_out = read_index(local_path(cfg["evaluation_index"]))
    validation_scenes = {s for s, row in validation.items() if row is not None}
    test_scenes = {s for s, row in held_out.items() if row is not None}
    # val64 and Test 500 were split from the official index, so it overlaps them by design.
    if plan["split"] != "full" and validation_scenes & test_scenes:
        raise ValueError("Validation and held-out test scene IDs overlap")
    if plan["command"] == "train" and plan["training"]["sampler"] == "fixed":
        from src.foundation.pairs import load_training_pairs
        pairs = load_training_pairs(local_path(cfg["train_pairs"]))
        if set(pairs) & (validation_scenes | test_scenes):
            raise ValueError("Training and validation/test scene IDs overlap")
        if plan["arm"] == "prior" and not local_path(cfg["prior"]["root"]).is_dir():
            raise ValueError("Real prior cache directory does not exist")


def save_manifest(path, plan):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def execute(plan, run=subprocess.run):
    preflight(plan)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)  # never reuse even a failed run
    manifest = output / "manifest.json"
    plan["checkpoint_sha256"] = file_hash(plan["checkpoint"])
    plan["index_sha256"] = file_hash(plan["index"])
    plan["validation_index_sha256"] = file_hash(local_path(plan["settings"]["validation_index"]))
    plan["evaluation_index_sha256"] = file_hash(local_path(plan["settings"]["evaluation_index"]))
    if plan["command"] == "train" and plan["training"]["sampler"] == "fixed":
        plan["train_pairs_sha256"] = file_hash(local_path(plan["settings"]["train_pairs"]))
    try:
        plan["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        plan["git_status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        plan["git_commit"] = None
    save_manifest(manifest, plan)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan["settings"]["gpu"]))
    for job in plan["jobs"]:
        job["status"] = "running"
        save_manifest(manifest, plan)
        try:
            print(f"Running {job['name']} (log: {output / (job['name'] + '.log')})", flush=True)
            with (output / f"{job['name']}.log").open("x", encoding="utf-8") as log:
                result = run(job["argv"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            job["returncode"] = result.returncode
            job["status"] = "completed" if result.returncode == 0 else "failed"
        except (Exception, KeyboardInterrupt) as error:
            job["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            job["error"] = str(error)
            save_manifest(manifest, plan)
            raise
        save_manifest(manifest, plan)
        if result.returncode:
            return result.returncode  # fail fast; remaining jobs stay pending
    return 0


def summarize(root, output):
    """Keep failed/missing results visible; never invent zero-valued metrics."""
    rows = []
    for path in sorted(Path(root).rglob("manifest.json")):
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("schema_version") != 1 or plan.get("command") != "eval-grid":
            continue
        for job in plan["jobs"]:
            row = {"run": path.parent.name, "job": job["name"], "status": job["status"],
                   "split": plan["split"], "checkpoint": plan["checkpoint"],
                   "checkpoint_sha256": plan.get("checkpoint_sha256", ""),
                   "index_sha256": plan.get("index_sha256", ""),
                   "seed": plan["settings"]["seed"],
                   "align_pose": plan["settings"]["evaluation"]["align_pose"],
                   "score": job["score"], "ratio": job["ratio"], "local": job["local"],
                   "retention": job["retention"], "metrics_status": "missing"}
            # Keep complete settings inspectable rather than silently pooling
            # results with different OLP/local/pose-alignment hyperparameters.
            row["manifest"] = str(path.resolve())
            metrics = Path(job["directory"]) / "test_metrics.json"
            if metrics.is_file():
                try:
                    values = json.loads(metrics.read_text(encoding="utf-8"))
                    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
                        raise ValueError("Expected one test dataloader's metric dictionary")
                    mapping = {"psnr": "psnr_ours", "ssim": "ssim_ours", "lpips": "lpips_ours",
                               "input_count": "eval/retention_input_count",
                               "output_count": "eval/retention_output_count"}
                    for key, source in mapping.items():
                        val = values[0].get(source)
                        if isinstance(val, (float, int)) and not isinstance(val, bool) and math.isfinite(val):
                            row[key] = val
                    row["metrics_status"] = "ok" if all(k in row for k in ("psnr", "ssim", "lpips")) else "incomplete"
                except (ValueError, TypeError):
                    row["metrics_status"] = "invalid"
            rows.append(row)
    if not rows:
        raise ValueError("No retention eval-grid manifests found")
    columns = list(rows[0].keys())
    columns = list(dict.fromkeys(columns + ["psnr", "ssim", "lpips", "input_count", "output_count"]))
    with Path(output).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default=str(TEMPLATE))
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("train", "eval-grid"):
        p = sub.add_parser(name)
        p.add_argument("--execute", action="store_true", help="Launch jobs; default only prints commands")
        p.add_argument("--run-id", help="New unique output directory name (never reused)")
        p.add_argument("--checkpoint", help="Override initial/evaluation checkpoint")
        if name == "train":
            p.add_argument("--arm", choices=("control", "legacy", "prior"), default="control")
            p.add_argument("--steps", type=int)
            p.add_argument("--train-scope", choices=("full", "heads"))
            p.add_argument("--accumulate", type=int)
            p.add_argument("--prune", action="store_true")
            p.add_argument("--ratio", type=float, default=.5)
            p.add_argument("--score", choices=SCORES, default="reliability")
            p.add_argument("--local", action="store_true")
            p.add_argument("--budget-weight", type=float, default=0.0,
                           help="Push reliability of Gaussians outside the budget toward r_min")
            p.add_argument("--soft-steps", type=int, default=0,
                           help="Render all Gaussians for this many steps before hard pruning")
            p.add_argument("--budget-warmup", action="store_true",
                           help="Ramp the budget weight from 0 over the soft steps")
            p.add_argument("--budget-hard-scale", type=float, default=1.0,
                           help="Budget weight multiplier once hard pruning starts")
        else:
            p.add_argument("--split", choices=("validation", "test", "full"), default="validation",
                           help="full = official RE10K test index (comparable to published results)")
            p.add_argument("--scores", nargs="*", choices=SCORES)
            p.add_argument("--baselines", choices=("both", "off", "none"), default="both",
                           help="Full-count baselines to run before the score variants")
            p.add_argument("--ratios", nargs="+", type=float)
            p.add_argument("--local", choices=("off", "on", "both"), default="off")
    p = sub.add_parser("summarize")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True, help="NEW CSV path; existing files are never overwritten")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "summarize":
            print(f"Wrote {summarize(args.root, args.output)} rows to {args.output}")
            return 0
        plan = build_plan(read_settings(args.config), args)
        print(f"{'EXECUTE' if args.execute else 'PREVIEW ONLY'}: {len(plan['jobs'])} job(s) -> {plan['output']}")
        for job in plan["jobs"]:
            print(f"\n[{job['name']}]\n{shlex.join(job['argv'])}")
        if not args.execute:
            print("\nNo files written or jobs started. Add --execute on the configured GPU server.")
            return 0
        return execute(plan)
    except (ValueError, OSError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
