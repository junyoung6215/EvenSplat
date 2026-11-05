"""Sequential launcher for pruning-aware fine-tuning and retention evaluation.

Planning needs only PyYAML and the standard library, never imports the model,
and never writes files. Each training or evaluation job runs ``src.main`` in its
own subprocess on one GPU.
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
SCORES = ("random", "base_opacity", "opacity", "reliability", "r_filter", "grid")
FULL_TEST_INDEX = "assets/evaluation_index_re10k.json"
PRESET = "+experiment=retention"


def read_settings(path):
    defaults = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    supplied = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

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


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def ratio_value(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Ratios must be finite and in (0, 1]")
    return value


def cell_size_value(value):
    if type(value) is not int or value < 0:
        raise ValueError("cell size must be a nonnegative integer (0 = one global top-k)")
    return value


def validate_settings(cfg):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(cfg["gpu"])):
        raise ValueError("gpu must select exactly one device, e.g. '0' or a GPU UUID")
    t, e = cfg["train"], cfg["evaluation"]
    for name, value in (("olp.enabled", cfg["olp"]["enabled"]),
                        ("reliability.enabled", cfg["reliability"]["enabled"]),
                        ("train.validation_videos", t["validation_videos"]),
                        ("evaluation.align_pose", e["align_pose"]),
                        ("evaluation.save_image", e["save_image"]),
                        ("evaluation.per_scene_stats", e["per_scene_stats"])):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a YAML boolean, not a string")
    nonnegative(t["lr"], "train.lr")
    nonnegative(t["backbone_lr_multiplier"], "train.backbone_lr_multiplier")
    if not 0 <= cfg["olp"]["mix"] <= 1 or not 0 <= cfg["reliability"]["r_min"] < 1:
        raise ValueError("Invalid olp.mix or reliability.r_min")
    if not 0 <= cfg["retention"]["r_threshold"] <= 1:
        raise ValueError("retention.r_threshold must be in [0, 1]")
    cell_size_value(cfg["retention"]["cell_size"])
    for key in ("steps", "batch_size", "accumulate", "validation_interval", "checkpoint_interval"):
        positive_int(t[key], f"train.{key}")
    if type(t["workers"]) is not int or t["workers"] < 0:
        raise ValueError("train.workers must be a nonnegative integer")
    if t["finetune_scope"] not in ("full", "heads"):
        raise ValueError("train.finetune_scope must be full or heads")


def validate_train_args(args):
    if not math.isfinite(args.budget_weight) or args.budget_weight < 0:
        raise ValueError("--budget-weight must be finite and nonnegative")
    if type(args.soft_steps) is not int or args.soft_steps < 0:
        raise ValueError("--soft-steps must be a nonnegative integer")
    if not math.isfinite(args.budget_hard_scale) or args.budget_hard_scale < 0:
        raise ValueError("--budget-hard-scale must be finite and nonnegative")
    if not math.isfinite(args.ratio_min) or not 0 <= args.ratio_min <= args.ratio:
        raise ValueError("--ratio-min must be in [0, --ratio]")
    if args.ratio_min > 0 and not args.prune:
        raise ValueError("--ratio-min requires --prune")
    if args.budget_warmup and args.soft_steps == 0:
        raise ValueError("--budget-warmup ramps over --soft-steps, which must be positive")
    if (args.budget_weight > 0 or args.soft_steps > 0) and not (args.prune and args.score == "reliability"):
        raise ValueError("--budget-weight/--soft-steps need --prune --score reliability")


def base_overrides(cfg, train, checkpoint, index, cell_size):
    """Hydra overrides shared by every training and evaluation job."""
    t, e = cfg["train"], cfg["evaluation"]
    return {
        "mode": "train" if train else "test",
        "wandb.mode": "disabled", "seed": cfg["seed"],
        "dataset.re10k.roots": [local_path(cfg["data_root"]).as_posix()],
        "dataset.re10k.view_sampler.evaluation_index_path": index.as_posix(),
        # EncoderNoPoSplatCfg.pretrained_weights is a str; null fails typed config load.
        "model.encoder.pretrained_weights": checkpoint.as_posix() if train else "",
        "checkpointing.load": None if train else checkpoint.as_posix(),
        "checkpointing.save_weights_only": False,
        "model.encoder.orthogonal_preconditioner.enabled": cfg["olp"]["enabled"],
        "model.encoder.orthogonal_preconditioner.seed": cfg["olp"]["seed"],
        "model.encoder.orthogonal_preconditioner.mix": cfg["olp"]["mix"],
        "model.encoder.orthogonal_preconditioner.target_mix": cfg["olp"]["mix"],
        "model.encoder.orthogonal_preconditioner.warmup_steps": 0,
        "model.encoder.gaussian_reliability.enabled": cfg["reliability"]["enabled"],
        "model.encoder.gaussian_reliability.r_min": cfg["reliability"]["r_min"],
        "model.encoder.gaussian_reliability.detach_gate": False,
        "test.align_pose": e["align_pose"],
        "test.pose_align_steps": e["pose_align_steps"],
        "test.save_image": e["save_image"],
        "test.save_video": False, "test.save_compare": False,
        # Per-scene metrics and OLP/Gaussian statistics (olp_stats_per_scene.csv).
        "test.save_olp_stats": e["per_scene_stats"],
        "model.gaussian_retention.random_seed": cfg["seed"],
        "model.gaussian_retention.r_threshold": cfg["retention"]["r_threshold"],
        "model.gaussian_retention.cell_size": cell_size,
        "model.gaussian_retention.start_step": 0,
        "model.gaussian_retention.diagnostics_every_n_steps": 1,
        "data_loader.test.num_workers": t["workers"],
        "data_loader.test.persistent_workers": False,
    }


def train_overrides(cfg, args):
    """Fine-tuning overrides and a summary of the optimization schedule."""
    validate_train_args(args)
    t = cfg["train"]
    scope = args.train_scope or t["finetune_scope"]
    accumulate = positive_int(args.accumulate if args.accumulate is not None else t["accumulate"],
                              "accumulate")
    steps = positive_int(args.steps if args.steps is not None else t["steps"], "steps")
    overrides = {
        # Original NoPoSplat bounded random pairs over the whole train split.
        # Warm-up 0: fine-tuning starts from a converged checkpoint, so use the
        # final 45-90 context gap immediately instead of re-ramping from 25.
        "dataset.re10k.view_sampler.warm_up_steps": 0,
        "trainer.max_steps": steps,
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
        "model.gaussian_retention.train_keep_ratio_min": args.ratio_min,
    }
    summary = {"finetune_scope": scope, "accumulate": accumulate,
               "effective_batch_size": t["batch_size"] * accumulate, "optimizer_steps": steps}
    return overrides, summary


def eval_variants(cfg, args, cell_size):
    """(job name, score, keep ratio, retention enabled) for every eval-grid job."""
    e = cfg["evaluation"]
    scores = e["scores"] if args.scores is None else args.scores
    ratios = [ratio_value(r) for r in (args.ratios or e["ratios"])]
    if set(scores) - set(SCORES) or len(set(scores)) != len(scores):
        raise ValueError("scores must be distinct supported scores")
    if not scores and args.baselines == "none":
        raise ValueError("eval-grid needs at least one baseline or score")
    if len(set(ratios)) != len(ratios):
        raise ValueError("ratios must be distinct")
    # Both baselines are intentional: OFF vs identity is a deployment gate,
    # not two independent scientific baselines.
    baselines = [("baseline-off", "opacity", 1.0, False),
                 ("baseline-identity", "opacity", 1.0, True)]
    variants = {"both": baselines, "off": baselines[:1], "none": []}[args.baselines]
    return variants + [(f"{score}-r{ratio:g}-cell{cell_size}", score, ratio, True)
                       for score in scores for ratio in ratios]


def build_plan(cfg, args):
    """Build JSON-serializable jobs without touching disk or loading a checkpoint."""
    validate_settings(cfg)
    train = args.command == "train"
    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_id):
        raise ValueError("run-id must be a short alphanumeric/underscore/hyphen name")
    output = local_path(cfg["output_root"]) / run_id
    checkpoint = local_path(args.checkpoint or cfg["initial_checkpoint"])
    split = "train" if train else args.split
    if split == "full":
        # Official RE10K test index, as reported by NoPoSplat and others.
        index = local_path(FULL_TEST_INDEX)
    else:
        index = local_path(cfg["evaluation_index" if split == "test" else "validation_index"])
    cell_size = cell_size_value(cfg["retention"]["cell_size"] if args.cell_size is None else args.cell_size)
    base = base_overrides(cfg, train, checkpoint, index, cell_size)
    training = None
    if train:
        extra, training = train_overrides(cfg, args)
        base.update(extra)
        variants = [("pruned" if args.prune else "full", args.score,
                     ratio_value(args.ratio) if args.prune else 1.0, args.prune)]
    else:
        variants = eval_variants(cfg, args, cell_size)
    jobs = []
    for name, score, ratio, enabled in variants:
        if enabled and score in ("reliability", "r_filter") and not cfg["reliability"]["enabled"]:
            raise ValueError("Reliability ranking requires reliability.enabled=true")
        if enabled and score == "grid" and (cell_size <= 0 or cell_size & (cell_size - 1)):
            raise ValueError("The grid score needs a power-of-two cell size")
        directory = output / name
        settings = dict(base, **{
            "wandb.name": name, "hydra.run.dir": directory.as_posix(),
            "test.output_path": (directory / "renders").as_posix(),
            "model.gaussian_retention.enabled": enabled,
            "model.gaussian_retention.score_source": score,
            "model.gaussian_retention.keep_ratio": ratio,
        })
        argv = [sys.executable, "-u", "-m", "src.main", PRESET]
        argv.extend(override(k, v) for k, v in settings.items())
        jobs.append({"name": name, "directory": str(directory), "argv": argv, "score": score,
                     "ratio": ratio, "cell_size": cell_size, "retention": enabled,
                     "status": "pending"})
    return {"schema_version": 2, "command": args.command, "training": training, "split": split,
            "checkpoint": str(checkpoint), "index": str(index), "settings": cfg,
            "output": str(output), "jobs": jobs}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_index(path):
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not any(v is not None for v in index.values()):
        raise ValueError(f"{path}: index must contain valid scene entries")
    for scene, row in index.items():
        if row is None:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("context"), list) or len(row["context"]) != 2:
            raise ValueError(f"{scene}: evaluation requires exactly two context indices")
        if not isinstance(row.get("target"), list) or not row["target"]:
            raise ValueError(f"{scene}: missing evaluation target indices")
        if any(type(i) is not int or i < 0 for i in row["context"] + row["target"]):
            raise ValueError(f"{scene}: invalid frame indices")
    return {scene for scene, row in index.items() if row is not None}


def preflight(plan):
    """Local inputs only. No model imports, downloads, or GPU probing."""
    cfg = plan["settings"]
    for path in (Path(plan["checkpoint"]), Path(plan["index"])):
        if not path.is_file():
            raise ValueError(f"Missing file: {path}")
    if not local_path(cfg["data_root"]).is_dir():
        raise ValueError("data_root is not an existing directory")
    validation = read_index(local_path(cfg["validation_index"]))
    held_out = read_index(local_path(cfg["evaluation_index"]))
    # val64 and Test 500 were split from the official index, so it overlaps them by design.
    if plan["split"] != "full" and validation & held_out:
        raise ValueError("Validation and held-out test scene IDs overlap")


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


METRICS = {"psnr": "psnr_ours", "ssim": "ssim_ours", "lpips": "lpips_ours",
           "input_count": "eval/retention_input_count",
           "output_count": "eval/retention_output_count"}


def summarize(root, output):
    """Keep failed/missing results visible; never invent zero-valued metrics."""
    rows = []
    for path in sorted(Path(root).rglob("manifest.json")):
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("schema_version") != 2 or plan.get("command") != "eval-grid":
            continue
        for job in plan["jobs"]:
            row = {"run": path.parent.name, "job": job["name"], "status": job["status"],
                   "split": plan["split"], "checkpoint": plan["checkpoint"],
                   "checkpoint_sha256": plan.get("checkpoint_sha256", ""),
                   "index_sha256": plan.get("index_sha256", ""),
                   "seed": plan["settings"]["seed"],
                   "align_pose": plan["settings"]["evaluation"]["align_pose"],
                   "score": job["score"], "ratio": job["ratio"], "cell_size": job["cell_size"],
                   "retention": job["retention"], "metrics_status": "missing",
                   "manifest": str(path.resolve())}
            metrics = Path(job["directory"]) / "test_metrics.json"
            if metrics.is_file():
                try:
                    values = json.loads(metrics.read_text(encoding="utf-8"))
                    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
                        raise ValueError("Expected one test dataloader's metric dictionary")
                    for key, source in METRICS.items():
                        val = values[0].get(source)
                        if isinstance(val, (float, int)) and not isinstance(val, bool) and math.isfinite(val):
                            row[key] = val
                    row["metrics_status"] = "ok" if all(k in row for k in ("psnr", "ssim", "lpips")) else "incomplete"
                except (ValueError, TypeError):
                    row["metrics_status"] = "invalid"
            rows.append(row)
    if not rows:
        raise ValueError("No retention eval-grid manifests found")
    columns = list(dict.fromkeys(list(rows[0]) + list(METRICS)))
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
        p.add_argument("--cell-size", type=int,
                       help="Stratified cell size in pixels; 0 = one global top-k (default: settings)")
        if name == "train":
            p.add_argument("--steps", type=int)
            p.add_argument("--train-scope", choices=("full", "heads"))
            p.add_argument("--accumulate", type=int)
            p.add_argument("--prune", action="store_true", help="Prune while training (pruning-aware FT)")
            p.add_argument("--ratio", type=float, default=.5)
            p.add_argument("--ratio-min", type=float, default=0.0,
                           help="Draw each step's keep ratio from [ratio-min, ratio]; 0 keeps --ratio fixed")
            p.add_argument("--score", choices=SCORES, default="r_filter")
            p.add_argument("--budget-weight", type=float, default=0.0,
                           help="Learned r: push reliability of Gaussians outside the budget toward r_min")
            p.add_argument("--soft-steps", type=int, default=0,
                           help="Learned r: render all Gaussians for this many steps before pruning")
            p.add_argument("--budget-warmup", action="store_true",
                           help="Learned r: ramp the budget weight from 0 over the soft steps")
            p.add_argument("--budget-hard-scale", type=float, default=1.0,
                           help="Learned r: budget weight multiplier once pruning starts")
        else:
            p.add_argument("--split", choices=("validation", "test", "full"), default="validation",
                           help="full = official RE10K test index (comparable to published results)")
            p.add_argument("--scores", nargs="*", choices=SCORES)
            p.add_argument("--baselines", choices=("both", "off", "none"), default="both",
                           help="Full-count baselines to run before the score variants")
            p.add_argument("--ratios", nargs="+", type=float)
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
