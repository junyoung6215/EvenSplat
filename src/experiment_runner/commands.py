from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

from .config import get_cfg, hydra_value
from .paths import RunPaths

Command = list[str]


def command_line(command: Command) -> str:
    return shlex.join(command)


def python_bin(config: dict[str, Any]) -> str:
    return str(get_cfg(config, "python_bin", sys.executable))


def require(config: dict[str, Any], key: str) -> Any:
    value = get_cfg(config, key)
    if value is None:
        raise SystemExit(f"Missing required preset key: {key}")
    return value


def build_train_command(config: dict[str, Any], run_dir: Path) -> Command:
    train = get_cfg(config, "train", {})
    olp = get_cfg(config, "olp", {})
    data_loader = get_cfg(config, "data_loader", {})
    warmup_steps = get_cfg(olp, "warmup_steps", get_cfg(train, "warmup_steps", 0))

    return [
        python_bin(config),
        "-u",
        "-m",
        "src.main",
        "+experiment=re10k",
        "mode=train",
        f"wandb.mode={get_cfg(config, 'wandb_mode', 'disabled')}",
        f"wandb.name={require(config, 'name')}",
        "model.encoder.pretrained_weights="
        f"{get_cfg(config, 'pretrained_weights', './pretrained_weights/re10k.ckpt')}",
        f"checkpointing.load={hydra_value(get_cfg(train, 'checkpoint_load'))}",
        f"dataset.re10k.roots=[{get_cfg(config, 'data_root', 'datasets/re10k')}]",
        "model.encoder.orthogonal_preconditioner.enabled="
        f"{hydra_value(get_cfg(olp, 'enabled', True))}",
        f"model.encoder.orthogonal_preconditioner.mode={get_cfg(olp, 'mode', 'hadamard')}",
        f"model.encoder.orthogonal_preconditioner.dim={get_cfg(olp, 'dim', 256)}",
        f"model.encoder.orthogonal_preconditioner.mix={get_cfg(olp, 'mix', 0.0)}",
        "model.encoder.orthogonal_preconditioner.target_mix="
        f"{get_cfg(olp, 'target_mix', 1.0)}",
        f"model.encoder.orthogonal_preconditioner.warmup_steps={warmup_steps}",
        f"model.encoder.orthogonal_preconditioner.seed={get_cfg(olp, 'seed', 0)}",
        "model.encoder.orthogonal_preconditioner.collect_stats="
        f"{hydra_value(get_cfg(olp, 'collect_stats', True))}",
        f"trainer.max_steps={require(config, 'train.max_steps')}",
        f"trainer.num_sanity_val_steps={get_cfg(train, 'num_sanity_val_steps', 0)}",
        f"trainer.val_check_interval={get_cfg(train, 'val_check_interval', 100000000)}",
        f"trainer.accumulate_grad_batches={get_cfg(train, 'accumulate_grad_batches', 1)}",
        f"optimizer.lr={require(config, 'train.lr')}",
        f"data_loader.train.batch_size={require(config, 'train.batch_size')}",
        f"data_loader.train.num_workers={get_cfg(data_loader, 'train_num_workers', 16)}",
        f"data_loader.val.num_workers={get_cfg(data_loader, 'val_num_workers', 1)}",
        f"data_loader.test.num_workers={get_cfg(data_loader, 'test_num_workers', 4)}",
        f"checkpointing.every_n_train_steps={get_cfg(train, 'ckpt_every', 5000)}",
        f"checkpointing.save_top_k={get_cfg(train, 'save_top_k', 1)}",
        "checkpointing.save_weights_only="
        f"{hydra_value(get_cfg(train, 'save_weights_only', False))}",
        f"train.print_log_every_n_steps={get_cfg(train, 'print_log_every_n_steps', 100)}",
        f"hydra.run.dir={run_dir}",
    ]


def build_eval_command(
    config: dict[str, Any],
    *,
    run_dir: Path,
    checkpoint: Path,
    run_name: str,
    index_path: Path,
    output_path: Path,
    save_image: bool,
    save_olp_stats: bool,
    olp_enabled: bool,
    olp_mix: float,
    olp_target_mix: float,
) -> Command:
    eval_cfg = get_cfg(config, "eval", {})
    olp = get_cfg(config, "olp", {})

    return [
        python_bin(config),
        "-u",
        "-m",
        "src.main",
        "+experiment=re10k",
        "mode=test",
        f"wandb.mode={get_cfg(config, 'wandb_mode', 'disabled')}",
        f"wandb.name={run_name}",
        "dataset/view_sampler@dataset.re10k.view_sampler=evaluation",
        f"dataset.re10k.view_sampler.index_path={index_path}",
        f"dataset.re10k.roots=[{get_cfg(config, 'data_root', 'datasets/re10k')}]",
        f"checkpointing.load={checkpoint}",
        f"model.encoder.orthogonal_preconditioner.enabled={hydra_value(olp_enabled)}",
        f"model.encoder.orthogonal_preconditioner.mode={get_cfg(olp, 'mode', 'hadamard')}",
        f"model.encoder.orthogonal_preconditioner.dim={get_cfg(olp, 'dim', 256)}",
        f"model.encoder.orthogonal_preconditioner.mix={olp_mix}",
        f"model.encoder.orthogonal_preconditioner.target_mix={olp_target_mix}",
        "model.encoder.orthogonal_preconditioner.warmup_steps=0",
        f"model.encoder.orthogonal_preconditioner.seed={get_cfg(olp, 'seed', 0)}",
        "model.encoder.orthogonal_preconditioner.collect_stats="
        f"{hydra_value(get_cfg(olp, 'collect_stats', True))}",
        f"test.save_image={hydra_value(save_image)}",
        f"test.save_olp_stats={hydra_value(save_olp_stats)}",
        f"test.olp_tau_alpha={get_cfg(eval_cfg, 'olp_tau_alpha', 0.95)}",
        f"test.olp_tau_scale={hydra_value(get_cfg(eval_cfg, 'olp_tau_scale'))}",
        f"test.olp_tau_volume={hydra_value(get_cfg(eval_cfg, 'olp_tau_volume'))}",
        "test.olp_adaptive_tau_scale_quantile="
        f"{get_cfg(eval_cfg, 'olp_adaptive_tau_scale_quantile', 0.95)}",
        f"test.output_path={output_path}",
        f"hydra.run.dir={run_dir}",
    ]


def build_compare_commands(
    config: dict[str, Any],
    paths: RunPaths,
    *,
    baseline_ckpt: Path | None,
    olp_ckpt: Path | None,
) -> list[Command]:
    compare = get_cfg(config, "compare", {})
    baseline = baseline_ckpt or Path(
        str(get_cfg(config, "checkpoints.baseline", "./pretrained_weights/re10k.ckpt"))
    )
    olp = olp_ckpt or get_cfg(config, "checkpoints.olp")
    if olp is None:
        raise SystemExit(
            "Compare stage requires --olp-ckpt or checkpoints.olp in the preset."
        )
    olp = Path(str(olp))

    metric_count = int(get_cfg(compare, "metric_count", 2000))
    vis_count = int(get_cfg(compare, "vis_count", 10))
    vis_targets = int(get_cfg(compare, "vis_targets_per_scene", 1))
    source_index = Path(str(get_cfg(compare, "source_index", "assets/evaluation_index_re10k.json")))
    metric_index = paths.root / f"evaluation_index_re10k_metric{metric_count}.json"
    vis_index = paths.root / f"evaluation_index_re10k_vis{vis_count}.json"
    metric_name = str(get_cfg(compare, "metric_run_name", f"re10k_olp_metric{metric_count}"))
    vis_original = str(
        get_cfg(compare, "vis_original_run_name", f"re10k_noposplat_vis{vis_count}")
    )
    vis_baseline = str(
        get_cfg(compare, "vis_baseline_run_name", f"re10k_olp_noft_vis{vis_count}")
    )
    vis_olp = str(get_cfg(compare, "vis_olp_run_name", f"re10k_olp_ft_vis{vis_count}"))
    side_by_side = paths.test_outputs / str(
        get_cfg(
            compare,
            "side_by_side_name",
            f"re10k_noposplat_vs_olp_noft_vs_olp_ft_vis{vis_count}",
        )
    )

    return [
        [
            python_bin(config),
            "scripts/make_eval_subset.py",
            "--input",
            str(source_index),
            "--output",
            str(metric_index),
            "--count",
            str(metric_count),
        ],
        [
            python_bin(config),
            "scripts/make_eval_subset.py",
            "--input",
            str(source_index),
            "--output",
            str(vis_index),
            "--count",
            str(vis_count),
            "--targets-per-scene",
            str(vis_targets),
        ],
        build_eval_command(
            config,
            run_dir=paths.root / metric_name,
            checkpoint=olp,
            run_name=metric_name,
            index_path=metric_index,
            output_path=paths.test_outputs,
            save_image=False,
            save_olp_stats=True,
            olp_enabled=True,
            olp_mix=1.0,
            olp_target_mix=1.0,
        ),
        build_eval_command(
            config,
            run_dir=paths.root / vis_original,
            checkpoint=baseline,
            run_name=vis_original,
            index_path=vis_index,
            output_path=paths.test_outputs,
            save_image=True,
            save_olp_stats=False,
            olp_enabled=False,
            olp_mix=0.0,
            olp_target_mix=0.0,
        ),
        build_eval_command(
            config,
            run_dir=paths.root / vis_baseline,
            checkpoint=baseline,
            run_name=vis_baseline,
            index_path=vis_index,
            output_path=paths.test_outputs,
            save_image=True,
            save_olp_stats=True,
            olp_enabled=True,
            olp_mix=1.0,
            olp_target_mix=1.0,
        ),
        build_eval_command(
            config,
            run_dir=paths.root / vis_olp,
            checkpoint=olp,
            run_name=vis_olp,
            index_path=vis_index,
            output_path=paths.test_outputs,
            save_image=True,
            save_olp_stats=True,
            olp_enabled=True,
            olp_mix=1.0,
            olp_target_mix=1.0,
        ),
        [
            python_bin(config),
            "-u",
            "-m",
            "src.scripts.compute_metrics",
            "+experiment=re10k",
            "+evaluation=re10k",
            f"dataset.re10k.view_sampler.index_path={vis_index}",
            f"dataset.re10k.roots=[{get_cfg(config, 'data_root', 'datasets/re10k')}]",
            (
                "evaluation.methods="
                f"[{{name:NoPoSplat,key:noposplat,path:{paths.test_outputs / vis_original}}},"
                f"{{name:OLP-NoFT,key:olp_noft,path:{paths.test_outputs / vis_baseline}}},"
                f"{{name:OLP-FT,key:olp_ft,path:{paths.test_outputs / vis_olp}}}]"
            ),
            f"evaluation.side_by_side_path={side_by_side}",
            "evaluation.animate_side_by_side=false",
            f"evaluation.output_metrics_path={side_by_side / 'metrics.json'}",
            f"hydra.run.dir={paths.root / 'side_by_side'}",
        ],
    ]
