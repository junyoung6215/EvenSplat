import json
import os
import signal
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

from src.model.distiller import get_distiller

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.foundation.validation import validate_prior_pipeline
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.checkpoint_audit import load_encoder_pretrained_weights
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.training_scope import validate_training_initialization, training_scope_report


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {key: jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    validate_prior_pipeline(cfg)
    validate_training_initialization(
        cfg.train.finetune_scope, cfg.mode,
        cfg.model.encoder.pretrained_weights, cfg.checkpointing.load,
    )
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []

    # Get the Slurm job ID from enviroment variable
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
    tags = cfg_dict.wandb.get("tags", [])
    tags += [f"job_id={slurm_job_id}"] if slurm_job_id != "unknown" else []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.name})",
            tags=tags,
            log_model=False,
            save_dir=output_dir,
            notes=f"outputs/{output_dir.parent.name}/{output_dir.name}",
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger(output_dir / "local")

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            save_last="link",
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        max_steps=cfg.trainer.max_steps,
        limit_test_batches=cfg.trainer.limit_test_batches,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    distiller = None
    if cfg.train.distiller:
        distiller = get_distiller(cfg.train.distiller)
        distiller = distiller.eval()

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        load_encoder_pretrained_weights(
            encoder,
            cfg.model.encoder.pretrained_weights,
            output_dir / "pretrained_load_audit.json",
            strict=False,
            require_backbone=cfg.train.finetune_scope == "heads",
        )

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        distiller=distiller,
        anchor_selection_cfg=cfg.model.anchor_selection,
        gaussian_retention_cfg=cfg.model.gaussian_retention,
        debug_cfg=cfg.debug,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        if trainer.global_rank == 0:
            scope_report = training_scope_report(encoder, cfg.train.finetune_scope)
            (output_dir / "training_scope.json").write_text(
                json.dumps(scope_report, indent=2) + "\n", encoding="utf-8"
            )
            print(cyan(
                f"Training scope={cfg.train.finetune_scope}: "
                f"{scope_report['trainable_parameters']:,} trainable / "
                f"{scope_report['frozen_parameters']:,} frozen encoder parameters."
            ))
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        metrics = trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )
        if trainer.global_rank == 0:
            metrics_path = output_dir / "test_metrics.json"
            with metrics_path.open("w") as f:
                json.dump(jsonable(metrics), f, indent=2)
                f.write("\n")
            print(cyan(f"Wrote test metrics to {metrics_path}."))


if __name__ == "__main__":
    train()
