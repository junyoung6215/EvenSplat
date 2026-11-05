from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import torch.nn.functional as F
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim

from ..analysis.olp_eval_recorder import OlpEvalRecorder
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..geometry.projection import project
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_reliability import validate_reliability_loss_encoder
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .retention import (
    GaussianGridLayout,
    GaussianRetention,
    GaussianRetentionCfg,
    RetentionResult,
    budget_loss,
    budget_weight,
    distribution_statistics,
    flatten_gaussian_scores,
    gaussian_cell_ids,
    score_statistics,
    selection_mask,
)
from .types import Gaussians
from .training_scope import (
    CHECKPOINT_SCOPE_KEY,
    apply_training_scope,
    validate_resume_scope,
)


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    save_olp_stats: bool = False
    olp_tau_alpha: float = 0.95
    olp_tau_scale: float | None = None
    olp_tau_volume: float | None = None
    olp_adaptive_tau_scale_quantile: float | None = 0.95
    olp_raw_dump: bool = False
    olp_raw_dump_max_locations: int = 256
    olp_raw_dump_full_maps: bool = False


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    # Preserve legacy visualization unless a retention experiment opts out.
    validation_videos: bool = True
    finetune_scope: str = "full"


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        distiller: Optional[nn.Module] = None,
        gaussian_retention_cfg: GaussianRetentionCfg | None = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.gaussian_retention = GaussianRetention(gaussian_retention_cfg)
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        apply_training_scope(self.encoder, self.train_cfg.finetune_scope)
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        validate_reliability_loss_encoder(self.encoder, self.losses)
        if any(
            bool(getattr(module, "reliability_enabled", False))
            for module in self.encoder.modules()
        ):
            self.strict_loading = False

        self.distiller = distiller
        self.distiller_loss = None
        if self.distiller is not None:
            convert_to_buffer(self.distiller, persistent=False)
            self.distiller_loss = Regr3D()

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.olp_eval_recorder = OlpEvalRecorder(
            tau_alpha=test_cfg.olp_tau_alpha,
            tau_scale=test_cfg.olp_tau_scale,
            tau_volume=test_cfg.olp_tau_volume,
            adaptive_tau_scale_quantile=test_cfg.olp_adaptive_tau_scale_quantile,
            save_raw_dump=test_cfg.olp_raw_dump,
            raw_dump_max_locations=test_cfg.olp_raw_dump_max_locations,
            save_full_raw_maps=test_cfg.olp_raw_dump_full_maps,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Lightning switches back to train() after validation. Keep the frozen
        # Transformer deterministic while geometry/Gaussian/reliability heads train.
        if self.train_cfg.finetune_scope == "heads":
            self.encoder.backbone.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint[CHECKPOINT_SCOPE_KEY] = self.train_cfg.finetune_scope

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        trainer = getattr(self, "_trainer", None)
        fn = getattr(getattr(trainer, "state", None), "fn", None)
        if getattr(fn, "value", fn) == "fit":
            validate_resume_scope(checkpoint, self.train_cfg.finetune_scope)
            if self.train_cfg.finetune_scope == "heads":
                missing = [name for name, _ in self.encoder.backbone.named_parameters()
                           if f"encoder.backbone.{name}" not in checkpoint.get("state_dict", {})]
                if missing:
                    raise ValueError(f"Frozen backbone weights missing from resume checkpoint: {missing[:5]}")

    def _reliability_loss_active(self) -> bool:
        global_step = int(self.global_step)
        return any(
            getattr(loss_fn, "name", None) == "reliability"
            and callable(getattr(loss_fn, "is_active", None))
            and bool(loss_fn.is_active(global_step))
            for loss_fn in self.losses
        )

    def _retention_budget_active(self) -> bool:
        cfg = self.gaussian_retention.cfg
        return bool(cfg.enabled) and bool(cfg.apply_during_train) and float(cfg.budget_weight) > 0

    def _retention_needs_dump(self, stage: str) -> bool:
        return (
            self.gaussian_retention.applies(stage, int(self.global_step))
            and self.gaussian_retention.encoder_map is not None
        ) or (stage == "train" and self._retention_budget_active())

    def _retention_layout(self, gaussians: Gaussians, batch: BatchedExample) -> GaussianGridLayout:
        batch_size, views, _, height, width = batch["context"]["image"].shape
        if int(batch_size) != int(gaussians.opacities.shape[0]):
            raise RuntimeError("Context and Gaussian batch sizes do not match.")
        surfaces = int(getattr(getattr(self.encoder, "cfg", None), "num_surfaces", 1))
        base_count = int(views) * int(height) * int(width) * surfaces
        gaussian_count = int(gaussians.opacities.shape[1])
        if gaussian_count % base_count != 0:
            raise RuntimeError(
                "Cannot align flattened Gaussians with the source image grid: "
                f"G={gaussian_count}, VHW*surfaces={base_count}."
            )
        return GaussianGridLayout(
            int(views), int(height), int(width), surfaces, gaussian_count // base_count
        )

    def _retention_budget_loss(
        self,
        gaussians: Gaussians,
        batch: BatchedExample,
        visualization_dump: dict,
    ) -> Tensor:
        """Learned-r ablation: train the reliability head to rank for the budget."""

        cfg = self.gaussian_retention.cfg
        reliability = flatten_gaussian_scores(
            visualization_dump["reliability"], *gaussians.opacities.shape, name="reliability"
        ).to(gaussians.opacities.dtype)
        layout = self._retention_layout(gaussians, batch)
        cell_size = int(cfg.cell_size)
        cells = layout.cell_count(cell_size) if cell_size > 0 else 1
        loss, stats = budget_loss(
            reliability,
            gaussian_cell_ids(layout, cell_size, reliability.device),
            cells,
            float(cfg.keep_ratio),
            float(self.encoder.gaussian_reliability_cfg.r_min),
        )
        for name, value in stats.items():
            self.log(f"retention_budget/{name}", value)
        weight = budget_weight(
            float(cfg.budget_weight),
            int(self.global_step),
            int(cfg.start_step),
            bool(cfg.budget_warmup),
            float(cfg.budget_hard_scale),
        )
        self.log("retention_budget/weight", weight)
        return loss * weight

    def _log_retention_statistics(
        self,
        stage: str,
        result: RetentionResult,
        layout: GaussianGridLayout,
    ) -> None:
        prefix = "train" if stage == "train" else "eval"
        self.log(f"{prefix}/retention_input_count", float(result.input_count))
        self.log(f"{prefix}/retention_output_count", float(result.output_count))
        self.log(f"{prefix}/retention_kept_fraction", float(result.kept_fraction))
        cfg = self.gaussian_retention.cfg
        due = stage != "train" or int(self.global_step) % int(cfg.diagnostics_every_n_steps) == 0
        if not bool(cfg.log_diagnostics) or not due or result.scores is None:
            return
        stats = score_statistics(result.scores, result.indices)
        mask = selection_mask(result.scores, result.indices)
        stats.update(distribution_statistics(mask, layout, result.cell_ids))
        for name, value in stats.items():
            self.log(f"{prefix}/retention_{name}", value)

    def _apply_gaussian_retention(
        self,
        gaussians: Gaussians,
        batch: BatchedExample,
        visualization_dump: dict | None,
        stage: str,
        selection_id: int,
    ) -> RetentionResult:
        retention = self.gaussian_retention
        if not retention.applies(stage, int(self.global_step)):
            return retention.identity(gaussians)
        keep_ratio = None
        if stage == "train":
            keep_ratio = retention.train_keep_ratio(selection_id)
            if keep_ratio is not None:
                self.log("train/retention_keep_ratio", keep_ratio)
        world_size = max(int(getattr(getattr(self, "_trainer", None), "world_size", 1)), 1)
        layout = self._retention_layout(gaussians, batch)
        key = retention.encoder_map
        result = retention(
            gaussians,
            layout,
            encoder_map=None if key is None or visualization_dump is None else visualization_dump.get(key),
            # Distinct random selections per process under data parallelism.
            selection_id=int(selection_id) * world_size + int(self.global_rank),
            keep_ratio=keep_ratio,
        )
        self._log_retention_statistics(stage, result, layout)
        return result

    def _denormalize_context_image(self, image: Tensor) -> Tensor:
        cfg = getattr(self.encoder, "cfg", None)
        mean = getattr(cfg, "input_mean", (0.5, 0.5, 0.5))
        std = getattr(cfg, "input_std", (0.5, 0.5, 0.5))
        mean = torch.as_tensor(mean, dtype=image.dtype, device=image.device).view(
            1, 1, 3, 1, 1
        )
        std = torch.as_tensor(std, dtype=image.dtype, device=image.device).view(
            1, 1, 3, 1, 1
        )
        return (image * std + mean).clamp(0.0, 1.0)

    def _flatten_reliability_dump(
        self,
        visualization_dump: dict,
    ) -> tuple[Tensor, Tensor, Tensor]:
        means = visualization_dump["means"]
        reliability = visualization_dump["reliability"]
        opacity_before = visualization_dump["opacities_before_reliability"]

        means = rearrange(means, "b v h w g xyz -> b (v h w g) xyz")
        reliability = rearrange(reliability, "b v h w s one -> b (v h w s) one")
        opacity_before = rearrange(
            opacity_before, "b v h w s one -> b (v h w s) one"
        )

        if means.shape[1] != reliability.shape[1]:
            if means.shape[1] % reliability.shape[1] != 0:
                raise RuntimeError(
                    "Cannot align reliability gates with Gaussian centers: "
                    f"means={tuple(means.shape)}, reliability={tuple(reliability.shape)}."
                )
            repeat_factor = means.shape[1] // reliability.shape[1]
            reliability = reliability.repeat_interleave(repeat_factor, dim=1)
            opacity_before = opacity_before.repeat_interleave(repeat_factor, dim=1)

        return means, reliability, opacity_before

    def _attach_reliability_intermediates(
        self,
        batch,
        gaussians,
        visualization_dump: dict | None,
    ) -> None:
        if not self._reliability_loss_active():
            return
        if visualization_dump is None:
            raise RuntimeError(
                "reliability loss requires encoder visualization_dump, but it was not created."
            )
        required = ["means", "reliability", "opacities_before_reliability"]
        missing = [key for key in required if key not in visualization_dump]
        if missing:
            raise RuntimeError(
                "reliability loss requires visualization_dump entries "
                f"{required}, missing {missing}."
            )

        context = batch["context"]
        means, reliability, opacity_before = self._flatten_reliability_dump(
            visualization_dump
        )
        _, _, _, h_ctx, w_ctx = context["image"].shape
        with torch.no_grad():
            context_output = self.decoder.forward(
                gaussians,
                context["extrinsics"],
                context["intrinsics"],
                context["near"],
                context["far"],
                (h_ctx, w_ctx),
                depth_mode=self.train_cfg.depth_mode,
            )
            context_gt = self._denormalize_context_image(context["image"]).to(
                dtype=context_output.color.dtype
            )
            residual_map = (context_output.color - context_gt).abs().mean(
                dim=2, keepdim=True
            )
            residual_map = residual_map.nan_to_num(
                nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)

            xy, in_front = project(
                means.detach()[:, None],
                context["extrinsics"][:, :, None],
                context["intrinsics"][:, :, None],
            )
            in_bounds = (xy >= 0.0).all(dim=-1) & (xy <= 1.0).all(dim=-1)
            valid_per_view = in_front & in_bounds

            b, v_ctx, _, _ = xy.shape
            grid = xy.mul(2.0).sub(1.0)
            grid = rearrange(grid, "b v g xy -> (b v) g 1 xy")
            sampled = F.grid_sample(
                rearrange(residual_map, "b v c h w -> (b v) c h w"),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = rearrange(
                sampled, "(b v) 1 g 1 -> b v g", b=b, v=v_ctx
            )
            valid_float = valid_per_view.to(dtype=sampled.dtype)
            valid_count = valid_float.sum(dim=1).clamp_min(1.0)
            residual_risk = (sampled * valid_float).sum(dim=1) / valid_count
            valid_mask = valid_per_view.any(dim=1).unsqueeze(-1)

        metadata = batch.setdefault("_reliability", {})
        metadata["reliability"] = reliability
        metadata["opacity_before_reliability"] = opacity_before
        metadata["residual_risk"] = residual_risk.unsqueeze(-1)
        metadata["valid_mask"] = valid_mask.to(dtype=reliability.dtype)
        metadata["context_residual_mean"] = residual_map.mean()

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        visualization_dump = None
        if (
            self.distiller is not None
            or self._reliability_loss_active()
            or self._retention_needs_dump("train")
        ):
            visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
        self._attach_reliability_intermediates(batch, gaussians, visualization_dump)
        retention = self._apply_gaussian_retention(
            gaussians,
            batch,
            visualization_dump,
            "train",
            int(self.global_step),
        )
        render_gaussians = retention.gaussians
        output = self.decoder.forward(
            render_gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(
                output,
                batch,
                render_gaussians,
                self.global_step,
            )
            self.log(f"loss/{loss_fn.name}", loss)
            if loss_fn.name == "reliability" and getattr(loss_fn, "last_info", None):
                for key, value in loss_fn.last_info.items():
                    self.log(f"{loss_fn.name}/{key}", value)
            total_loss = total_loss + loss

        preconditioner_stats = self.olp_eval_recorder.collect_preconditioner_stats(
            self.encoder
        )
        for key, value in preconditioner_stats.items():
            self.log(f"olp/{key}", value)

        # distillation
        if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            distillation_loss = self.distiller_loss(pseudo_gt1['pts3d'], pseudo_gt2['pts3d'],
                                                    visualization_dump['means'][:, 0].squeeze(-2),
                                                    visualization_dump['means'][:, 1].squeeze(-2),
                                                    pseudo_gt1['conf'], pseudo_gt2['conf'], disable_view1=False) * 0.1
            self.log("loss/distillation_loss", distillation_loss)
            total_loss = total_loss + distillation_loss

        if self._retention_budget_active():
            budget_loss = self._retention_budget_loss(gaussians, batch, visualization_dump)
            self.log("loss/retention_budget", budget_loss)
            total_loss = total_loss + budget_loss

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.
        visualization_dump = (
            {}
            if self.test_cfg.save_olp_stats
            or self._retention_needs_dump("eval")
            else None
        )
        with self.benchmarker.time("encoder"):
            full_gaussians = self.encoder(
                batch["context"],
                self.global_step,
                visualization_dump=visualization_dump,
            )
        retention = self._apply_gaussian_retention(
            full_gaussians,
            batch,
            visualization_dump,
            "eval",
            int(batch_idx),
        )
        gaussians = retention.gaussians

        # align the target pose
        if self.test_cfg.align_pose:
            output = self.test_step_align(batch, gaussians)
        else:
            with self.benchmarker.memory("decoder"), self.benchmarker.time("decoder", num_calls=v):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )

        # compute scores
        overlap = None
        overlap_tag = None
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
            }
            methods = ['ours']

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        if self.test_cfg.save_olp_stats and visualization_dump is not None:
            self.olp_eval_recorder.record_test_batch(
                self.encoder,
                visualization_dump,
                scene,
                batch_idx,
                all_metrics if self.test_cfg.compute_scores else None,
                float(overlap.detach().cpu()) if overlap is not None else None,
                overlap_tag,
            )
            self.olp_eval_recorder.write_full_raw_maps(
                path, scene, batch_idx, self.encoder, visualization_dump
            )

        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in output.color[0]],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0])
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")

    def test_step_align(self, batch, gaussians):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()
            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

        # Render Gaussians. Timed here too: with align_pose the final render is the decoder cost.
        with torch.no_grad(), self.benchmarker.memory("decoder"), self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                extrinsics,
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )

        return output

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        self.benchmarker.dump(path / "benchmark.json")
        self.benchmarker.dump_memory(
            path / "peak_memory.json"
        )
        self.benchmarker.dump_stage_memory(path / "stage_memory.json")
        if self.test_cfg.save_olp_stats:
            self.olp_eval_recorder.write(path)
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
        retention = self._apply_gaussian_retention(
            gaussians,
            batch,
            visualization_dump,
            "eval",
            int(batch_idx),
        )
        gaussians = retention.gaussians
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            "depth",
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # Construct comparison image.
        context_img = inverse_normalize(batch["context"]["image"][0])
        context_img_depth = vis_depth_map(gaussian_means)
        context = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            context.append(context_img_depth[i])
        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
        )

        if self.distiller is not None:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            depth1, depth2 = pseudo_gt1['pts3d'][..., -1], pseudo_gt2['pts3d'][..., -1]
            conf1, conf2 = pseudo_gt1['conf'], pseudo_gt2['conf']
            depth_dust = torch.cat([depth1, depth2], dim=0)
            depth_dust = vis_depth_map(depth_dust)
            conf_dust = torch.cat([conf1, conf2], dim=0)
            conf_dust = confidence_map(conf_dust)
            dust_vis = torch.cat([depth_dust, conf_dust], dim=0)
            comparison = hcat(add_label(vcat(*dust_vis), "Context"), comparison)

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = hcat(
                *render_projections(
                    gaussians,
                    256,
                    extra_label="",
                )[0]
            )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        self._render_validation_videos(batch, gaussians)

    def _render_validation_videos(
        self, batch: BatchedExample, gaussians: Gaussians
    ) -> None:
        if not self.train_cfg.validation_videos:
            return
        # Reuse the exact subset used for validation metrics, including random
        # retention. Re-encoding or re-selecting here would change the result.
        self.render_video_interpolation(batch, gaussians)
        self.render_video_wobble(batch, gaussians)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch, gaussians)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample, gaussians: Gaussians) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", gaussians, num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample, gaussians: Gaussians) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb", gaussians)

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample, gaussians: Gaussians) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            gaussians,
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        gaussians: Gaussians,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Caller supplies the already-selected validation Gaussians. Do not
        # bypass retention by predicting another full-count scene here.

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=20)
                dir = self.logger.output_dir / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
