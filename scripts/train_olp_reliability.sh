#!/usr/bin/env bash
# Stage 2 of the base model: add the Gaussian reliability head to an OLP-H
# checkpoint and fine-tune 75K steps with the context-residual reliability loss.
# Produces the starting checkpoint of every retention run.
#
#   BASE_CKPT=/path/to/olp_h_300k.ckpt DATA_ROOT=/path/to/re10k \
#     bash scripts/train_olp_reliability.sh
set -euo pipefail

: "${BASE_CKPT:?Set BASE_CKPT to the OLP-H checkpoint}"
: "${DATA_ROOT:?Set DATA_ROOT to the RE10K root}"
RUN_DIR="${RUN_DIR:-outputs/olp_reliability/$(date +%Y%m%d_%H%M%S)}"

python -m src.main \
  +experiment=re10k \
  mode=train \
  wandb.mode="${WANDB_MODE:-disabled}" \
  wandb.name=re10k_olp_reliability \
  model.encoder.pretrained_weights="${BASE_CKPT}" \
  checkpointing.load=null \
  "dataset.re10k.roots=[${DATA_ROOT}]" \
  model.encoder.orthogonal_preconditioner.enabled=true \
  model.encoder.orthogonal_preconditioner.mix=1.0 \
  model.encoder.orthogonal_preconditioner.target_mix=1.0 \
  model.encoder.orthogonal_preconditioner.warmup_steps=0 \
  model.encoder.orthogonal_preconditioner.seed=0 \
  model.encoder.orthogonal_preconditioner.collect_stats=true \
  model.encoder.gaussian_reliability.enabled=true \
  model.encoder.gaussian_reliability.r_min=0.05 \
  model.encoder.gaussian_reliability.r_init=0.95 \
  "loss=[mse,lpips,reliability]" \
  loss.reliability.enabled=true \
  loss.reliability.weight=1.0e-4 \
  loss.reliability.warmup_start=3000 \
  loss.reliability.warmup_end=8000 \
  trainer.max_steps=75000 \
  trainer.accumulate_grad_batches=4 \
  trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval=100000000 \
  optimizer.lr=2e-4 \
  data_loader.train.batch_size=4 \
  checkpointing.every_n_train_steps=25000 \
  checkpointing.save_top_k=3 \
  checkpointing.save_weights_only=true \
  train.print_log_every_n_steps=100 \
  hydra.run.dir="${RUN_DIR}"
