# Per-scene scores + kept-mask dumps for the method section (base model without FT, and Ours).
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate noposplat-olp-cuda118
cd /workspace/EvenSplat
python /workspace/supp/patch3.py
export SUPP_MAP_DIR=/workspace/supp/maps
mkdir -p /workspace/supp/scores $SUPP_MAP_DIR
W=/workspace/EvenSplat/pretrained_weights
COMMON="+experiment=retention mode=test wandb.mode=disabled seed=111123 dataset.re10k.roots=[/workspace/data/re10k] dataset.re10k.view_sampler.index_path=/workspace/supp/idx_sel.json model.encoder.pretrained_weights=\"\" checkpointing.save_weights_only=false model.encoder.orthogonal_preconditioner.seed=0 model.encoder.orthogonal_preconditioner.mix=1.0 model.encoder.orthogonal_preconditioner.target_mix=1.0 model.encoder.orthogonal_preconditioner.warmup_steps=0 model.encoder.gaussian_reliability.r_min=0.05 model.encoder.gaussian_reliability.detach_gate=false test.align_pose=true test.pose_align_steps=100 test.save_image=false test.save_video=false test.save_compare=false test.save_olp_stats=false model.gaussian_retention.random_seed=111123 model.gaussian_retention.r_threshold=0.35 model.gaussian_retention.start_step=0 model.gaussian_retention.diagnostics_every_n_steps=1 data_loader.test.num_workers=2 data_loader.test.persistent_workers=false model.encoder.orthogonal_preconditioner.enabled=true model.encoder.gaussian_reliability.enabled=true model.gaussian_retention.enabled=true model.gaussian_retention.keep_ratio=0.5"
run() { # name ckpt score_source cell_size
  rm -f /workspace/supp/scores/$1.jsonl
  SUPP_SCORE_FILE=/workspace/supp/scores/$1.jsonl eval python -m src.main $COMMON checkpointing.load=$2 model.gaussian_retention.score_source=$3 model.gaussian_retention.cell_size=$4 wandb.name=$1 hydra.run.dir=/workspace/supp/out/m/$1 test.output_path=/workspace/supp/out/m/$1/r > /workspace/supp/scores/$1.log 2>&1 || echo "FAIL $1"
  echo "DONE $1 $(wc -l < /workspace/supp/scores/$1.jsonl)"
}
run abl_cell0 $W/base/epoch_1-step_75000.ckpt reliability 0
run abl_cell2 $W/base/epoch_1-step_75000.ckpt reliability 2
run ours_map $W/evensplat/retention_r50_re10k.ckpt r_filter 2
echo METHODRUNS
