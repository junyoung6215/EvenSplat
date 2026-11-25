export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate noposplat-olp-cuda118
cd /workspace/EvenSplat
python /workspace/supp/patch2.py
mkdir -p /workspace/supp/scores /workspace/supp/orbit
export SUPP_ORBIT_DIR=/workspace/supp/orbit
W=/workspace/EvenSplat/pretrained_weights
COMMON="+experiment=retention mode=test wandb.mode=disabled seed=111123 dataset.re10k.roots=[/workspace/data/re10k] dataset.re10k.view_sampler.index_path=/workspace/EvenSplat/assets/evaluation_index_re10k_val64.json model.encoder.pretrained_weights=\"\" checkpointing.save_weights_only=false model.encoder.orthogonal_preconditioner.seed=0 model.encoder.orthogonal_preconditioner.mix=1.0 model.encoder.orthogonal_preconditioner.target_mix=1.0 model.encoder.orthogonal_preconditioner.warmup_steps=0 model.encoder.gaussian_reliability.r_min=0.05 model.encoder.gaussian_reliability.detach_gate=false test.align_pose=true test.pose_align_steps=100 test.save_image=false test.save_video=false test.save_compare=false test.save_olp_stats=false model.gaussian_retention.random_seed=111123 model.gaussian_retention.r_threshold=0.35 model.gaussian_retention.cell_size=2 model.gaussian_retention.start_step=0 model.gaussian_retention.diagnostics_every_n_steps=1 data_loader.test.num_workers=4 data_loader.test.persistent_workers=false"
run() { # name ckpt olp rel ret extra
  rm -f /workspace/supp/scores/$1.jsonl
  SUPP_SCORE_FILE=/workspace/supp/scores/$1.jsonl eval python -m src.main $COMMON checkpointing.load=$2 model.encoder.orthogonal_preconditioner.enabled=$3 model.encoder.gaussian_reliability.enabled=$4 model.gaussian_retention.enabled=$5 $6 wandb.name=$1 hydra.run.dir=/workspace/supp/out/score/$1 test.output_path=/workspace/supp/out/score/$1/renders > /workspace/supp/scores/$1.log 2>&1 || echo "FAIL $1"
  echo "DONE $1 $(wc -l < /workspace/supp/scores/$1.jsonl)"
}
run nopo $W/re10k.ckpt false false false ""
run olp $W/base/epoch_1-step_75000.ckpt true true false ""
run ours_r50 $W/evensplat/retention_r50_re10k.ckpt true true true "model.gaussian_retention.score_source=r_filter model.gaussian_retention.keep_ratio=0.5"
run ours_r75 $W/evensplat/retention_r75_re10k.ckpt true true true "model.gaussian_retention.score_source=r_filter model.gaussian_retention.keep_ratio=0.75"
echo EVENSCORED
while ! grep -q PF3DONE /workspace/supp/pf3_extract_run.log 2>/dev/null; do sleep 30; done
conda activate pf3plat
cd /workspace/PF3plat
python /workspace/supp/pf3_score_patch.py
rm -f /workspace/supp/scores/pf3plat.jsonl
SUPP_SCORE_FILE=/workspace/supp/scores/pf3plat.jsonl python -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=/workspace/EvenSplat/assets/evaluation_index_re10k_val64.json 'dataset.roots=[/workspace/data/re10k]' checkpointing.load=ckpt/re10k.ckpt wandb.mode=disabled wandb.name=pf3plat_score test.save_image=false test.compute_scores=true dataset.near=1. dataset.far=100. dataset.baseline_scale_bounds=false dataset.make_baseline_1=false data_loader.test.num_workers=0 > /workspace/supp/scores/pf3plat.log 2>&1 || echo "FAIL pf3"
echo "DONE pf3 $(wc -l < /workspace/supp/scores/pf3plat.jsonl)"
echo ALLSCORED
