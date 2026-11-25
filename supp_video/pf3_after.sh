cd /workspace/supp
python3 -c "
import json
d=json.load(open('/workspace/EvenSplat/assets/evaluation_index_re10k_val64.json'))
for k in range(3):
    json.dump({s:(dict(v,target=[v['target'][k]]) if v else v) for s,v in d.items()},open(f'/workspace/supp/idx_t{k}.json','w'))
print('IDX OK')"
while ! grep -q ALLSCORED score_orbit.log; do sleep 30; done
sed -i 's#/workspace/EvenSplat/assets/evaluation_index_re10k_val64.json#/workspace/supp/idx_t0.json#' pf3_run.sh
bash pf3_run.sh > pf3_run2.log 2>&1
echo "PF3 RENDERED $(ls videos/pf3plat/*.mp4 | wc -l)"
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate pf3plat
cd /workspace/PF3plat
python /workspace/supp/pf3_score_patch.py
rm -f /workspace/supp/scores/pf3plat.jsonl
for k in 0 1 2; do
SUPP_SCORE_FILE=/workspace/supp/scores/pf3plat.jsonl python -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=/workspace/supp/idx_t$k.json 'dataset.roots=[/workspace/data/re10k]' checkpointing.load=ckpt/re10k.ckpt wandb.mode=disabled wandb.name=pf3plat_score test.save_image=false test.compute_scores=true dataset.near=1. dataset.far=100. dataset.baseline_scale_bounds=false dataset.make_baseline_1=false data_loader.test.num_workers=0 > /workspace/supp/scores/pf3plat_t$k.log 2>&1 || echo "FAIL pf3 t$k"
echo "pf3 t$k $(wc -l < /workspace/supp/scores/pf3plat.jsonl)"
done
echo PF3ALL
