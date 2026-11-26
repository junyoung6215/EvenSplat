# Adds new scenes: PF3plat render + per-target scores, distant-orbit renders for NoPo / Ours.
cd /workspace/supp
SC=${1:-23a6c9168abdb38e,cc6384ff2733d7c6}
python3 - "$SC" <<'PY'
import json, sys
src = json.load(open('/workspace/EvenSplat/assets/evaluation_index_re10k_val64.json'))
sel = {s: src[s] for s in sys.argv[1].split(',')}
json.dump(sel, open('idx_new.json', 'w'))
for k in range(3):
    json.dump({s: dict(v, target=[v['target'][k]]) for s, v in sel.items()}, open(f'idx_t{k}.json', 'w'))
print('IDX', len(sel))
PY
bash pf3_run.sh > pf3_run_new.log 2>&1; echo "PF3 RENDER $(ls videos/pf3plat | wc -l)"
sed '/rm -f/d' pf3_score.sh > pf3_score_new.sh; bash pf3_score_new.sh > pf3_score_new.log 2>&1; echo "PF3 SCORES $(wc -l < scores/pf3plat.jsonl)"
sed 's/idx_sel.json/idx_new.json/' orbit2.sh > orbit3.sh; bash orbit3.sh > orbit3.log 2>&1; echo "ORBIT $(ls orbit2/nopo | wc -l) $(ls orbit2/ours_r50 | wc -l)"
echo NEWDONE
