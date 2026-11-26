import json
def load(n):
    acc={}
    for l in open(f'/workspace/supp/scores/{n}.jsonl'):
        d=json.loads(l); acc.setdefault(d['scene'],[]).append(d['psnr'])
    return {k:sum(v)/len(v) for k,v in acc.items()}
N,O,E=load('nopo'),load('olp'),load('ours_r50')
import os
have=set(f[:-4] for f in os.listdir('/workspace/supp/videos/noposplat') if f.endswith('.npy') and '_ctx' not in f)
rows=sorted([(E[s]-max(N[s],O[s]),s,N[s],O[s],E[s]) for s in E if s in N and s in O and N[s]>=21],reverse=True)
for g,s,n,o,e in rows[:12]: print(f'{s} nopo {n:.2f} olp {o:.2f} ours {e:.2f} gain {g:+.2f} vid {s in have}')
