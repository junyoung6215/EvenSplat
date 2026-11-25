import json
S='/workspace/supp/scores/'
def L(n):
    a={}
    for d in map(json.loads,open(S+n+'.jsonl')): a.setdefault(d['scene'],[]).append(d)
    return {k:{'n':v[0]['n'],'psnr':sum(x['psnr'] for x in v)/len(v),'cnt':len(v)} for k,v in a.items()}
pf,no,ol,o5,o7=[L(n) for n in ('pf3plat','nopo','olp','ours_r50','ours_r75')]
for n,d in (('pf3',pf),('nopo',no),('olp',ol),('r50',o5),('r75',o7)): print(n, len(d), round(sum(x['psnr'] for x in d.values())/len(d),3), list(d.values())[0]['n'])
rows=[]
for s in o5:
    if s not in pf or s not in no or s not in ol: continue
    r=(o5[s]['psnr']-no[s]['psnr'], s, pf[s]['psnr'], no[s]['psnr'], ol[s]['psnr'], o5[s]['psnr'], o7[s]['psnr'])
    rows.append(r)
rows.sort(reverse=True)
print('d(r50-nopo) scene pf3 nopo olp r50 r75')
for r in rows[:15]: print(' '.join(f'{x:.2f}' if isinstance(x,float) else x[:16] for x in r))
