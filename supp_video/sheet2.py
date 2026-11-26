import numpy as np, imageio.v2 as io, sys
from PIL import Image, ImageDraw
S='/workspace/supp/videos'
def u8(a):
    a=np.asarray(a)
    if a.dtype!=np.uint8: a=(np.clip(a,0,1)*255).round().astype(np.uint8)
    if a.ndim==4 and a.shape[1]==3: a=a.transpose(0,2,3,1)
    return a
rows=[]
for s in sys.argv[1].split(','):
    n=u8(np.load(f'{S}/noposplat/{s}.npy')); o=u8(np.load(f'{S}/r_filter-r0.5-cell2/{s}.npy'))
    m=min(len(n),len(o)); d=[np.abs(n[i].astype(float)-o[i].astype(float)).mean() for i in range(m)]
    i=int(np.argmax(d)); j=m//2
    im=Image.fromarray(np.concatenate([n[i],o[i],n[j],o[j]],1)); ImageDraw.Draw(im).text((4,4),f'{s[:6]} maxdiff {d[i]:.1f}',fill=(255,0,0))
    rows.append(np.asarray(im))
io.imwrite('/workspace/supp/prev/cand.png',np.concatenate(rows,0)); print('CAND')
