import json, sys, os, numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
S = os.environ.get('SUPP', '/workspace/supp')
SCENES = sys.argv[1].split(',')
OUT = sys.argv[2] if len(sys.argv) > 2 else f'{S}/final/evensplat_supp.mp4'
PREVIEW = os.environ.get('PREVIEW')  # write PNG stills instead of video
W, H, FPS = 1920, 1080, 30
FD = os.environ.get('FD', '/usr/share/fonts/truetype/msttcorefonts/')  # apt install ttf-mscorefonts-installer
FB, FR = ('Times_New_Roman_Bold.ttf', 'Times_New_Roman.ttf') if 'msttcore' in FD else ('LiberationSerif-Bold.ttf', 'LiberationSerif-Regular.ttf')
def F(sz, bold=False):
    return ImageFont.truetype(FD + (FB if bold else FR), sz)
M = [  # key, label, video dir, score file
    ('pf3', 'PF3plat', 'pf3plat', 'pf3plat'),
    ('nopo', 'NoPoSplat', 'noposplat', 'nopo'),
    ('olp', 'OLP + Reliability', 'baseline-off', 'olp'),
    ('ours', 'EvenSplat (Ours)', 'r_filter-r0.5-cell2', 'ours_r50'),
]
SC = {}
for _, _, _, s in M:
    SC[s] = {}
    acc = {}
    for l in open(f'{S}/scores/{s}.jsonl'):
        d = json.loads(l); acc.setdefault(d['scene'], []).append(d)
    for k_, v_ in acc.items(): SC[s][k_] = {'scene': k_, 'n': v_[0]['n'], 'psnr': sum(x['psnr'] for x in v_) / len(v_)}
def u8(a):
    a = np.asarray(a)
    if a.dtype != np.uint8: a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    if a.ndim == 4 and a.shape[1] in (1, 3) and a.shape[-1] not in (1, 3): a = a.transpose(0, 2, 3, 1)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3): a = a.transpose(1, 2, 0)
    return a
def loop180(v):  # resample a ping-pong clip to exactly 180 frames (6 s)
    idx = np.round(np.arange(180) * len(v) / 180).astype(int) % len(v)
    return v[idx]
def fit(img, size):
    return Image.fromarray(img).resize((size, size), Image.LANCZOS)
def text(d, xy, s, font, anchor='la', fill=(0, 0, 0)):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)
def header(d, main, sub=None):
    f1, f2 = F(60), F(38)
    text(d, (60, 30), main, f1)
    if sub:
        w = d.textlength(main, font=f1)
        text(d, (60 + w + 16, 30 + 60 - 40), sub, f2, anchor='la')
def metric(d, cx, y, sc, fs=30):
    text(d, (cx, y), f"PSNR {sc['psnr']:.2f} dB", F(fs), 'ma')
    text(d, (cx, y + fs + 8), f"{sc['n']:,} Gaussians", F(fs), 'ma', fill=(70, 70, 70))
frames_written = [0]
writer = None if PREVIEW else imageio.get_writer(OUT, fps=FPS, codec='libx264', quality=None, ffmpeg_params=['-crf', '16', '-pix_fmt', 'yuv420p', '-preset', 'slow'], macro_block_size=1)
def emit(img, fade=None):
    a = np.asarray(img.convert('RGB'))
    if fade is not None and fade < 1: a = (a * fade + 255 * (1 - fade)).astype(np.uint8)
    if PREVIEW:
        if frames_written[0] in PREV_AT: Image.fromarray(a).save(f'{PREVIEW}/f{frames_written[0]:05d}.png')
    else:
        writer.append_data(a)
    frames_written[0] += 1
PREV_AT = set(int(x) for x in os.environ.get('PREV_AT', '').split(',') if x)
def fadein(i, n=8): return min(1.0, (i + 1) / n)

def title_card(lines, secs):
    n = int(secs * FPS)
    base = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(base)
    y = H // 2 - sum(l[1] for l in lines) // 2 - 20
    for s, sz, b in lines:
        text(d, (W // 2, y), s, F(sz, b), 'ma'); y += int(sz * 1.5)
    for i in range(n):
        emit(base, min(fadein(i, 15), (n - i) / 15))

def scene_block(k, scene):
    vids = {key: loop180(u8(np.load(f'{S}/videos/{vd}/{scene}.npy'))) for key, _, vd, _ in M}
    ctx = u8(np.load(f'{S}/videos/noposplat/{scene}_ctx.npy'))
    if ctx.ndim == 5: ctx = ctx[0]
    sc = {key: SC[s][scene] for key, _, _, s in M}
    # ---- Part A: 10 s comparison ----
    base = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(base)
    header(d, 'Novel View Synthesis', '(RealEstate10K)')
    text(d, (W // 2, 118), 'Input views', F(36, True), 'ma')
    cs, gap = 290, 30
    x0 = W // 2 - cs - gap // 2
    for j in range(2): base.paste(fit(ctx[j], cs), (x0 + j * (cs + gap), 166))
    vs, vg = 420, 40
    vx0 = (W - 4 * vs - 3 * vg) // 2; vy = 540
    for j, (key, lab, _, _) in enumerate(M):
        cx = vx0 + j * (vs + vg) + vs // 2
        if key != 'ours':
            text(d, (cx, vy - 52), lab, F(38, True), 'ma'); metric(d, cx, vy + vs + 12, sc[key])
    ours_lay = Image.new('RGB', (W, H), 'white'); d2 = ImageDraw.Draw(ours_lay)
    cxo = vx0 + 3 * (vs + vg) + vs // 2
    text(d2, (cxo, vy - 52), M[3][1], F(38, True), 'ma'); metric(d2, cxo, vy + vs + 12, sc['ours'])
    box = (cxo - vs // 2 - 10, vy - 60, cxo + vs // 2 + 10, vy + vs + 90)
    for f in range(300):
        img = base.copy()
        for j, (key, *_r) in enumerate(M):
            x = vx0 + j * (vs + vg)
            if key == 'ours':
                if f < 120: continue
                a = min(1.0, (f - 120 + 1) / 9)
                img.paste(Image.blend(img.crop(box), ours_lay.crop(box), a), box[:2])
                fr = fit(vids[key][f - 120], vs)
                img.paste(Image.blend(img.crop((x, vy, x + vs, vy + vs)), fr, a), (x, vy))
            else:
                img.paste(fit(vids[key][(f - 120) % 180], vs), (x, vy))
        emit(img, fadein(f))
    # ---- Part B: 10 s split slider NoPo | Ours ----
    base = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(base)
    header(d, 'NoPoSplat vs. EvenSplat (Ours)')
    bs = 780; bx = (W - bs) // 2; by = 150
    for cx, key, lab in ((bx - 200, 'nopo', 'NoPoSplat'), (bx + bs + 200, 'ours', 'EvenSplat (Ours)')):
        text(d, (cx, by + bs // 2 - 40), lab, F(44, True), 'ma'); metric(d, cx, by + bs // 2 + 20, sc[key])
    NB = 300  # 10 s, one constant-speed sweep: center -> right -> left -> center
    for f in range(NB):
        u = (f / NB * 4) % 4
        p = 0.5 + 0.38 * (u if u < 1 else 2 - u if u < 3 else u - 4)
        xs = int(bs * p)
        A = np.asarray(fit(vids['nopo'][f % 180], bs)).copy(); B = np.asarray(fit(vids['ours'][f % 180], bs))
        A[:, xs:] = B[:, xs:]
        A[:, max(0, xs - 2):xs + 2] = 255
        img = base.copy(); img.paste(Image.fromarray(A), (bx, by))
        emit(img, fadein(f))
    # ---- Part C: 8 s 3D overview ----
    orb = {key: u8(np.stack(imageio.mimread(f'{S}/orbit2/{s}/{scene}.mp4', memtest=False))) for key, _, _, s in M if key in ('nopo', 'ours')}
    base = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(base)
    header(d, '3D Gaussians from a Distant Viewpoint')
    os_ = 760; og = 80; ox0 = (W - 2 * os_ - og) // 2; oy = 200
    for j, key in enumerate(('nopo', 'ours')):
        cx = ox0 + j * (os_ + og) + os_ // 2
        text(d, (cx, oy - 58), M[1 if key == 'nopo' else 3][1], F(42, True), 'ma')
        text(d, (cx, oy + os_ + 14), f"{sc[key]['n']:,} Gaussians", F(32), 'ma', fill=(70, 70, 70))
    nf = len(orb['nopo'])
    for f in range(240):
        img = base.copy()
        for j, key in enumerate(('nopo', 'ours')):
            img.paste(fit(orb[key][f % nf], os_), (ox0 + j * (os_ + og), oy))
        emit(img, fadein(f))


# ================= Method section (pipeline zoom + per-step renders) =================
MSCENE = os.environ.get('MSCENE', 'c48f19e2ffa52523')
ORANGE = (200, 90, 20)
def ease(t): t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)
def cmap(x):  # viridis-like
    a = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    x = np.clip(x, 0, 1) * 4; i = np.minimum(x.astype(int), 3); f = (x - i)[..., None]
    return (a[i] * (1 - f) + a[i + 1] * f).astype(np.uint8)
def load_scores(name):
    acc = {}
    p = f'{S}/scores/{name}.jsonl'
    if not os.path.exists(p): return {}
    for l in open(p):
        d = json.loads(l); acc.setdefault(d['scene'], []).append(d)
    return {k: {'n': v[0]['n'], 'psnr': sum(x['psnr'] for x in v) / len(v)} for k, v in acc.items()}
def mask_view(ctx0, keep, P, inset_at=None):
    kv = keep.reshape(-1, ctx0.shape[0], ctx0.shape[1])[0]
    a = ctx0.astype(float)
    a = np.where(kv[..., None], a, a * 0.22 + 255 * 0.78).astype(np.uint8)
    im = Image.fromarray(a).resize((P, P), Image.NEAREST)
    if inset_at is not None:
        y, x = inset_at; n = 16; z = 7
        sub = Image.fromarray(a[y:y + n, x:x + n]).resize((n * z, n * z), Image.NEAREST)
        d = ImageDraw.Draw(sub)
        for k in range(0, n + 1, 2):
            d.line([(k * z, 0), (k * z, n * z)], fill=(150, 150, 150), width=1)
            d.line([(0, k * z), (n * z, k * z)], fill=(150, 150, 150), width=1)
        fr = Image.new('RGB', (n * z + 6, n * z + 6), ORANGE); fr.paste(sub, (3, 3))
        im.paste(fr, (P - fr.size[0] - 6, P - fr.size[1] - 6))
        ImageDraw.Draw(im).rectangle([x * P / ctx0.shape[1], y * P / ctx0.shape[0], (x + n) * P / ctx0.shape[1], (y + n) * P / ctx0.shape[0]], outline=ORANGE, width=2)
    return im

def rich(d, xy, parts, font, fill, anchor='la'):
    # parts: ints draw a circled number, strings draw text; anchor 'la' or 'ma'
    sz = font.size; r = int(sz * 0.42); gap = int(sz * 0.25)
    widths = [(2 * r + gap) if isinstance(p, int) else d.textlength(p, font=font) + gap for p in parts]
    x, y = xy
    if anchor == 'ma': x -= sum(widths) // 2
    for p, w in zip(parts, widths):
        if isinstance(p, int):
            cy = y + int(sz * 0.55)
            d.ellipse([x, cy - r, x + 2 * r, cy + r], outline=fill, width=max(2, sz // 14))
            d.text((x + r, cy), str(p), font=F(int(sz * 0.62), True), fill=fill, anchor='mm')
        else:
            d.text((x, y), p, font=font, fill=fill)
        x += w

def method_block():
    PIPE = Image.open(f'{S}/pipeline.png').convert('RGB')
    PW, PH = PIPE.size; RW = 2000.0; RH = RW * PH / PW; SX = PW / RW
    BW = 1800; BH = int(BW * PH / PW); BX = (W - BW) // 2; BY = 92
    def band(cx, cy, z, hl, ha):
        w = RW / z; h = RH / z
        x0 = min(max(cx - w / 2, 0), RW - w); y0 = min(max(cy - h / 2, 0), RH - h)
        im = PIPE.crop((int(x0 * SX), int(y0 * SX), int((x0 + w) * SX), int((y0 + h) * SX))).resize((BW, BH), Image.LANCZOS)
        if hl is not None and ha > 0:
            k = BW / w
            r = [int((hl[0] - x0) * k), int((hl[1] - y0) * k), int((hl[2] - x0) * k), int((hl[3] - y0) * k)]
            a = np.asarray(im).astype(float); m = np.ones(a.shape[:2], bool)
            m[max(r[1], 0):max(r[3], 0), max(r[0], 0):max(r[2], 0)] = False
            a[m] = a[m] * (1 - 0.6 * ha) + 255 * 0.6 * ha
            im = Image.fromarray(a.astype(np.uint8))
            ImageDraw.Draw(im).rounded_rectangle(r, radius=10, outline=ORANGE, width=max(1, int(4 * ha)))
        return im
    mp = {n: np.load(f'{S}/maps/{n}_{MSCENE}.npz') for n in ('abl_cell0', 'abl_cell2', 'ours_map')}
    ctx0 = mp['abl_cell0']['ctx'][0]
    Hc, Wc = ctx0.shape[:2]
    rel = mp['abl_cell0']['rel'].reshape(-1, Hc, Wc)[0]
    rn = (rel - rel.min()) / max(1e-6, rel.max() - rel.min())
    vid = lambda d: loop180(u8(np.load(f'{S}/videos/{d}/{MSCENE}.npy')))
    V = {'olp': vid('baseline-off'), 'g': vid('reliability-r0.5-cell0'), 's': vid('reliability-r0.5-cell2'), 'ours': vid('r_filter-r0.5-cell2')}
    sc = {k: load_scores(n).get(MSCENE) for k, n in (('olp', 'olp'), ('g', 'abl_cell0'), ('s', 'abl_cell2'), ('ours', 'ours_r50'))}
    def met(k, n):
        out = []
        if sc[k]: out.append(f"PSNR {sc[k]['psnr']:.2f} dB")
        out.append(f"{n:,} Gaussians"); return out
    # find an inset location with a visible mix in the global mask
    kg = mp['abl_cell0']['keep'].reshape(-1, Hc, Wc)[0].astype(float)
    best, at = -1, (Hc // 2, Wc // 2)
    for y in range(16, Hc - 32, 8):
        for x in range(16, Wc - 32, 8):
            v = kg[y:y + 16, x:x + 16].mean(); sc_ = -abs(v - 0.5) * 0 + (1 - abs(v - 0.15) * 4)
            if sc_ > best: best, at = sc_, (y, x)
    full = 2 * Hc * Wc; half = full // 2
    stages = [
        dict(n=90, view=(1000, RH / 2, 1.0, None), tag=None),
        dict(n=210, view=(745, 212, 1.3, (515, 2, 975, 425)), tag=[1, 'OLP-H  +', 2, 'Reliability'],
             desc=['Each pixel predicts one Gaussian', 'and a reliability r in [r_min, 1].', 'Opacity becomes α·r, so unreliable', 'Gaussians fade out.'],
             panels=[('img', Image.fromarray(ctx0), 'Input view', []),
                     ('img', Image.fromarray(cmap(rn)), 'Reliability r', ['low r', None]),
                     ('vid', V['olp'], 'Render (all Gaussians)', met('olp', full))]),
        dict(n=240, view=(1372, 235, 1.18, (1010, 2, 1735, 470)), tag=[3, 'Stratified retention'],
             desc=['Keep 50% in every 2×2 cell:', 'low r first, the rest at random.', 'A global top-k by r removes', 'whole regions instead.', '(no fine-tuning yet)'],
             panels=[('mask', mp['abl_cell0']['keep'], 'Global top-k: kept', []),
                     ('vid', V['g'], 'Global top-k', met('g', half)),
                     ('mask', mp['abl_cell2']['keep'], '2×2 stratified: kept', []),
                     ('vid', V['s'], '2×2 stratified', met('s', half))]),
        dict(n=210, view=(1000, RH / 2, 1.0, (335, 468, 1995, 553)), tag=[4, 'Pruning-aware fine-tuning'],
             desc=['Training uses the same selection,', 'so only the kept Gaussians', 'receive gradients and learn to', 'cover the pruned ones.'],
             panels=[('mask', mp['ours_map']['keep'], 'Kept (r filter, 2×2)', []),
                     ('vid', V['s'], 'Before fine-tuning', met('s', half)),
                     ('vid', V['ours'], 'EvenSplat (Ours)', met('ours', half))]),
    ]
    prev = stages[0]['view']
    for si, st in enumerate(stages):
        base = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(base)
        header(d, 'Method')
        if st['tag'] is None:
            text(d, (W // 2, 700), 'Pose-free 3D Gaussian Splatting from two unposed views,', F(38), 'ma')
            text(d, (W // 2, 752), 'keeping an even 50% of the pixel-aligned Gaussians.', F(38), 'ma')
            rich(d, (W // 2, 840), [1, 'OLP-H   ', 2, 'Reliability   ', 3, 'Stratified retention   ', 4, 'Pruning-aware fine-tuning'], F(34, True), ORANGE, 'ma')
            ps = []
        else:
            rich(d, (60, 640), st['tag'], F(40, True), ORANGE)
            for i, l in enumerate(st['desc']):
                text(d, (60, 706 + i * 42), l, F(31), fill=(40, 40, 40))
            n = len(st['panels']); P = 300 if n <= 3 else 268; g = 26
            x0 = 660 + (1200 - (n * P + (n - 1) * g)) // 2; py = 690
            ps = []
            for j, (kind, src, lab, lines) in enumerate(st['panels']):
                x = x0 + j * (P + g); cx = x + P // 2
                text(d, (cx, py - 44), lab, F(29, True), 'ma')
                if kind == 'img': base.paste(src.resize((P, P), Image.LANCZOS), (x, py))
                if kind == 'mask': base.paste(mask_view(ctx0, src, P, at if si == 2 else None), (x, py))
                if lab == 'Reliability r':
                    bar = Image.fromarray(np.repeat(cmap(np.linspace(0, 1, P))[None], 14, 0)); base.paste(bar, (x, py + P + 10))
                    text(d, (x, py + P + 30), 'low', F(24), 'la', fill=(70, 70, 70)); text(d, (x + P, py + P + 30), 'high', F(24), 'ra', fill=(70, 70, 70))
                else:
                    for i, l in enumerate(lines):
                        if l: text(d, (cx, py + P + 10 + i * 32), l, F(27), 'ma', fill=(0, 0, 0) if i == 0 and 'PSNR' in l else (70, 70, 70))
                if kind == 'vid': ps.append((src, x, py, P))
        nxt = st['view']; T = 24
        for f in range(st['n']):
            t = ease(f / T)
            cx = prev[0] + (nxt[0] - prev[0]) * t; cy = prev[1] + (nxt[1] - prev[1]) * t; z = prev[2] + (nxt[2] - prev[2]) * t
            img = base.copy()
            img.paste(band(cx, cy, z, nxt[3], ease((f - T) / 12)), (BX, BY))
            for src, x, y, P in ps:
                img.paste(fit(src[f % 180], P), (x, y))
            fa = fadein(f, 10) if si > 0 else fadein(f, 15)
            if si == len(stages) - 1: fa = min(fa, (st['n'] - f) / 12)
            emit(img, fa)
        prev = nxt

def results_card(secs=8):
    base = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(base)
    header(d, 'Results', '(RealEstate10K, full test set)')
    fig = Image.open(f'{S}/psnr_vs_gaussians.png').convert('RGB')
    fh = 780; fw = int(fig.size[0] * fh / fig.size[1])
    base.paste(fig.resize((fw, fh), Image.LANCZOS), (70, 170))
    cols = [('Model', 'la', 0), ('#Gaussians', 'ma', 330), ('PSNR ↑', 'ma', 480), ('SSIM ↑', 'ma', 610), ('LPIPS ↓', 'ma', 740)]
    rows = [('NoPoSplat*', '131K', '25.03', '0.838', '0.160', False),
            ('EvenSplat 75%', '98K', '25.26', '0.837', '0.161', True),
            ('EvenSplat 50%', '66K', '25.01', '0.829', '0.169', True),
            ('EvenSplat 25%', '33K', '24.62', '0.817', '0.184', True)]
    tx = 70 + fw + 50; ty = 330
    for name, an, dx in cols: text(d, (tx + dx, ty), name, F(32, True), an)
    d.line([(tx, ty + 50), (tx + 800, ty + 50)], fill=(0, 0, 0), width=2)
    for i, r in enumerate(rows):
        y = ty + 72 + i * 58
        for j, (name, an, dx) in enumerate(cols): text(d, (tx + dx, y), r[j], F(32, r[5] and j == 0), an)
    d.line([(tx, ty + 72 + 4 * 58), (tx + 800, ty + 72 + 4 * 58)], fill=(0, 0, 0), width=2)
    text(d, (tx, ty + 72 + 4 * 58 + 24), '*Reported in NoPoSplat.', F(26), fill=(90, 90, 90))
    text(d, (tx, ty + 72 + 4 * 58 + 90), 'Half the Gaussians at the same quality as NoPoSplat;', F(32))
    text(d, (tx, ty + 72 + 4 * 58 + 136), 'peak rendering memory 167 MB → 87 MB.', F(32))
    n = int(secs * FPS)
    for i in range(n): emit(base, min(fadein(i, 15), (n - i) / 15))

title_card([('EvenSplat', 120, True), ('Kyoosuk Hwang, Junyoung Kwon', 48, False)], 4)
if os.path.exists(f'{S}/maps'): method_block()
for k, s in enumerate(SCENES): scene_block(k, s)
if os.path.exists(f'{S}/psnr_vs_gaussians.png'): results_card()
title_card([('Thank you for watching.', 90, False)], 3)
if writer: writer.close()
print('COMPOSED', frames_written[0], OUT)
