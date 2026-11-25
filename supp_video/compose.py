import json, sys, os, numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
S = os.environ.get('SUPP', '/workspace/supp')
SCENES = sys.argv[1].split(',')
OUT = sys.argv[2] if len(sys.argv) > 2 else f'{S}/final/evensplat_supp.mp4'
PREVIEW = os.environ.get('PREVIEW')  # write PNG stills instead of video
W, H, FPS = 1920, 1080, 30
FD = '/usr/share/fonts/truetype/liberation/'
def F(sz, bold=False):
    return ImageFont.truetype(FD + ('LiberationSerif-Bold.ttf' if bold else 'LiberationSerif-Regular.ttf'), sz)
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
    # ---- Part B: 7 s split slider NoPo | Ours ----
    base = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(base)
    header(d, 'NoPoSplat vs. EvenSplat (Ours)')
    bs = 780; bx = (W - bs) // 2; by = 150
    text(d, (bx - 30, by + bs // 2 - 40), 'NoPoSplat', F(44, True), 'ra'); metric(d, bx - 30 - 150, by + bs // 2 + 20, sc['nopo'])
    text(d, (bx + bs + 30, by + bs // 2 - 40), 'EvenSplat (Ours)', F(44, True), 'la'); metric(d, bx + bs + 30 + 165, by + bs // 2 + 20, sc['ours'])
    for f in range(210):
        t = f / 210
        p = 0.5 + 0.38 * np.sin(2 * np.pi * 2 * t)  # two smooth sweeps
        xs = int(bs * p)
        A = np.asarray(fit(vids['nopo'][f % 180], bs)).copy(); B = np.asarray(fit(vids['ours'][f % 180], bs))
        A[:, xs:] = B[:, xs:]
        A[:, max(0, xs - 2):xs + 2] = 255
        img = base.copy(); img.paste(Image.fromarray(A), (bx, by))
        emit(img, fadein(f))
    # ---- Part C: 8 s 3D overview ----
    orb = {key: u8(np.stack(imageio.mimread(f'{S}/orbit/{s}/{scene}.mp4', memtest=False))) for key, _, _, s in M if key in ('nopo', 'ours')}
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

title_card([('EvenSplat', 120, True), ('Kyoosuk Hwang, Junyoung Kwon', 48, False)], 4)
for k, s in enumerate(SCENES): scene_block(k, s)
title_card([('Thank you for watching.', 90, False)], 3)
if writer: writer.close()
print('COMPOSED', frames_written[0], OUT)
