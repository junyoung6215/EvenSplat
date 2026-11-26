# Adds a SUPP_MAP_DIR hook to EvenSplat test_step: dumps context images, reliability r and kept mask.
p = '/workspace/EvenSplat/src/model/model_wrapper.py'
s = open(p).read()
if '_save_supp_map' in s:
    print('ALREADY'); raise SystemExit
b = '        gaussians = retention.gaussians\n'
assert s.count(b) == 1
s = s.replace(b, b + '''        if os.environ.get("SUPP_MAP_DIR"):
            self._save_supp_map(batch, full_gaussians, retention, visualization_dump)
''')
c = '    def test_step_align(self, batch, gaussians):\n'
assert s.count(c) == 1
s = s.replace(c, '''    @torch.no_grad()
    def _save_supp_map(self, batch, full_gaussians, retention, dump):
        import numpy as np
        root = Path(os.environ["SUPP_MAP_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        G = int(full_gaussians.opacities.shape[1])
        keep = torch.zeros(G, dtype=torch.bool, device=full_gaussians.opacities.device)
        if retention.indices is None:
            keep[:] = True
        else:
            keep[retention.indices[0]] = True
        rel = None if not dump else dump.get("reliability")
        rel = np.zeros(G, np.float32) if rel is None else flatten_gaussian_scores(rel, 1, G, "reliability")[0].float().cpu().numpy()
        ctx = (batch["context"]["image"][0].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        name = get_cfg()["wandb"]["name"]
        np.savez_compressed(root / f"{name}_{batch['scene'][0]}.npz", ctx=ctx, rel=rel, keep=keep.cpu().numpy())

''' + c)
open(p, 'w').write(s)
print('PATCHED3')
