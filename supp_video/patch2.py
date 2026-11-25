p='/workspace/EvenSplat/src/model/model_wrapper.py'
s=open(p).read()
if '_save_supp_orbit' in s: print('ALREADY'); raise SystemExit
a='            self.log_dict(all_metrics)\n'
assert s.count(a)==1
s=s.replace(a,a+'''            if os.environ.get("SUPP_SCORE_FILE"):
                import json as _json
                with open(os.environ["SUPP_SCORE_FILE"], "a") as _f:
                    _f.write(_json.dumps({"scene": batch["scene"][0], "psnr": float(all_metrics["psnr_ours"]), "ssim": float(all_metrics["ssim_ours"]), "lpips": float(all_metrics["lpips_ours"]), "n": int(gaussians.means.shape[1])}) + "\\n")
''')
b='        gaussians = retention.gaussians\n'
assert s.count(b)==1
s=s.replace(b,b+'''        if os.environ.get("SUPP_ORBIT_DIR"):
            self._save_supp_orbit(batch, gaussians)
''')
c='    def test_step_align(self, batch, gaussians):\n'
assert s.count(c)==1
s=s.replace(c,'''    @torch.no_grad()
    def _save_supp_orbit(self, batch, gaussians):
        import json, imageio
        root = Path(os.environ["SUPP_ORBIT_DIR"])
        scene = batch["scene"][0]
        out = root / get_cfg()["wandb"]["name"]
        out.mkdir(parents=True, exist_ok=True)
        (root / "params").mkdir(parents=True, exist_ok=True)
        pf = root / "params" / f"{scene}.json"
        E = batch["context"]["extrinsics"][0]
        dev = E.device
        if pf.exists():
            P = json.loads(pf.read_text())
        else:
            w2c0 = E[0].inverse()
            pc = (w2c0[:3, :3] @ gaussians.means[0].T + w2c0[:3, 3:]).T
            P = {"d": float(pc[:, 2].median())}
            pf.write_text(json.dumps(P))
        d = P["d"]
        back = float(os.environ.get("SUPP_ORBIT_BACK", "1.0"))
        up = float(os.environ.get("SUPP_ORBIT_UP", "0.3"))
        rad = float(os.environ.get("SUPP_ORBIT_RAD", "0.35"))
        n = int(os.environ.get("SUPP_ORBIT_FRAMES", "240"))
        res = int(os.environ.get("SUPP_ORBIT_RES", "512"))
        mid = (E[0].inverse() @ E[-1])[:3, 3] * 0.5
        target = mid + torch.tensor([0.0, 0.0, d], device=dev)
        base = mid + torch.tensor([0.0, -up * d, -back * d], device=dev)
        t = torch.arange(n, device=dev, dtype=torch.float32) / n * 2 * torch.pi
        pos = base[None] + torch.stack([torch.sin(t) * rad * d, (1 - torch.cos(t)) * 0.5 * rad * d, torch.zeros_like(t)], -1)
        dn = torch.tensor([0.0, 1.0, 0.0], device=dev)
        mats = []
        for p_ in pos:
            z = target - p_; z = z / z.norm()
            x = torch.linalg.cross(dn, z); x = x / x.norm()
            y = torch.linalg.cross(z, x)
            M = torch.eye(4, device=dev)
            M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, p_
            mats.append(E[0] @ M)
        extr = torch.stack(mats)[None]
        K = batch["context"]["intrinsics"][0, :1].clone()
        K[:, 0, 0] *= float(os.environ.get("SUPP_ORBIT_FOCAL", "0.8"))
        K[:, 1, 1] *= float(os.environ.get("SUPP_ORBIT_FOCAL", "0.8"))
        near = batch["context"]["near"][0, :1] * 0.1
        far = torch.full_like(near, max(float(batch["context"]["far"][0, 0]), 20 * d))
        frames = []
        for i in range(0, n, 30):
            k = min(30, n - i)
            o = self.decoder.forward(gaussians, extr[:, i:i + k], K[None].expand(1, k, 3, 3), near[None].expand(1, k), far[None].expand(1, k), (res, res))
            frames.append((o.color[0].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu())
        frames = torch.cat(frames).numpy()
        imageio.mimwrite(str(out / f"{scene}.mp4"), list(frames), fps=30, quality=10, macro_block_size=1)

'''+c)
open(p,'w').write(s); print('PATCH2 OK')
