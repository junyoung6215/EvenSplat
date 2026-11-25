p='/workspace/PF3plat/src/model/model_wrapper.py'
s=open(p).read()
if 'SUPP_SCORE_FILE' in s: print('ALREADY'); raise SystemExit
a='            self.metrics["all"]["lpips"].append(compute_lpips(rgb_gt, rgb).mean().item())\n'
assert s.count(a)==1
s=s.replace(a,a+'''            if os.environ.get("SUPP_SCORE_FILE"):
                import json as _json
                with open(os.environ["SUPP_SCORE_FILE"], "a") as _f:
                    _f.write(_json.dumps({"scene": scene, "psnr": self.test_step_outputs["psnr"][-1], "ssim": self.test_step_outputs["ssim"][-1], "lpips": self.test_step_outputs["lpips"][-1], "n": int(gaussians.means.shape[1])}) + "\\n")
''')
if '\nimport os\n' not in s: s='import os\n'+s
open(p,'w').write(s); print('PF3 SCORE PATCH OK')
