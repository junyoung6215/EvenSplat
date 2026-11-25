# EvenSplat supplementary video

Scripts used on the AstraGo server (`/workspace/supp`) to build `final/evensplat_supp.mp4`
(1920x1080, 30 fps, 82 s: title, 3 scenes x 25 s, closing).

- `patch2.py`: hooks for EvenSplat `model_wrapper.py` (per-scene scores via `SUPP_SCORE_FILE`, distant orbit video via `SUPP_ORBIT_DIR`).
- `score_orbit.sh`: per-scene scores (val64, align_pose) for NoPo / OLP / Ours r50 / r75. The orbit used in the video was rerun with `SUPP_ORBIT_BACK=2.5 SUPP_ORBIT_FOCAL=1.2 SUPP_ORBIT_UP=0.6`, align off, on `idx_sel.json`, into `orbit2/`.
- `pf3_score_patch.py`, `pf3_after.sh`: PF3plat scoring. PF3plat feeds targets to its pose estimator, so it runs one target per pass (`idx_t{0,1,2}.json`); its test step also needs `[:, 1:2]` on target intrinsics/near/far/image.
- `pick_scenes.py`: ranks scenes where Ours beats NoPo/OLP.
- `compose.py`: `python compose.py c48f19e2ffa52523,79ce50fb7b021e1d,9d8ca3b8ce050769`. `PREVIEW=dir PREV_AT=60,400` writes stills instead. Fonts: `apt install ttf-mscorefonts-installer`.
- `pipeline.png`: method figure for the planned method section.

Planned method section (~26 s): pipeline on top, zooming into OLP-H/Reliability, stratified retention, fine-tuning; below, "that step only" renders from `videos/reliability-r0.5-cell0` (global top-k, no FT), `videos/reliability-r0.5-cell2` (2x2, no FT), `videos/r_filter-r0.5-cell2` (Ours).
