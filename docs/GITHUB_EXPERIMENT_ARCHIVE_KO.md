# GitHub 실험 보존 가이드

이 문서는 6월 서버 초기화 전에 GitHub에 올릴 것과 따로 백업할 것을 나누기 위한
체크리스트입니다. 목표는 "어떤 코드와 config로 실험을 돌렸는지"와 "결과 분석에
필요한 작은 산출물"을 남기는 것입니다.

## GitHub에 올릴 것

코드와 재현 정보:

- `src/analysis/`
- `src/model/encoder/heads/`
- `src/model/model_wrapper.py`
- `config/experiment_runs/`
- `scripts/run_re10k_*.sh`
- `scripts/run_re10k.py`
- `scripts/archive_re10k_experiment_artifacts.py`
- `env/environment-cuda118.yml`
- `env/requirements-cuda118.lock`
- `docs/`
- `README_OLP_H_KO.md`, `README_OLP_H_EN.md`

작은 분석 산출물:

- `benchmark.json`
- `peak_memory.json`
- `olp_stats_summary.json`
- `olp_stats_summary_by_overlap.json`
- `olp_stats_per_scene.json`
- `olp_stats_per_scene.csv`
- `.hydra/`
- `command.sh`
- `pretrained_load_audit.json`
- `fingerprint/`
- `manifest.txt`
- `run_config.env`
- `logs/*.log`
- 대표 render image 몇 장

## GitHub에 올리지 않을 것

- `outputs/` 전체
- `datasets/`
- `*.torch`
- `*.ckpt`
- 전체 render image 묶음
- smoke/sanity 결과 로그
- CUDA build cache, virtualenv, `__pycache__/`

checkpoint가 2학기에 필요하면 GitHub가 아니라 Hugging Face, Google Drive,
학교 스토리지, external disk 같은 별도 저장소에 백업합니다.

## 분석 산출물 모으기

서버에서 실험이 끝난 뒤 아래 명령으로 GitHub에 올릴 수 있는 작은 파일만
`experiment_records/<ARCHIVE_NAME>/` 아래로 모읍니다.

```bash
python scripts/archive_re10k_experiment_artifacts.py \
  --name re10k_olp_h_main_2026 \
  --include-images \
  --max-images 12
```

기본 동작은 `outputs/olp_re10k` 아래에서 분석용 JSON/CSV/log/config만 찾고,
경로 이름에 `smoke` 또는 `sanity`가 들어간 결과는 제외합니다.

추가로 별도 로그를 같이 남기려면:

```bash
python scripts/archive_re10k_experiment_artifacts.py \
  --name re10k_olp_h_main_2026 \
  --extra-path logs/olp_ft_7k_bs4.txt
```

먼저 복사 대상을 확인만 하려면:

```bash
python scripts/archive_re10k_experiment_artifacts.py \
  --name dry_run_check \
  --include-images \
  --dry-run
```

## 업로드 전 확인

```bash
git status --short
git add README_OLP_H_KO.md README_OLP_H_EN.md docs scripts config env tests src experiment_records
git status --short
```

`git status`에 `outputs/`, `datasets/`, `*.ckpt`, `*.torch`가 보이면 add하지
않습니다.

커밋 메시지 예시:

```bash
git commit -m "Archive OLP-H RE10K experiment code and records"
git push origin main
```

## 2학기 재개 때 볼 순서

1. `README_OLP_H_KO.md`
2. `docs/SERVER_RUNBOOK_KO.md`
3. `docs/OLP_H_EXPERIMENT_PROTOCOL_KO.md`
4. `experiment_records/<ARCHIVE_NAME>/archive_manifest.json`
5. `experiment_records/<ARCHIVE_NAME>/outputs/**/.hydra/`
6. `experiment_records/<ARCHIVE_NAME>/outputs/**/command.sh`
7. `experiment_records/<ARCHIVE_NAME>/outputs/**/benchmark.json`
8. `experiment_records/<ARCHIVE_NAME>/outputs/**/olp_stats_summary*.json`
9. `experiment_records/<ARCHIVE_NAME>/outputs/**/olp_stats_per_scene.csv`
