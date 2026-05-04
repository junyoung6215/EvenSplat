# NoPoSplat OLP-H A100 재현 실험

이 저장소는 `cvg/NoPoSplat` 위에 OLP-H 실험 레이어를 추가한 버전입니다.

중요 문서:

- 한글 변경 보고서: `docs/OLP_H_DIFF_REPORT_KO.md`
- 영어 변경 보고서: `docs/OLP_H_DIFF_REPORT_EN.md`
- A100 package lock: `env/requirements-a100.lock`
- A100 conda 환경: `env/environment-a100.yml`

기본 config에서는 OLP-H가 꺼져 있습니다. 따라서 run script에서 명시적으로
켜지 않는 한 원본 NoPoSplat 동작을 유지합니다.

## A100 최소 실행 순서

```bash
scripts/create_a100_env.sh
conda activate noposplat-olp-a100

scripts/run_re10k_smoke.sh
scripts/run_re10k_original_eval.sh
scripts/run_re10k_ft_control.sh
scripts/run_re10k_olp_ft.sh
```

fine-tuning 이후 비교:

```bash
BASELINE_CKPT=/path/to/re10k_ft_control.ckpt \
OLP_CKPT=/path/to/re10k_olp_ft.ckpt \
scripts/run_re10k_collect_stats.sh
```
