# NoPoSplat OLP-H A100 재현 실험

이 저장소는 `cvg/NoPoSplat` 위에 OLP-H 실험 레이어를 추가한 버전입니다.
핵심 OLP-H 로직은 새 모듈로 분리했고, 기존 NoPoSplat 파일에는
config 전달과 head 호출을 위한 얇은 연결부만 있습니다.

중요 문서:

- 한글 변경 보고서: `docs/OLP_H_DIFF_REPORT_KO.md`
- 영어 변경 보고서: `docs/OLP_H_DIFF_REPORT_EN.md`
- A100 package lock: `env/requirements-a100.lock`
- A100 conda 환경: `env/environment-a100.yml`

기본 config에서는 OLP-H가 꺼져 있습니다. 따라서 run script에서 명시적으로
켜지 않는 한 원본 NoPoSplat 동작을 유지합니다.
`scripts/run_re10k_original_eval.sh`는 원본 재현용이라 OLP-H를 끈 상태로
실행합니다. baseline latent stats는 fine-tuning control 또는
`scripts/run_re10k_collect_stats.sh`에서 `mix=0.0` identity 모드로 수집합니다.

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
