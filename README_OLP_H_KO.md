# NoPoSplat OLP-H A100 재현 실험

이 저장소는 `cvg/NoPoSplat` 위에 OLP-H 실험 레이어를 추가한 버전입니다.
핵심 OLP-H 로직은 새 모듈로 분리했고, 기존 NoPoSplat 파일에는
config 전달과 head 호출을 위한 얇은 연결부만 있습니다.

중요 문서:

- 한글 실험 프로토콜: `docs/OLP_H_EXPERIMENT_PROTOCOL_KO.md`
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

밤새 자동 실행:

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=8 \
nohup bash scripts/run_re10k_overnight_protocol.sh \
  > outputs/olp_re10k/overnight_nohup.log 2>&1 &
```

직접 확인 후 fine-tuning부터 자동 실행:

```bash
CUDA_VISIBLE_DEVICES=0 START_STAGE=4 BATCH_SIZE=8 \
nohup bash scripts/run_re10k_overnight_protocol.sh \
  > outputs/olp_re10k/overnight_nohup.log 2>&1 &
```

동일 조건의 baseline FT checkpoint가 이미 있으면:

```bash
CUDA_VISIBLE_DEVICES=0 START_STAGE=5 \
BASELINE_CKPT=/path/to/baseline_ft_control_last.ckpt \
BATCH_SIZE=8 \
nohup bash scripts/run_re10k_overnight_protocol.sh \
  > outputs/olp_re10k/overnight_nohup.log 2>&1 &
```

자세한 실행 순서와 산출물은 `docs/OLP_H_EXPERIMENT_PROTOCOL_KO.md`를
확인합니다.

## OLP-H 핵심 설정

`RandomizedHadamardPreconditioner`는 Gaussian parameter head 직전 feature
`x`에 고정 직교 변환 `Q = H D P / sqrt(C)`를 적용합니다. `H`는
Walsh-Hadamard transform, `D`는 random sign, `P`는 channel permutation입니다.
정규화된 Hadamard transform은 `C`가 2의 거듭제곱일 때 L2 norm을 보존합니다.
NoPoSplat `dpt_gs` 기본 channel은 `C=256`이라 이 조건을 만족합니다.

권장값:

- 원본 재현: `enabled=false`
- baseline latent stats: `enabled=true`, `mix=0.0`, `target_mix=0.0`,
  `warmup_steps=0`, `collect_stats=true`
- zero-shot diagnostic: `enabled=true`, `mix=1.0`, `target_mix=1.0`,
  `warmup_steps=0`
- OLP-H fine-tuning: `enabled=true`, `mix=0.0`, `target_mix=1.0`,
  `warmup_steps=5000`, `seed=0`

참고:

- Hadamard transform: https://en.wikipedia.org/wiki/Hadamard_transform
- OLP-H 상세 변경 보고서: `docs/OLP_H_DIFF_REPORT_KO.md`

## 테스트

OLP-H 단위테스트는 `tests/test_olp_preconditioner.py`에 있습니다. 외부
NoPoSplat dataset dependency 없이 OLP 모듈만 isolated import해서 검증합니다.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/noposplat_pycache \
python3 tests/test_olp_preconditioner.py
```

검증 항목:

- disabled mode가 입력 tensor를 그대로 반환
- `mix=1`에서 Hadamard preconditioner가 L2 norm 보존
- warm-up schedule이 `mix`에서 `target_mix`로 이동
- `collect_stats=true`에서 latent stats 저장
- 같은 `seed`에서 rotation 재현
- random sign/permutation buffer가 checkpoint `state_dict`에 저장되지 않음
- Gaussian tail stats에 `floater_proxy` 포함
