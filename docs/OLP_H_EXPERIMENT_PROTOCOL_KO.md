# OLP-H RE10K A100 실험 프로토콜

이 문서는 NoPoSplat OLP-H 실험을 A100 서버에서 재현 가능하게 실행하기 위한
실행 순서와 예상 산출물을 정리한 문서입니다.

목표 비교는 다음입니다.

```text
NoPoSplat + FT
vs.
NoPoSplat + OLP-H + FT
```

기본 fine-tuning batch size는 `8`로 둡니다. A100에서 OOM이 나면 실험 순서는
유지하고 `BATCH_SIZE=4`로 낮춥니다. Smoke run은 반드시 full run 전에 먼저
실행합니다.

## 내일까지 결과 제출용 Pilot Protocol

시간이 부족한 경우 오늘 실행할 우선순위는 아래 3개 비교입니다.

```text
Run 0: official NoPoSplat re10k.ckpt
Run 1: official NoPoSplat re10k.ckpt + OLP-H zero-shot
Run 3: official NoPoSplat re10k.ckpt -> OLP-H fine-tuning
```

이 pilot은 baseline FT control을 생략합니다. 따라서 결과 해석은
`official pretrained baseline 대비 OLP-H adaptation 결과`로 제한합니다.
`same-budget FT control 대비 OLP-H의 순수 효과`라고 주장하려면 이후
baseline FT control을 추가로 돌려야 합니다.

처음부터 밤새 pilot 전체를 자동 실행하려면 아래 명령을 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=8 \
nohup bash scripts/run_re10k_pilot_overnight.sh \
  > outputs/olp_re10k/pilot_overnight_nohup.log 2>&1 &
```

처음 2~3시간 동안 환경 검증, smoke, original eval, zero-shot을 직접 확인한 뒤
OLP-H fine-tuning부터 자동 실행하려면 아래 명령을 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
START_STAGE=5 \
BASELINE_CKPT=./pretrained_weights/re10k.ckpt \
BASELINE_EVAL_NAME=re10k_official_eval_stats \
OLP_EVAL_NAME=re10k_olp_ft_eval \
BATCH_SIZE=8 \
nohup bash scripts/run_re10k_overnight_protocol.sh \
  > outputs/olp_re10k/pilot_overnight_nohup.log 2>&1 &
```

pilot final eval 산출물:

```text
outputs/olp_re10k/re10k_official_eval_stats/
  benchmark.json
  peak_memory.json
  olp_stats_summary.json
  olp_stats_summary_by_overlap.json
  olp_stats_per_scene.json
  olp_stats_per_scene.csv
  <scene>/color/*.png

outputs/olp_re10k/re10k_olp_ft_eval/
  benchmark.json
  peak_memory.json
  olp_stats_summary.json
  olp_stats_summary_by_overlap.json
  olp_stats_per_scene.json
  olp_stats_per_scene.csv
  <scene>/color/*.png
```

`small`, `medium`, `large`는 이미지 크기가 아니라 RE10K evaluation pair의
overlap 구간입니다. 코드 기준은 다음입니다.

```text
small:  0.05 <= overlap <= 0.30
medium: 0.30 < overlap <= 0.55
large:  0.55 < overlap <= 0.80
ignore: 그 외
```

이미지 해상도는 `+experiment=re10k`와 `config/dataset/re10k.yaml` 기준으로
`256x256`입니다.

controlled comparison까지 밤새 자동 실행하려면 아래 wrapper를 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=8 \
nohup bash scripts/run_re10k_overnight_protocol.sh \
  > outputs/olp_re10k/overnight_nohup.log 2>&1 &
```

실행 상태는 다음 파일에서 확인합니다.

```text
outputs/olp_re10k/overnight/<SESSION_ID>/manifest.txt
outputs/olp_re10k/overnight/<SESSION_ID>/logs/*.log
```

처음 2~3시간은 직접 확인하고 이후만 자동으로 돌릴 수도 있습니다. 이 경우
환경 검증, smoke, original eval, zero-shot까지 직접 확인한 뒤 아래처럼
fine-tuning부터 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
START_STAGE=4 \
BATCH_SIZE=8 \
nohup bash scripts/run_re10k_overnight_protocol.sh \
  > outputs/olp_re10k/overnight_nohup.log 2>&1 &
```

이미 동일 조건의 baseline FT checkpoint가 있다면 baseline FT는 다시 돌리지
않고 OLP-H FT부터 시작할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
START_STAGE=5 \
BASELINE_CKPT=/path/to/baseline_ft_control_last.ckpt \
BATCH_SIZE=8 \
nohup bash scripts/run_re10k_overnight_protocol.sh \
  > outputs/olp_re10k/overnight_nohup.log 2>&1 &
```

단, `BASELINE_CKPT=./pretrained_weights/re10k.ckpt`를 쓰는 경우는 내일까지
결과 제출용 pilot입니다. main controlled comparison의 baseline FT control을
대체하지는 못합니다. OLP-H 모델만 추가 fine-tuning을 받으면 성능 차이가
OLP-H 때문인지 추가 학습 때문인지 분리할 수 없기 때문입니다.

## 0. 공통 전제

저장소 위치:

```bash
cd /Users/kwon_junyoung/Desktop/4rd_1st/캡스톤디자인/NoPoSplat
```

공통 기준:

- GPU: A100
- Python: `3.10`
- CUDA: `11.8`
- PyTorch: `2.1.2+cu118`
- 환경 lock: `env/requirements-a100.lock`
- pretrained checkpoint: `pretrained_weights/re10k.ckpt`
- primary dataset: RE10K 256x256, 2-view
- evaluation index: `assets/evaluation_index_re10k.json`
- 기본 OLP seed: `OLP_SEED=0`

`OLP_SEED`는 학습과 평가에서 반드시 동일하게 유지합니다. OLP-H의 random
permutation/sign buffer는 `persistent=False`라 checkpoint에 저장되지 않기
때문입니다.

## 1. 산출물 위치 규칙

이 실험은 산출물이 두 종류의 경로에 나뉘어 저장됩니다.

### Hydra run directory

`command.sh`, `.hydra`, checkpoint, fingerprint, pretrained load audit는
`hydra.run.dir` 아래에 저장됩니다.

예:

```text
outputs/olp_re10k/re10k_ft_control/<RUN_ID>/
  command.sh
  fingerprint/
  pretrained_load_audit.json
  .hydra/
  checkpoints/
```

### Test output directory

test image, benchmark, peak memory, OLP stats는 NoPoSplat wrapper 로직에 따라
`test.output_path / wandb.name` 아래에 저장됩니다.

예:

```text
outputs/olp_re10k/re10k_ft_control_eval/
  benchmark.json
  peak_memory.json
  olp_stats_summary.json
  olp_stats_per_scene.json
  olp_stats_per_scene.csv
  <scene>/color/*.png
```

따라서 train checkpoint는 `<RUN_ID>`가 붙은 Hydra run directory에서 찾고,
eval 통계와 이미지는 `wandb.name` 기반 output directory에서 찾습니다.

## 2. A100 환경 준비

```bash
bash scripts/create_a100_env.sh
conda activate noposplat-olp-a100

python scripts/check_a100_env.py --output-dir outputs/olp_re10k/env_check
```

예상 산출물:

```text
outputs/olp_re10k/env_check/
  env_check.json
  pip_freeze.txt
  conda_explicit.txt
  nvidia_smi.txt
```

확인할 항목:

- `torch.__version__ == "2.1.2+cu118"`
- `torch.version.cuda == "11.8"`
- `torch.cuda.is_available() == true`
- GPU 이름에 `A100` 포함
- pinned package mismatch 없음

## 3. Smoke Run

목적:

- CUDA extension compile 확인
- pretrained checkpoint load 확인
- OLP-H forward/backward 확인
- checkpoint 저장 확인
- A100 memory footprint 1차 확인

실행:

```bash
BATCH_SIZE=1 MAX_STEPS=2 CKPT_EVERY_N_STEPS=1 \
bash scripts/run_re10k_smoke.sh
```

예상 산출물:

```text
outputs/olp_re10k/smoke/<RUN_ID>/
  re10k_ft_control/
    command.sh
    fingerprint/
    pretrained_load_audit.json
    .hydra/
    checkpoints/
      last.ckpt
      epoch_*.ckpt

  re10k_olp_ft/
    command.sh
    fingerprint/
    pretrained_load_audit.json
    .hydra/
    checkpoints/
      last.ckpt
      epoch_*.ckpt
```

중단 기준:

- smoke run에서 CUDA error가 발생하면 full fine-tuning으로 넘어가지 않습니다.
- `pretrained_load_audit.json`에서 matched key 수가 0이면 checkpoint load 문제를
  먼저 해결합니다.

## 3.1. 밤새 자동 실행 Wrapper

아래 명령은 환경 검증부터 final eval까지 순서대로 실행합니다. 중간 단계가
실패하면 즉시 멈추고 해당 단계 log를 남깁니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=8 \
MAX_STEPS=18751 \
LR=2e-4 \
OLP_SEED=0 \
nohup bash scripts/run_re10k_overnight_protocol.sh \
  > outputs/olp_re10k/overnight_nohup.log 2>&1 &
```

기본값:

- `CUDA_VISIBLE_DEVICES=0`
- `START_STAGE=0`
- `END_STAGE=6`
- `SMOKE_BATCH_SIZE=1`
- `BATCH_SIZE=8`
- `MAX_STEPS=18751`
- `LR=2e-4`
- `CKPT_EVERY_N_STEPS=5000`
- `OLP_WARMUP_STEPS=5000`
- `FINAL_SAVE_IMAGE=true`
- `OLP_SEED=0`

stage 번호:

- `0`: A100 environment check
- `1`: smoke run
- `2`: original RE10K eval
- `3`: OLP zero-shot diagnostic
- `4`: baseline FT control
- `5`: OLP-H FT
- `6`: final eval + tail stats

이미 있는 checkpoint 재사용:

- `BASELINE_CKPT=/path/to/baseline.ckpt`: stage 4를 건너뛰고 이 checkpoint를
  final eval baseline으로 사용합니다.
- `OLP_CKPT=/path/to/olp.ckpt`: stage 5를 건너뛰고 이 checkpoint를 final eval
  OLP-H 모델로 사용합니다.

자동 실행 산출물:

```text
outputs/olp_re10k/overnight/<SESSION_ID>/
  manifest.txt
  run_config.env
  env_check/
  logs/
    00_env_check.log
    01_smoke.log
    02_original_eval.log
    03_olp_zeroshot.log
    04_ft_control.log
    05_olp_ft.log
    06_collect_stats.log
```

현재 진행 상황 확인:

```bash
tail -f outputs/olp_re10k/overnight/<SESSION_ID>/manifest.txt
```

전체 nohup log 확인:

```bash
tail -f outputs/olp_re10k/overnight_nohup.log
```

OOM이 나면 같은 명령에서 `BATCH_SIZE=4`로 낮춰 다시 실행합니다.

## 4. Run 0: Original RE10K Evaluation

목적:

- 공식 `re10k.ckpt`가 현재 코드에서 OLP-H 없이 정상 재현되는지 확인합니다.
- 이 run은 OLP-H가 완전히 꺼진 원본 재현 기준입니다.

실행:

```bash
bash scripts/run_re10k_original_eval.sh
```

Hydra run 산출물:

```text
outputs/olp_re10k/re10k_original/<RUN_ID>/
  command.sh
  fingerprint/
    git_commit.txt
    olp_seed.txt
    env_check.json
    pip_freeze.txt
    conda_explicit.txt
    nvidia_smi.txt
  .hydra/
    config.yaml
    hydra.yaml
    overrides.yaml
```

test 산출물:

```text
outputs/olp_re10k/re10k_original/
  benchmark.json
  peak_memory.json
  <scene>/color/*.png
```

확인할 항목:

- RE10K eval이 끝까지 완료되는지
- reconstruction image가 저장되는지
- `benchmark.json`, `peak_memory.json`가 생성되는지

## 5. Run 1: OLP Zero-Shot Diagnostic

목적:

- fine-tuning 없이 OLP-H를 켰을 때 latent tail과 Gaussian tail이 어떻게
  변하는지 진단합니다.
- 성능은 떨어질 수 있습니다. 이 run은 main result가 아니라 diagnostic입니다.

실행:

```bash
bash scripts/run_re10k_olp_zeroshot.sh
```

Hydra run 산출물:

```text
outputs/olp_re10k/re10k_olp_zeroshot/<RUN_ID>/
  command.sh
  fingerprint/
  .hydra/
```

test 산출물:

```text
outputs/olp_re10k/re10k_olp_zeroshot/
  benchmark.json
  peak_memory.json
  olp_stats_summary.json
  olp_stats_summary_by_overlap.json
  olp_stats_per_scene.json
  olp_stats_per_scene.csv
  <scene>/color/*.png
```

확인할 항목:

- `latent_input_max_rms`
- `latent_output_max_rms`
- `latent_input_kurtosis`
- `latent_output_kurtosis`
- `opacity_p95`, `opacity_p99`
- `scale_max_p95`, `scale_max_p99`
- `floater_proxy`

## 6. Run 2: Baseline Fine-Tuning Control

목적:

- fine-tuning 자체의 효과를 측정합니다.
- OLP-H module은 켜지만 `mix=0.0`, `target_mix=0.0`으로 두어 출력은
  identity와 같게 유지합니다.
- 이렇게 해야 baseline latent stats도 수집할 수 있습니다.

실행:

```bash
BATCH_SIZE=8 MAX_STEPS=18751 LR=2e-4 CKPT_EVERY_N_STEPS=5000 \
bash scripts/run_re10k_ft_control.sh
```

예상 산출물:

```text
outputs/olp_re10k/re10k_ft_control/<RUN_ID>/
  command.sh
  fingerprint/
    git_commit.txt
    olp_seed.txt
    env_check.json
    pip_freeze.txt
    conda_explicit.txt
    nvidia_smi.txt
  pretrained_load_audit.json
  .hydra/
    config.yaml
    hydra.yaml
    overrides.yaml
  checkpoints/
    last.ckpt
    epoch_*.ckpt
```

중요 체크:

- `pretrained_load_audit.json`에서 pretrained encoder weight가 정상 load됐는지
  확인합니다.
- `checkpoints/last.ckpt`가 생성됐는지 확인합니다.
- OOM이 나면 `BATCH_SIZE=4`로 낮춰서 같은 run을 다시 실행합니다.

## 7. Run 3: OLP-H Fine-Tuning

목적:

- baseline fine-tuning control과 동일한 budget에서 OLP-H만 켠 모델을
  fine-tuning합니다.
- warm-up mixing은 `0 -> 1`로 증가합니다.

실행:

```bash
BATCH_SIZE=8 MAX_STEPS=18751 LR=2e-4 CKPT_EVERY_N_STEPS=5000 \
OLP_WARMUP_STEPS=5000 \
bash scripts/run_re10k_olp_ft.sh
```

예상 산출물:

```text
outputs/olp_re10k/re10k_olp_ft/<RUN_ID>/
  command.sh
  fingerprint/
    git_commit.txt
    olp_seed.txt
    env_check.json
    pip_freeze.txt
    conda_explicit.txt
    nvidia_smi.txt
  pretrained_load_audit.json
  .hydra/
    config.yaml
    hydra.yaml
    overrides.yaml
  checkpoints/
    last.ckpt
    epoch_*.ckpt
```

중요 체크:

- baseline FT와 동일한 `BATCH_SIZE`, `MAX_STEPS`, `LR`인지 확인합니다.
- `OLP_SEED=0`을 유지합니다.
- `pretrained_load_audit.json`에서 matched key 수가 0이 아닌지 확인합니다.

## 8. Run 4: Final Eval + Tail Stats 비교

목적:

- baseline FT checkpoint와 OLP-H FT checkpoint를 같은 RE10K evaluation index로
  평가합니다.
- rendering metric과 OLP tail stats를 함께 저장합니다.

실행:

```bash
BASELINE_CKPT=outputs/olp_re10k/re10k_ft_control/<RUN_ID>/checkpoints/last.ckpt \
OLP_CKPT=outputs/olp_re10k/re10k_olp_ft/<RUN_ID>/checkpoints/last.ckpt \
BATCH_SIZE=8 \
SAVE_IMAGE=true \
bash scripts/run_re10k_collect_stats.sh
```

`SAVE_IMAGE=true`를 권장합니다. 최종 발표에서는 metric뿐 아니라 baseline FT와
OLP-H FT의 reconstruction image를 직접 비교해야 하기 때문입니다.

Hydra run 산출물:

```text
outputs/olp_re10k/collect_stats/<RUN_ID>/
  fingerprint/
    git_commit.txt
    olp_seed.txt
    env_check.json
    pip_freeze.txt
    conda_explicit.txt
    nvidia_smi.txt

  re10k_ft_control_eval/
    command.sh
    .hydra/

  re10k_olp_ft_eval/
    command.sh
    .hydra/
```

test 산출물:

```text
outputs/olp_re10k/re10k_ft_control_eval/
  benchmark.json
  peak_memory.json
  olp_stats_summary.json
  olp_stats_summary_by_overlap.json
  olp_stats_per_scene.json
  olp_stats_per_scene.csv
  <scene>/color/*.png

outputs/olp_re10k/re10k_olp_ft_eval/
  benchmark.json
  peak_memory.json
  olp_stats_summary.json
  olp_stats_summary_by_overlap.json
  olp_stats_per_scene.json
  olp_stats_per_scene.csv
  <scene>/color/*.png
```

최종 비교 핵심 파일:

```text
outputs/olp_re10k/re10k_ft_control_eval/olp_stats_summary.json
outputs/olp_re10k/re10k_olp_ft_eval/olp_stats_summary.json
```

overlap bucket별 비교 핵심 파일:

```text
outputs/olp_re10k/re10k_ft_control_eval/olp_stats_summary_by_overlap.json
outputs/olp_re10k/re10k_olp_ft_eval/olp_stats_summary_by_overlap.json
```

## 9. 최종 비교 metric

Rendering metric:

- `psnr_ours`
- `ssim_ours`
- `lpips_ours`

Latent tail metric:

- `latent_input_max_rms`
- `latent_output_max_rms`
- `latent_input_kurtosis`
- `latent_output_kurtosis`

Gaussian tail metric:

- `opacity_p95`
- `opacity_p99`
- `opacity_tail_095`
- `scale_max_p95`
- `scale_max_p99`
- `scale_tail`
- `floater_proxy`

`floater_proxy`는 높은 opacity와 큰 scale이 동시에 나타나는 Gaussian의 비율입니다.
현재 recorder는 `OLP_TAU_SCALE`이 명시되지 않으면 batch의 scale quantile 기반
adaptive threshold를 사용합니다. 기본값은
`OLP_ADAPTIVE_TAU_SCALE_QUANTILE=0.95`입니다.

## 10. 결과 해석 기준

성공:

```text
PSNR/SSIM/LPIPS 유지 또는 개선
+ latent max/RMS, kurtosis 감소
+ opacity/scale tail 감소
+ floater_proxy 감소
```

흡수:

```text
rendering metric 거의 동일
+ latent/Gaussian tail metric도 거의 동일
```

해석:

```text
fine-tuning 중 Gaussian head가 OLP-H rotation을 흡수했을 가능성이 큼
```

실패:

```text
rendering metric 악화
+ tail metric도 감소하지 않음
```

해석:

```text
rotated basis가 Gaussian head semantics를 깨뜨렸거나,
warm-up / LR / freeze policy를 재조정해야 함
```

## 11. 빠른 체크리스트

- [ ] `python scripts/check_a100_env.py` 통과
- [ ] smoke run 통과
- [ ] `pretrained_load_audit.json`에서 matched key 수 확인
- [ ] original eval image 저장 확인
- [ ] OLP zero-shot stats 저장 확인
- [ ] baseline FT `last.ckpt` 저장 확인
- [ ] OLP-H FT `last.ckpt` 저장 확인
- [ ] final eval에서 `SAVE_IMAGE=true`로 reconstruction image 저장
- [ ] baseline/OLP `olp_stats_summary.json` 비교
- [ ] `OLP_SEED`가 train/eval에서 동일한지 확인
