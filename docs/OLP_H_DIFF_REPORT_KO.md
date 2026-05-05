# OLP-H 변경 사항 보고서

이 저장소는 upstream `cvg/NoPoSplat`을 기반으로 하고, Gaussian parameter
head 입력 안정화를 위한 OLP-H 실험 코드를 모듈 형태로 추가한 버전입니다.

## 기준 버전과 실험 버전

- 기준 저장소: `cvg/NoPoSplat`
- 로컬 기준 commit: `a097a78c5bdd0486493f74abb1165614d86ae952`
- 실험 버전: `olp-h-mvp-cuda118-v0.2`
- 목표 환경: CUDA 11.8 + PyTorch 2.1.2 cu118
- 목표 CUDA/PyTorch: CUDA 11.8, `torch==2.1.2+cu118`

## NoPoSplat 원본과 다른 부분

### 새로 추가한 OLP-H 모듈

- `src/model/encoder/heads/latent_preconditioner.py`
  - `RandomizedHadamardPreconditioner`를 구현합니다.
  - 입력 shape은 `[B, C, H, W]`입니다.
  - channel 축에 `Q = H D P / sqrt(C)`를 적용합니다.
  - warm-up mixing과 latent tail 통계 수집을 지원합니다.

- `src/model/encoder/heads/preconditioner_cfg.py`
  - OLP-H 설정 dataclass와 config field reader를 분리했습니다.

- `src/model/encoder/heads/preconditioner_math.py`
  - power-of-two 검증과 FWHT 구현을 분리했습니다.

- `src/model/encoder/heads/preconditioner_stats.py`
  - latent max/RMS와 kurtosis 계산을 분리했습니다.

- `src/model/encoder/heads/preconditioner_utils.py`
  - `global_step`을 `set_step()`을 가진 모듈로 전달하는 helper입니다.

### 기존 NoPoSplat 파일의 얇은 연결부

기존 파일에는 실험 로직을 넣지 않고, 호출부만 최소로 추가했습니다.

- `src/model/encoder/heads/dpt_gs_head.py`
  - 기본값으로 `self.preconditioner = nn.Identity()`를 둡니다.
  - Gaussian parameter output head 직전에
    `path_1 = self.preconditioner(path_1)`만 삽입합니다.
  - config가 켜져 있을 때만 OLP-H 모듈을 생성합니다.

- `src/model/encoder/heads/__init__.py`
  - `dpt_gs`의 `gs_params` head에만 optional `preconditioner_cfg`를 전달합니다.

- `src/model/encoder/encoder_noposplat.py`
  - optional `orthogonal_preconditioner` config field를 추가합니다.
  - 두 Gaussian parameter head에 같은 config를 전달합니다.
  - forward 시작 시 warm-up step을 전달합니다.

- `src/model/encoder/encoder_noposplat_multi.py`
  - multi-view encoder에도 같은 연결부를 적용했습니다.

center/pts3d branch는 변경하지 않았습니다.

### 평가 통계와 checkpoint 검증 모듈

- `src/analysis/olp_stats.py`
  - latent tail, opacity tail, scale tail, floater proxy를 계산합니다.

- `src/analysis/olp_eval_recorder.py`
  - evaluation 중 OLP latent stats와 Gaussian tail stats를 수집합니다.
  - `olp_stats_summary.json`, `olp_stats_per_scene.json`, CSV를 저장합니다.

- `src/misc/checkpoint_audit.py`
  - pretrained encoder weight load 결과를 출력하고 JSON으로 저장합니다.
  - matched/missing/unexpected key count를 확인할 수 있습니다.

### Config 추가

- `config/model/encoder/noposplat.yaml`
  - `orthogonal_preconditioner` block을 추가했습니다.
  - 기본값은 `enabled: false`라 원본 NoPoSplat 동작을 유지합니다.

- `config/main.yaml`
  - OLP stats 저장용 test field를 추가했습니다.
  - 기본값은 `save_olp_stats: false`입니다.

### 재현성 관련 추가 파일

- `env/environment-cuda118.yml`
- `env/requirements-cuda118.lock`
- `scripts/create_cuda118_env.sh`
- `scripts/check_cuda118_env.py`
- `scripts/run_re10k_*.sh`
- `tests/test_olp_preconditioner.py`

## 실험 시 주의할 점

- `RandomizedHadamardPreconditioner`는 `Q = H D P / sqrt(C)`를 사용합니다.
  `C`는 2의 거듭제곱이어야 하며, NoPoSplat `dpt_gs`의 기본 `C=256`은 이
  조건을 만족합니다. Hadamard transform 배경은
  https://en.wikipedia.org/wiki/Hadamard_transform 를 참고하면 됩니다.
- baseline latent stats를 찍을 때는 OLP-H 모듈을 켜고 `mix=0.0`,
  `target_mix=0.0`, `collect_stats=true`로 둡니다. 출력은 identity와 같고
  latent stats만 수집됩니다.
- `scripts/run_re10k_original_eval.sh`는 공식 재현용이므로 OLP-H를 끈
  상태로 실행합니다. 원본 재현 run과 latent stats 수집 run을 구분합니다.
- OLP-H sign/permutation buffer는 `persistent=False`입니다. 학습과 평가에서
  같은 `OLP_SEED`를 사용해야 같은 rotation이 재현됩니다.
- `floater_proxy`는 fixed `tau_scale`을 주거나 adaptive scale quantile을
  켜면 저장됩니다.
- GPU 서버에서 먼저 `BATCH_SIZE=1`, `MAX_STEPS=2` smoke test로 메모리와
  checkpoint load를 확인하는 것을 권장합니다.
- 단위테스트는 `tests/test_olp_preconditioner.py`에 있으며, disabled mode,
  norm 보존, warm-up schedule, stats 수집, seed 재현성, checkpoint buffer
  제외를 검증합니다.
