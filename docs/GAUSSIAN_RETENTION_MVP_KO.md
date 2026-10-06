# NoPoSplat Reliability 기반 Adaptive Gaussian Retention MVP

최종 갱신: 2026-10-06 — 선택형 backbone freeze / head fine-tuning, 실행기 옵션 및 validation 경로 수정

현재 구현은 **학습 시에만 offline teacher target으로 Reliability를 보정하고,
추론에서는 NoPoSplat의 예측 Reliability와 가벼운 image-detail demand로 retention하는
구조**다. 기존 V1 loss와 foundation 없는 경로는 유지한다. 실제 VGGT-Ω/DA3 실행 및
native output → teacher target 변환기는 아직 구현·검증하지 않았다. 아래 9~12절은
서버 할당 후 실행할 최신 작업 순서이며, 성능이 검증됐다는 뜻은 아니다.

이번에는 기존 checkpoint를 활용한 짧은 실험을 위해 **전체 학습(`full`) / backbone을
고정한 출력 head 학습(`heads`)**을 선택할 수 있게 추가했다. 기본값은 기존 `full`이다.
`heads`는 Reliability만 학습하는 것이 아니라 **geometry·Gaussian·Reliability head를
함께 학습**한다. 학습 범위만 바꾸며 retention, loss, prior, local demand는 독립적으로
선택한다. 서버에서는 먼저 50% budget의 짧은 heads/full pilot으로 품질과 비용을 비교한다.

이 문서는 NoPoSplat에 추가한 Gaussian Reliability 기반 hard retention과
local-cell allocation의 현재 구현 상태, 연구 의도, 설정 방법, 검증 범위를 한곳에
정리한다.

현재 단계의 목표는 새로운 backbone이나 Gaussian decoder를 만드는 것이 아니다.
기존 NoPoSplat과 checkpoint를 최대한 보존하면서, 이미 예측된 Gaussian 중 중요한
Gaussian만 rasterizer에 전달해 quality--compactness trade-off를 개선하는 것이다.

## 1. 연구 전체에서의 위치

현재 모델 흐름은 다음과 같다.

```text
NoPoSplat feature
-> OLP-H feature preconditioning
-> Gaussian parameter head + Reliability head
-> soft suppression: alpha'_g = alpha_g * r_g
-> hard retention: score에 따른 Gaussian 선택
-> 선택된 Gaussian만 rasterizer에 전달
-> reconstruction loss 및 pruning-aware fine-tuning
```

각 구성요소의 역할은 서로 다르다.

- OLP-H: Gaussian parameter 예측 직전 feature의 channel spike를 완화한다.
- Reliability: 각 Gaussian을 얼마나 신뢰할 수 있는지 `r_g`로 예측한다.
- Soft suppression: 낮은 Reliability의 rendering contribution을 연속적으로 줄인다.
- Hard retention: 실제 Gaussian 수를 줄여 rasterization 비용 감소로 연결한다.
- Local demand: 신뢰도와 별개로, 어느 영역에 표현 용량이 더 필요한지 나타낸다.

따라서 Reliability와 hard pruning은 중복 기능이 아니다. Soft suppression은
Reliability를 학습하고 Gaussian의 기여도를 조절하는 단계이며, hard retention은
그 점수를 실제 primitive 선택과 비용 감소에 사용하는 단계다.

## 2. 현재 구현된 기능

### 2.1 Gaussian hard retention

구현 파일:

- `src/model/gaussian_retention.py`
- `src/model/model_wrapper.py`
- `src/config.py`
- `src/main.py`
- `config/main.yaml`

`GaussianRetention`은 Gaussian 생성이 끝난 뒤, decoder/rasterizer에 전달하기 전에
동작한다. 선택된 index는 모든 Gaussian 속성에 동일하게 적용된다.

```text
means:       [B, G, 3]
covariances: [B, G, 3, 3]
harmonics:   [B, G, 3, D_sh]
opacities:   [B, G]
```

지원하는 budget mode는 다음과 같다.

- `fixed_ratio`: 전체 Gaussian의 일정 비율을 유지한다.
- `fixed_count`: 지정한 개수를 정확히 유지한다.
- `elastic`: threshold 이상만 유지하되 `max_gaussians` upper bound를 적용한다.

`elastic` mode는 장면마다 Gaussian 수가 달라지므로 현재 batch size 1만 지원한다.
학습에는 fixed budget을 우선 사용하고, elastic mode는 장면별 추론 평가에 사용하는
것이 안전하다.

### 2.2 공정한 pruning 비교군

동일한 Gaussian budget에서 `score_source`만 바꾸어 비교할 수 있다.

| `score_source` | 실제 점수 | 실험 목적 |
| --- | --- | --- |
| `random` | deterministic random score | 단순 개수 감소 효과 측정 |
| `base_opacity` | Reliability 적용 전 `alpha_g` | 원래 opacity 순위의 정보량 측정 |
| `opacity` | Reliability 적용 후 `alpha_g * r_g` | opacity와 Reliability 결합 효과 측정 |
| `reliability` | 순수 `r_g` | Reliability 순위 자체의 유효성 측정 |
| `external` | 외부에서 공급한 Gaussian score | 향후 foundation prior 연결용 |

`opacity`는 이미 soft suppression이 적용된 값이라는 점에 주의해야 한다. 따라서
Reliability가 켜진 상태의 opacity top-k는 `alpha_g * r_g`를 기준으로 선택한다.

Random baseline은 별도의 local random generator를 사용한다. 같은 seed와 sample
identifier에는 같은 결과를 만들며, PyTorch global RNG를 변경하지 않는다.

### 2.3 Pruning-aware fine-tuning

Retention은 평가뿐 아니라 학습 중에도 적용할 수 있다.

```text
전체 Gaussian 예측
-> hard top-k 선택
-> 선택된 Gaussian만 rendering
-> target image와 reconstruction loss 계산
-> 선택된 Gaussian을 생성한 encoder/head로 gradient 전달
```

Hard top-k index 자체는 미분할 수 없으므로 detached score로 결정한다. 그러나
선택된 Gaussian의 mean, covariance, color, opacity tensor는 gather 이후에도 gradient를
유지한다. 따라서 모델은 Gaussian이 줄어든 조건에서 남은 Gaussian을 조정할 수 있다.

Reliability ranking에는 top-k 선택으로부터 직접 gradient가 흐르지 않는다.
Reliability는 기존 soft opacity gate와 optional Reliability auxiliary loss를 통해
학습된다. 기존 auxiliary loss용 metadata는 hard retention 전에 전체 Gaussian에서
계산되므로, pruning이 켜져도 loss의 의미가 바뀌지 않는다.

`start_step`을 사용하면 초기 warm-up 이후 hard retention을 시작할 수 있다.

### 2.4 기존 Reliability auxiliary loss(V1) 안전장치

Reliability auxiliary loss와 soft opacity gate는 gradient 경로를 분리해 제어한다.
Reliability head가 출력한 원본 tensor를 `r_g`라고 하면 rendering에는 다음 gate를
사용한다.

```text
detach_gate = false: r_render = r_g
detach_gate = true:  r_render = stop_gradient(r_g)
alpha'_g = alpha_g * r_render
```

`detach_gate=true`는 reconstruction loss가 opacity gate를 통해 Reliability head를
학습시키는 경로만 차단한다. Auxiliary loss와 retention ranking에는 detach되지 않은
원본 `r_g`를 사용하므로 Reliability head는 auxiliary supervision으로 계속 학습할 수
있다. 이전처럼 원본 tensor 자체를 detach하면 soft gate와 auxiliary loss의 gradient가
모두 끊기므로 두 목적을 분리할 수 없다.

다만 기존 V1 auxiliary loss는 다음처럼 비음수 위험 신호에 대한 단방향 페널티다.

```text
L_float = mean_g[r_g * stopgrad(opacity_g) * stopgrad(residual_risk_g)]
dL_float / dr_g >= 0
```

따라서 `detach_gate=true`로 reconstruction의 반대 gradient까지 제거하면 auxiliary
loss만의 최적점은 유효 Gaussian도 `r_min` 방향으로 보내는 퇴화해가 될 수 있다.
`detach_gate=false`를 기본 학습 설정으로 유지하면 reconstruction loss가 품질 보존에
필요한 Reliability를 높이는 반대 압력을 제공해 위험 억제와 재구성 품질 사이의
trade-off가 생긴다. 현재의 `detach_gate=true`는 gradient 경로를 분리해 확인하는
진단용 ablation으로만 사용하고, 실제 학습에 사용할 때는 budget/anchor 또는 양방향
target 같은 balancing objective를 먼저 추가해야 한다. 이 조합을 사용하면 모델 초기화
시에도 해당 위험을 경고한다.

Reliability loss가 `enabled=true`이고 weight가 0이 아닌데 encoder의
`gaussian_reliability.enabled=false`이면 모델 초기화 시 즉시 실패한다. 이 조합을
허용하면 상수 1인 Reliability에 loss를 계산하게 되어 loss 값은 기록되지만 학습 가능한
Reliability head가 없는 잘못된 실험이 될 수 있기 때문이다.

Auxiliary loss warm-up 계수가 0인 step에서는 loss가 metadata 없이 정확히 0을 반환한다.
`ModelWrapper`도 이 구간에는 Reliability용 visualization dump, full-Gaussian context
rendering, residual sampling을 실행하지 않는다. Warm-up이 실제로 양수가 되는 step부터
해당 경로를 활성화한다. 이는 loss의 수학적 값은 유지하면서 불필요한 초기 연산과
메모리 사용을 제거한다.

### 2.5 Local-cell allocator

구현 파일:

- `src/model/local_cell_retention.py`
- `src/dataset/types.py`
- `src/model/model_wrapper.py`

Gaussian을 source image의 local cell로 묶고, Gaussian score와 cell demand를 결합한다.

```text
q_g = s_g * d_c(g)
```

`score_source=reliability`로 설정하면 연구에서 의도한 다음 식이 된다.

```text
q_g = r_g * d_c(g)
```

현재 구현은 NoPoSplat의 실제 dense flatten 순서를 따른다.

```text
view -> pixel -> surface -> sample
```

Cell size는 8, 16 등으로 설정할 수 있다. Full-resolution demand
`[B, V, H, W]`를 받으면 cell 평균으로 pooling하며, 이미 pooling된
`[B, V, cell_h, cell_w]`도 받을 수 있다.

각 cell에 동일한 quota를 배정하지 않는다. 두 source view와 모든 cell의 Gaussian이
하나의 global budget 안에서 경쟁한다. 다만 cell이 완전히 비는 현상을 막기 위한
optional `min_per_cell=1`을 지원한다. 현재는 ablation을 위해 0 또는 1만 허용한다.

중요한 현재 범위는 다음과 같다.

- 구현됨: cell grouping, demand pooling, `score * demand`, global budget, optional 최소 유지,
  ON/OFF 가능한 image-detail 휴리스틱 demand(9절)
- 미구현: 실제 foundation output에서 좋은 `d_c`를 생성하는 방법

즉 allocator 구조는 구현되었지만, foundation-grounded demand의 성능은 아직
검증되지 않았다.

### 2.6 Retention diagnostics

다음 값을 학습 또는 평가 log로 기록한다.

- retention 입력 및 출력 Gaussian 수
- 전체 kept fraction
- 전체, 선택, 제거 score 평균
- view별 kept fraction의 최소, 최대, 표준편차
- cell별 kept fraction의 최소, 최대, 표준편차
- Gaussian이 하나도 남지 않은 empty-cell 비율

이 통계는 성능 저하가 단순한 Gaussian 수 감소 때문인지, 특정 view나 cell을
과도하게 제거했기 때문인지 구분하는 데 사용한다.

기존 OLP evaluation recorder에는 opacity, scale, covariance volume tail 통계가 있다.
다만 retention 전후 tail을 한 번에 직접 비교하는 전용 log는 아직 구현하지 않았다.

## 3. 기본 설정과 모듈성

기본 설정은 `config/main.yaml`에 있다.

```yaml
model:
  gaussian_retention:
    enabled: false
    apply_during_train: true
    apply_during_eval: true
    score_source: opacity
    external_score_key: retention_score
    random_seed: 0

    mode: fixed_ratio
    keep_ratio: 0.5
    keep_count: null
    threshold: 0.5
    max_gaussians: null
    start_step: 0

    log_diagnostics: true
    diagnostics_every_n_steps: 100

    local_cells:
      enabled: false
      cell_size: 16
      demand_key: cell_demand
      demand_floor: 0.0
      min_per_cell: 0
```

기본값은 `enabled: false`다. 이 경우 기존 Gaussian 객체를 gather나 copy 없이 그대로
통과시킨다. `keep_ratio: 1.0`도 동일한 identity path를 사용한다.

Retention과 local allocator는 자체 trainable parameter가 없고 checkpoint state를
추가하지 않는다. 따라서 기능이 꺼져 있을 때 기존 NoPoSplat checkpoint 구조에
영향을 주지 않는다.

주요 ON/OFF 단위:

- 전체 retention: `gaussian_retention.enabled`
- 학습 적용: `apply_during_train`
- 평가 적용: `apply_during_eval`
- warm-up: `start_step`
- score 비교군: `score_source`
- budget 방식: `mode`
- local allocation: `local_cells.enabled`
- cell 최소 유지: `local_cells.min_per_cell`
- 학습 범위: `train.finetune_scope=full|heads` — 아래 설명 참조
- 학습 validation 영상: `train.validation_videos` — 기본 true, retention preset/runner false

Reliability 관련 추가 의미:

- `gaussian_reliability.detach_gate=false`: reconstruction과 auxiliary loss가 모두
  Reliability head를 학습할 수 있으며 현재 MVP의 권장 기본값이다.
- `gaussian_reliability.detach_gate=true`: reconstruction gradient만 차단하고
  auxiliary loss gradient는 유지하지만, 기존 V1 단방향 loss에서는 `r_min` 수축 위험이
  있으므로 진단용 ablation으로만 사용한다.
- V1 Reliability auxiliary loss의 warm-up 계수가 0이면 추가 context rendering을 생략한다.
  새 prior loss는 활성화된 뒤에도 V1용 context rendering을 요구하지 않는다.

### 학습 범위 선택: 전체 학습 / 출력 head fine-tuning

구현: `src/model/training_scope.py`의 `apply_training_scope`, `ModelWrapper.__init__/train`,
`src/main.py`의 초기화 검사·학습 범위 보고서. 모델 구조나 state_dict key를 바꾸지 않는다.

| 범위 | `full` (기본값) | `heads` |
| --- | --- | --- |
| `encoder.backbone` 전체 | 기존 학습 설정 유지 | gradient OFF, eval mode 고정 |
| Transformer encoder·두 decoder·intrinsic embedding | 기존 학습 설정 유지 | 모두 고정 |
| `downstream_head1/2` | 학습 | 학습 — Gaussian center/geometry 보정 |
| `gaussian_param_head` / `gaussian_param_head2` | 학습 | 학습 — scale·rotation·opacity·SH 및 활성화된 Reliability branch |
| OLP-H, retention, local allocator | 기존 설정 유지 | 동일 — 학습 범위와 별개 |

여기서 고정하는 Transformer decoder는 **backbone 내부 decoder**다. Gaussian 출력 head나
rasterizer를 고정한다는 뜻이 아니다. `heads`에서도 선택된 Gaussian 속성을 통해 geometry와
Gaussian head로 reconstruction gradient가 흐른다. Hard top-k의 index 결정 자체는 미분하지
않으며, Reliability의 reconstruction gradient에는 기존 soft gate가 사용된다.

- Frozen backbone은 optimizer에서 제외된다. Validation 뒤 `train()`으로 돌아와도 backbone만
  eval 상태를 유지하고 출력 heads는 train 상태로 복귀한다.
- 일반적인 RGB/intrinsics 입력은 gradient를 요구하지 않으므로 frozen backbone 내부 backward
  graph가 만들어지지 않는다. **전체 encoder를 no_grad/inference_mode로 감싸지는 않는다.**
- Backbone backward·gradient·optimizer state 비용은 줄지만, backbone forward·출력 feature·head
  backward·rasterizer는 남는다. 추론 구조 및 encoder forward 비용은 바뀌지 않는다.
- `full`은 기존 requires_grad/LR 설정을 그대로 사용한다. 기존 LR 분류도 유지: Gaussian 및
  Reliability head는 `optimizer.lr`, geometry heads는 `lr × backbone_lr_multiplier` 그룹이다.
  Runner 기본값에서는 각각 `2e-5`, `2e-6`이며 warm-up/scheduler도 기존 방식이다.
- `heads`는 저장된 NoPoSplat checkpoint가 필요하다. 초기화 시 backbone parameter 누락은 오류로
  처리해 무작위 backbone을 고정하는 실수를 막는다. Head 누락은 기존 non-strict 로드처럼 허용하므로
  `pretrained_load_audit.json`에서 **기존 geometry/Gaussian/Reliability 가중치가 복원됐는지** 확인한다.
- 학습 범위 변경은 **새 run에서** 한다. `train --checkpoint`로 가중치만 가져오면 optimizer와
  step은 초기화된다. 같은 범위의 정확한 resume는 `checkpointing.load`를 사용하며,
  `full ↔ heads` optimizer resume는 차단한다. 범위 표식이 없는 과거 checkpoint는 full로 간주한다.
- `heads` checkpoint에도 전체 가중치가 저장된다. 기존 평가 경로로 불러올 수 있고 추론 시 추가 모듈,
  teacher 실행, per-scene post-optimization은 요구하지 않는다.

성능 보장은 아니다. 기존 feature로 충분히 보정되지 않으면 heads가 full보다 낮은 품질에 머물 수 있다.
따라서 2,000-step pilot 후 회복 추세가 있으면 5,000 step까지 연장하고, 같은 비용 조건의
짧은 full fine-tuning보다 지속적으로 불리하면 full로 돌아간다. 부분 block unfreeze는 이번 범위 밖이다.

## 4. 권장 ablation 묶음

한 번에 모든 기능을 켜지 않고 다음 순서로 효과를 분리한다.

| 단계 | OLP-H | Reliability soft gate | Hard retention | Local demand | 목적 |
| --- | --- | --- | --- | --- | --- |
| A0 | OFF | OFF | OFF | OFF | Original NoPoSplat |
| A1 | ON | OFF | OFF | OFF | OLP-H 단독 효과 |
| A2 | ON | ON | OFF | OFF | Reliability soft suppression 효과 |
| A3 | A2 기준 | 유지 | random | OFF | Gaussian 수 감소 자체의 영향 |
| A4 | A2 기준 | 유지 | opacity | OFF | opacity ranking baseline |
| A5 | A2 기준 | 유지 | reliability | OFF | Reliability ranking 유효성 |
| A6 | A2 기준 | 유지 | reliability | ON | local allocation 추가 효과 |

A3--A6은 반드시 같은 Gaussian budget으로 비교한다. 우선 evaluation-only pruning으로
ranking 품질을 비교한 뒤, 가장 유망한 방식만 pruning-aware fine-tuning한다.

Elastic budget은 fixed-budget 비교가 끝난 이후 별도 실험으로 진행한다. Elastic
결과는 장면별 Gaussian 수가 다르므로 PSNR 하나만 비교하지 않고 Gaussian 수와 함께
Pareto curve로 보고해야 한다.

## 5. 기대 효과와 줄어들지 않는 비용

기대 효과:

- 같은 Gaussian 수에서 random pruning보다 높은 PSNR/SSIM, 낮은 LPIPS
- 비슷한 품질을 더 적은 Gaussian으로 유지
- 불필요하거나 불안정한 Gaussian의 rendering contribution 제거
- rasterization 시간과 peak memory 감소
- 장면별 elastic budget 적용 가능성

Hard retention을 rasterizer 전에 수행하므로 줄어들 수 있는 부분:

- rasterizer가 처리하는 Gaussian 수
- Gaussian projection, sorting, blending 관련 비용
- selected Gaussian attribute를 보관하는 후반부 메모리

줄어들지 않는 부분:

- NoPoSplat encoder/backbone 연산
- Gaussian parameter head 연산
- 전체 후보 Gaussian을 처음 예측하는 비용
- foundation prior를 online으로 실행할 경우의 비용

따라서 최종 속도 향상은 rasterizer가 전체 inference에서 차지하는 비율에 달려 있으며,
A100에서 실제 profiling하기 전에는 수치를 단정하지 않는다.

## 6. 구현 근거와 안전 장치

이 구조를 선택한 근거는 다음과 같다.

1. Gaussian 선택을 rasterizer 전에 해야 실제 rendering 비용을 줄일 수 있다.
2. 동일 budget 비교가 있어야 compactness와 score quality를 분리할 수 있다.
3. Reliability와 local demand를 분리해야 신뢰도와 표현 복잡도를 혼동하지 않는다.
4. Global top-k만 사용하면 특정 view 또는 영역이 비어 버릴 수 있으므로 cell 통계를
   기록하고 optional 최소 유지를 제공한다.
5. Pruning-aware fine-tuning을 통해 단순 삭제 후 발생할 수 있는 구멍을 남은
   Gaussian이 일부 보완할 수 있게 한다.
6. 기존 checkpoint를 유지하기 위해 retention/allocator를 parameter-free 모듈로
   구성하고 기본값을 OFF로 둔다.

잘못된 설정이나 입력은 fail-fast하도록 검사한다.

- score와 Gaussian 수 또는 batch가 맞지 않는 경우
- score 또는 demand에 NaN/Inf가 있는 경우
- 음수 demand가 들어오는 경우
- cell 최소 유지 수가 global budget보다 큰 경우
- fixed-count mode에서 count를 지정하지 않은 경우
- elastic mode에 batch size 1보다 큰 입력을 넣은 경우
- 활성화된 Reliability loss에 대응하는 encoder Reliability head가 없는 경우

## 7. 검증 상태와 검증 환경 구분

### Head fine-tuning 선택형 모드 (2026-10-06 최신 검사)

전체 unittest **134개 중 128개 통과, 6개 skip**. 신규 범위/실행기 회귀 검사는 17개다.
실제 CPU torch optimizer로 frozen weight·buffer 불변, geometry/Gaussian/Reliability head 갱신,
backbone gradient 부재, full 모드 gradient 유지, optimizer 포함 범위와 기존 LR 그룹을 검사했다.
또한 parameter key 보존, pretrained backbone 누락 차단, checkpoint scope resume 검사,
학습·검증 모드 전환, runner의 scope/accumulation 옵션과 manifest 기록을 확인했다.
작은 모델에서 동일 범위 optimizer state 재로드 및 저장 가중치를 full/heads로 불러온 뒤
eval 출력이 같은지도 확인했다.

현재 로컬에 전체 모델 의존성이 없어 Lightning wrapper 메서드는 원본 AST를 경량 harness에서
실행했다. Skip은 Hydra compose 4개와 jaxtyping/beartype 의존 테스트 2개다. 실제 NoPoSplat
checkpoint 복원, Lightning fit/resume 및 GPU rasterizer까지 검증했다는 의미는 아니다.
서버에서 첫 smoke 때 scope 보고서·실제 gradient·validation 복귀·동일 범위 resume를 확인한다.
Smoke 전용 서버 YAML에서는 `steps=4`, `accumulate=1`, `validation_interval=1`,
`checkpoint_interval=2`로 두고 `train --train-scope heads --prune`를 실행하면 학습 도중 validation까지
점검할 수 있다. 이 임시 설정과 결과는 연구 pilot에 섞지 않고, 실제 pilot 전에 원 설정으로 복귀한다.

### Validation 경로 수정 (2026-10-06)

- Validation 영상(interpolation/wobble/extended)은 정량 지표 계산에 사용한 **동일한
  retention 결과**를 재사용한다. 영상마다 encoder를 다시 실행하거나 random selection을
  다시 뽑지 않는다. Retention OFF/100%에서는 전달받은 전체 Gaussian을 그대로 사용한다.
- 학습 중 영상 생성은 `train.validation_videos`로 제어한다. 기존 NoPo 설정의 기본값은
  `true`로 유지하고, retention prior/control/mock preset과 retention runner는 `false`로
  시작한다. 비교 이미지와 validation 지표는 계속 기록한다. 영상이 필요하면 runner YAML의
  `train.validation_videos: true` 또는 직접 Hydra 실행의 `train.validation_videos=true`를 사용한다.
  `test.save_video`와는 별개이며, 영상을 켜면 기존 30/60-frame 렌더링 비용은 발생한다.
- 작은 validation subset의 iterator가 소진되면 새 iterator로 다시 시작한다. 새 iterator에서도
  sample이 없으면 명시적인 오류를 내므로 잘못된 index나 필터링 문제를 숨기지 않는다.
- 기존 **validation당 한 sample** 방식은 변경하지 않았다. `val/psnr`은 subset 전체 평균이
  아니며, subset 전체 비교는 기존 `eval-grid --split validation`으로 수행한다.
- 기존 NoPo의 test pose alignment, 평가 loss/지표, benchmark 방식은 변경하지 않았다.

새 회귀 검사 13개는 CPU DataLoader의 반복/빈 데이터 처리, 실제 validation 메서드의
Gaussian 재사용 경로, 영상 ON/OFF·trajectory 전달, config/runner 옵션을 확인한다.
모델 메서드 검사는 실제 AST를 실행하되 renderer와 영상 저장을 대역으로 바꾼 검사이며,
실제 GPU 렌더링·Lightning 실행 검증을 대신하지 않는다.

```bash
python -B -m unittest discover -s tests -p 'test_validation_retention.py' -v
```

### 설정·실행 오류 수정 (2026-10-01)

- `config/loss/reliability_prior.yaml`에 누락된 `reliability_prior:` wrapper를 추가했다.
  기존 파일은 YAML 문법은 유효했지만 설정이 `loss.weight` 등에 배치되어, 코드가 기대하는
  `loss.reliability_prior.weight`와 맞지 않았다. Loss 식·weight·학습 구조는 변경하지 않았다.
- Retention 테스트의 임시 `jaxtyping` 모듈은 필요한 import 구간에서만 사용하고 제거한다.
  OLP 격리 테스트도 실제 `src` package를 덮어쓰지 않는 전용 namespace를 사용한다.
- Prior/fixed-pair 테스트의 직접 실행 경로를 수정했다. 기존 함수형 테스트도 전체
  `unittest discover`에서 실제 실행하도록 등록했다(이전에는 일부가 수집만 되고 실행되지 않음).
- Loss YAML 구조 회귀 검사와 실제 Hydra preset compose 검사 3개를 추가했다.
  Hydra 검사는 설치된 환경에서 자동 실행하며, 없으면 명시적으로 skip한다.
- 재실행 결과: **80개 통과, skip 항목 5개, 실패/오류 0개**.
  Skip은 Hydra preset 검사 3개와 `test_reliability_loss`, `test_olp_eval_recorder` 모듈 2개다.
  두 모듈에는 총 8개 테스트가 있으므로 skip 5개를 개별 테스트 5개로 해석하면 안 된다.
  Skip된 검증은 통과로 계산하지 않는다. 전체 학습·checkpoint·GPU 검증은 여전히 미실행이다.

집 컴퓨터에서 pull 후 **기존 프로젝트 환경의 Python**으로 실행:

```bash
# 전체 CPU 검사. 마지막 skip 사유까지 확인할 것.
python -B -m unittest discover -s tests -p 'test_*.py' -v

# Hydra가 설치된 환경에서는 prior/control/mock compose 3개가 skip 없이 통과해야 함.
python -B tests/test_retention_config.py -v

# 직접 실행 경로 회귀 확인
python -B tests/test_prior_retention.py
python -B tests/test_fixed_pair_pipeline.py
```

Hydra compose 검사는 모델·dataset을 로드하지 않는다. 이 검사 통과만으로 전체
`load_typed_root_config` 또는 실제 학습 성공을 보장하지 않으므로 Gate 0/1은 별도로 수행한다.
의존성이 없는 legacy 테스트를 직접 실행하면 `SKIP` 이유를 출력하고 nonzero로 종료한다.
필요 패키지가 있는 환경에서 import/실행 오류가 발생하면 정상적인 skip으로 숨기지 않는다.

### 이번 변경 검증 (2026-09-29)

환경은 Python 3.12 / PyTorch 2.4.1+cpu다. 환경 설치·변경이나 GPU 실행은 하지 않았다.

- `test_prior_retention.py`: 27개 통과. cache key/FP16 round-trip/누락·오정렬 거부,
  mock 오사용 방지, teacher detach, 양방향 gradient, unknown/zero-weight 처리,
  surface broadcast, warm-up, dense supervision 연결, image demand와 global budget 확인.
- `test_fixed_pair_pipeline.py`: 10개 통과. 실제 sampler 및 dataset 제어 흐름을 격리하여
  frame 범위, scene filtering, worker sharding, epoch 반복, 누락·중복 거부 확인.
- standalone 테스트: hard retention 15개, local-cell 10개(기존 9 + 신규 1), OLP 11개 통과.
  신규 통합 gradient 테스트에서는 실제 gather 이후 selected geometry만 gradient를 받고,
  제거된 후보의 gate는 dense prior loss로 보정되는 것을 확인했다. 합계 73개 통과.
- 새 loss/wrapper/sampler/dataset 일부는 실제 소스 정의를 AST로 로드한 CPU harness다.
  dependency import만 우회하며, dataset 이미지 decode/crop은 테스트용 입력으로 대체한다.
  **전체 애플리케이션 integration test 통과를 의미하지 않는다.**
- 기존 `test_reliability_loss.py`는 `jaxtyping` 미설치로 실행 불가. Hydra/OmegaConf,
  Lightning, einops 등도 없어 실제 config compose/typed load와 checkpoint end-to-end는
  서버에서 검증해야 한다. 새 YAML은 정적 검사만 가능하다.
- 새 코드에는 학습 파라미터가 추가되지 않는다. 따라서 기존 state_dict key를 늘리지
  않지만, 실제 checkpoint load 및 OFF/100% 이미지 parity 검증은 서버에서 해야 한다.

```bash
python -B -m unittest discover -s tests -p test_prior_retention.py -v
python -B -m unittest discover -s tests -p test_fixed_pair_pipeline.py -v
python -B tests/test_gaussian_retention.py
python -B tests/test_local_cell_retention.py
python -B tests/test_olp_preconditioner.py
```

### 이전 검증 기록 (2026-09-27 문서에서 보존)

아래 43개 및 RTX 4060 Ti 기록은 이전 작업의 기록이며, 이번 CPU 환경에서 재현한
결과로 인용하면 안 된다.

로컬에서 가능한 CPU 단위 검증 결과:

- Gaussian retention 신규 테스트 15개 통과
- Local-cell retention 신규 테스트 9개 통과
- 기존 OLP-H 테스트 11개 통과
- OLP evaluation recorder 테스트 3개 통과
- Reliability loss/gate 테스트 5개 통과
- 총 43개 테스트 통과
- 수정 Python 파일 AST parsing 통과
- Hydra 설정에서 retention/local-cell dataclass 변환 통과
- `git diff --check` 통과

검증한 주요 항목:

- OFF 상태 exact identity
- 100% retention exact identity
- 모든 Gaussian attribute에 동일 index 적용
- 선택된 Gaussian attribute의 gradient 유지
- hard selection score detach
- deterministic random과 global RNG 독립성
- fixed ratio/count 및 elastic upper bound
- source grid와 Gaussian flatten 순서 정렬
- full-resolution demand pooling 및 Gaussian broadcast
- optional cell 최소 유지와 global budget 결합
- batch별 fixed-budget 선택
- 잘못된 shape/config에 대한 fail-fast
- Reliability loss와 encoder head 설정 불일치 fail-fast
- `detach_gate=true`에서 opacity 경로만 detach되고 auxiliary gradient는 유지
- warm-up 계수가 0일 때 metadata 없이 zero loss를 반환

이전 로컬 RTX 4060 Ti에서 수행했다고 기록된 CUDA smoke 검증:

- 실제 입력 규모와 같은 131,072개 synthetic Gaussian에서 50% retention,
  local-cell 최소 유지, gather 및 backward 통과
- 4,096개 synthetic Gaussian을 2,048개로 retention한 뒤 실제
  `diff_gaussian_rasterization` CUDA forward/backward 통과
- CUDA tensor에서 `detach_gate=true`일 때 opacity gradient는 유지되고 Reliability에는
  auxiliary gradient만 전달되는 것 확인
- PyTorch 2.1.2+cu118, CUDA toolkit 11.8, RTX 4060 Ti compute capability 8.9에서 확인

현재 로컬에는 실제 RE10K dataset과 NoPoSplat/MASt3R checkpoint가 없으므로 다음은
아직 확인하지 않았다.

- 실제 NoPo checkpoint end-to-end 실행
- 실제 encoder output에서 retention을 거쳐 CUDA rasterizer까지 연결되는 end-to-end 경로
- A100 40GB 메모리와 속도
- 실제 학습 안정성 및 수렴
- 분산 학습
- 실제 RE10K PSNR/SSIM/LPIPS

## 8. 의도적으로 구현하지 않은 범위

현재 구현에 포함되지 않는 항목:

- VGGT-Omega/DA3 checkpoint 실행
- VGGT-Ω/DA3 native depth/confidence/pose → source-aligned teacher target adapter
- 실제 teacher confidence calibration 및 support/conflict/unknown 정책 검증
- 실제 foundation-grounded cell demand
- 학습 가능한 demand predictor
- retention 전후 tail 전용 비교 log
- bounded local expansion
- child Gaussian residual head
- 별도의 post-optimization stage

pair cache key·NoPo-facing supervision 계약·loader는 이번에 구현했다. 이것과
VGGT native output 스키마는 다르다. 실제 출력을 한 sample에서 확인하고 좌표를
정렬한 후 변환기를 구현해야 한다. mock 데이터로 geometry 효과를 주장하면 안 된다.

Bounded expansion은 MVP에 포함하지 않는다. 먼저 retention-only 실험에서 다음 세
조건이 반복적으로 확인될 때만 추가 여부를 판단한다.

1. 특정 high-demand cell이 지속적으로 100% retention된다.
2. 그 cell의 reconstruction residual이 여전히 높다.
3. 전체 budget을 늘렸을 때 해당 영역의 품질이 실제로 개선된다.

## 9. 이번 추가 구현: 역할과 독립적인 ON/OFF

| 모듈 | 코드 | 목적 / 기대효과 | OFF 방법 |
| --- | --- | --- | --- |
| Fixed pair sampler | `src/foundation/pairs.py`, `src/dataset/view_sampler/view_sampler_fixed_pairs.py` | 같은 학습 pair를 반복하여 cache를 재사용하고 비교군의 데이터 차이를 제거 | 기존 `+experiment=re10k`의 bounded sampler 사용 |
| Prior cache loader | `src/foundation/prior.py`, `src/dataset/dataset_re10k.py` | source image와 정렬된 target·mask·weight를 CPU에서 읽기 | `dataset.re10k.foundation_prior.enabled=false` |
| 양방향 prior auxiliary loss | `src/foundation/supervision.py`, `src/loss/loss_reliability_prior.py` | support는 신뢰도를 올리고 conflict는 낮추며 unknown은 제외 | `loss.reliability_prior.enabled=false` 또는 loss 목록에서 제거 |
| Image-detail demand | `src/model/image_demand.py`, `ModelWrapper._apply_local_cell_allocation` | 이미지 경계·detail 영역의 기존 후보를 더 남기는 휴리스틱 | `model.gaussian_retention.local_cells.enabled=false` |
| 설정 사전 검사 | `src/foundation/validation.py`, `src/main.py` | loss만 켜고 cache를 빼는 등의 오류를 warm-up 전에 검출 | prior loss가 OFF이면 개입하지 않음 |
| Input export / cache 검사 | `scripts/prepare_prior_inputs.py`, `scripts/check_prior_cache.py` | teacher 입력과 NoPo crop의 동일성·cache coverage 검증 | 명시적으로 실행할 때만 동작 |

기존 `src/loss/loss_reliability.py`는 수정하지 않았다. V1과 prior loss는 별개이며
MVP에서는 둘을 동시에 켜지 못하게 했다. 새 optimizer parameter / decoder / backbone은
없다. OLP 설정·seed·mix·r_min은 복원할 checkpoint의 설정과 반드시 일치시킨다.
새 preset의 OLP mix=1, seed=0은 **기본 가정**이며 checkpoint 설정을 확인해 바꿔야 한다.

### Prior loss가 하는 일

```text
학습 입력 pair ──offline teacher──> target t / valid m / weight w (cache)
       │
       └──NoPo encoder──> r [B,V,H,W,S,1]
                            │
                    prior loss (전체 후보)
                            │
             soft opacity gate → hard selection → render → MSE + LPIPS

추론: NoPo → r × image-detail cell demand → top-k → render
      teacher 실행 / cache 조회 / 장면별 Gaussian post-opt 없음
```

정의는 `r_hat=(r-r_min)/(1-r_min)`,
`L_prior=sum(m*w*(r_hat-t)^2)/max(sum(m*w),eps)`다.
pixel target `[B,V,H,W]`를 surface 축에 broadcast하고 분모에도 같은 축을 반영한다.
`L_total=L_MSE+L_LPIPS+lambda*warmup*L_prior`이며 lambda=0.001은 **검증 전 pilot 값**이다.

- teacher target/weight/mask는 detach한다. Encoder Reliability는 detach하지 않는다.
- hard top-k의 선택 index는 미분하지 않지만, prior loss는 **선택 전 전체 r**를 감독한다.
  따라서 제거된 후보도 prior loss gradient를 받을 수 있다.
- selected Gaussian에는 rendering loss가 흐르므로 pruning-aware fine-tuning이 가능하다.
- prior-only 학습에서는 V1용 context render를 추가 실행하지 않는다.
- valid가 전부 false이거나 weight가 0이면 미분 가능한 0을 반환한다. 학습이 계속돼도
  teacher signal이 없는 것이므로 `effective_weight`, `valid_fraction`을 반드시 감시한다.
- teacher support는 “그 pixel의 geometry evidence”이지, 예측 Gaussian의 기하학적 정확성
  자체를 증명하지 않는다. Reliability 순위가 좋아지는지는 동일 budget 실험으로 검증한다.

### Local demand가 하는 일

입력은 정규화를 되돌린 RGB `[B,V,3,H,W]`. 인접 pixel RGB 차이의 평균을 구해 두 view
전체에서 공통 scale로 정규화한다. `d_pixel=1+strength*detail/max(detail)`를 cell 평균
pooling한 후 기존 allocator가 `q_g=r_g*d_cell`로 global competition을 수행한다.
기본 strength=1, cell=16, min_per_cell=0. strength=0 또는 평탄한 입력은 demand=1이다.

이는 **image-detail 휴리스틱**이다. 실제 geometry demand, reconstruction residual 또는
학습된 capacity predictor라고 부르면 안 된다. Texture가 강한 불량 영역을 과대평가할 수
있어 반드시 global Reliability-only와 비교한다. cell마다 동일 quota를 주지 않으며,
최대 후보 수를 늘리거나 새로운 Gaussian을 생성하지 않는다.

### Cache 계약 및 데이터 정렬

- manifest는 `schema_version=1`, `split=train`, scene당 **하나의 ordered context pair**와
  고정 개수 target을 담는다. context와 target은 겹치지 않는다. 다중 pair/scene은 미지원.
- 실제 `train/index.json`에 없는 scene, frame 범위 초과, 잘못된 shape/FOV/baseline은
  조용히 건너뛰지 않고 오류를 낸다. worker별 chunk를 먼저 나누고 shuffle하여 중복을
  막으며 persistent worker가 다음 epoch에서도 같은 chunk 집합을 맡는다. 단일 GPU 전용.
- augmentation=false를 prior arm뿐 아니라 **모든 matched control arm**에 적용한다.
  현재 flip된 prior map 정렬은 구현하지 않았기 때문에 augmentation=true는 거부한다.
- dataset의 기존 `apply_crop_shim` 이후, encoder normalize 이전 RGB를 기준으로 한다.
  360×640 원본은 기존 resize/center-crop을 거쳐 256×256이 된다.
- key는 scene, **순서를 보존한 context indices**, H/W, teacher checkpoint revision,
  teacher settings ID, preprocessing ID, schema version의 SHA256이다. RGB uint8 hash도
  payload에서 대조한다. target frame은 teacher 입력이 아니므로 key에 넣지 않는다.
- cache `.npz`: `teacher_target`, `teacher_weight`는 FP16 `[2,H,W]`, `teacher_valid`는
  bool `[2,H,W]`, metadata는 JSON 문자열. 로드 후 float map은 FP32로 변환된다.
- target와 weight는 유한한 `[0,1]`이어야 한다. VGGT raw confidence를 그대로 넣지 않는다.
- loader는 **train에서만** 활성화된다. val/test는 training manifest 대신 별도 evaluation
  index를 쓰며 cache 없이 동작한다. cache가 누락되면 train은 실패한다. 일부 cache로
  전체 랜덤 pair를 학습하는 자동 fallback은 없다: **manifest 자체가 cached subset**이다.
- mock은 좌/우 0/1 label과 첫 행 unknown을 만드는 배선 점검용이다. 저장 시 `kind=mock`,
  dataset/loss 양쪽 명시적 허용이 필요하며 실제 teacher 실험에 섞지 않는다.

저장량 상한 추정(압축·metadata 제외, 256² / 2 view): maps는 pair당 0.625 MiB,
10,000 pair 약 6.1 GiB, 100,000 pair 약 61 GiB. Exported RGB는 pair당 0.375 MiB를
추가 사용한다. 압축률·파일시스템 overhead·teacher native depth/pose 보관량은 별도다.
학습 batch의 세 map 자체는 FP32/Bool 기준 pair당 약 1.125 MiB이며, loss 연산 중간값과
prefetch는 별도다. Frozen teacher가 NoPo GPU 메모리를 점유하지 않는 것이 주된 이점이다.

## 10. 서버에서 실행할 순서와 명령

### 짧은 명령으로 실행하기 — retention 전용 runner (2026-10-06 갱신)

반복 실험은 **`scripts/run_retention.py`**를 사용한다. 기존 `run_re10k.py`는 변경하지
않았고, 서로 config 형식이 다르다. 새 runner 설정은
**`config/experiment_runs/retention_suite.yaml`**에 있다. 이 파일을 서버의 별도 경로로
복사해 한 번 수정한 후 `--config`로 지정한다. 이 YAML을 `+experiment=`에 넣지 않는다.

한 번 설정할 항목:

- `data_root`, `initial_checkpoint`, `train_pairs`, `validation_index`, `evaluation_index`, `output_root`.
  상대 경로는 현재 터미널 위치나 YAML 위치가 아니라 **저장소 루트 기준**이다.
- `gpu`: 한 장만 지정. 모든 작업을 순차 실행하며 병렬 GPU 실행은 하지 않는다.
- `olp_enabled/olp_mix/olp_seed`, `reliability_enabled/reliability_r_min`: 기존 checkpoint의 설정과 일치시킨다.
  기본 템플릿은 OLP+Reliability checkpoint 기준이지 original NoPo checkpoint 기준이 아니다.
- 학습 범위 `train.finetune_scope`, step·batch·accumulation·LR·loss weight, local cell 크기 및 demand 강도.
- 실제 teacher 검증을 마친 경우에만 `prior.root/teacher_id/teacher_settings_id`.

**기본은 미리보기**다. `--execute`가 없으면 파일 생성, 학습, 평가를 하지 않는다.
출력되는 긴 명령을 다시 복사할 필요 없이 같은 짧은 명령에 `--execute`만 붙이면 된다.
Windows에서도 미리보기가 가능하지만 `/server/...` 경로와 실험 실행은 서버에서 설정한다.

**우선 실행할 pilot: checkpoint 복원·OFF/100% parity와 아래 Gate 1 smoke를 통과한 뒤**, 같은
checkpoint·pair·seed·50% budget·LR·step·accumulation에서 학습 범위만 비교한다. 실제 teacher cache가
없어도 control arm으로 가능하다. 기존 OLP/신뢰도 설정은 checkpoint와 맞춰야 한다.

```bash
CFG=/server/config/retention_suite.yaml

# 기본은 미리보기. 확인 후 같은 명령에 --execute를 붙인다.
python scripts/run_retention.py --config "$CFG" train --arm control --train-scope heads --accumulate 4 --steps 2000 --prune --ratio 0.5 --score reliability --run-id pilot_heads_r50
python scripts/run_retention.py --config "$CFG" train --arm control --train-scope full --accumulate 4 --steps 2000 --prune --ratio 0.5 --score reliability --run-id pilot_full_r50

# 학습 전 초기 checkpoint의 50% 기준선
python scripts/run_retention.py --config "$CFG" eval-grid --scores reliability --ratios 0.5 --run-id before_ft

# heads 학습 후. full pilot도 checkpoint 경로만 바꿔 동일하게 평가한다.
python scripts/run_retention.py --config "$CFG" eval-grid --checkpoint /server/outputs/retention/pilot_heads_r50/control-pruned/checkpoints/last.ckpt --scores reliability --ratios 0.5 --run-id after_heads_ft
```

`run-id`는 실제 실행한 폴더를 재사용할 수 없다. 출력 경로 예시는 YAML의 `output_root`에 맞춰
바꾼다. `--train-scope`와 `--accumulate`는 이번 run만 YAML을 덮어쓴다. 결정한 범위와 accumulation을
서버용 YAML에 저장하면 이후 A/B/C 및 FT/continuation에 매번 긴 옵션을 입력할 필요가 없다.
폴더의 `control-full`은 **Gaussian full-count**라는 의미이며 backbone 학습 범위와는 다르다.

판단 기준: heads의 화질 회복 추세·full 대비 품질 차이·초/optimizer step·peak memory를 함께 확인한다.
허용 품질 차이는 pilot 전에 정하고 같은 validation pair의 PSNR/SSIM/LPIPS 및 실패 사례로 판단한다.
짧은 학습의 마지막 값 하나만으로 결론 내리지 않는다. 처음부터 모든 loss/local 조합을 학습하지 않는다.

**Step과 시간 해석:** batch=1, accumulation=4이면 한 optimizer update에 pair 4개를 사용한다
(단일 GPU 기준). 2,000 step은 대략 8,000 pair 노출이다. 과거 300K step/약 5일이 같은 설정이었다면
동일 속도 단순 환산은 2K 약 48분, 5K 약 2시간이지만 이는 보장 시간이 아니다. 실제 데이터 I/O,
validation 빈도, loss와 freeze 효과를 측정해 예상 시간을 다시 계산한다.
현재 `val_check_interval=500`은 **train batch 기준**이라 accumulation=4에서 약 125 update마다
validation하며, `checkpoint_interval=500`은 optimizer update 기준이다. 비교군의 설정을 동일하게 유지한다.

```bash
# 아래 명령은 저장소 루트에서 실행. CFG는 본인이 수정한 runner YAML 경로.
CFG=/server/config/retention_suite.yaml

# 1. matched control: 먼저 미리보기 → 서버 준비 후 실제 실행
python scripts/run_retention.py --config "$CFG" train --arm control
python scripts/run_retention.py --config "$CFG" train --arm control --run-id control_s1 --execute

# 2. 같은 초기 checkpoint/pair/schedule의 legacy V1 loss 대조군
python scripts/run_retention.py --config "$CFG" train --arm legacy --run-id legacy_s1 --execute

# 3. 실제 teacher cache가 검증된 뒤에만 prior loss 대조군
python scripts/run_retention.py --config "$CFG" train --arm prior --run-id prior_s1 --execute
```

위 세 학습은 별도 요청으로 실행한다. 하나가 끝났다고 다른 arm을 자동 시작하거나
기존 checkpoint를 자동으로 다음 arm에 넘기지 않는다. 기본 5,000 optimizer step,
batch=1, accumulation=1, scope=full이다. 이는 기존 명령의 호환 기본값이며 새로운 권장 pilot은
위처럼 scope와 accumulation을 명시한다. `--steps`, `--accumulate`, `--train-scope`로 이번 run만 바꿀 수 있다.
새 runner는 mock 학습을 연구 arm으로 제공하지 않는다. mock smoke는 아래 Gate 1을 사용한다.

```bash
# 4. 선택한 checkpoint 하나에서 평가. 처음에는 validation 50%만 확인
CKPT=/server/outputs/retention/control_s1/control-full/checkpoints/last.ckpt
python scripts/run_retention.py --config "$CFG" eval-grid --checkpoint "$CKPT" --ratios 0.5 --execute

# 기본 sweep: 4 score × 25/50/75% + OFF/100% 확인 = 14개 순차 평가
python scripts/run_retention.py --config "$CFG" eval-grid --checkpoint "$CKPT" --execute

# 같은 checkpoint·budget에서 image-detail local demand ON/OFF만 비교
python scripts/run_retention.py --config "$CFG" eval-grid --checkpoint "$CKPT" --scores reliability --ratios 0.5 --local both --execute

# 최종 설정 확정 후에만 held-out test index로 전환
python scripts/run_retention.py --config "$CFG" eval-grid --checkpoint "$CKPT" --split test --execute
```

`eval-grid`의 기본 split은 **validation**이다. 파일 이름만 다른 동일 scene을 사용하지
않도록 실행 전에 validation/test의 유효 scene ID 중복을 차단한다. 학습 실행에서는
train pair와 두 평가 index의 scene 중복도 차단한다. 따라서 두 index를 실제로 분리하여
준비해야 한다. 데이터 자체의 frame 범위·카메라·실제 cache coverage 검증을 대신하는 것은 아니다.

OFF와 100%는 배선 확인용이며 별개의 방법으로 중복 집계하지 않는다. 이 runner는 두 조건을
실행할 뿐 **영상 pixel parity를 자동 판정하지는 않는다**. 첫 서버 smoke에서 별도로 확인한다.
모든 평가 조건은 동일한 pose alignment 정책을 사용한다. 기본 `align_pose=true`에는
기존 camera pose optimization 비용이 포함되므로 latency 실험으로 해석하지 않는다.

```bash
# 5. 유망 checkpoint에서 pruning-aware FT와 같은 step의 full-count continuation
# checkpoint/arm/step은 두 명령에서 동일하게 유지. 예시는 control arm.
# 학습 범위와 accumulation은 앞서 선택해 서버 YAML에 저장한 값을 두 run에 동일하게 사용한다.
python scripts/run_retention.py --config "$CFG" train --arm control --checkpoint "$CKPT" --steps 2000 --prune --ratio 0.5 --score reliability --run-id ft_r50_s1 --execute
python scripts/run_retention.py --config "$CFG" train --arm control --checkpoint "$CKPT" --steps 2000 --run-id continuation_s1 --execute

# local을 켜고 FT하려면 첫 명령에 --local 추가. 평가에서도 --local on으로 맞춘다.
# prior/legacy loss를 유지할 경우 두 명령 모두 --arm prior 또는 --arm legacy로 바꾼다.

# 6. 여러 run의 평가 지표를 하나의 CSV로 정리. 새 파일명만 허용.
python scripts/run_retention.py summarize --root /server/outputs/retention --output /server/outputs/retention_summary_v1.csv
```

**초기화와 resume 구분:** `train --checkpoint`는 encoder weight 초기화이며
optimizer/global step은 새로 시작한다. FT와 continuation도 같은 원칙이다. 모든 새 학습은
`save_weights_only=false`로 저장하지만, 이 runner에는 중단 지점의 optimizer를 복원하는
full-resume 옵션이 없다. 정확한 resume가 필요하면 보존된 Hydra 설정을 사용하여 기존
`checkpointing.load` 경로로 별도 실행해야 한다. `--checkpoint`를 resume로 오해하지 않는다.
OLP mix는 고정하고, auxiliary loss warm-up은 새 학습 step 기준으로 다시 적용한다.

결과 폴더 구조:

```text
output_root/run-id/
  manifest.json                  # argv, 설정, checkpoint/index SHA256, Git 상태, 작업 상태
  control-full.log               # 각 subprocess의 stdout/stderr (평가는 조건별 .log)
  control-full/
    .hydra/                     # Hydra가 저장한 실제 config/override
    checkpoints/
    pretrained_load_audit.json
    training_scope.json         # 학습 범위, 실제 trainable/frozen parameter 이름·개수
    local/metrics.jsonl         # W&B OFF에서도 학습/검증 scalar 기록
    local/...                   # 해당 run의 시각화
  baseline-off/test_metrics.json # eval-grid인 경우 조건별 평가 결과
```

`run-id`를 생략하면 UTC 시각+무작위 ID를 생성한다. 같은 폴더 재사용은 실패 처리하여
기존 결과를 덮어쓰지 않는다. 한 작업이 실패하면 다음 작업을 시작하지 않고 상태를 남긴다.
진행 상황은 해당 `.log`를 확인한다. 강제 종료 시 manifest에 `running`이 남을 수 있으므로
그것만으로 현재 프로세스가 살아 있다고 판단하지 않는다. 실패 후에는 새 run-id로 재실행한다.

CSV는 PSNR/SSIM/LPIPS, 점수·비율·local 여부, 기록된 실제 Gaussian 수와 checkpoint/index
hash를 모은다. 누락/실패는 0점으로 채우지 않고 상태와 빈칸으로 남긴다. OFF run은
retention count가 기록되지 않을 수 있다. 서로 다른 checkpoint/split/alignment의 결과를
자동 평균하지 않으며, 전체 설정은 CSV의 manifest 경로에서 확인한다.
학습 manifest의 `training`에는 CLI까지 반영된 scope·accumulation·effective batch·optimizer step이
기록된다. 평가 CSV에 학습 범위를 자동 추정해 넣지는 않으므로 checkpoint와 원 학습 manifest를 함께 보관한다.
**CUDA 동기화 시간·단계별 peak memory·tail 통계의 새 자동 측정 기능은 추가하지 않았다.**
그 부분은 아래 측정 주의와 Gate 절차에 따라 서버에서 별도로 검증한다.

이번 자동화 검증은 작은 가짜 파일·subprocess 대역으로 명령 생성, 옵션 검증, 데이터 split
중복, 출력 충돌 방지, 실패 중단, CSV 및 로거 경로 분리를 확인한다. 실제 GPU 학습 검증은
아니다. 서버의 정식 의존성 환경에서는 아래 테스트의 Hydra compose 항목까지 통과해야 한다.
2026-10-01 로컬 재검사 결과는 전체 **98개 통과, 6개 항목 skip**이다. 새 runner/로거 검사는
18개 통과했으며, 새 Hydra 명령 조합 검사 1개는 Hydra/OmegaConf 부재로 skip했다.
기존 skip 5개도 그대로다. 따라서 전체 GPU 실행까지 오류가 없음을 보장하는 수치가 아니다.

```bash
python -B -m unittest discover -s tests -p 'test_retention_runner.py' -v
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

### 수동 명령과 단계별 점검

이하 명령은 Linux 서버 Bash 예시다. 경로는 실제 할당 경로로 바꾸고, 기존 checkpoint의
OLP/reliability 설정을 먼저 확인한다. 아래 `+experiment`는 `src.main`용 Hydra experiment다.
반복 실험은 위 runner를 우선 사용하고, mock/cache 준비 및 원인 점검은 아래 명령을 사용한다.

### Gate 0 — 환경·checkpoint·OFF parity

1. 서버에서만 기존 NoPo 환경/CUDA rasterizer를 설치·검증한다. VGGT 환경은 별도로 둔다.
   저장소 README의 환경을 우선하며, teacher 패키지를 NoPo 환경에 덮어 설치하지 않는다.
2. HF의 기존 checkpoint를 받아 `pretrained_load_audit.json` 및 load log를 확인한다.
   missing/unexpected Reliability head key가 있으면 기존 head가 복원된 것으로 간주하지 않는다.
3. 7절 CPU 테스트 + 기존 `test_reliability_loss.py`를 정식 의존성 환경에서 모두 재실행한다.
4. 기존 evaluation pair 8~32개에서 retention OFF vs ON/100% output parity를 확인한다.
   preset 변경으로 OLP나 camera frame이 달라진 비교는 parity가 아니다.
5. `test.align_pose=true`는 기존 pose alignment다. 새 Gaussian post-opt가 아니다.
   품질 비교는 모든 arm에서 동일 설정, 속도 비교는 alignment OFF를 별도로 측정한다.

### Gate 1 — 작은 fixed pair와 mock end-to-end

먼저 train split에서 8~32 scene만 선택한다. 실제 frame 번호를 확인한 뒤 아래 형식으로
`train_pairs_pilot.json`을 만든다. 이 예시 scene/번호는 실행 가능한 데이터가 아니다.
overlap을 알면 기록하고, 기본 0.5는 placeholder이므로 실제 overlap별 성능 분석에 쓰지 않는다.

```json
{"schema_version":1,"split":"train","pairs":[
  {"scene":"ACTUAL_TRAIN_SCENE","context":[10,70],"target":[25,40,55,60],"overlap":0.5}
]}
```

```bash
DATA=/server/data/re10k
CKPT=/server/checkpoints/existing_olp_reliability.ckpt
PAIRS=/server/manifests/train_pairs_pilot.json

# 먼저 config만 확인. --cfg는 Hydra compose 확인이며 typed config 실행 검증은 아님.
python -m src.main +experiment=retention_mock \
  model.encoder.pretrained_weights="$CKPT" \
  dataset.re10k.roots="[$DATA]" \
  dataset.re10k.view_sampler.index_path="$PAIRS" --cfg job

# 실제 4-step mock smoke. 연구 결과에 포함 금지.
python -m src.main +experiment=retention_mock \
  model.encoder.pretrained_weights="$CKPT" \
  dataset.re10k.roots="[$DATA]" \
  dataset.re10k.view_sampler.index_path="$PAIRS" \
  hydra.run.dir=outputs/DIAGNOSTIC_prior_mock
```

모든 새 preset은 batch=1, worker=0에서 시작한다. GPU 실행 전 실제 composed config와
typed load가 정상인지 확인한다. mock loss finite / head gradient nonzero / mock_fraction=1,
retention ON 추가 실행 시 정확한 50% count와 backward를 확인한다. 이후 worker=2에서
pair coverage를 다시 확인한다. mock checkpoint는 main 실험 초기화에 사용하지 않는다.

### Gate 2 — 실제 teacher adapter 검증 후 작은 cache만 생성

```bash
python -m scripts.prepare_prior_inputs \
  --data-root "$DATA" --pairs "$PAIRS" --output /server/prior_inputs/pilot \
  --teacher-id vggt-omega-EXACT_REVISION --teacher-settings-id repro416-policy-v1

# 아래 명령은 실제 teacher가 아니라 file-cache 배선 점검용이다.
python -m scripts.check_prior_cache --inputs /server/prior_inputs/pilot \
  --cache /server/prior_cache/DIAGNOSTIC --make-mock
python -m scripts.check_prior_cache --inputs /server/prior_inputs/pilot \
  --cache /server/prior_cache/DIAGNOSTIC --allow-mock
```

서버에서 **추가 구현해야 하는 부분**은 teacher adapter다:

1. VGGT-Ω checkpoint 접근 및 2-view/416 한 pair의 실제 depth/confidence/pose tensor와
   preprocessing을 확인한다. 실행·메모리·전처리 시간을 측정한다. 접근/환경 문제가 계속되면
   DA3로 바꾸되 동일한 NoPo-facing 계약을 사용하고 새 teacher/settings ID를 부여한다.
2. Exported RGB source grid를 teacher에 넣고 resize/pad 좌표변환을 기록한다. 416 map을
   단순 reshape하지 말고 inverse resize/pad로 256 source grid에 되돌린다. valid mask는
   nearest/보수적 유효성 처리, depth/confidence는 의미에 맞는 보간을 검증한다.
3. Teacher 내부 depth/pose로 cross-view reprojection support/conflict를 계산한다.
   NoPo center/coordinate/canonical/GT target pose를 이 teacher pose로 대체하지 않는다.
4. 관측 범위 밖, occluded, low-confidence, invalid depth는 우선 unknown으로 제외한다.
   충분히 관측된 high-confidence mismatch만 conflict로 만든다. confidence 낮음만으로
   target=0을 주지 않는다. confidence calibration과 depth-relative threshold를 train pilot에서
   정하고 settings ID에 버전 고정한다. test quality로 threshold를 튜닝하지 않는다.
5. source RGB 위 target/valid/weight를 overlay하여 최소 8 pair를 직접 검토한다.
   이후 `write_prior_cache(..., kind="teacher")`로 저장하고 아래 검사로 100% coverage 확인.

```bash
python -m scripts.check_prior_cache --inputs /server/prior_inputs/pilot \
  --cache /server/prior_cache/real_pilot
```

여기에는 `--allow-mock`를 넣지 않는다. 실제 adapter의 confidence를 확인하기 전에는
공식 teacher output과 스키마가 이미 호환된다고 주장하지 않는다. 8~32 pair 확인 후
128~512 pair 비용 pilot, 그 다음에만 1k~10k pair subset으로 확장한다. 전체 RE10K cache는
만들지 않는다. subset 규모는 측정한 pair/sec·디스크·학습 다양성으로 결정한다.

### Gate 3 — loss 비교부터, local demand와 pruning 학습은 나중에

아래는 실제 cache 완성 후의 명령이다. 각 arm은 **같은 기존 checkpoint에서 별도로**
초기화한다. `checkpointing.load=null`로 optimizer/global_step을 새로 시작한다.
`model.encoder.pretrained_weights`가 weights 초기화, `checkpointing.load`는 Lightning
resume라는 차이를 유의한다. resume하면 warm-up/max_steps도 기존 step 기준이 된다.

```bash
# A: matched control — reconstruction loss만, 같은 fixed pair/augmentation OFF
python -m src.main +experiment=retention_control \
  model.encoder.pretrained_weights="$CKPT" dataset.re10k.roots="[$DATA]" \
  dataset.re10k.view_sampler.index_path="$PAIRS" \
  hydra.run.dir=outputs/pilot_control

# B: legacy V1 — 식은 그대로, pilot warm-up은 prior arm과 맞춤
python -m src.main +experiment=retention_control 'loss=[mse,lpips,reliability]' \
  loss.reliability.enabled=true loss.reliability.warmup_start=500 \
  loss.reliability.warmup_end=1500 \
  model.encoder.pretrained_weights="$CKPT" dataset.re10k.roots="[$DATA]" \
  dataset.re10k.view_sampler.index_path="$PAIRS" \
  hydra.run.dir=outputs/pilot_legacy

# C: foundation prior — local allocation OFF, hard retention OFF
python -m src.main +experiment=retention_prior \
  model.encoder.pretrained_weights="$CKPT" dataset.re10k.roots="[$DATA]" \
  dataset.re10k.view_sampler.index_path="$PAIRS" \
  dataset.re10k.foundation_prior.root=/server/prior_cache/real_pilot \
  dataset.re10k.foundation_prior.teacher_id=vggt-omega-EXACT_REVISION \
  dataset.re10k.foundation_prior.teacher_settings_id=repro416-policy-v1 \
  hydra.run.dir=outputs/pilot_prior
```

각 run의 resolved config, checkpoint SHA/이름, pair manifest hash, seed, 실제 optimizer
update 수, teacher ID/settings를 보관한다. 기존 historical 결과는 삭제하지 않고 참고선으로
남긴다. 데이터/schedule이 다른 historical V1과 C를 직접 비교하여 loss 효과라고 결론내리지 않는다.
loss weight sweep은 pilot에서 0 / 1e-4 / 1e-3 정도로 제한하며 성능·gradient 크기를 보고 확정한다.

### Gate 4 — 동일 budget 평가 → demand → pruning-aware fine-tuning

```bash
EVAL_CKPT=/server/checkpoints/CHOSEN_PILOT.ckpt
EVAL_INDEX=/server/manifests/held_out_eval.json
python -m src.main +experiment=re10k mode=test 'loss=[mse,lpips]' \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path="$EVAL_INDEX" dataset.re10k.roots="[$DATA]" \
  checkpointing.load="$EVAL_CKPT" model.encoder.pretrained_weights=null \
  model.encoder.orthogonal_preconditioner.enabled=true \
  model.encoder.orthogonal_preconditioner.mix=1.0 \
  model.encoder.orthogonal_preconditioner.target_mix=1.0 \
  model.encoder.orthogonal_preconditioner.warmup_steps=0 \
  model.encoder.gaussian_reliability.enabled=true \
  model.gaussian_retention.enabled=true model.gaussian_retention.keep_ratio=0.5 \
  model.gaussian_retention.score_source=reliability \
  test.output_path=outputs/eval_r50 hydra.run.dir=outputs/eval_r50_run
```

- 먼저 같은 checkpoint에서 `score_source=random/base_opacity/opacity/reliability`만 바꾼다.
  50%에서 유망하면 25/50/75%를 평가한다. 모든 출력 경로는 다르게 지정한다.
- 동일 checkpoint/ratio에서 다음 세 override를 추가하여 demand 효과만 비교한다:
  `model.gaussian_retention.local_cells.enabled=true`,
  `model.gaussian_retention.local_cells.demand_source=image_detail`,
  `model.gaussian_retention.local_cells.detail_strength=1.0`.
  cell=16, min_per_cell=0으로 시작하고 빈 cell 문제가 관측될 때만 min_per_cell=1.
- 유망한 arm에만 50% pruning-aware fine-tuning. 해당 pilot checkpoint를
  `model.encoder.pretrained_weights`로 넣고 optimizer를 reset한다. 학습 명령에
  `model.gaussian_retention.enabled=true`, `keep_ratio=0.5`, `score_source=reliability`를
  추가한다. demand는 ON/OFF 비교한 설정 그대로 유지하고 eval에서도 같은 설정을 사용한다.
- Fine-tuning 효과만 보려면 동일 pilot checkpoint에서 **동일 추가 step의 full-count
  continuation**도 수행한다. 전후 비교만 하면 추가 학습 시간 효과와 pruning 효과가 섞인다.
- Elastic은 마지막 선택 실험이다. batch=1, mode=elastic, threshold, max_gaussians를
  고정해 실제 count/quality를 보고한다. fixed-budget Pareto와 혼합하지 않는다.

### 측정 주의

PSNR/SSIM/LPIPS, 실제 count, view/cell retention, empty-cell 비율, opacity·scale·covariance
tail을 함께 기록한다. floater proxy가 실제 geometry correctness를 증명하지는 않는다.
현재 `Benchmarker.time`은 CPU wall clock이며 CUDA 동기화가 없고 memory는 전체 peak다.
따라서 그 숫자만으로 rasterizer 가속을 주장하지 않는다. 서버에서 별도 synchronized
CUDA event / warm-up / 반복 측정 / 단계별 reset_peak_memory_stats 기반 protocol을 검증한 뒤
encoder, selection, decoder, total을 분리 기록한다. 공정한 속도 run은 이미지 저장·진단 log와
pose alignment OFF. 품질 평가의 pose alignment ON 결과는 별도 표로 구분한다.

## 11. 6주 일정, 필수 ablation, stop/go

| 주 | 실행 | 다음 단계 진입 조건 / 중단 조건 |
| --- | --- | --- |
| 1 | 환경·checkpoint 복원, OFF/100% parity, 4-step mock 및 heads fit/resume smoke, baseline 50%, heads/full 2K pilot | parity/count/backward 실패 시 배선부터 해결. heads의 비용 절감·품질 회복을 확인해 범위 선택 |
| 2 | 실제 teacher 8~32 pair adapter, 128~512 pair 비용/정렬 pilot | unknown 처리·overlay·coverage·저장비용이 검증될 때만 subset cache 확장 |
| 3 | matched reconstruction-only / V1 / prior pilot, local OFF | full-count 품질 심각한 하락, r collapse, valid signal 부재면 weight/labels 점검; 1회 수정 후에도 개선 없으면 prior는 보류 |
| 4 | 선택 checkpoint의 25/50/75% Pareto, image-detail demand ON/OFF | 동일 budget에서 개선 없으면 local OFF를 최종모델로 채택 가능 |
| 5 | 유망 arm pruning-aware FT + 같은 step full-count continuation | 회복 여부 및 비용 비교; expansion으로 범위 확대하지 않음 |
| 6 | seed 반복/held-out 최종평가, synchronized 성능측정, 사례·표·논문 정리 | test로 설정 재튜닝 금지, 검증된 결과만 contribution으로 주장 |

필수: 원 checkpoint 기준선, 짧은 heads/full pilot로 학습 범위 결정, matched A/B/C loss 비교(teacher 성공 시), 동일 count의
random/base-opacity/reliability 비교, local ON/OFF, pruning-aware FT 전후와 continuation.
`opacity=alpha*r`도 기록하면 soft suppression과 ranking 효과를 구분하기 쉽다.
필요 이상으로 모든 조합을 학습하지 않는다. score/budget/demand의 첫 비교는 **재학습 없는
같은 checkpoint 평가**로 끝낸다. 최종 유망 설정에 한해 최소 2 seed 반복을 우선한다.
학습 범위를 결정한 뒤에는 loss/local/FT ablation에서 범위·accumulation을 고정한다.
300K 전체 재학습을 반복하는 계획이 아니라 기존 checkpoint의 2K~5K pilot과 유망 설정의 추가 학습이 우선이다.

선택: 8 vs 16 cell, min_per_cell=1, elastic threshold, 두 번째 teacher(DA3), 확대된 subset,
depth-derived demand. 제외: bounded expansion, learned demand head, 전체 데이터 cache,
online foundation backbone, anchor/canonical 변경, scene별 Gaussian post-optimization.

Foundation이 막히면 `retention_control` 또는 matched V1로 진행하고,
`foundation_prior.enabled=false`, prior loss OFF를 유지한다. **fallback에서도 hard retention,
image-detail local allocator, pruning-aware fine-tuning은 사용 가능**하다. Teacher를 빼도
기존 checkpoint와 결과는 보존되지만, foundation-grounded라는 논문 주장은 제거해야 한다.

## 12. 면담용 요약

“기존 Reliability를 실제 Gaussian 압축에 쓰는 구조에, 학습 때만 외부 geometry 근거로
신뢰도를 보정할 수 있는 분리된 loss와 안전한 cache 입력 경로를 추가했습니다. 이미지
detail에 따라 남길 위치를 조절하는 가벼운 demand도 구현했습니다. 각각 끄고 켤 수 있어
loss의 효과, 선택 기준의 효과, pruning-aware fine-tuning의 효과를 따로 비교할 수 있습니다.
학습 시간을 줄이기 위해 기존 backbone을 고정하고 geometry·Gaussian·Reliability head만
fine-tuning하는 선택형 모드도 추가했습니다. 먼저 짧은 전체 fine-tuning과 비교해 품질을
얼마나 보존하는지 확인하고, 부족하면 기존 전체 학습으로 돌아갈 수 있습니다.
현재 확인한 것은 CPU 수준의 동작과 gradient이며, geometry prior의 실제 효용과 품질·속도
개선은 서버 실험으로 검증해야 합니다. Foundation 모델을 추론마다 추가 실행하는 설계는
아니며, 잘 안 될 경우 기존 방식으로 돌아갈 수 있습니다.”
