# NoPoSplat Reliability 기반 Adaptive Gaussian Retention MVP

작성 기준일: 2026-09-27

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

### 2.4 Reliability auxiliary loss 안전장치

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

다만 현재 MVP auxiliary loss는 다음처럼 비음수 위험 신호에 대한 단방향 페널티다.

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

- 구현됨: cell grouping, demand pooling, `score * demand`, global budget, optional 최소 유지
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

Reliability 관련 추가 의미:

- `gaussian_reliability.detach_gate=false`: reconstruction과 auxiliary loss가 모두
  Reliability head를 학습할 수 있으며 현재 MVP의 권장 기본값이다.
- `gaussian_reliability.detach_gate=true`: reconstruction gradient만 차단하고
  auxiliary loss gradient는 유지하지만, 현재 단방향 loss에서는 `r_min` 수축 위험이
  있으므로 진단용 ablation으로만 사용한다.
- Reliability auxiliary loss의 warm-up 계수가 0이면 추가 context rendering을 생략한다.

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

## 7. 현재 검증 상태

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

현재 로컬 RTX 4060 Ti에서 수행한 CUDA smoke 검증:

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
- foundation depth/confidence cache schema
- pair cache key, preprocessing metadata, dataloader cache loader
- 실제 foundation-grounded cell demand
- 학습 가능한 demand predictor
- retention 전후 tail 전용 비교 log
- bounded local expansion
- child Gaussian residual head
- 별도의 post-optimization stage

`ModelWrapper`에는 외부 tensor를 받을 수 있는 일반적인 연결 지점만 있다. 이것은
foundation cache schema가 구현되었다는 뜻이 아니다. 실제 VGGT-Omega/DA3 output과
preprocessing을 한 sample에서 확인한 후 cache 계약을 정해야 한다.

Bounded expansion은 MVP에 포함하지 않는다. 먼저 retention-only 실험에서 다음 세
조건이 반복적으로 확인될 때만 추가 여부를 판단한다.

1. 특정 high-demand cell이 지속적으로 100% retention된다.
2. 그 cell의 reconstruction residual이 여전히 높다.
3. 전체 budget을 늘렸을 때 해당 영역의 품질이 실제로 개선된다.

## 9. 서버 할당 후 검증 순서

1. Retention OFF로 기존 checkpoint 결과와 parity 확인
2. Retention ON, `keep_ratio=1.0` parity 확인
3. 50% fixed budget에서 정확한 Gaussian 수와 renderer 실행 확인
4. Evaluation-only random/opacity/reliability top-k 비교
5. 25%, 50%, 75% budget Pareto curve 작성
6. 가장 유망한 score로 짧은 pruning-aware fine-tuning
7. view/cell empty ratio와 품질 저하 위치 확인
8. 단순 dummy 또는 image/depth-gradient demand로 local allocator sanity check
9. VGGT-Omega/DA3 한 sample의 실제 output/preprocessing 확인
10. 확인된 output을 기준으로 cache schema와 foundation experiment 설계

각 단계의 stop/go 기준:

- OFF 또는 100% parity가 깨지면 다음 실험으로 진행하지 않는다.
- Random과 차이가 없으면 Reliability ranking 및 loss부터 재검토한다.
- 빈 cell 증가가 품질 저하와 연결되면 `min_per_cell=1`을 ablation한다.
- Fine-tuning으로 품질이 회복되지 않으면 expansion보다 score/demand 설계를 먼저
  재검토한다.
