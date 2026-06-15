# OLP-H Gaussian Reliability Head

이 문서는 OLP-H 위에 추가한 Gaussian Reliability Head의 현재 구현 상태와,
다음 단계로 붙일 context residual 기반 reliability regularization 계획을 정리한다.

현재 코드에는 reliability head와 opacity modulation만 구현되어 있다. 아직 context
residual loss, volume risk, OLP-H tail risk는 구현되어 있지 않다.

## 핵심 아이디어

OLP-H는 DPT Gaussian head의 latent feature를 orthogonal preconditioning으로
안정화한다. 하지만 최종 Gaussian primitive가 novel view에서 믿을 만한지, 또는
floater처럼 동작할 위험이 있는지를 별도로 예측하지는 않는다.

Gaussian Reliability Head는 이 문제를 primitive-level reliability prediction으로
본다.

```text
DPT / NoPoSplat feature
-> OLP-H preconditioned feature
-> existing Gaussian head: opacity, scale, color, ...
-> reliability head: r_g in [r_min, 1]
-> opacity'_g = opacity_g * r_g
```

즉 Gaussian 개수를 바로 줄이거나 후처리 pruning을 하는 것이 아니라, feed-forward
prediction 안에서 각 Gaussian의 rendering contribution을 soft하게 조절한다.

## 현재 구현

현재 구현은 기존 Gaussian head의 출력 채널을 바꾸지 않는다. 대신
`dpt_gs_head` 안에서 OLP-H preconditioner를 지난 `path_1` feature를 공유하고,
별도의 reliability branch를 추가한다.

관련 파일:

- `src/model/encoder/heads/reliability_cfg.py`
- `src/model/encoder/heads/dpt_gs_head.py`
- `src/model/encoder/encoder_noposplat.py`
- `src/model/encoder/encoder_noposplat_multi.py`
- `config/model/encoder/noposplat.yaml`

최종 opacity는 다음과 같이 계산된다.

```text
r_g = r_min + (1 - r_min) * sigmoid(a_g)
opacity'_g = opacity_g * r_g
```

기본 설정은 다음과 같다.

```yaml
model:
  encoder:
    gaussian_reliability:
      enabled: false
      r_min: 0.05
      r_init: 0.95
      detach_gate: false
```

`enabled=false`가 기본값이므로, 기본 학습과 추론에는 영향을 주지 않는다.
`enabled=true`로 켜면 reliability gate가 opacity에 곱해진다.

`r_init=0.95`는 학습 시작 시 reliability gate가 거의 identity처럼 동작하도록 하기
위한 초기화다. 기존 pretrained checkpoint에는 reliability head parameter가 없으므로,
reliability가 켜진 경우 `ModelWrapper`는 Lightning checkpoint loading을
non-strict로 전환한다.

## 현재 구현의 한계

현재 구현은 floater-aware loss를 포함하지 않는다.

구현된 것은 아래 구조까지다.

```text
feature
-> Gaussian head
-> opacity_g
-> Reliability head
-> r_g
-> opacity'_g = opacity_g * r_g
```

아직 구현되지 않은 항목:

- context view rendering
- context residual map 계산
- Gaussian projection을 통한 residual sampling
- residual 기반 `L_float`
- volume 또는 robust volume risk
- OLP-H latent tail risk
- budget loss

따라서 현재 상태에서 reliability를 켜면, `r_g`는 일반 rendering loss를 통해서만
간접적으로 학습된다. 이 경우 reliability head는 floater detector라기보다 추가
opacity gate에 가깝다. Floater suppression contribution을 만들려면 다음 단계에서
context residual 기반 regularization을 붙여야 한다.

## 1차 Sanity Check

가장 먼저 확인해야 할 것은 reliability head만 켰을 때 기존 OLP-H와 거의 같은
성능을 유지하는지다.

```bash
python -m src.main +experiment=re10k mode=test \
  wandb.name=olp_reliability_identity \
  checkpointing.load=./pretrained_weights/re10k.ckpt \
  model.encoder.orthogonal_preconditioner.enabled=true \
  model.encoder.orthogonal_preconditioner.mix=1.0 \
  model.encoder.orthogonal_preconditioner.target_mix=1.0 \
  model.encoder.gaussian_reliability.enabled=true \
  model.encoder.gaussian_reliability.r_min=0.05 \
  model.encoder.gaussian_reliability.r_init=0.95
```

이 단계에서는 `L_float=0`이다. 성능이 크게 흔들리면 residual loss를 붙이기 전에
reliability 초기화, opacity modulation 위치, checkpoint loading부터 확인해야 한다.

## Visualization Dump

`visualization_dump`가 활성화된 경우 다음 항목이 추가된다.

- `reliability`: `[B, V, H, W, S, 1]`
- `opacities_before_reliability`: reliability 적용 전 opacity
- `opacities`: reliability 적용 후 opacity

이 값들은 이후 residual regularization과 ablation debugging에 사용한다.

## 다음 단계: Context Residual

다음 단계의 목표는 context image reconstruction residual을 약한 위험 신호로 사용해
reliability head를 학습시키는 것이다.

전체 흐름은 다음과 같다.

```text
predicted Gaussians
-> render context cameras
-> residual_ctx = |render_ctx - input_ctx|
-> project Gaussian centers to context views
-> sample residual_ctx at projected positions
-> residual_risk_g
-> conservative reliability loss
```

여기서 `residual_risk_g`는 floater ground truth가 아니다. Context residual이 큰
이유는 floater 외에도 조명 차이, view-dependent effect, occlusion boundary, texture
오차, normalization mismatch 등 여러 가지가 있을 수 있다. 따라서 residual은 강한
label이 아니라 weak risk signal로만 사용해야 한다.

## 보수적 1차 Loss

이전 geometry loss 실험에서 NaN이 발생했고, 현재는 OLP-H 위에 새 head를 처음
붙이는 단계다. 따라서 첫 residual 실험은 기존 Gaussian opacity/scale head를 직접
흔들지 않는 보수적인 식으로 시작한다.

구현 config는 `config/loss/reliability.yaml`에 있다. 기본값은 disabled라서
loss list에 추가하고 명시적으로 켜야 한다.

```yaml
loss:
  reliability:
    enabled: true
    weight: 1.0e-4
    warmup_start: 3000
    warmup_end: 8000
```

예시 실행 override:

```bash
python -m src.main +experiment=re10k mode=train \
  loss=[mse,reliability] \
  model.encoder.gaussian_reliability.enabled=true \
  loss.reliability.enabled=true \
  loss.reliability.weight=1.0e-4 \
  loss.reliability.warmup_start=3000 \
  loss.reliability.warmup_end=8000
```


1차 MVP loss:

```text
L_float_v1 =
mean_g [
  r_g
  * stopgrad(opacity_g)
  * stopgrad(residual_risk_g)
]
```

이 식의 목적은 위험도가 높게 관측된 Gaussian에서 기존 opacity 자체를 바로 낮추는
것이 아니라, 새로 추가한 reliability gate `r_g`가 먼저 낮아지도록 유도하는 것이다.
Gradient는 주로 reliability head로 흐르고, 기존 Gaussian head의 opacity/scale/color
prediction은 최대한 보존한다.

초기 구현에서는 volume이나 tail risk를 포함하지 않는다.

```text
사용:
residual_risk_g

사용하지 않음:
robust_volume_g
tail_risk_g
budget loss
monocular depth prior
hard pruning
```

## 이후 확장 Loss

1차 MVP가 안정적으로 동작하면 더 직접적인 형태로 확장할 수 있다.

2차:

```text
L_float_v2 =
mean_g [
  opacity'_g
  * stopgrad(residual_risk_g)
]
```

3차:

```text
L_float_v3 =
mean_g [
  opacity'_g
  * stopgrad(residual_risk_g)
  * stopgrad(additional_risk_g)
]
```

여기서 `additional_risk_g`는 나중에 robust volume, OLP-H latent tail, Gaussian
parameter outlier 등을 포함할 수 있다. 다만 이 항들은 1차 구현 범위에 포함하지
않는다.

## Loss Warmup

Floater regularization은 매우 약하게 시작한다.

```text
0 ~ 3k steps: lambda_float = 0
3k ~ 8k steps: linear warmup
after 8k: lambda_float = 1e-4 ~ 1e-3
```

`0.01` 수준의 weight는 초기 실험에는 너무 강할 가능성이 높다.

## 권장 실험 순서

1. `OLP-H`
2. `OLP-H + Reliability Head`, `L_float=0`
3. `OLP-H + Reliability Head + context residual`, conservative `L_float_v1`
4. `OLP-H + Reliability Head + context residual`, `L_float_v2`
5. `OLP-H + Reliability Head + additional risk`, optional extension

2번은 반드시 baseline과 거의 같아야 한다. 2번부터 깨지면 residual loss 문제가
아니라 reliability head 자체의 초기화나 opacity modulation 문제다.

## 논문 관점의 주장

이 모듈의 주장은 단순 후처리 pruning이 아니다.

```text
OLP-H improves latent conditioning, but primitive-level reliability remains
unmodeled. We introduce a reliability branch that learns soft opacity
modulation for each Gaussian primitive. In the first regularized version,
context residuals are lifted into weak primitive-level risk signals, allowing
floater-aware reliability learning without breaking the feed-forward pipeline.
```

핵심은 Gaussian을 삭제하는 것이 아니라, feed-forward Gaussian prediction 내부에
reliability라는 제어 변수를 추가하고 이를 약한 context consistency 신호로 학습하는
것이다.
