# 후속 연구: 채널 적응형 의미 압축 실험 보고서

> 작성 기준일: 2026-07-12  
> 핵심 결론: 최종 강화 실험은 proxy 대신 **실제 5-rate neural codec 측정값**을 사용했다. Adaptive는 25개 area×policy 조합 중 23개에서 fixed-80 대비 완료 수를 늘리고 평균 시간을 줄였으며, 조합별 변화율의 단순 평균은 각각 +18.8%와 -26.6%였다. 다만 Stochastic 400m·500m에서는 반대 결과가 나왔고, 5개 rate의 mIoU가 거의 평탄하고 비단조였으며, paired 차이의 신뢰구간·다중비교 보정까지는 수행하지 않았다. 따라서 이 결과는 **유망하지만 확정적 성능 증명은 아닌 가설 생성 결과**다.

## 1. 후속 연구 질문

AirTalking 논문 Table III는 의미 압축률 \(\rho_c=0.104\)를 하나의 고정값으로 사용한다. 하지만 무선 채널은 계속 변한다.

- 채널이 나쁘면: 큰 payload는 전송에 오래 걸리고 request가 막힌다.
- 채널이 좋으면: 더 많은 semantic feature를 보내도 지연 부담이 작다.

그래서 이 후속 연구는 다음을 묻는다.

> “매 request 후보의 SINR에 따라 semantic payload 크기와 품질을 바꾸면, 고정 압축률보다 더 많은 작업을 더 빨리 끝낼 수 있는가?”

여기서 **적응형(adaptive)**은 실행 중 관측한 상태에 따라 선택을 바꾸는 방식이고, **고정형(fixed)**은 모든 상태에서 같은 압축 단계를 쓰는 방식이다.

## 2. 연구 가설

강화 codec 결과를 넣기 전부터 판단 기준을 명확히 하기 위해 가설을 다음처럼 둔다.

- **H1 완료율 가설**: adaptive는 fixed paper-like보다 완료 request 수를 늘린다.
- **H2 지연 가설**: adaptive는 fixed paper-like보다 완료 request의 평균 시간을 줄인다.
- **H3 품질 제약 가설**: H1/H2 개선을 얻으면서 semantic quality 감소를 미리 정한 허용 범위 안에 둔다.
- **H4 rate-quality 단조성 가설**: 활성 latent 채널이 늘면 payload는 늘고, 평균 품질은 대체로 나빠지지 않는다.
- **H5 일반성 가설**: 효과가 특정 면적·특정 scheduler 하나에만 국한되지 않는다.

H1/H2만 보고 “성공”이라고 하면 가장 작은 payload만 계속 보내는 정책이 이길 수 있다. 의미 통신은 전달량과 task 품질을 함께 봐야 하므로 H3가 필수다.

## 3. 용어 설명

- **policy/scheduler**: 가능한 UAV relay 행동 중 무엇을 실행할지 고르는 규칙이다.
- **SINR**: 원하는 신호를 간섭+잡음으로 나눈 비율이다. dB가 높을수록 대체로 채널이 좋다.
- **operating point**: codec을 특정 payload 크기와 품질로 쓰는 한 설정이다.
- **rate-quality trade-off**: payload를 줄이면 전송은 빨라지지만 품질이 낮아질 수 있는 맞교환 관계다.
- **Pareto frontier**: 다른 선택보다 payload도 크고 품질도 낮은 열등점을 제거한 뒤 남는 최선의 경계다.
- **proxy**: 직접 측정하기 어려운 것을 대신하는 간접 지표다. proxy가 원래 대상을 완전히 대표한다는 보장은 없다.
- **paired experiment**: 두 방식이 같은 random request·위치·채널 표본을 공유해 방식 차이만 비교하는 실험이다.
- **ablation**: 구성 요소 하나를 제거하거나 바꿔 그 요소의 기여를 확인하는 실험이다.

## 4. 기존 연구의 전체 흐름

```text
Cityscapes gtFine 정답 label
  -> 5개 크기로 nearest downsample/upscale
  -> payload ratio와 label 보존 mIoU 측정
  -> 기존 AirTalking simulator의 SINR 표본으로 probe
  -> SINR 구간별 요구 품질을 만족하는 최소 payload 선택
  -> UAV scheduler 후보 평가 안에 adaptive 선택 삽입
  -> nonsemantic / fixed / adaptive 비교
```

관련 코드는 다음과 같다.

- `run_adaptive_probe.py`: proxy rate-quality 표 생성과 간단한 SINR probe
- `run_full_adaptive_research.py`: 적응형 선택을 full scheduler 안에 삽입
- `validate_full_adaptive_results.py`: 결과 구조와 방향성 invariant 검사

## 5. 사용 데이터와 기존 proxy 생성

### 5.1 데이터

Cityscapes `gtFine`의 train/val `*_gtFine_labelIds.png`를 사용했다. 기존 probe 기본값은 정렬된 목록에서 등간격으로 최대 400개를 고른다. 저장된 `compression_quality.csv`는 이 표본의 집계다.

중요하게도 RGB 이미지를 encoder에 넣지 않았다. 이미 알고 있는 정답 segmentation map을 줄였다가 키웠다. 따라서 이 실험이 측정하는 것은 “RGB에서 의미를 추출하는 능력”이 아니라 “정답 label의 공간 해상도를 낮췄을 때 경계가 얼마나 보존되는가”다.

### 5.2 다섯 proxy mode

원본 label 크기를 \(s\)배로 줄이면 1채널 feature byte는 대략 \(s^2\)에 비례한다. raw RGB는 3채널이므로 고정 길이 payload ratio는 대략 \(s^2/3\)이다.

| Mode | scale | 저장 feature ratio | 저장 proxy mIoU |
|---|---:|---:|---:|
| emergency | 0.125 | 0.005208 | 0.813617 |
| low | 0.250 | 0.020833 | 0.891526 |
| medium | 0.375 | 0.046875 | 0.934981 |
| paper_like | 0.560 | 0.104464 | 0.950513 |
| high | 0.750 | 0.187500 | 0.969392 |

paper-like 0.104464는 \(0.56^2/3\)으로 구조적으로 만들어진 값이다. 논문 식 (22)의 비율과 숫자는 가깝지만, 논문 modified U-Net의 encoded payload가 아니다.

### 5.3 Proxy 품질 계산

1. label을 nearest-neighbor로 작은 크기로 줄인다.
2. 다시 원래 크기로 키운다.
3. `ignore=255`를 제외하고 pixel accuracy와 mIoU를 계산한다.
4. 작은 uint8 배열 자체의 byte와 zlib byte를 센다.

nearest-neighbor는 label 번호를 섞지 않는 올바른 resize 방법이지만, 정답에서 시작하므로 encoder 오분류가 없다. 그래서 emergency mode도 mIoU 0.81 이상이라는 매우 높은 값이 나올 수 있다. 실제 RGB neural codec 품질로 해석하면 심각한 과대평가다.

## 6. 적응형 선택 규칙

### 6.1 Legacy proxy rule

이미 저장된 legacy proxy run의 구간별 목표 품질은 다음과 같았다.

| SINR 구간 | 요구 proxy quality |
|---|---:|
| \(SINR < -15\) dB | 0.800 |
| \(-15 \le SINR < -10\) dB | 0.880 |
| \(-10 \le SINR < -5\) dB | 0.925 |
| \(-5 \le SINR < 0\) dB | 0.950 |
| \(SINR \ge 0\) dB | 0.965 |

각 candidate link에서 다음처럼 고른다.

1. SINR 구간으로 target quality를 찾는다.
2. payload ratio가 작은 순서로 mode를 본다.
3. target quality 이상인 첫 mode를 고른다.
4. 아무 mode도 못 맞추면 가장 품질이 높은 mode를 고른다.

이 규칙은 채널이 나쁠수록 낮은 target을 허용해 작은 payload를 보내고, 좋을수록 더 높은 품질을 요구한다. 표의 threshold와 target quality는 논문값도 neural 측정값도 아닌 legacy label-proxy heuristic이다. **heuristic**은 최적임이 증명되지는 않았지만 합리적이고 계산이 쉬운 규칙이다.

### 6.2 강화 5-rate rule

강화 neural 실험은 위 `0.800` 등의 proxy 목표를 그대로 복사하지 않는다. runner의 기본 `auto`는 실제 다섯 neural rate를 감지하면 `measured_ordered`를 선택하고, 고정 SINR bin `[-∞,-15), [-15,-10), [-10,-5), [-5,0), [0,∞)`에 **실측 5-rate 품질 순서와 맞춘 목표**를 연결한다. 적용한 rule 종류, bin, 목표 품질과 source를 run metadata에 기록해 결과 뒤에 기준을 바꿀 수 없게 한다. rate별 mIoU가 비단조이면 일부 mode가 어느 bin에서도 선택되지 않을 수 있으므로 reachable mode 목록과 개수도 함께 기록하고 실제 결과에서 감사한다.

별도 사전등록 목표를 시험할 때만 `explicit` rule과 다섯 품질값을 명시한다. `legacy_proxy`는 과거 결과 재검사용 선택지이며 강화 neural 주 분석의 기본값이 아니다.

## 7. Full scheduler에 들어가는 방식

고정형과 적응형은 simulator 바깥에서 전송 시간을 사후 계산하는 것이 아니다. 각 UAV pair/semantic state 후보를 만들 때 다음 값이 달라진다.

- 선택 mode와 \(\rho_c\)
- D2D payload bit와 전송 시간
- decode 시간
- semantic quality
- 전체 duration과 energy
- scheduler가 비교하는 cost

즉 압축 선택이 scheduler 행동 선택에도 영향을 준다. 이 점이 단순 probe보다 발전한 부분이다.

비교 mode는 다음 세 가지다.

- `nonsemantic`: raw payload, quality를 1로 기록
- `fixed_paper_like`: 항상 paper-like 한 단계
- `adaptive_semantic`: candidate SINR에 따라 다섯 단계 중 선택

면적은 100~500 m, 정책은 Stochastic/LinUCB/SA/Greedy/MCTS, 반복 10회, 각 반복 1,000 slots다.

## 8. 기존 full adaptive 결과

### 8.1 300×300 m 상세 결과

다음은 이미 저장된 `summary_metrics.csv`와 생성 보고서의 10회 평균이다.

| Policy | Mode | 완료 수 | 평균 시간(s) | 비행 J/request | 품질 | 평균 payload ratio |
|---|---|---:|---:|---:|---:|---:|
| Stochastic | nonsemantic | 66.5 | 78.61 | 17,885.3 | 1.000 | 1.000 |
| Stochastic | fixed | 75.0 | 70.29 | 15,931.2 | 0.987 | 0.769 |
| Stochastic | adaptive | 78.6 | 65.83 | 14,937.8 | 0.965 | 0.742 |
| LinUCB | nonsemantic | 66.5 | 81.98 | 18,569.2 | 1.000 | 1.000 |
| LinUCB | fixed | 127.5 | 29.53 | 6,489.2 | 0.967 | 0.398 |
| LinUCB | adaptive | 155.5 | 19.96 | 4,283.5 | 0.922 | 0.385 |
| SA | nonsemantic | 68.3 | 76.57 | 17,400.8 | 1.000 | 1.000 |
| SA | fixed | 90.1 | 54.86 | 12,394.8 | 0.974 | 0.529 |
| SA | adaptive | 103.2 | 44.41 | 10,012.7 | 0.929 | 0.488 |
| Greedy | nonsemantic | 75.9 | 66.19 | 15,033.2 | 1.000 | 1.000 |
| Greedy | fixed | 187.4 | 11.18 | 2,416.0 | 0.961 | 0.301 |
| Greedy | adaptive | 211.7 | 6.26 | 1,276.6 | 0.940 | 0.371 |
| MCTS | nonsemantic | 77.0 | 63.76 | 14,465.3 | 1.000 | 1.000 |
| MCTS | fixed | 158.5 | 19.33 | 4,244.8 | 0.962 | 0.318 |
| MCTS | adaptive | 183.6 | 11.98 | 2,529.0 | 0.927 | 0.360 |

여기서 fixed인데 평균 payload ratio가 0.104가 아닌 이유는 모든 완료 request가 semantic encode를 선택하는 것이 아니기 때문이다. raw로 처리된 request까지 평균에 들어간다. 품질도 같은 방식으로 완료 request 전체에 평균된다.

### 8.2 Adaptive 대 fixed 변화

| Policy | 완료 수 변화 | 평균 시간 변화 | 품질 변화 | payload-ratio 변화 |
|---|---:|---:|---:|---:|
| Stochastic | +4.8% | -6.3% | -0.022 | -3.5% |
| LinUCB | +22.0% | -32.4% | -0.045 | -3.3% |
| SA | +14.5% | -19.0% | -0.045 | -7.8% |
| Greedy | +13.0% | -44.0% | -0.021 | +23.5% |
| MCTS | +15.8% | -38.0% | -0.036 | +13.1% |

Greedy와 MCTS에서는 adaptive 평균 payload가 오히려 커졌다. 이것은 모순이 아니다. scheduler가 좋은 링크에서는 high-quality 큰 payload를 선택하고도 action을 더 빨리 완료할 수 있기 때문이다. 핵심 가설은 “payload를 항상 줄인다”가 아니라 “상태에 맞는 rate-quality 선택으로 시스템 효율을 높인다”다.

### 8.3 전체 25개 면적×정책 비교

기존 proxy 실험 `full_adaptive_results`의 구버전 `result_validation.json`은 adaptive가 fixed에 비해 다음 invariant를 모두 통과했다고 기록했다.

- 25개 조합 모두 완료 수가 fixed 이상
- 25개 조합 모두 평균 시간이 fixed 이하
- 가장 큰 저장 품질 감소: 약 -0.0893
- 기대한 75개 mode×area×policy 행이 모두 존재
- non-finite 수치 없음

그러나 구버전은 산출물 무결성과 평균값 방향 가설을 하나의 `passed`로 섞었고, repeat별 분산·paired 차이의 신뢰구간·다중비교 보정·효과 크기의 불확실성을 보지 않았다. 강화 schema v2는 상위 `passed`(무결성)와 `all_comparisons_pass`(방향 가설)를 분리한다. 두 값 모두 과학적 가설의 통계적 확정을 의미하지 않는다.

## 9. 기존 결과가 좋아 보이는 이유와 위험

### 9.1 합리적인 원인

- 약한 링크에서 작은 payload가 D2D 시간을 줄인다.
- action이 빨리 끝나 UAV/device가 다음 request에 빨리 풀린다.
- scheduler가 candidate별 payload를 알고 더 나은 행동을 고른다.
- 좋은 링크에서는 품질을 위해 더 큰 mode를 선택할 여지가 있다.

### 9.2 과대평가될 수 있는 원인

- 정답 label proxy라 실제 encoder 오류가 0이다.
- proxy mIoU가 모든 mode에서 매우 높다.
- rate-quality 값이 이미지 내용과 무관한 전체 평균이다.
- calibrated simulator의 request/workload/power가 논문 비공개 가정이다.
- density interference penalty가 로컬 설계다.
- mode threshold를 같은 simulator SINR 분포를 본 뒤 정했다면 과적합 가능성이 있다.
- 완료된 request만의 품질 평균은 오래 걸려 미완료된 어려운 request를 제외하는 survivorship bias가 있다.

**survivorship bias**는 성공적으로 끝난 사례만 보고 전체가 좋다고 판단하는 오류다. 따라서 생성된 모든 request 기준 품질/지연, deadline miss, drop count도 함께 기록하는 것이 좋다.

## 10. Neural anchor 연결의 구버전과 현재 버전

이미 저장된 `results/full_adaptive_results`는 **neural anchor를 실제로 적용하지 않았다**. `run_metadata.json`에는 `source_neural_encoder_decoder`가 없고, source reproduction metadata와 다섯 quality mode 모두 Cityscapes label proxy를 가리킨다. base codec 처리율도 neural 측정값이 아니라 91.30/23.23 Mbps이며, 과거 절대경로가 남아 있다. 따라서 이 legacy 결과를 “single-rate neural anchor가 반영된 결과” 또는 “다섯 단계 neural codec 실험”이라고 부르면 안 된다.

현재 코드에서 single-rate summary를 `record_only`로 주는 경우에도 neural 정보는 기록되고 measured bitrate 쌍이 있으면 simulator 처리율에 적용될 수 있지만, 다섯 mode의 선택 품질 전체가 neural로 바뀌지는 않는다. 이는 위 저장 legacy artifact의 provenance와도 다른 새 실행이다.

현재 코드는 이 제한을 보완했다. 강화 summary에 `multi_rate_profiles` 다섯 점이 있으면 다음을 수행한다.

1. 각 점에서 활성 채널, payload ratio, mIoU를 엄격히 읽는다.
2. payload ratio가 작은 순서로 정렬한다.
3. 정확히 다섯 점인지, ratio와 채널 수가 중복되지 않는지 검사한다.
4. emergency/low/medium/paper-like/high의 ratio와 선택 quality를 **모두 neural 값으로 교체**한다.
5. raw/zlib ratio, PSNR과 프로젝트 로컬 SSIM 정의를 metadata에 보존한다.
6. 측정 encode/decode bitrate 쌍이 있으면 simulator의 codec throughput도 함께 교체한다. 한쪽만 있으면 명확히 실패한다.

즉 5-rate 연결 경로와 `adaptive_enhanced_scalable_verified` full run이 완료됐다. 최종 표는 result/provenance와 local validator를 strict finalizer가 검사한 뒤 반영하며, legacy full 파일만으로 강화 full 완료를 추정하지 않는다.

## 11. 강화 codec 기반 후속 연구 설계

### 11.1 입력 rate-quality 표

강화 codec은 같은 val 표본에서 각 활성 채널의 다음 값을 낸다.

- 실제 `uint8` payload ratio
- zlib payload ratio
- mIoU, pixel accuracy, per-class IoU
- RGB PSNR, 프로젝트 로컬 7×7 uniform-window SSIM proxy
- neural encode/decode latency(median·mean·min·max; CPU 직렬화·zlib 시간 제외)

| 활성 채널 | 지점 | ρ uint8 | ρ zlib | mIoU | pixel acc. | PSNR(dB) | SSIM | 평가 표본 |
|---|---|---|---|---|---|---|---|---|
| 20 | rate_20 | 0.026042 | 0.025968 | 0.304828 | 0.828432 | 18.448 | 0.566652 | 500 |
| 40 | rate_40 | 0.052083 | 0.050739 | 0.305751 | 0.829247 | 18.473 | 0.567122 | 500 |
| 60 | rate_60 | 0.078125 | 0.075046 | 0.304969 | 0.828745 | 18.467 | 0.567432 | 500 |
| 80 | paper_like | 0.104167 | 0.099276 | 0.305416 | 0.828616 | 18.435 | 0.567506 | 500 |
| 120 | rate_120 | 0.156250 | 0.147679 | 0.306280 | 0.829142 | 18.469 | 0.568032 | 500 |

기존 proxy 표를 위의 실제 neural rate-quality 표로 교체했다. primary analysis는 논문 식 (22)에 가장 직접적인 `rho_uint8`를 쓰고, zlib은 별도 analysis로 둔다.

### 11.2 Mode 매핑

| Adaptive 이름 | 활성 채널 | 이론 \(\rho_c\) |
|---|---:|---:|
| emergency | 20 | 0.0260417 |
| low | 40 | 0.0520833 |
| medium | 60 | 0.0781250 |
| paper_like | 80 | 0.1041667 |
| high | 120 | 0.1562500 |

기존 proxy의 emergency ratio 0.0052와 새 neural emergency 0.0260은 다르다. 따라서 기존 결과와 새 결과의 절대 크기가 달라질 수 있으며, 이것이 실패가 아니라 더 현실적인 전송 표현의 영향일 수 있다.

### 11.3 선택 규칙 개선

현재 강화 baseline은 실측 5-rate 품질에 맞춘 recorded `measured_ordered` rule이다. 이 기준과 함께 다음 정책을 비교한다.

1. **fixed-80**: 논문 0.104 근처 기준선
2. **measured-order adaptive**: 고정 SINR bin에 실측 rate-quality 순서를 연결하고 metadata에 rule을 기록
3. **latency-min under quality constraint**: 예측 mIoU가 기준 이상인 mode 중 예상 전송+codec 시간이 최소
4. **utility policy**: \(U=-latency-\lambda_E energy+\lambda_Q quality\)
5. **oracle upper bound**: 실제 사후 결과를 아는 비현실적 상한선; 배포 정책이 아니라 비교 기준

quality constraint는 전체 mIoU 하나뿐 아니라 person/rider/car 같은 안전 중요 클래스의 최소 IoU도 고려해야 한다.

### 11.4 Codec latency 반영

기존 simulator는 bitrate로 encode/decode 시간을 환산한다. 강화 모델은 이미지 크기별 실측 latency를 내므로 두 접근을 구분한다.

- `throughput model`: workload bit에 선형 비례
- `measured frame model`: 한 frame당 측정 latency 사용
- `hybrid`: frame 수×측정 latency + 마지막 partial frame 보정

논문 Table II와 다른 GPU에서 측정한 값을 Jetson latency라고 부르면 안 된다. local hardware scenario와 paper-hardware Table III scenario를 별도 표로 낸다.

## 12. 통계 실험 계획

### 12.1 Primary endpoint

- 첫 번째: fixed-80 대비 완료 request 수의 paired 차이
- 두 번째: fixed-80 대비 평균 completion time의 paired 차이
- 품질 guardrail: 생성 request 기준 semantic quality 또는 task success 감소

### 12.2 Randomness 통제

fixed와 adaptive가 다음 random stream을 공유해야 한다.

- 초기 UAV/device 위치
- device 이동
- request arrival
- source/destination
- workload
- channel fading

policy 선택에 필요한 추가 random stream은 별도로 파생한다. 이렇게 해야 차이가 단순히 더 쉬운 request를 우연히 받은 탓이 아니다.

### 12.3 반복과 신뢰구간

각 area×policy×mode에 repeat별 원시값을 저장한다. 평균만 저장하지 않는다.

현재 실행기는 `repeat_metrics.csv`에 각 repeat를 저장하고, `statistical_summary.csv`에 평균·표준편차·양측 Student-t 95% 신뢰구간을 기록하도록 강화됐다. `--workers N`은 독립 repeat를 Windows spawn process로 병렬 실행하되 repeat 순서를 보존한다. 다만 fixed-adaptive **paired difference 자체**의 신뢰구간과 Holm 보정은 별도 분석으로 추가해야 한다.

- paired 차이의 평균과 median
- 95% bootstrap confidence interval
- 표준편차와 seed별 scatter
- 효과 크기
- 5개 정책×5개 면적 다중 비교 시 Holm 보정

**95% 신뢰구간**은 같은 실험을 반복했을 때 추정 불확실성이 어느 범위인지 보여 주는 구간이다. 0을 넓게 포함하면 개선 방향이 우연일 가능성을 배제하기 어렵다.

### 12.4 사전 성공 기준

숫자를 본 뒤 기준을 바꾸지 않도록 실행 전에 예를 들어 다음을 고정한다.

- 완료 수: paired mean이 fixed보다 증가하고 95% CI 하한이 0 이상
- 평균 시간: paired mean이 감소하고 95% CI 상한이 0 이하
- mIoU: 감소가 사전 허용치 이내
- 중요 클래스 IoU: 어느 하나도 사전 허용치보다 크게 하락하지 않음

실제 허용치는 응용 목적에 따라 연구자가 정해야 하므로 여기서 임의 숫자를 만들지 않는다.

| SINR 구간 | metadata 요구 품질 |
|---|---|
| SINR < -15.0 dB | 0.304828 |
| -15.0 ≤ SINR < -10.0 dB | 0.304969 |
| -10.0 ≤ SINR < -5.0 dB | 0.305416 |
| -5.0 ≤ SINR < 0.0 dB | 0.305751 |
| SINR ≥ 0.0 dB | 0.306280 |

이 threshold는 `run_metadata.json`에 기록된 실행 설정이다. 별도 사전등록 문서·시각·해시는 산출물에 없으므로 **외부 사전등록을 완료했다는 증거는 없음**으로 해석한다.

## 13. 필요한 ablation

| Ablation | 확인하려는 질문 |
|---|---|
| label proxy vs neural multi-rate | 기존 이득이 proxy 과대평가 때문인가? |
| fixed vs heuristic adaptive | 적응 선택 자체의 효과가 있는가? |
| heuristic vs optimized constraint | threshold 설계가 결과를 좌우하는가? |
| uint8 vs uint8+zlib | 무손실 entropy coding이 추가 이득을 주는가? |
| codec latency 포함/제외 | 신경망 계산비용을 빼서 이득이 부풀었는가? |
| RGB loss 제거 | visual reconstruction이 task/통신 성능에 미치는 영향은? |
| Dice/SSIM 제거 | 각 loss의 기여는? |
| prefix model vs rate별 전용 model | scalable 모델의 공유 비용은? |
| density penalty on/off | 로컬 간섭 보정에 얼마나 민감한가? |
| calibrated/public-only params | hidden parameter 보정에 얼마나 의존하는가? |

| system mode | 조합 | 완료 평균 | 시간 평균 | 비행 J/request | 품질 평균 | payload 평균 |
|---|---|---|---|---|---|---|
| 비의미 전송 | 25 | 63.40 | 94.34 | 21,483.3 | 1.000000 | 1.000000 |
| 고정 paper-like | 25 | 119.41 | 42.45 | 9,546.3 | 0.530499 | 0.394466 |
| 적응형 | 25 | 139.72 | 32.90 | 7,349.2 | 0.539839 | 0.375843 |

이는 저장된 세 system mode 비교이며, neural loss·threshold·zlib·latency를 한 요인씩 제거한 인과적 ablation은 아니다. 해당 세부 ablation은 **미실행/증거 없음**이다.

## 14. 재현 명령

모든 명령은 저장소 루트 PowerShell에서 실행한다.

### 14.1 기존 proxy 표 다시 만들기

```powershell
.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\run_adaptive_probe.py `
  --gt-root dataset\gtFine_trainvaltest\gtFine `
  --sample-limit 400 `
  --splits train,val
```

기존 CSV를 그대로 쓰고 SINR 그림만 다시 계산하려면:

```powershell
.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\run_adaptive_probe.py --reuse-quality
```

### 14.2 Legacy proxy-only 조건 재실행

```powershell
.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\run_full_adaptive_research.py `
  --metadata studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified\run_metadata.json `
  --quality studies\adaptive_semantic_compression\results\probe_outputs\compression_quality.csv `
  --neural-summary studies\adaptive_semantic_compression\results\legacy_proxy_only.NO_NEURAL_FILE `
  --neural-quality-mode record_only `
  --adaptive-threshold-rule legacy_proxy `
  --out studies\adaptive_semantic_compression\results\full_adaptive_recheck
```

`legacy_proxy_only.NO_NEURAL_FILE`은 의도적으로 존재하지 않는 경로여야 하며, 현재 loader가 neural anchor를 적용하지 않게 한다. 이 명령은 저장 legacy artifact와 같은 proxy-only 과학적 조건을 재구성하지만, 현재 코드/provenance schema까지 과거와 byte-identical하다는 뜻은 아니다. full run은 오래 걸린다. 기존 metadata는 약 60,987초를 기록하며 실행 환경에 따라 달라진다.

### 14.3 구조 검증

```powershell
.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\validate_full_adaptive_results.py `
  --summary studies\adaptive_semantic_compression\results\full_adaptive_recheck\summary_metrics.csv `
  --out studies\adaptive_semantic_compression\results\full_adaptive_recheck\result_validation.json
```

현재 validator의 schema v2는 source summary 절대경로·SHA-256, `passed`, expected combinations와 missing/duplicate/non-finite/zero-denominator 진단을 기록한다. strict finalizer는 이 snapshot과 실제 summary를 대조하며, 구버전 `passed=true`만으로는 강화 결과 증거가 충분하지 않다.

### 14.4 강화 multi-rate codec 연결

별도 변환 파일을 손으로 만들 필요는 없다. 현재 loader가 강화 summary의 다섯 profile을 검증·정렬·매핑한다. 2개 worker를 사용하는 full run 예시는 다음과 같다.

```powershell
.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\run_full_adaptive_research.py `
  --metadata studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified\run_metadata.json `
  --quality studies\adaptive_semantic_compression\results\probe_outputs\compression_quality.csv `
  --neural-summary studies\neural_encoder_decoder\results\enhanced_scalable_full_256x128_verified\airtalking_semantic_summary.json `
  --neural-quality-mode selection `
  --adaptive-threshold-rule measured_ordered `
  --workers 2 `
  --out studies\adaptive_semantic_compression\results\enhanced_multi_rate_full
```

새 명령은 강화 summary의 `multi_rate_profiles` 다섯 점을 읽었는지, proxy CSV를 실제 선택 품질에 사용하지 않았는지, `measured_ordered`의 실제 bin별 target이 무엇이었는지를 metadata로 증명해야 한다.

### 14.5 통계 분석

실제 실행 명령은 metadata에 다음과 같이 기록됐다.

```powershell
"C:\Users\firep\OneDrive\바탕 화면\Replication-Experiment-of-Airtalking\.venv\Scripts\python.exe" studies\adaptive_semantic_compression\code\run_full_adaptive_research.py --metadata studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified\run_metadata.json --neural-summary studies\neural_encoder_decoder\results\enhanced_scalable_full_256x128_verified\airtalking_semantic_summary.json --neural-quality-mode selection --adaptive-threshold-rule measured_ordered --workers 6 --out studies\adaptive_semantic_compression\results\adaptive_enhanced_scalable_verified
```

## 15. 강화 후속 결과

### 15.1 전체 결과

| 면적(m) | 정책 | 완료 | 평균 시간(s) | 비행 J/request | 품질 | payload ratio | encode | decode |
|---|---|---|---|---|---|---|---|---|
| 100 | Stochastic | 35.4 | 121.81 | 27,932.9 | 0.791250 | 0.709822 | 10.8 | 5.4 |
| 100 | LinUCB | 136.5 | 23.97 | 5,405.5 | 0.408258 | 0.176568 | 115.8 | 51.2 |
| 100 | SA | 56.9 | 73.46 | 16,797.2 | 0.562363 | 0.390676 | 35.6 | 17.0 |
| 100 | Greedy | 200.0 | 7.72 | 1,708.1 | 0.341061 | 0.117249 | 189.7 | 0.0 |
| 100 | MCTS | 190.1 | 11.52 | 2,571.6 | 0.384836 | 0.153637 | 168.3 | 68.2 |
| 200 | Stochastic | 63.0 | 74.38 | 16,925.4 | 0.777156 | 0.693525 | 20.4 | 11.3 |
| 200 | LinUCB | 157.1 | 19.06 | 4,152.1 | 0.495117 | 0.303299 | 114.1 | 49.1 |
| 200 | SA | 91.7 | 46.66 | 10,580.4 | 0.573025 | 0.409232 | 56.5 | 27.8 |
| 200 | Greedy | 205.1 | 6.29 | 1,328.3 | 0.331701 | 0.120176 | 197.4 | 0.0 |
| 200 | MCTS | 188.2 | 11.65 | 2,520.2 | 0.435114 | 0.229057 | 153.0 | 64.8 |
| 300 | Stochastic | 80.0 | 59.23 | 13,413.9 | 0.794648 | 0.717395 | 23.5 | 12.6 |
| 300 | LinUCB | 149.4 | 20.83 | 4,492.5 | 0.537049 | 0.364180 | 99.6 | 40.6 |
| 300 | SA | 108.7 | 38.52 | 8,660.8 | 0.609726 | 0.462448 | 61.0 | 32.1 |
| 300 | Greedy | 210.6 | 6.27 | 1,278.9 | 0.331153 | 0.124760 | 202.9 | 0.0 |
| 300 | MCTS | 184.1 | 12.98 | 2,771.8 | 0.469644 | 0.279418 | 140.4 | 60.7 |
| 400 | Stochastic | 84.7 | 57.95 | 12,997.1 | 0.826782 | 0.763784 | 21.0 | 11.3 |
| 400 | LinUCB | 140.4 | 26.17 | 5,609.5 | 0.531016 | 0.355226 | 94.6 | 41.7 |
| 400 | SA | 110.8 | 39.23 | 8,815.6 | 0.638754 | 0.504651 | 57.8 | 28.9 |
| 400 | Greedy | 211.9 | 6.81 | 1,368.5 | 0.322966 | 0.114585 | 206.7 | 0.0 |
| 400 | MCTS | 173.5 | 13.27 | 2,791.6 | 0.502060 | 0.325190 | 124.7 | 53.7 |
| 500 | Stochastic | 87.6 | 59.73 | 13,401.6 | 0.827870 | 0.765779 | 21.7 | 11.5 |
| 500 | LinUCB | 137.2 | 24.32 | 5,110.8 | 0.554340 | 0.388875 | 87.6 | 39.2 |
| 500 | SA | 107.9 | 38.42 | 8,532.8 | 0.615318 | 0.473568 | 59.8 | 30.6 |
| 500 | Greedy | 207.7 | 7.43 | 1,476.2 | 0.318306 | 0.108462 | 204.0 | 0.0 |
| 500 | MCTS | 174.4 | 14.85 | 3,086.1 | 0.516453 | 0.344524 | 121.7 | 51.4 |

Greedy·MCTS를 포함한 정책별 fixed/adaptive 완료 수를 면적에 따라 비교한다. 막대는 10회 반복 평균이며, 신뢰구간은 뒤의 통계 표에서 확인한다.

![고정과 적응형의 면적별 완료 request](../../studies/adaptive_semantic_compression/results/adaptive_enhanced_scalable_verified/figures/finished_by_area_greedy_mcts.png)

300×300 m에서 실제로 선택된 mode를 보여 준다. 실측 mIoU가 비단조여서 emergency·low·high만 도달 가능했고 medium·paper_like는 선택되지 않았다.

![300m 적응형 압축 mode 사용량](../../studies/adaptive_semantic_compression/results/adaptive_enhanced_scalable_verified/figures/adaptive_mode_usage_300m.png)

300×300 m의 평균 시간과 기록된 semantic quality를 함께 본다. 별도 점은 system mode·정책별 평균이며 paired 유의성 검정을 대체하지 않는다.

![300m 지연-품질 균형](../../studies/adaptive_semantic_compression/results/adaptive_enhanced_scalable_verified/figures/latency_quality_tradeoff_300m.png)

### 15.2 Fixed-80 대비 paired 변화

동일한 area·policy·repeat 번호의 `adaptive_semantic - fixed_paper_like`를 먼저 계산한 뒤 평균했다. 음의 시간 Δ는 단축, 음의 품질 Δ는 품질 하락이다.

| 면적 | 정책 | paired n | 완료 Δ | 시간 Δ(s) | 품질 Δ |
|---|---|---|---|---|---|
| 100 | Greedy | 10 | 77.600 | -26.628 | 0.032283 |
| 100 | LinUCB | 10 | 71.200 | -22.913 | 0.010456 |
| 100 | MCTS | 10 | 66.200 | -20.414 | 0.047022 |
| 100 | SA | 10 | 12.700 | -29.529 | 0.004845 |
| 100 | Stochastic | 10 | 3.900 | -18.896 | -0.021075 |
| 200 | Greedy | 10 | 31.100 | -6.485 | 0.013875 |
| 200 | LinUCB | 10 | 37.600 | -12.362 | 0.033772 |
| 200 | MCTS | 10 | 33.800 | -9.645 | 0.058102 |
| 200 | SA | 10 | 10.700 | -9.083 | -0.027537 |
| 200 | Stochastic | 10 | 7.700 | -23.880 | -0.031748 |
| 300 | Greedy | 10 | 19.900 | -4.249 | 0.015387 |
| 300 | LinUCB | 10 | 11.100 | -4.987 | 0.007816 |
| 300 | MCTS | 10 | 19.700 | -6.059 | 0.043295 |
| 300 | SA | 10 | 8.400 | -7.891 | -0.005975 |
| 300 | Stochastic | 10 | 7.800 | -8.819 | -0.010967 |
| 400 | Greedy | 10 | 26.700 | -2.926 | 0.006819 |
| 400 | LinUCB | 10 | 14.200 | -6.017 | -0.006433 |
| 400 | MCTS | 10 | 10.000 | -3.477 | 0.029734 |
| 400 | SA | 10 | 3.700 | -3.614 | 0.008670 |
| 400 | Stochastic | 10 | -2.400 | 1.638 | 0.020941 |
| 500 | Greedy | 10 | 13.100 | -1.504 | 0.004184 |
| 500 | LinUCB | 10 | 6.900 | -6.305 | -0.013293 |
| 500 | MCTS | 10 | 15.900 | -3.671 | 0.021996 |
| 500 | SA | 10 | 0.600 | -1.649 | -0.022945 |
| 500 | Stochastic | 10 | -0.500 | 0.616 | 0.014261 |

### 15.3 95% 신뢰구간과 유의성

| 면적 | 정책 | n | 완료 평균 [95% CI] | 시간 평균 [95% CI] | 품질 평균 [95% CI] | 방법 |
|---|---|---|---|---|---|---|
| 100 | Stochastic | 10 | 35.40 [29.60, 41.20] | 121.81 [93.53, 150.09] | 0.791250 [0.749859, 0.832641] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 100 | LinUCB | 10 | 136.50 [124.28, 148.72] | 23.97 [18.89, 29.04] | 0.408258 [0.384174, 0.432342] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 100 | SA | 10 | 56.90 [51.00, 62.80] | 73.46 [59.86, 87.06] | 0.562363 [0.534167, 0.590560] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 100 | Greedy | 10 | 200.00 [192.92, 207.08] | 7.72 [6.61, 8.84] | 0.341061 [0.332183, 0.349939] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 100 | MCTS | 10 | 190.10 [180.18, 200.02] | 11.52 [10.11, 12.92] | 0.384836 [0.364022, 0.405650] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 200 | Stochastic | 10 | 63.00 [52.82, 73.18] | 74.38 [57.73, 91.04] | 0.777156 [0.753994, 0.800319] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 200 | LinUCB | 10 | 157.10 [151.35, 162.85] | 19.06 [17.36, 20.75] | 0.495117 [0.470453, 0.519780] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 200 | SA | 10 | 91.70 [80.90, 102.50] | 46.66 [40.72, 52.60] | 0.573025 [0.550140, 0.595909] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 200 | Greedy | 10 | 205.10 [197.70, 212.50] | 6.29 [5.88, 6.69] | 0.331701 [0.325778, 0.337623] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 200 | MCTS | 10 | 188.20 [181.65, 194.75] | 11.65 [10.63, 12.66] | 0.435114 [0.419297, 0.450932] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 300 | Stochastic | 10 | 80.00 [69.68, 90.32] | 59.23 [51.88, 66.58] | 0.794648 [0.771000, 0.818296] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 300 | LinUCB | 10 | 149.40 [142.13, 156.67] | 20.83 [19.15, 22.51] | 0.537049 [0.516322, 0.557777] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 300 | SA | 10 | 108.70 [100.65, 116.75] | 38.52 [32.47, 44.58] | 0.609726 [0.594321, 0.625130] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 300 | Greedy | 10 | 210.60 [204.43, 216.77] | 6.27 [5.67, 6.87] | 0.331153 [0.322413, 0.339893] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 300 | MCTS | 10 | 184.10 [177.57, 190.63] | 12.98 [12.25, 13.71] | 0.469644 [0.449424, 0.489865] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 400 | Stochastic | 10 | 84.70 [77.61, 91.79] | 57.95 [50.99, 64.90] | 0.826782 [0.806299, 0.847265] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 400 | LinUCB | 10 | 140.40 [132.61, 148.19] | 26.17 [23.32, 29.02] | 0.531016 [0.513034, 0.548998] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 400 | SA | 10 | 110.80 [104.69, 116.91] | 39.23 [34.42, 44.05] | 0.638754 [0.613542, 0.663966] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 400 | Greedy | 10 | 211.90 [205.37, 218.43] | 6.81 [6.44, 7.18] | 0.322966 [0.319424, 0.326507] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 400 | MCTS | 10 | 173.50 [163.05, 183.95] | 13.27 [12.72, 13.82] | 0.502060 [0.484388, 0.519733] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 500 | Stochastic | 10 | 87.60 [77.13, 98.07] | 59.73 [51.67, 67.79] | 0.827870 [0.807840, 0.847899] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 500 | LinUCB | 10 | 137.20 [129.82, 144.58] | 24.32 [22.50, 26.13] | 0.554340 [0.522356, 0.586323] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 500 | SA | 10 | 107.90 [100.47, 115.33] | 38.42 [33.86, 42.98] | 0.615318 [0.592025, 0.638611] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 500 | Greedy | 10 | 207.70 [198.39, 217.01] | 7.43 [6.63, 8.23] | 0.318306 [0.314533, 0.322078] | two-sided Student t; n=1 uses zero-width descriptive interval |
| 500 | MCTS | 10 | 174.40 [166.35, 182.45] | 14.85 [14.09, 15.61] | 0.516453 [0.494243, 0.538663] | two-sided Student t; n=1 uses zero-width descriptive interval |

이 파일은 각 mode의 평균 CI를 기록한다. paired difference 자체의 CI, p-value, Holm 다중비교 보정은 별도 열이 없으므로 **증거 없음**이며 통계적 유의성을 주장하지 않는다.

### 15.4 품질 guardrail

구조 무결성 `passed`=True, 방향 가설 `all_comparisons_pass`=False, 기록된 최저 품질 Δ=-0.031748이다.

| 면적 | 정책 | 완료 변화 | 시간 변화 | 품질 Δ | 조합 가설 pass |
|---|---|---|---|---|---|
| 100 | Stochastic | +12.4% | -13.4% | -0.021075 | True |
| 100 | LinUCB | +109.0% | -48.9% | 0.010456 | True |
| 100 | SA | +28.7% | -28.7% | 0.004845 | True |
| 100 | Greedy | +63.4% | -77.5% | 0.032283 | True |
| 100 | MCTS | +53.4% | -63.9% | 0.047022 | True |
| 200 | Stochastic | +13.9% | -24.3% | -0.031748 | True |
| 200 | LinUCB | +31.5% | -39.3% | 0.033772 | True |
| 200 | SA | +13.2% | -16.3% | -0.027537 | True |
| 200 | Greedy | +17.9% | -50.8% | 0.013875 | True |
| 200 | MCTS | +21.9% | -45.3% | 0.058102 | True |
| 300 | Stochastic | +10.8% | -13.0% | -0.010967 | True |
| 300 | LinUCB | +8.0% | -19.3% | 0.007816 | True |
| 300 | SA | +8.4% | -17.0% | -0.005975 | True |
| 300 | Greedy | +10.4% | -40.4% | 0.015387 | True |
| 300 | MCTS | +12.0% | -31.8% | 0.043295 | True |
| 400 | Stochastic | -2.8% | +2.9% | 0.020941 | False |
| 400 | LinUCB | +11.3% | -18.7% | -0.006433 | True |
| 400 | SA | +3.5% | -8.4% | 0.008670 | True |
| 400 | Greedy | +14.4% | -30.1% | 0.006819 | True |
| 400 | MCTS | +6.1% | -20.8% | 0.029734 | True |
| 500 | Stochastic | -0.6% | +1.0% | 0.014261 | False |
| 500 | LinUCB | +5.3% | -20.6% | -0.013293 | True |
| 500 | SA | +0.6% | -4.1% | -0.022945 | True |
| 500 | Greedy | +6.7% | -16.8% | 0.004184 | True |
| 500 | MCTS | +10.0% | -19.8% | 0.021996 | True |

상위 `passed`는 coverage·고유 키·finite 값·비교 분모 무결성만 뜻한다. 조합별 `pass`와 `all_comparisons_pass`는 저장된 방향·품질 guardrail이며, 논문 일치나 통계적 유의성을 뜻하지 않는다.

### 15.5 Mode 사용 빈도

면적 5개에 저장된 평균 count를 정책별로 합산했다. 괄호는 해당 정책 내 기록 count 비율이다.

| 정책 | emergency | fixed_paper_like | high | low | medium | paper_like | raw | 합계 |
|---|---|---|---|---|---|---|---|---|
| Stochastic | 48.2 (13.7%) | 0.0 (0.0%) | 7.6 (2.2%) | 41.6 (11.9%) | 0.0 (0.0%) | 0.0 (0.0%) | 253.3 (72.2%) | 350.7 |
| LinUCB | 254.3 (35.3%) | 0.0 (0.0%) | 14.9 (2.1%) | 242.5 (33.7%) | 0.0 (0.0%) | 0.0 (0.0%) | 208.9 (29.0%) | 720.6 |
| SA | 130.8 (27.5%) | 0.0 (0.0%) | 9.9 (2.1%) | 130.0 (27.3%) | 0.0 (0.0%) | 0.0 (0.0%) | 205.3 (43.1%) | 476.0 |
| Greedy | 174.9 (16.9%) | 0.0 (0.0%) | 375.6 (36.3%) | 450.2 (43.5%) | 0.0 (0.0%) | 0.0 (0.0%) | 34.6 (3.3%) | 1035.3 |
| MCTS | 246.6 (27.1%) | 0.0 (0.0%) | 70.8 (7.8%) | 390.7 (42.9%) | 0.0 (0.0%) | 0.0 (0.0%) | 202.2 (22.2%) | 910.3 |

### 15.6 면적·정책 일반화

각 조합의 fixed 대비 상대 변화를 먼저 구한 다음 그룹 안에서 평균했다.

**면적별(정책 평균)**

| 면적 | 정책 수 | 완료 변화 | 시간 변화 | 품질 Δ |
|---|---|---|---|---|
| 100 | 5 | +53.4% | -46.5% | 0.014706 |
| 200 | 5 | +19.7% | -35.2% | 0.009293 |
| 300 | 5 | +9.9% | -24.3% | 0.009911 |
| 400 | 5 | +6.5% | -15.0% | 0.011946 |
| 500 | 5 | +4.4% | -12.1% | 0.000841 |

**정책별(면적 평균)**

| 정책 | 면적 수 | 완료 변화 | 시간 변화 | 품질 Δ |
|---|---|---|---|---|
| Greedy | 5 | +22.6% | -43.1% | 0.014510 |
| LinUCB | 5 | +33.0% | -29.4% | 0.006463 |
| MCTS | 5 | +20.7% | -36.3% | 0.040030 |
| SA | 5 | +10.9% | -14.9% | -0.008588 |
| Stochastic | 5 | +6.8% | -9.3% | -0.005718 |

### 15.7 논문 재현 simulator에 미친 영향

| 감사 항목 | 결과 |
|---|---|
| fixed-vs-reproduction 대응 조합 | 25 |
| 공통 지표 최대 절대차 | 0.000000000 |
| neural quality mode | selection |
| neural summary source | C:\Users\firep\OneDrive\바탕 화면\Replication-Experiment-of-Airtalking\studies\neural_encoder_decoder\results\enhanced_scalable_full_256x128_verified\airtalking_semantic_summary.json |

이 비교는 simulator 입력 연결의 일관성 감사다. adaptive 결과 자체를 Fig. 3~6 `verify_against_paper.py`로 다시 평가한 verifier CSV는 필수 adaptive 산출물에 없으므로 논문 그림 일치도는 **미실행/증거 없음**이다.

## 16. 결과 신뢰성 감사표

| 주장 | 현재 증거 | 신뢰 판단 |
|---|---|---|
| adaptive 코드가 scheduler 안에 있다 | candidate 생성 때 mode·payload·quality를 선택 | 높음 |
| 기존 proxy 75개 조합 결과 파일이 완전하다 | legacy validator의 row count와 finite 검사 통과 | 높음(구조) |
| 기존 proxy 설정에서 adaptive 평균이 fixed보다 좋다 | legacy 25개 조합 방향 검사 통과 | 중간 |
| 강화 결과의 개선이 통계적으로 확실하다 | 조합별 평균 CI는 있으나 paired 차이 CI·p-value·다중비교 보정은 미제공 | 미확인 |
| 기존 저장 결과의 다섯 mode가 neural codec이다 | neural source 미적용, label proxy 사용 | 아니오 |
| 강화 품질 수치가 실제 RGB encoder 오류를 포함한다 | RGB 입력에서 얻은 5-rate uint8 decoded mIoU 사용 | 예(내부 val 기준) |
| 논문 조건에서도 같은 개선이 난다 | calibrated hidden params와 독립 구현 사용 | 미확인 |
| 새 강화 codec이 개선을 유지한다 | 실제 강화 산출물·provenance·validator로 판정 | 강화 5-rate codec 연결과 로컬 validator 통과 확인; mode·area·policy 조합별 최소 repeat 10개 기록; 논문 정확 재현·인과 효과·통계적 유의성은 별도 증거 없이는 주장하지 않음 |

## 17. 주장할 수 있는 것과 없는 것

### 17.1 현재 주장할 수 있는 것

- 구현된 simulator 안에서 candidate별 adaptive compression이 동작한다.
- 강화 run은 실제 RGB encoder에서 측정한 20/40/60/80/120채널 uint8 rate-quality profile을 사용한다.
- 저장된 10×1,000-slot 평균에서는 25개 area×policy 조합 중 23개에서 fixed-80 대비 완료 수가 늘고 평균 시간이 줄었다.
- Stochastic 400 m와 500 m에서는 같은 방향의 개선이 나오지 않았다.
- 750개 repeat 원시 행과 조합별 평균·sample 표준편차·양측 Student-t 95% 신뢰구간이 저장돼 있다.
- 성능 변화에는 semantic quality와 payload 선택의 trade-off가 있다.
- 적응형이 반드시 평균 payload를 줄이지는 않으며, 좋은 링크에서 품질을 위해 더 큰 payload를 쓸 수 있다.

### 17.2 현재 주장하면 안 되는 것

- AirTalking 논문의 원래 neural codec을 재현했다.
- 기존 label proxy mIoU를 현재 neural decoded visual/segmentation 품질과 같은 측정값으로 취급한다.
- adaptive가 모든 실제 무선 환경에서 우월하다.
- 구조 무결성 `passed=true`이므로 통계적으로 유의하다.
- 23/25 조합의 평균 방향만으로 paired 인과 효과나 다중비교 보정 후 유의성을 입증했다.
- 강화 모델의 mIoU 상승이 모델 구조 하나만의 순수 효과다. 학습 표본·해상도·decoder 목표도 함께 바뀌었다.

## 18. 최종 결론

기존 후속 연구의 아이디어는 타당하다. 무선 상태가 달라지는데 압축률을 하나로 고정할 이유는 없고, scheduler가 payload-quality 선택까지 함께 고려하면 자원 점유 시간을 줄일 수 있다. 실제 강화 결과도 25개 조합 중 23개에서 이 방향을 보였지만, 실패한 두 조합 때문에 보편적 우월성은 주장할 수 없다.

기존 결과의 가장 큰 약점이었던 정답 label proxy는 현재 강화 run에서 실제 uint8 multi-rate neural profile로 대체됐다. 남은 핵심 한계는 다섯 rate의 mIoU가 거의 평탄하고 비단조라는 점, 같은 Cityscapes val을 checkpoint 선택과 내부 평가에 함께 썼다는 점, paired 차이의 신뢰구간·p-value·다중비교 보정이 없다는 점, simulator가 비공개 상수에 대한 calibrated 가정에 의존한다는 점이다. 따라서 다음 단계에서는 독립 test split 또는 별도 holdout, paired repeat 차이 분석, 여러 seed와 무선 조건의 외부 검증이 필요하다. 현재 결과도 “AirTalking 원본 구현의 확정 개선”이 아니라 “공개 상수와 명시적 가정으로 만든 독립 재현 환경에서의 후속 실험”으로 보고해야 한다.
