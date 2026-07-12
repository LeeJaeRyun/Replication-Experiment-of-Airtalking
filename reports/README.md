# AirTalking 재현 및 후속 연구 핵심 정리

## 결론부터

- **논문 재현:** 의미 압축을 허용하면 원본만 보낼 때보다 요청을 더 많이 처리한다는 방향은 나왔다. 하지만 논문 그림의 수치까지 맞추는 데는 실패했다.
- **후속 연구:** 논문의 고정 압축률 대신 무선 상태에 따라 전송량을 바꾸자, 25개 조건 중 23개에서 완료 요청 수가 늘고 처리 시간이 줄었다. 2개 조건에서는 실패했다.
- **딥러닝 모델:** 재현과 후속 연구에 서로 다른 모델을 쓴 것이 아니다. Cityscapes로 학습한 **하나의 인코더·디코더**를 두 실험이 같이 썼다.

## 공통: 인코더·디코더는 무엇을 보고 어떻게 학습했나

논문은 `Cityscapes 사진 → modified U-Net → 압축 정보 전송 → modified Pix2PixHD → 사진 복원`이라는 큰 흐름만 공개했다. 정확한 층 구성, 전송할 숫자 묶음의 모양, 8비트 변환법, 오차 계산법, 학습 설정·횟수와 가중치는 공개하지 않았다. 따라서 이 저장소의 모델은 **논문 저자 모델의 정확한 복제본이 아니라, 공개된 흐름을 참고해 새로 만든 모델**이다.

학습에는 Cityscapes의 다음 두 정보를 한 쌍으로 사용했다.

- 입력: 도로 장면 RGB 사진
- 정답: 사진의 각 픽셀이 도로·건물·사람·자동차 등 19종 중 무엇인지 표시한 지도
- 데이터: train 2,975장, val 500장
- 크기: 256×128

실제 학습 흐름은 다음과 같다.

1. 인코더가 RGB 사진을 가로·세로 각각 16분의 1인 작은 숫자 묶음(feature)으로 바꾼다.
2. 이 값을 8비트 정수로 바꾼 뒤, 앞쪽 20·40·60·80·120채널 중 하나만 전송한다.
3. 디코더는 **전송된 숫자 묶음만** 받아 RGB 사진과 19종 픽셀 지도를 함께 복원한다.
4. 복원 사진과 원본 사진의 차이, 예측한 픽셀 종류와 정답의 차이를 동시에 줄이도록 인코더와 디코더를 함께 학습한다.
5. 한 학습 묶음마다 20채널, 120채널, 임의의 중간 채널 하나를 같이 연습시켰다. 전체 데이터를 20회 학습했고, 검증 성능이 가장 좋았던 19회차 가중치를 사용했다.

80채널 결과는 다음과 같다.

| 항목 | 결과 | 뜻 |
|---|---:|---|
| 전송 byte / 원본 RGB byte | 10.42% | 원본의 약 10분의 1을 전송 |
| mIoU | 30.54% | 19종을 종류별로 구분한 평균 점수 |
| 전체 픽셀 정확도 | 82.86% | 모든 유효 픽셀 중 맞힌 비율 |
| RGB 복원 PSNR / SSIM | 18.43 dB / 0.568 | 복원 사진 품질. 논문에 대응 수치가 없어 직접 비교 불가 |
| 인코딩 / 디코딩 중앙 시간 | 1.87 ms / 3.01 ms | RTX 4060 Ti에서 신경망 계산만 측정 |

mIoU 30.54%는 압축 전송이 작동했다는 뜻이지, 의미 복원이 충분히 정확하다는 뜻은 아니다. 또한 val 500장을 모델 선택과 최종 내부 평가에 같이 썼으므로 별도 시험 데이터로 확인한 결과도 아니다.

근거: [학습 코드](../studies/neural_encoder_decoder/code/train_enhanced_semantic_codec.py), [실제 학습 설정](../studies/neural_encoder_decoder/results/enhanced_scalable_full_256x128_verified/launch_manifest.json), [학습 결과](../studies/neural_encoder_decoder/results/enhanced_scalable_full_256x128_verified/result_summary.json), [채널별 품질](../studies/neural_encoder_decoder/results/enhanced_scalable_full_256x128_verified/rate_quality.csv)

## 1. 논문을 바탕으로 한 실험 재현

### 무엇을 어떻게 재현했나

`base/Airtalking_Aerial_D2D_for_Multi-UAV_Systems_Based_on_Semantic_Communication.pdf`에 보관된 원 논문의 저자 코드를 다시 실행한 것이 아니라, 논문 수식과 공개 설정을 보고 Python 시뮬레이터를 새로 만들었다.

- UAV 20대와 지상 단말 20대
- 100×100m부터 500×500m까지 5개 영역
- 한 실험당 1,000초, 조건마다 10회 반복
- Stochastic, LinUCB, SA, Greedy, MCTS의 5개 선택 방식 비교

요청이 생길 때마다 사용 가능한 UAV 두 대와 전송 방식을 후보로 만든다. 각 후보의 UAV 이동시간, 무선 전송시간, 인코딩·디코딩시간과 에너지를 계산한 뒤 정책이 하나를 고른다. 최종적으로 완료 요청 수, 평균 처리시간, 이동거리와 에너지를 기록했다.

시뮬레이션 중에 신경망을 매번 실행한 것은 아니다. 앞에서 학습한 모델의 **압축률과 처리속도 측정값을 JSON에서 읽어 시간·전송량 계산에 넣었다.** 논문에 없는 요청 발생 확률, 작업 크기, UAV 이동·대기 전력, 간섭 보정값 등은 별도 가정값으로 넣었다.

### 결과

300×300m에서 의미 압축을 선택할 수 있을 때와 원본만 보낼 때의 완료 요청 수는 다음과 같았다. 값은 10회 평균이다.

| 정책 | 의미 압축 선택 가능 | 원본만 전송 |
|---|---:|---:|
| LinUCB | 138.3 | 66.5 |
| SA | 100.3 | 68.3 |
| Greedy | 190.7 | 75.9 |
| MCTS | 164.4 | 77.0 |

네 정책 모두 이 시뮬레이터 안에서는 의미 압축을 선택할 수 있을 때 더 많은 요청을 끝냈다. 다만 논문 그림에서 읽은 73개 근삿값과 비교하면 다음과 같았다.

- 차이 25% 이내: 17개
- 차이 25~50%: 17개
- 차이 50% 초과: 39개

따라서 **논문의 방향성은 일부 재현했지만, 논문 수치의 정량 재현에는 실패했다.** 주된 이유는 원본 신경망·시뮬레이터 코드와 중요한 설정값이 공개되지 않았고, 현재 정책들도 논문 설명을 바탕으로 만든 독립 근사이기 때문이다. 완료 요청 수 계산은 복원된 의미가 실제 업무에 충분히 정확한지도 평가하지 않는다.

근거: [재현 시뮬레이터](../studies/airtalking_reproduction/code/airtalking_reproduction.py), [실제 적용값](../studies/airtalking_reproduction/results/airtalking_enhanced_scalable_verified/run_metadata.json), [전체 결과](../studies/airtalking_reproduction/results/airtalking_enhanced_scalable_verified/summary_metrics.csv), [논문 수치 비교](../studies/airtalking_reproduction/results/airtalking_enhanced_scalable_verified/verification_against_paper_enhanced_scalable_verified.csv)

## 2. 논문의 한계를 바탕으로 한 후속 연구

### 어떤 한계를 해결하려 했나

논문은 의미 압축률을 `0.104` 하나로 고정했다. 하지만 무선 상태는 계속 달라진다. 채널이 나쁠 때는 더 작게 보내 전송을 빨리 끝내고, 채널이 좋을 때는 더 많은 정보를 보내 품질을 지키는 편이 나을 수 있다.

그래서 앞의 **같은 모델 하나**에서 전송 채널 수만 바꾸는 다섯 단계를 만들었다.

| 전송 채널 | 원본 대비 전송량 | mIoU |
|---:|---:|---:|
| 20 | 2.60% | 30.48% |
| 40 | 5.21% | 30.58% |
| 60 | 7.81% | 30.50% |
| 80 (논문 비율과 유사) | 10.42% | 30.54% |
| 120 | 15.63% | 30.63% |

딥러닝 때 무선 채널 정보는 사용하지 않았다. 모델은 RGB 사진과 픽셀 정답만 보고 학습했다. 학습이 끝난 뒤 실행 단계에서 전송 방식 선택기가 후보 링크의 SINR, 즉 무선 신호 상태를 보고 이미 측정한 다섯 전송량 중 하나를 고른다. 선택한 전송량은 전송시간, 디코딩시간, 에너지와 전체 비용 계산에 바로 반영된다.

### 결과

원본 전송, 고정 80채널, 채널 적응형 전송을 `5개 영역 × 5개 정책 × 10회`로 비교했다. 25개 영역·정책 조합 중 **23개에서 적응형이 고정 80채널보다 완료 수를 늘리고 평균 시간을 줄였다.** 조합별 변화율을 단순 평균하면 완료 수는 `+18.78%`, 평균 시간은 `-26.61%`였다.

300×300m 결과는 다음과 같다.

| 정책 | 고정 완료 수 → 적응형 | 고정 시간 → 적응형 |
|---|---:|---:|
| Stochastic | 72.2 → 80.0 | 68.05s → 59.23s |
| LinUCB | 138.3 → 149.4 | 25.81s → 20.83s |
| SA | 100.3 → 108.7 | 46.41s → 38.52s |
| Greedy | 190.7 → 210.6 | 10.52s → 6.27s |
| MCTS | 164.4 → 184.1 | 19.04s → 12.98s |

하지만 결과를 확정적 개선으로 보면 안 된다.

- Stochastic 400m와 500m에서는 오히려 완료 수가 줄고 시간이 늘었다.
- 전송량을 6배 바꿔도 mIoU가 30.48~30.63%로 거의 같고 순서도 일정하지 않았다. 실제 선택 가능한 단계도 5개 중 3개뿐이었다.
- 10회 반복값은 저장했지만, 고정 방식과 적응형의 차이가 우연인지 확인하는 통계 검정은 하지 않았다.
- 같은 독립 시뮬레이터와 가정값을 사용했으므로, 원 논문 시스템이나 실제 UAV 환경에서도 좋아진다고 말할 수 없다.

즉 후속 연구는 **채널에 맞춰 전송량을 바꾸는 방식이 이 시뮬레이터에서는 유망했다**는 결과다. 원 논문의 신경망을 개선했다고 입증한 결과는 아니다.

근거: [적응형 실행 코드](../studies/adaptive_semantic_compression/code/run_full_adaptive_research.py), [실제 선택 규칙과 모델 정보](../studies/adaptive_semantic_compression/results/adaptive_enhanced_scalable_verified/run_metadata.json), [전체 결과](../studies/adaptive_semantic_compression/results/adaptive_enhanced_scalable_verified/summary_metrics.csv), [23개 성공·2개 실패 검증](../studies/adaptive_semantic_compression/results/adaptive_enhanced_scalable_verified/result_validation.json)
