# AirTalking Reproduction Study

AirTalking의 공개 수식과 표를 바탕으로 UAV D2D 의미통신 실험을 독립적으로 재구현한 폴더입니다. 원본 simulator·neural codec·raw plotting data와 여러 상수가 공개되지 않았으므로, 결과는 **명시한 가정을 포함한 부분 재현**입니다.

## 공개값과 가정값

`airtalking_reproduction.py`는 두 묶음을 분리해 `run_metadata.json`에 저장합니다.

- `paper_params`: 논문 Table III에서 옮긴 UAV/device 수, 시간·이동·채널 상수, \(\rho_c\), \(\rho_r\), codec 처리율 등
- `assumed_params`: request 확률, workload 분포, device 이동, 이동/hover·codec power, energy weight, 간섭 보정, 정책 sampling/iteration, seed 등 논문에 수치가 없거나 구현상 추가한 값

실행 시 우선순위는 `--assumed KEY=VALUE` > `--assumed-metadata` > 코드 default입니다. 이 저장소의 대표 비교는 calibrated metadata의 `assumed_params`를 명시적으로 다시 읽습니다. `calibrated`는 논문 공개값이라는 뜻이 아니라 Fig. 3/4/6에서 읽은 근삿값에 가까워지도록 미공개 가정을 조정했다는 뜻입니다.

## 실제 실험 coverage

일반 재현기는 다음 조합을 실행합니다.

- semantic: 5개 면적(100~500 m) × 5개 정책(Stochastic, LinUCB, SA, Greedy, MCTS) = 25개 조합
- nonsemantic: 300 m × 4개 최적화 정책(LinUCB, SA, Greedy, MCTS) = 4개 조합

즉 semantic과 nonsemantic을 모두 5×5로 실행하지 않습니다. 기본 10 repeats이면 `repeat_metrics.csv`에는 29×10개의 repeat 행이 생기며, Fig. 6식 semantic/nonsemantic 비교는 300 m의 네 정책만 대상으로 합니다.

## Semantic profile의 종류

- summary 없음: Table III의 \(\rho_c=0.104\), \(\rho_r=3\), 91.30/23.23 Mbps를 사용
- 과거 Cityscapes proxy: 정답 label map을 zlib 또는 resize한 값으로 실제 neural encoder 측정이 아님
- 기존 neural 기준선: RGB→latent→segmentation 모델의 한 rate; 실제 uint8 전송과 RGB 복원은 없음
- 강화 후속 모델: 실제 uint8 5-rate RGB+segmentation codec의 80채널 paper-like ratio·품질과 GPU feature timing을 사용

강화 summary는 `--semantic-profile-kind feature --semantic-raw-basis uncompressed`로 읽습니다. PNG 분모는 측정하지 않았고, GPU timing은 CPU 직렬화·zlib transport를 제외하므로 PNG ratio나 end-to-end zlib 처리율을 만들어내지 않습니다. 강화 codec은 논문 저자의 비공개 codec이 아니라 교체 민감도를 보기 위한 독립 후속 모델입니다.

## 실행

Table III codec 상수 + 대표 calibrated 가정:

```powershell
.\.venv\Scripts\python.exe studies\airtalking_reproduction\code\airtalking_reproduction.py `
  --assumed-metadata studies\airtalking_reproduction\results\airtalking_cityscapes_calibrated_final_p012\run_metadata.json `
  --workers 6 `
  --out studies\airtalking_reproduction\results\airtalking_tableiii_codec_calibrated_assumptions_recheck
```

강화 neural profile + 같은 calibrated 가정:

```powershell
.\.venv\Scripts\python.exe studies\airtalking_reproduction\code\airtalking_reproduction.py `
  --semantic-summary studies\neural_encoder_decoder\results\enhanced_scalable_full_256x128_verified\airtalking_semantic_summary.json `
  --semantic-profile-kind feature --semantic-raw-basis uncompressed `
  --semantic-encoder-mode measured --semantic-decoder-mode measured `
  --assumed-metadata studies\airtalking_reproduction\results\airtalking_cityscapes_calibrated_final_p012\run_metadata.json `
  --workers 6 `
  --out studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified

.\.venv\Scripts\python.exe studies\airtalking_reproduction\code\verify_against_paper.py `
  --summary studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified\summary_metrics.csv `
  --out-dir studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified `
  --label enhanced_scalable_verified
```

verifier의 paper-side 값은 Fig. 3/4/6의 raw data가 아니라 렌더링된 그림에서 읽은 근삿값입니다. 새 strict verifier 산출물은 단 하나의 verifier CSV와 같은 stem의 `status=completed` JSON에 입력 summary의 경로·SHA-256과 verdict/행 수를 고정해, 다른 CSV를 잘못 결합하는 일을 막습니다.

## 현재 verified 결과

대표 결과는 [`results/airtalking_enhanced_scalable_verified`](results/airtalking_enhanced_scalable_verified)입니다. 10 repeats의 원시 행과 조합별 sample 표준편차·양측 Student-t 95% 신뢰구간을 저장했고, verifier 판정은 `match 17 / partial 17 / mismatch 39`입니다. mismatch가 다수이므로 정량 재현 성공으로 해석하지 않습니다.

300 m의 LinUCB·SA·Greedy·MCTS에서는 semantic 완료 수가 nonsemantic보다 큰 방향이 관찰됐지만, calibrated `AssumedParams`와 독립 simulator 안의 결과입니다. 상세 내용은 [`reports/final/02_AirTalking_논문_실험_재현_final.md`](../../reports/final/02_AirTalking_논문_실험_재현_final.md)에서 확인할 수 있습니다.

## 결론

현재 강화 결과는 semantic codec의 실제 측정 profile과 반복 통계를 포함하지만, 여러 지표와 정책 순위가 논문 그림의 추정치와 정량적으로 맞지 않습니다. 원본 simulator·raw data·비공개 상수가 없으므로 완전 재현이라고 주장하지 않습니다.
