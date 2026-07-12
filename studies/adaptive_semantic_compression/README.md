# Adaptive Semantic Compression Study

채널 상태에 따라 의미 payload의 rate-quality operating point를 바꾸는 후속 연구 폴더입니다. 고정 80채널 paper-like 방식과 5-rate adaptive 방식을 같은 UAV scheduler 안에서 비교합니다.

## Legacy 결과와 강화 실험의 차이

저장된 `results/full_adaptive_results`는 Cityscapes **정답 label map을 다섯 크기로 축소·확대한 proxy**를 사용했습니다. 그 run의 `run_metadata.json`이 가리키는 source reproduction/quality도 label proxy이며 `source_neural_encoder_decoder`가 없습니다. 따라서 과거 문서의 표현과 달리, 이 legacy 결과에는 neural anchor가 실제로 적용되지 않았고 다섯 mode를 neural codec 결과로 부를 수 없습니다.

현재 강화 경로는 `airtalking_semantic_summary.json`의 실제 5-rate neural profile을 읽어 다음을 수행합니다.

- 20/40/60/80/120채널의 실제 uint8 payload ratio와 measured mIoU를 payload 순서대로 `emergency/low/medium/paper_like/high`에 매핑
- fixed-80과 adaptive 모두 같은 neural rate-quality 표 사용
- measured feature encode/decode throughput 쌍을 함께 적용
- raw repeat, 평균·sample 표준편차·양측 Student-t 95% 신뢰구간 저장

`record_only`에 single-rate summary를 넣는 것은 다섯 선택 품질을 neural로 바꾸지 않습니다. 5-rate 강화 실험은 `--neural-quality-mode selection`을 사용해야 합니다.

## 선택 규칙

legacy proxy run은 `0.80/0.88/0.925/0.95/0.965`라는 휴리스틱 목표를 썼습니다. 이 숫자는 논문값도 neural 품질 기준도 아닙니다. 강화 5-rate run의 기본 `auto` 규칙은 **실측 다섯 rate의 품질 순서에 맞춘 `measured_ordered` rule**을 만들고, 실제 적용한 SINR bin·목표 품질·rule source를 metadata에 기록합니다. 측정 mIoU가 rate에 따라 비단조이면 일부 mode가 어느 bin에서도 선택되지 않을 수 있으므로 reachable mode 목록과 개수도 기록합니다. 따라서 proxy의 0.8 threshold를 neural 결과에 그대로 재사용했다고 서술하지 않습니다.

명시적 품질 목표를 시험할 때만 `--adaptive-threshold-rule explicit --adaptive-quality-thresholds q1,q2,q3,q4,q5`를 사용하고, 그 선택도 metadata에 남깁니다.

## 실제 full 실험 구조

adaptive runner는 세 mode 각각에 대해 5개 면적 × 5개 정책을 실행합니다.

- `nonsemantic`: raw payload
- `fixed_paper_like`: 항상 80채널 operating point
- `adaptive_semantic`: candidate SINR에 따라 다섯 neural rate 중 선택

따라서 full summary는 3×5×5=75개 조합입니다. 기본 10 repeats이면 repeat 원시 행은 750개입니다. 구조 validator의 상위 `passed=true`는 행 coverage·고유 키·finite 값·비교 분모 같은 **산출물 무결성**을 뜻합니다. Adaptive 방향 가설은 `adaptive_vs_fixed.all_comparisons_pass`와 조합별 `pass`로 별도 기록되며, 이 값들은 논문 일치나 통계적 유의성을 뜻하지 않습니다.

## 실행

강화 5-rate full run:

```powershell
.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\run_full_adaptive_research.py `
  --metadata studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified\run_metadata.json `
  --quality studies\adaptive_semantic_compression\results\probe_outputs\compression_quality.csv `
  --neural-summary studies\neural_encoder_decoder\results\enhanced_scalable_full_256x128_verified\airtalking_semantic_summary.json `
  --neural-quality-mode selection `
  --adaptive-threshold-rule measured_ordered `
  --workers 6 `
  --out studies\adaptive_semantic_compression\results\adaptive_enhanced_scalable_verified
```

결과 구조 검증:

```powershell
.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\validate_full_adaptive_results.py `
  --summary studies\adaptive_semantic_compression\results\adaptive_enhanced_scalable_verified\summary_metrics.csv `
  --out studies\adaptive_semantic_compression\results\adaptive_enhanced_scalable_verified\result_validation.json
```

validator schema v2는 source summary의 절대경로·SHA-256, `passed`, 75개 expected combinations, missing/duplicate/non-finite/zero-denominator 진단을 기록합니다. strict finalizer는 이 snapshot과 summary가 실제로 일치하는지 다시 확인합니다.

## 현재 verified 결과

대표 결과는 [`results/adaptive_enhanced_scalable_verified`](results/adaptive_enhanced_scalable_verified)입니다. 실제 5-rate neural profile과 fixed-80을 같은 scheduler에서 비교한 결과, adaptive는 25개 area×policy 조합 중 23개에서 완료 수를 늘리고 평균 시간을 줄였습니다. 조합별 변화율의 단순 평균은 완료 수 `+18.78%`, 평균 시간 `-26.61%`이며 Stochastic 400 m·500 m는 두 조건을 함께 만족하지 못했습니다.

`result_validation.json`의 상위 `passed=true`는 75개 행의 구조·수치 무결성 통과를 뜻하고, `all_comparisons_pass=false`는 두 실패 조합을 숨기지 않았다는 뜻입니다. 기록된 `measured_ordered` rule에서 실제 도달 가능한 mode는 `emergency`, `low`, `high` 3개입니다. 750개 repeat 원시 행과 조합별 Student-t 95% 신뢰구간은 저장했지만 paired 차이의 신뢰구간·p-value·다중비교 보정은 수행하지 않았습니다. 상세 내용은 [`reports/final/03_후속연구_적응형_의미압축_final.md`](../../reports/final/03_후속연구_적응형_의미압축_final.md)에서 확인할 수 있습니다.

probe만 다시 만들 때:

```powershell
.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\run_adaptive_probe.py `
  --gt-root dataset\gtFine_trainvaltest\gtFine --sample-limit 400 --splits train,val
```

현재 workspace에는 강화 full 산출물과 strict 최종 보고서가 생성돼 있습니다. 다른 clone이나 장비에서는 해당 결과 디렉터리와 Git에서 제외된 checkpoint가 실제로 존재하는지 다시 확인해야 하며, legacy full 결과만으로 강화 neural full run 완료를 추정하면 안 됩니다.

## 해석 한계

강화 codec도 AirTalking 저자의 비공개 neural 구현은 아닙니다. simulator는 calibrated `AssumedParams`에 의존하고, 현재 품질은 이미지 전체 평균이며, 같은 Cityscapes val이 모델 선택과 내부 평가에 함께 쓰입니다. 완료 수·지연 개선은 semantic quality guardrail, repeat별 불확실성, 면적·정책별 일반성과 함께 해석해야 합니다.
