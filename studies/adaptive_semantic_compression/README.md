# Adaptive Semantic Compression Study

채널 상태에 따라 semantic compression 비율을 바꾸는 후속 연구 폴더입니다.

## 무엇을 했나

기존 재현 실험은 semantic compression이 거의 고정값처럼 들어갑니다.
이 폴더에서는 네트워크 상태에 따라 압축률을 바꾸면 성능이 좋아지는지 봅니다.

- 채널이 나쁘면 더 강하게 압축해서 전송량을 줄입니다.
- 채널이 좋으면 덜 압축해서 semantic 품질을 지킵니다.
- 고정 `paper_like` 방식과 adaptive 방식을 비교합니다.

## Encoder/Decoder 반영

직접 학습한 neural encoder/decoder 결과를 adaptive 실험에도 반영했습니다.

- `paper_like` 압축 단계는 neural encoder/decoder의 `rho_c = 0.10417`을 기준값으로 연결했습니다.
- encode/decode 시간과 mIoU도 분석/보고서에 기록합니다.
- 기본값은 `record_only`입니다. 즉, 직접 학습한 encoder/decoder 결과를 기록하고 기준값으로 연결하되, adaptive의 모든 압축 단계 선택 품질을 전부 neural mIoU로 바꾸지는 않습니다.

이유는 현재 neural network로 직접 학습한 단계가 `paper_like` 1개뿐이기 때문입니다.
진짜 완전한 adaptive neural codec이 되려면 `emergency`, `low`, `medium`, `paper_like`, `high` 각각에 해당하는 encoder/decoder를 따로 학습해야 합니다.

품질 선택까지 neural 값으로 강제로 쓰고 싶으면:

```powershell
python studies\adaptive_semantic_compression\code\run_full_adaptive_research.py --neural-quality-mode selection
```

## 구조

```text
code/
  run_adaptive_probe.py              압축률 후보별 pilot 측정
  run_full_adaptive_research.py      scheduler에 adaptive compression을 넣은 full experiment
  validate_full_adaptive_results.py  결과 검증 스크립트

reports/
  Adaptive_Semantic_Compression_Easy_Final_KR.docx

results/
  probe_outputs/          압축률 후보 측정 결과
  full_adaptive_results/  full adaptive experiment 결과
```

## 실행 예시

```powershell
python studies\adaptive_semantic_compression\code\run_adaptive_probe.py --sample-limit 400
python studies\adaptive_semantic_compression\code\run_adaptive_probe.py --reuse-quality
python studies\adaptive_semantic_compression\code\run_full_adaptive_research.py
python studies\adaptive_semantic_compression\code\validate_full_adaptive_results.py
```

## 현재 결론

300m 조건 예시에서 adaptive 방식은 fixed `paper_like`보다 완료 task 수가 늘고 평균 시간이 줄었습니다.
다만 현재 단계는 “완전한 adaptive neural encoder/decoder”가 아니라, neural encoder/decoder로 측정한 `paper_like` 기준값을 adaptive 실험에 연결한 버전입니다.
