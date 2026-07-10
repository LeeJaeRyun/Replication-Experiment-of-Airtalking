# Adaptive Semantic Compression Study

AirTalking 재현 시뮬레이터 위에 채널 상태 기반 adaptive semantic compression을
추가한 후속 연구 폴더입니다.

## 무엇을 했는가

- Cityscapes label map을 semantic feature proxy로 사용해 압축률과 복원 품질의 trade-off를 측정했습니다.
- `emergency`, `low`, `medium`, `paper_like`, `high` compression mode를 만들었습니다.
- SINR/링크 상태에 따라 compression mode를 고르는 adaptive policy를 AirTalking scheduler에 통합했습니다.
- fixed paper-like semantic compression과 adaptive semantic compression을 비교했습니다.
- 연구 제안서 DOCX를 생성했습니다.

## 구조

```text
code/
  run_adaptive_probe.py             SINR-only pilot probe
  run_full_adaptive_research.py     scheduler 통합 full adaptive experiment
  validate_full_adaptive_results.py 결과 검증 스크립트
  build_research_proposal.py        연구 제안서 DOCX 생성

reports/
  Adaptive_Semantic_Compression_Proposal_EASY_KR.docx
  Adaptive_Semantic_Compression_Research_Proposal_KR.docx

results/
  probe_outputs/          pilot probe 결과
  full_adaptive_results/  full adaptive experiment 결과
```

## 입력

```text
dataset/gtFine_trainvaltest/gtFine/{train,val}/
studies/airtalking_reproduction/results/airtalking_cityscapes_calibrated_final_p012/run_metadata.json
studies/airtalking_reproduction/results/airtalking_cityscapes_calibrated_final_p012/timeseries_and_sinr_samples.npz
```

## 실행 예시

```powershell
python studies\adaptive_semantic_compression\code\run_adaptive_probe.py --sample-limit 400
python studies\adaptive_semantic_compression\code\run_adaptive_probe.py --reuse-quality
python studies\adaptive_semantic_compression\code\run_full_adaptive_research.py
python studies\adaptive_semantic_compression\code\validate_full_adaptive_results.py
```

핵심 결과는 `results/full_adaptive_results/summary_metrics.csv`와
`results/full_adaptive_results/figures/`에 있습니다.
