# AirTalking Reproduction Study

AirTalking 논문의 UAV D2D semantic communication 실험을 공개 정보와
Cityscapes 데이터로 근사 재현한 폴더입니다.

## 무엇을 했는가

- 논문 Table III의 공개 파라미터를 기반으로 UAV/device 이동, 링크 품질, task offloading, semantic/nonsemantic 전송을 Python으로 구현했습니다.
- Cityscapes train/val 이미지와 gtFine label을 이용해 semantic payload profile을 측정했습니다.
- 논문 Figure 3/4/6의 시각 추정값과 재현 결과를 `match/partial/mismatch`로 비교했습니다.
- 재현 실험 보고서와 쉬운 설명용 DOCX를 생성했습니다.

## 구조

```text
code/
  airtalking_reproduction.py          메인 UAV D2D 시뮬레이터
  measure_cityscapes_semantics.py     Cityscapes semantic payload 측정
  verify_against_paper.py             논문 그래프 값과 재현 결과 비교
  calibrate_airtalking_params.py      미공개 파라미터 보정 helper
  build_airtalking_report.py          정식 재현 보고서 생성
  build_airtalking_easy_report.py     쉬운 재현 보고서 생성
  build_plain_language_docs.py        쉬운 제안서/구현서 생성
  legacy/measure_camvid_semantics.py  현재 실험에서는 쓰지 않는 CamVid legacy 코드

reports/
  AirTalking_Cityscapes_Reproduction_Report_KR.docx
  AirTalking_Cityscapes_Reproduction_Report_KR_revised.docx
  AirTalking_easy_reproduction_report_KR.docx
  AirTalking_Experiment_Implementation_EASY_KR.docx

results/
  cityscapes_semantic_measurement/
  airtalking_cityscapes_feature_paperhw/
  airtalking_cityscapes_calibrated_final_p012/
```

## 실행 예시

```powershell
python studies\airtalking_reproduction\code\measure_cityscapes_semantics.py
python studies\airtalking_reproduction\code\airtalking_reproduction.py --out studies\airtalking_reproduction\results\airtalking_reproduction
python studies\airtalking_reproduction\code\verify_against_paper.py --summary studies\airtalking_reproduction\results\airtalking_cityscapes_calibrated_final_p012\summary_metrics.csv --label _calibrated_final_p012
```

현재 최종 재현 결과는 `results/airtalking_cityscapes_calibrated_final_p012/`를 기준으로 보면 됩니다.
