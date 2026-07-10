# AirTalking Reproduction Study

AirTalking 논문의 UAV D2D semantic communication 실험을 공개 정보와 Cityscapes 기반 측정값으로 재현한 폴더입니다.

## 무엇을 했나

- 논문에 공개된 통신/계산 파라미터를 바탕으로 UAV, device, task offloading, semantic/nonsemantic 전송을 Python으로 구현했습니다.
- Cityscapes train/val 이미지와 gtFine 라벨로 semantic payload profile을 측정했습니다.
- 논문 Figure 3/4/6에서 읽은 대표값과 재현 결과를 `match`, `partial`, `mismatch`로 비교했습니다.
- 후속 실험에서 학습한 neural encoder/decoder 결과를 재현 실험에도 넣을 수 있게 했습니다.

## Encoder/Decoder 반영

기존 재현은 논문 표와 Cityscapes 측정값을 바탕으로 semantic 전송의 크기와 시간을 넣었습니다.
이제는 `neural_encoder_decoder`에서 직접 학습한 결과도 읽을 수 있습니다.

반영되는 값은 다음입니다.

- 압축률: `rho_c`
- 품질: best mIoU
- 시간: encode/decode median time

실행 예시:

```powershell
python studies\airtalking_reproduction\code\airtalking_reproduction.py --semantic-summary studies\neural_encoder_decoder\results\paperlike_timed_latent20\airtalking_semantic_summary.json
```

## 구조

```text
code/
  airtalking_reproduction.py       메인 UAV D2D 시뮬레이터
  measure_cityscapes_semantics.py  Cityscapes semantic payload 측정
  verify_against_paper.py          논문 그래프 값과 재현 결과 비교
  calibrate_airtalking_params.py   공개되지 않은 일부 시뮬레이터 상수 보정 도구

reports/
  AirTalking_Reproduction_Easy_Final_KR.docx

results/
  cityscapes_semantic_measurement/
  airtalking_cityscapes_feature_paperhw/
  airtalking_cityscapes_calibrated_final_p012/
```

## 현재 결론

정량값은 논문과 완전히 같지는 않습니다. 이유는 논문의 원본 코드, 원본 raw plotting data, 일부 시뮬레이터 세부값이 공개되어 있지 않기 때문입니다.
그래도 공개 데이터와 공개 파라미터만으로 semantic 전송이 nonsemantic 전송보다 완료 task 수와 평균 시간에서 유리해지는 경향은 재현했습니다.
