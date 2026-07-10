# Neural Semantic Encoder/Decoder Study

AirTalking 논문에서 핵심으로 쓰인 semantic encoder/decoder를 실제 학습 가능한
작은 neural network 형태로 구현한 후속 연구 폴더입니다.

## 무엇을 했는가

- Cityscapes RGB 이미지를 입력으로 받는 convolutional semantic encoder를 구현했습니다.
- encoder가 만든 compressed latent feature를 decoder가 semantic segmentation map으로 복원하도록 학습했습니다.
- latent 크기를 조정해 논문 Table III의 semantic compression ratio `rho_c=0.104`에 가깝게 맞췄습니다.
- 학습된 encoder/decoder의 encode/decode 시간을 측정하고 AirTalking timed simulation에 넣어 비교했습니다.
- 후속 연구 최종 보고서 DOCX를 생성했습니다.

## 구조

```text
code/
  train_semantic_encoder_decoder.py  encoder/decoder 학습 및 평가 코드

reports/
  AirTalking_Followup_Final_Report_KR.docx

results/
  paperlike_timed_latent20/              encoder/decoder 학습 결과
  airtalking_neural_encoder_decoder_timed/ AirTalking timed simulation 결과
```

## 실행 예시

```powershell
python studies\neural_encoder_decoder\code\train_semantic_encoder_decoder.py --out studies\neural_encoder_decoder\results\paperlike_timed_latent20 --model paperlite --epochs 30 --train-limit 512 --val-limit 256 --image-width 128 --image-height 64 --batch-size 8 --width 8 --latent-channels 20 --class-balanced-loss --timing-runs 10 --save-checkpoint
```

## 대표 결과

```text
rho_c: 0.10417
pixel accuracy: 0.7408
mIoU: 0.2135
encode/decode/full median time: 1.75 / 2.53 / 4.48 ms
```
