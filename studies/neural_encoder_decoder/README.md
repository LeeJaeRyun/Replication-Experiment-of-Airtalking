# Neural Semantic Encoder/Decoder Study

AirTalking 논문에서 핵심으로 쓰인 semantic encoder/decoder를 직접 학습 가능한 작은 neural network 형태로 구현한 폴더입니다.

## 무엇을 했나

- 입력: Cityscapes RGB 이미지
- encoder 출력: 작은 latent semantic feature
- decoder 출력: semantic segmentation map
- 목표: 논문 Table III의 semantic compression ratio `rho_c = 0.104`에 가깝게 맞추기

쉽게 말하면, 이미지를 그대로 보내는 대신 작은 feature로 줄여서 보내고, 받은 쪽에서 semantic map으로 복원하는 실험입니다.

## 주요 결과

```text
rho_c: 0.10417
pixel accuracy: 0.7408
mIoU: 0.2135
encode/decode/full median time: 1.75 / 2.53 / 4.48 ms
```

이 결과는 두 곳에 다시 사용됩니다.

- `airtalking_reproduction`: semantic 전송 profile로 반영
- `adaptive_semantic_compression`: `paper_like` 압축 단계의 기준값으로 반영

## 구조

```text
code/
  train_semantic_encoder_decoder.py  encoder/decoder 학습, 평가, 시간 측정 코드

reports/
  Neural_Encoder_Decoder_Easy_Final_KR.docx

results/
  paperlike_timed_latent20/                encoder/decoder 학습 결과
  airtalking_neural_encoder_decoder_timed/ AirTalking timed simulation 결과
```

## 실행 예시

```powershell
python studies\neural_encoder_decoder\code\train_semantic_encoder_decoder.py --out studies\neural_encoder_decoder\results\paperlike_timed_latent20 --model paperlite --epochs 30 --train-limit 512 --val-limit 256 --image-width 128 --image-height 64 --batch-size 8 --width 8 --latent-channels 20 --class-balanced-loss --timing-runs 10 --save-checkpoint
```

## 한계

논문 원본 encoder/decoder 코드와 weight는 공개되어 있지 않습니다.
그래서 여기서는 논문과 같은 구조를 복사한 것이 아니라, 공개 데이터로 재학습 가능한 작은 대체 모델을 만든 것입니다.
