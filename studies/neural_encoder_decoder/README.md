# Neural Semantic Encoder/Decoder Study

Cityscapes RGB를 작은 전송 표현으로 바꾸고, 수신 측에서 RGB와 19-class 의미 분할을 복원하는 모델을 학습하는 폴더입니다. AirTalking 논문은 정확한 neural network·학습법·전송 tensor를 공개하지 않았으므로, 이 모델은 논문 저자 codec의 정확한 복제가 아니라 공개 개요에서 출발한 독립 후속 모델입니다.

## 두 모델을 구분해서 보기

- **기존 기준선** `train_semantic_encoder_decoder.py`: `RGB → 실수 latent → segmentation` 모델입니다. 20채널/stride 8을 8-bit라고 계산상 가정해 \(\rho_c=0.10417\)을 만들었지만, 실제 양자화·byte 전송·RGB 복원은 없습니다.
- **강화 모델** `train_enhanced_semantic_codec.py`: `RGB → 8-bit latent → RGB+segmentation` 모델입니다. 수신기는 전송된 latent만 사용하며 encoder skip tensor를 받지 않습니다. 한 모델의 20/40/60/80/120채널 prefix가 다섯 전송률을 만듭니다.

기존 `paperlike_timed_latent20`의 내부 val 결과는 mIoU 0.2135, pixel accuracy 0.7408입니다. 이는 128×64, train 512장/val 256장, segmentation-only 기준선의 값이며 AirTalking의 visual decoder 품질로 해석하면 안 됩니다.

같은 계열 checkpoint를 CPU에서 다시 평가한 `baseline_retrained_cpu_20260711_best_eval`은 mIoU 0.223736입니다. 04 비교 보고서는 작업 전 원래 저장 결과 0.213473을, 01 상세 보고서는 provenance가 보강된 CPU 재평가 0.223736을 기준선으로 사용하므로 두 수치를 같은 run으로 보면 안 됩니다.

## 현재 verified 결과

대표 결과는 [`results/enhanced_scalable_full_256x128_verified`](results/enhanced_scalable_full_256x128_verified)입니다. Cityscapes train 2,975장과 val 500장, 256×128 입력으로 20 epoch 학습했고, epoch 19의 80채널 best checkpoint가 최종 선택됐습니다.

- 80채널 mIoU `0.305416`, pixel accuracy `0.828616`
- RGB PSNR `18.435 dB`, 프로젝트 로컬 SSIM `0.567506`
- full forward median `4.998 ms`(RTX 4060 Ti, batch 1, CUDA 동기화, 30회)
- 실제 uint8 5-rate payload와 byte round trip 검증 완료

데이터·해상도·decoder 목표·loss가 기존 기준선과 함께 바뀌었으므로 정확도 차이를 구조 하나나 GPU 하나의 효과로 해석하면 안 됩니다. 상세 내용은 [`reports/final/01_인코더_디코더_딥러닝_과정_final.md`](../../reports/final/01_인코더_디코더_딥러닝_과정_final.md)에서 확인할 수 있습니다.

## 데이터와 감사 범위

입력은 Cityscapes `leftImg8bit` RGB, 정답은 `gtFine/*_gtFine_labelIds.png`입니다. train 2,975장과 val 500장을 학습·내부 평가에 사용할 수 있고, 공개 test의 동봉 labelIds는 19-class 유효 픽셀이 없어 로컬 mIoU에 사용하지 않습니다.

```powershell
.\.venv\Scripts\python.exe tools\audit_cityscapes_dataset.py `
  --out studies\neural_encoder_decoder\results\dataset_audit_20260711
```

감사는 발견된 RGB·color·instanceIds·labelIds PNG 전부의 헤더상 크기와 mode, 모든 polygons JSON의 파싱·기본 schema, 모든 labelIds 픽셀을 검사합니다. 재현 지문은 전체 inventory의 상대 경로·크기와 **모든 RGB 및 labelIds 파일의 content SHA-256**을 포함합니다. 모든 PNG 파일의 전체 content를 hash했다는 뜻은 아닙니다.

## 강화 모델의 학습·평가 계약

강화 모델은 다음을 함께 학습합니다.

- class-balanced cross entropy와 Dice로 19-class segmentation
- RGB L1과 프로젝트 로컬 SSIM loss로 RGB 복원
- 학습 순전파에 포함된 `[0,1] ↔ uint8` STE 양자화
- 매 batch의 최소·최대 rate와 seeded 중간 rate 하나를 쓰는 sandwich training

평가에서는 실제 contiguous uint8 latent를 직렬화하고 zlib 압축·해제 뒤 byte 동일성을 확인합니다. `rho_uint8`은 실제 latent byte/raw uint8 RGB byte이며 논문 식 (22)과 일차 비교할 값입니다. `rho_zlib`은 내용 의존적인 별도 후속 지표입니다.

SSIM은 7×7 uniform window, zero padding, `C1=0.01²`, `C2=0.03²`를 쓰는 **프로젝트 로컬 differentiable proxy**입니다. Gaussian-window `torchmetrics`/`skimage` SSIM과 같은 구현이라고 가정하면 안 됩니다. GPU timing은 neural encode/decode/full forward만 재며 CPU 전송, 직렬화, zlib 압축·해제 시간은 포함하거나 측정한 것으로 보고하지 않습니다.

학습 중 `best_checkpoint.pt`는 80채널 val mIoU가 가장 좋았던 가중치를 보존합니다. 종료 후 그 가중치와 마지막 epoch 가중치를 같은 val의 다섯 rate에서 모두 평가하고, **5-rate mIoU 평균**이 큰 후보를 최종 결과로 고릅니다. 평균이 같으면 최저-rate mIoU, 다시 같으면 80채널 mIoU로 결정합니다. 따라서 최종 `rate_quality.csv`는 단순히 마지막 epoch나 80채널 best를 무조건 택한 표가 아닙니다.

## 실행

기존 기준선:

```powershell
.\.venv\Scripts\python.exe studies\neural_encoder_decoder\code\train_semantic_encoder_decoder.py `
  --device cpu --model paperlite --epochs 30 `
  --train-limit 512 --val-limit 256 --image-width 128 --image-height 64 `
  --batch-size 8 --width 8 --latent-channels 20 `
  --class-balanced-loss --save-checkpoint `
  --out studies\neural_encoder_decoder\results\baseline_retrained
```

강화 5-rate 모델:

```powershell
.\.venv\Scripts\python.exe studies\neural_encoder_decoder\code\train_enhanced_semantic_codec.py `
  --device cuda --epochs 20 --full-data `
  --image-width 256 --image-height 128 --base-width 16 `
  --batch-size 4 --gradient-accumulation 4 --num-workers 4 `
  --out studies\neural_encoder_decoder\results\enhanced_scalable_full_256x128_verified
```

완전한 새 형식 checkpoint에서 이어갈 때는 같은 핵심 설정과 `--resume ...\last_checkpoint.pt`를 사용합니다. 예전 checkpoint에 dataset/source provenance가 없으면 기본 resume가 실패하며, `--allow-legacy-resume-provenance`는 그 결손을 이해하고 명시적으로 허용할 때만 사용합니다.

```powershell
.\.venv\Scripts\python.exe studies\neural_encoder_decoder\code\train_enhanced_semantic_codec.py --smoke --device cpu
.\.venv\Scripts\python.exe -m unittest discover -s studies\neural_encoder_decoder\tests -v
```

smoke 수치는 배관 검사용 비수렴 값이므로 성능 결과로 인용하지 않습니다.

## 산출물과 provenance

정상 완료 run은 `result_summary.json`, `airtalking_semantic_summary.json`, 세 rate-quality CSV(최종 선택/best-80/last), 학습 이력, per-class IoU, confusion matrix, 정성 panel과 checkpoint를 남깁니다. 실행 시작 시 `training_source_snapshot.py`, 그 SHA-256, `launch_manifest.json`, dataset fingerprint도 기록합니다.

`*checkpoint.pt`는 크기가 커 `.gitignore` 대상입니다. git에 없다는 사실이 곧 로컬 실험 증거가 없다는 뜻은 아니지만, strict 최종 보고서는 metadata가 가리키는 best/last/final checkpoint 3종이 모두 **실제 로컬 파일**이어야 통과합니다. 다른 컴퓨터에서 재현하려면 checkpoint와 hash를 별도 보존·전달해야 합니다.
