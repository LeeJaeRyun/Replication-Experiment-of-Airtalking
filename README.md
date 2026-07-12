# Replication Experiment of AirTalking

AirTalking 논문의 UAV 의미통신 실험을 공개 정보로 재현하고, Cityscapes로 직접 학습한 neural semantic codec과 채널 적응형 압축을 후속 연구로 검증하는 저장소입니다.

## 과학적 범위

논문은 `RGB → modified U-Net semantic representation → 압축·전송 → modified Pix2PixHD visual reconstruction`이라는 개요와 장치 처리율만 공개합니다. 전송 tensor, 양자화, 모델 수정 내역, loss, optimizer, 학습 epoch·seed·입력 크기는 공개하지 않았습니다. 따라서 이 저장소는 다음을 명확히 구분합니다.

- **논문 시스템 재현**: 공개 수식·Table III 상수와 명시한 가정으로 UAV 시뮬레이터를 재구현합니다.
- **기존 기준선**: `RGB → latent → semantic segmentation` 소형 모델입니다. Pix2PixHD식 RGB 복원은 없고 8-bit payload도 계산상 가정이었습니다.
- **강화 후속 모델**: 실제 8-bit STE 양자화와 byte 왕복을 사용하고, 수신기가 latent만 받아 RGB 복원과 19-class segmentation을 함께 수행하는 독립 설계입니다. 논문 저자의 비공개 모델을 그대로 재현했다는 주장은 하지 않습니다.

## 현재 검증 결과

현재 대표 산출물은 `enhanced_scalable_full_256x128_verified` neural codec, `airtalking_enhanced_scalable_verified` 재현 run, `adaptive_enhanced_scalable_verified` 적응형 run입니다.

| 구분 | 현재 결과 | 해석 |
|---|---:|---|
| 강화 codec 80채널 mIoU / pixel accuracy | 0.305416 / 0.828616 | 작업 전 결과 0.213473 / 0.740795보다 높지만 데이터·해상도·모델·학습법이 함께 바뀐 시스템 수준 비교 |
| RGB 복원 PSNR / SSIM | 18.435 dB / 0.567506 | SSIM은 프로젝트 로컬 7×7 uniform-window proxy |
| 논문 verifier match / partial / mismatch | 17 / 17 / 39 | 일부 방향성만 재현했으며 정량 재현에는 실패 |
| Adaptive가 fixed-80보다 완료 수 증가·시간 감소 | 23 / 25 조합 | Stochastic 400 m·500 m는 실패; 보편적 우월성이나 통계적 유의성 주장이 아님 |

기준선은 두 개를 구분해야 합니다. 04 비교 보고서는 작업 전 원래 저장 결과 mIoU `0.213473`을 사용하고, 01 상세 보고서의 자동 비교표는 provenance를 보강해 CPU에서 다시 평가한 `baseline_retrained_cpu_20260711_best_eval`의 `0.223736`을 사용합니다.

제출·검토용 최신 보고서는 [`reports/final`](reports/final)에 있으며, 이전 컴퓨터와 현재 결과의 차이는 [`04_이전_현재_컴퓨터_실험_결과_비교_final.md`](reports/final/04_이전_현재_컴퓨터_실험_결과_비교_final.md)에 정리되어 있습니다.

## 구조

```text
base/                                  AirTalking 논문 PDF(로컬, git 제외)
dataset/                               Cityscapes 원본(로컬, git 제외)
reports/                               상세 한국어 보고서 Markdown 템플릿
reports/final/                         최종 수치가 채워진 Markdown/DOCX
studies/airtalking_reproduction/       논문 시스템 재현
studies/adaptive_semantic_compression/ 5-rate 적응형 후속 연구
studies/neural_encoder_decoder/        기준선·강화 codec 학습
tools/                                 데이터 감사·보고서 생성 도구
tests/                                 통합·회귀 테스트
```

## 환경 설치

검증 환경은 Windows, Python 3.12, PyTorch 2.12.1+CUDA 12.6, RTX 4060 Ti 8GB입니다.

저장소 루트에 Python 3.12 가상환경 `.venv`를 먼저 만든 뒤, 모든 프로젝트 명령은 그 interpreter를 명시적으로 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 데이터셋

공개 데이터는 Cityscapes의 다음 패키지입니다.

- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

```text
dataset/
  leftImg8bit_trainvaltest/leftImg8bit/{train,val,test}/
  gtFine_trainvaltest/gtFine/{train,val,test}/
```

train 2,975장과 val 500장에는 의미 라벨이 있습니다. 공개 test 1,525장의 gtFine은 평가용 19-class 정답이 아니므로 test mIoU를 계산하지 않습니다.

전수 무결성·class histogram·재현 지문 생성:

```powershell
.\.venv\Scripts\python.exe tools\audit_cityscapes_dataset.py `
  --out studies\neural_encoder_decoder\results\dataset_audit_20260711
```

감사 범위는 발견된 RGB·color·instanceIds·labelIds PNG 전부의 헤더상 크기/mode, 모든 polygons JSON의 파싱·기본 schema, 모든 labelIds 픽셀입니다. fingerprint는 전체 inventory의 상대 경로·크기와 **모든 RGB 및 labelIds 파일의 content SHA-256**을 포함합니다. 모든 PNG 파일의 전체 content를 hash했다는 뜻은 아닙니다.

## 1. 인코더·디코더 학습

### 검증 가능한 기존 기준선

```powershell
.\.venv\Scripts\python.exe studies\neural_encoder_decoder\code\train_semantic_encoder_decoder.py `
  --device cpu --model paperlite --epochs 30 `
  --train-limit 512 --val-limit 256 --image-width 128 --image-height 64 `
  --batch-size 8 --width 8 --latent-channels 20 `
  --class-balanced-loss --save-checkpoint `
  --out studies\neural_encoder_decoder\results\baseline_retrained
```

### 강화 scalable semantic codec

20/40/60/80/120 활성 채널을 하나의 모델에서 학습합니다. stride 16, 80채널, 8-bit일 때 raw RGB 대비 이론 payload 비율은 `80/(3×16²)=0.1041667`입니다.

```powershell
.\.venv\Scripts\python.exe studies\neural_encoder_decoder\code\train_enhanced_semantic_codec.py `
  --device cuda --epochs 20 --full-data `
  --image-width 256 --image-height 128 --base-width 16 `
  --batch-size 4 --gradient-accumulation 4 --num-workers 4 `
  --out studies\neural_encoder_decoder\results\enhanced_scalable_full_256x128_verified
```

중단 시 `last_checkpoint.pt`에서 optimizer·scheduler·AMP scaler·RNG까지 이어집니다. 핵심 설정이 다르면 resume는 실패하도록 되어 있습니다.

학습 중 80채널 val mIoU가 가장 좋은 checkpoint와 마지막 epoch checkpoint를 종료 시 같은 다섯 rate에서 비교하고, **5-rate mIoU 평균**이 큰 후보를 최종 `rate_quality.csv`에 사용합니다. 동률이면 최저-rate mIoU, 다시 동률이면 80채널 mIoU로 결정합니다.

대형 `*checkpoint.pt`는 `.gitignore` 대상입니다. git에 없더라도 로컬 실험 artifact로는 유효하지만 strict 최종 보고서는 metadata가 가리키는 `best_checkpoint.pt`, `last_checkpoint.pt`, `final_checkpoint.pt`가 모두 실제 로컬 파일이어야 합니다. 다른 장비에서의 감사·재현에는 checkpoint와 hash를 별도 보존해야 합니다.

## 2. AirTalking 시스템 재현

강화 summary는 GPU에서 실제 측정한 **neural feature codec** 처리율을 제공합니다. PNG 분모나 CPU 직렬화·zlib 압축/해제 시간은 측정하지 않았으므로 PNG ratio나 end-to-end zlib 처리율을 만들어내지 않습니다. RGB SSIM은 7×7 uniform window와 zero padding을 쓰는 프로젝트 로컬 differentiable proxy로, `torchmetrics`/`skimage`의 Gaussian-window SSIM과 동일하지 않습니다.

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

## 3. 적응형 의미압축 후속 연구

강화 summary의 5개 neural rate를 payload 순서대로 `emergency/low/medium/paper_like/high`에 연결합니다. proxy의 `0.8` 품질 목표를 그대로 쓰지 않고, 실측 5-rate 품질 순서에 맞춘 `measured_ordered` rule과 실제 bin별 목표를 metadata에 기록합니다. 개별 반복과 Student-t 95% 신뢰구간을 함께 저장합니다.

```powershell
.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\run_full_adaptive_research.py `
  --metadata studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified\run_metadata.json `
  --quality studies\adaptive_semantic_compression\results\probe_outputs\compression_quality.csv `
  --neural-summary studies\neural_encoder_decoder\results\enhanced_scalable_full_256x128_verified\airtalking_semantic_summary.json `
  --neural-quality-mode selection `
  --adaptive-threshold-rule measured_ordered `
  --workers 6 `
  --out studies\adaptive_semantic_compression\results\adaptive_enhanced_scalable_verified

.\.venv\Scripts\python.exe studies\adaptive_semantic_compression\code\validate_full_adaptive_results.py `
  --summary studies\adaptive_semantic_compression\results\adaptive_enhanced_scalable_verified\summary_metrics.csv `
  --out studies\adaptive_semantic_compression\results\adaptive_enhanced_scalable_verified\result_validation.json
```

## 상세 보고서 생성

원본 템플릿은 `reports/*.md`에 보존되고, 최종 수치가 채워진 Markdown과 DOCX는 `reports/final/`에 생성됩니다. 기본 strict 모드는 필요한 산출물이나 provenance가 빠지면 실패합니다. 구체적으로 다음을 교차검증합니다.

- enhanced: best/last/final checkpoint 3종, launch manifest, 학습 source snapshot과 SHA-256의 실제 로컬 파일
- reproduction: summary/repeat/statistics와 단 하나의 verifier CSV, 같은 stem의 `status=completed` JSON(source summary path/hash, row/verdict counts)
- adaptive: summary/repeat/statistics/mode-usage와 schema v2 validator JSON(source summary path/hash, `passed`, 75개 expected combinations)
- reproduction/adaptive 공통: 원시 repeat에서 mean·sample 표준편차·양측 Student-t 95% CI를 독립 재계산한 값과 저장 통계의 일치

provenance의 상대경로는 결과 디렉터리→저장소 루트→현재 작업 디렉터리 순으로 해석하며, 빈 경로나 디렉터리만 있는 값은 파일 증거로 인정하지 않습니다. `--allow-incomplete`는 배관 확인용이며 완성 보고서의 증거를 대신하지 않습니다.

```powershell
.\.venv\Scripts\python.exe tools\finalize_research_reports.py `
  --enhanced-result-dir studies\neural_encoder_decoder\results\enhanced_scalable_full_256x128_verified `
  --reproduction-result-dir studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified `
  --adaptive-result-dir studies\adaptive_semantic_compression\results\adaptive_enhanced_scalable_verified `
  --baseline-dir neural=studies\neural_encoder_decoder\results\baseline_retrained_cpu_20260711_best_eval `
  --baseline-dir reproduction=studies\airtalking_reproduction\results\airtalking_cityscapes_calibrated_final_p012 `
  --baseline-dir adaptive=studies\adaptive_semantic_compression\results\full_adaptive_results
```

최종 보고서는 다음 네 가지입니다. 1~3번은 strict finalizer가 실험 산출물에서 생성하고, 4번은 작업 이전의 상세 하드웨어가 기록되지 않은 CPU 결과 JSON과 현재 verified JSON을 같은 Markdown→DOCX 변환기로 비교한 보조 보고서입니다. `reports/*.md`는 원본 템플릿이고 `reports/final/*_final.md`와 `.docx`가 최신 수치가 반영된 최종본입니다.

1. [인코더·디코더 딥러닝 과정 상세 보고서](reports/final/01_인코더_디코더_딥러닝_과정_final.md)
2. [AirTalking 논문 실험 재현 보고서](reports/final/02_AirTalking_논문_실험_재현_final.md)
3. [적응형 의미압축 후속 연구 실험 보고서](reports/final/03_후속연구_적응형_의미압축_final.md)
4. [이전 컴퓨터와 현재 컴퓨터의 실험 결과 비교 보고서](reports/final/04_이전_현재_컴퓨터_실험_결과_비교_final.md)

`reports/final/finalization_manifest.json`은 1~3번, `reports/final/04_이전_현재_컴퓨터_실험_결과_비교_manifest.json`은 4번의 입력·출력 SHA-256과 생성 정보를 기록합니다. `reports/final`이 정본이며, `studies/*/reports/*Easy_Final_KR.docx`는 호환 경로를 위해 최신 final Markdown에서 다시 만드는 복사본입니다.

```powershell
.\.venv\Scripts\python.exe tools\build_easy_study_reports.py
```

## 검증

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s studies\neural_encoder_decoder\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

결과를 해석할 때는 mIoU·PSNR·SSIM뿐 아니라 데이터 split, 실제 payload byte 수, checkpoint 선택, 반복별 분산, 논문 비공개 항목을 함께 확인해야 합니다.
