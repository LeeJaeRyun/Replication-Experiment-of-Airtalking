# Replication Experiment of AirTalking

AirTalking 논문의 공개 파라미터와 공개 데이터셋인 Cityscapes를 이용해 semantic UAV D2D communication 실험을 재구현한 저장소입니다.

## 재현 범위

이 저장소는 공식 저자 코드가 아니라, 논문 본문과 Table III에 공개된 값들을 바탕으로 작성한 Python 기반 재구현입니다.

- 사용 데이터셋: Cityscapes train/val
- 사용 목적: semantic compression profile 측정 및 AirTalking 시스템 시뮬레이션 입력
- 공개 파라미터: UAV/device 수, channel setting, `rho_c`, `rho_r`, encoder/decoder bitrate 등
- 미공개 파라미터: request probability, workload distribution, propulsion/hover power, policy hyperparameter 등은 `run_metadata.json`의 `assumed_params`에 명시

따라서 결과는 공식 코드 기반 exact reproduction이 아니라, 공개 정보 기반 approximate reproduction입니다.

## 데이터 출처

원본 데이터는 저장소에 포함하지 않습니다. Cityscapes 라이선스와 계정 정책에 따라 사용자가 직접 다운로드해야 합니다.

- Cityscapes 공식 사이트: https://www.cityscapes-dataset.com/
- Download page: https://www.cityscapes-dataset.com/downloads/

실험에 사용한 패키지:

- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

압축 해제 후 기본 경로는 다음과 같이 둡니다.

```text
dataset/
  leftImg8bit_trainvaltest/
    leftImg8bit/
      train/
      val/
      test/
  gtFine_trainvaltest/
    gtFine/
      train/
      val/
      test/
```

Cityscapes test split의 gtFine label은 dummy/ignore region이므로 semantic profile 측정에는 train/val만 사용합니다.

## 주요 파일

```text
airtalking_reproduction.py
  AirTalking 시스템 모델과 정책별 시뮬레이션 코드

measure_cityscapes_semantics.py
  Cityscapes 이미지/라벨에서 semantic payload profile 측정

verify_against_paper.py
  논문 Figure 3/4/6 시각 추정값과 재현 결과 비교

measure_camvid_semantics.py
  이전 CamVid 측정용 보조 스크립트

build_airtalking_report.py
  로컬 DOCX 보고서 생성용 스크립트. 보고서 파일 자체는 저장소에 포함하지 않음
```

## 포함된 결과

원본 데이터와 보고서는 제외하고, 재현 확인에 필요한 생성 결과만 포함합니다.

```text
outputs/cityscapes_semantic_measurement/cityscapes_semantic_summary.json
  Cityscapes 기반 semantic profile 요약

outputs/airtalking_cityscapes_feature_paperhw/run_metadata.json
  공개 파라미터와 가정 파라미터 기록

outputs/airtalking_cityscapes_feature_paperhw/summary_metrics.csv
  정책/면적별 핵심 실험 결과

outputs/airtalking_cityscapes_feature_paperhw/verification_against_paper_cityscapes_feature_paperhw.*
  논문 Figure 3/4/6 대비 비교 결과

outputs/airtalking_cityscapes_feature_paperhw/figures/
  생성된 결과 그래프
```

## 실행 방법

필요 패키지 예시:

```bash
pip install numpy pillow matplotlib python-docx
```

Cityscapes semantic profile 측정:

```bash
python measure_cityscapes_semantics.py --root dataset --out outputs/cityscapes_semantic_measurement --splits train,val --feature-scale 0.56 --repeats 1
```

AirTalking 시뮬레이션 실행:

```bash
python airtalking_reproduction.py --out outputs/airtalking_cityscapes_feature_paperhw --semantic-summary outputs/cityscapes_semantic_measurement/cityscapes_semantic_summary.json --semantic-profile-kind feature --semantic-encoder-mode paper --semantic-decoder-mode paper
```

논문 그래프와 비교:

```bash
python verify_against_paper.py --summary outputs/airtalking_cityscapes_feature_paperhw/summary_metrics.csv --label _cityscapes_feature_paperhw
```

## 현재 결과 요약

- Cityscapes train/val 3,475장 사용
- 측정된 semantic feature compression ratio: `rho_c = 0.104464`
- 논문 Table III의 `rho_c = 0.104`와 거의 동일
- semantic processing은 300m x 300m 환경에서 non-semantic baseline보다 finished requests를 증가시킴
- 논문 Figure 3/4/6과의 정량 비교는 `match 6`, `partial 14`, `mismatch 53`

정량 차이의 주요 원인은 논문에 공개되지 않은 workload, request generation, energy power, interference scheduling, policy hyperparameter입니다.
