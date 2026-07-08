# Replication Experiment of AirTalking

AirTalking 논문의 공개 파라미터와 공개 데이터셋인 Cityscapes를 이용해 semantic UAV D2D communication 실험을 재구현한 저장소입니다.

이 저장소는 공식 저자 코드가 아닙니다. 논문 본문, Table III, 공개 데이터셋, 그리고 재현을 위해 분리 기록한 가정값을 이용한 approximate reproduction입니다.

## 데이터 출처

원본 데이터셋은 저장소에 포함하지 않습니다. Cityscapes 라이선스와 계정 정책에 따라 사용자가 직접 다운로드해야 합니다.

- Cityscapes 공식 사이트: https://www.cityscapes-dataset.com/
- Download page: https://www.cityscapes-dataset.com/downloads/

실험에 사용한 Cityscapes 패키지는 다음 2개입니다.

- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

실제로 계산에 사용한 split은 `train`과 `val`입니다. `test` split은 gtFine label이 dummy/ignore region이라 semantic profile 측정에 넣지 않았습니다.

예상 경로:

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

## 주요 파일

```text
airtalking_reproduction.py
  AirTalking UAV/device 이동, 무선 링크, semantic processing, 정책별 시뮬레이션 코드

measure_cityscapes_semantics.py
  Cityscapes RGB 이미지와 semantic label에서 payload 비율 측정

verify_against_paper.py
  논문 Figure 3/4/6의 시각 추정값과 재현 결과 비교

calibrate_airtalking_params.py
  논문에 공개되지 않은 파라미터 후보를 빠르게 비교하는 보정 helper

build_airtalking_easy_report.py
  로컬 DOCX 보고서 생성 스크립트. 생성된 DOCX 파일은 저장소에 포함하지 않음
```

## 포함한 결과

원본 데이터셋과 DOCX 보고서는 제외하고, 재현 확인에 필요한 생성 결과만 포함합니다.

```text
outputs/cityscapes_semantic_measurement/cityscapes_semantic_summary.json
  Cityscapes train/val 3,475쌍 기반 semantic profile 요약

outputs/airtalking_cityscapes_calibrated_final_p012/run_metadata.json
  논문 공개 파라미터와 보정 가정 파라미터 기록

outputs/airtalking_cityscapes_calibrated_final_p012/summary_metrics.csv
  정책/면적별 통합 실험 결과

outputs/airtalking_cityscapes_calibrated_final_p012/verification_against_paper_calibrated_final_p012.*
  논문 Figure 3/4/6 대비 비교 결과

outputs/airtalking_cityscapes_calibrated_final_p012/figures/
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

보정된 AirTalking 시뮬레이션 실행:

```bash
python airtalking_reproduction.py ^
  --out outputs/airtalking_cityscapes_calibrated_final_p012 ^
  --semantic-summary outputs/cityscapes_semantic_measurement/cityscapes_semantic_summary.json ^
  --semantic-profile-kind feature ^
  --semantic-encoder-mode paper ^
  --semantic-decoder-mode paper ^
  --assumed request_probability=0.012 ^
  --assumed workload_mean_bits=140000000 ^
  --assumed workload_std_bits=35000000 ^
  --assumed workload_min_bits=60000000 ^
  --assumed workload_max_bits=260000000 ^
  --assumed p_move=130 ^
  --assumed p_hover=115 ^
  --assumed density_interference_scale=12 ^
  --assumed linucb_candidate_samples=10 ^
  --assumed sa_iterations=2 ^
  --assumed random_semantic_encode_probability=0.25 ^
  --assumed random_semantic_decode_probability=0.5
```

논문 그래프와 비교:

```bash
python verify_against_paper.py --summary outputs/airtalking_cityscapes_calibrated_final_p012/summary_metrics.csv --label _calibrated_final_p012
```

## 현재 결과 요약

- Cityscapes train/val 3,475쌍 사용
- 측정한 semantic feature compression ratio: `rho_c = 0.104464`
- 논문 Table III의 `rho_c = 0.104`와 거의 동일
- semantic processing은 300m x 300m 환경에서 non-semantic baseline보다 finished requests를 증가시킴
- 보정 전 검증: `match 6`, `partial 14`, `mismatch 53`
- 보정 후 검증: `match 18`, `partial 15`, `mismatch 40`

정량 차이는 여전히 남아 있습니다. 주요 원인은 논문에 공개되지 않은 request generation, workload distribution, propulsion/hover power, interference scheduling, policy hyperparameter입니다. 이 값들은 `run_metadata.json`의 `assumed_params`에 분리 기록했습니다.
