# Replication Experiment of AirTalking

AirTalking 논문의 UAV 의미통신 실험을 공개 정보로 다시 구현하고, 직접 학습한 인코더·디코더와 채널 적응형 압축을 검증한 프로젝트입니다.

## 먼저 볼 문서

- [통합 실험 보고서](reports/README.md): 인코더·디코더 학습, 논문 재현 방법과 결과, 후속 연구를 쉬운 표현으로 정리했습니다.
- 원시 수치와 실행 조건은 각 실험의 `results/` 폴더에 JSON·CSV로 보존되어 있습니다.

## 프로젝트 구조

### 최상위

```text
.
|-- README.md          # 프로젝트 안내와 파일 구조(현재 문서)
|-- requirements.txt  # 실행에 필요한 Python 패키지 목록
|-- reports/           # 사람이 읽기 위한 통합 결과 보고서
|-- studies/           # 모델 학습, 논문 재현, 후속 연구 코드와 결과
|-- tools/             # 데이터셋 검사 보조 도구
|-- tests/             # 프로젝트 전체 회귀·무결성 테스트
|-- dataset/           # Cityscapes 원본 데이터(로컬 전용, Git 제외)
`-- base/              # AirTalking 원 논문 PDF(로컬 전용, Git 제외)
```

`base/`와 `dataset/`은 용량과 배포 조건 때문에 Git에 포함되지 않습니다. 새로 저장소를 내려받으면 사용자가 직접 준비해야 합니다.

### `reports/`: 통합 보고서

| 파일 | 역할 |
|---|---|
| `reports/README.md` | 모델 학습 결과, 논문 재현 결과, 후속 연구 결과와 한계를 쉬운 표현으로 정리한 통합 보고서 |

### `studies/neural_encoder_decoder/`: 모델 학습

#### 실행 코드

| 파일 | 역할 |
|---|---|
| `code/train_semantic_encoder_decoder.py` | 초기 기준 모델의 구조, 학습, 평가를 한 파일에 구현한 코드. 작은 실험과 파이프라인 검증용 |
| `code/train_enhanced_semantic_codec.py` | 최종 확장형 모델의 인코더, 8비트 양자화, 디코더, 다중 전송률 학습, 평가, 체크포인트 저장을 구현한 핵심 코드 |
| `tests/test_enhanced_semantic_codec.py` | 양자화, 전송 데이터 변환, 모델 입출력 등 확장형 모델의 단위 테스트 |

두 학습 파일은 인코더 코드와 디코더 코드를 따로 나눈 것이 아닙니다. 각 파일 안에 인코더와 디코더가 모두 있으며, 첫 번째는 초기 기준 구현이고 두 번째는 최종 실험 구현입니다.

#### 최종 모델 실험 결과

대표 폴더는 `results/enhanced_scalable_full_256x128_verified/`입니다.

| 파일 | 역할 |
|---|---|
| `launch_manifest.json` | 학습을 실행할 때 사용한 명령, 데이터 경로, 하이퍼파라미터와 환경 기록 |
| `result_summary.json` | 모델 구조, 학습 설정, 평가 수치, 선택된 체크포인트를 모은 전체 결과 |
| `airtalking_semantic_summary.json` | 논문 재현 시뮬레이터가 읽는 압축률과 인코딩·디코딩 시간 요약 |
| `training_history.csv` | epoch별 학습 손실과 검증 성능 |
| `rate_quality.csv` | 20·40·60·80·120 활성 채널별 전송량과 mIoU, PSNR, SSIM |
| `rate_quality_best_80ch_checkpoint.csv` | 검증 성능이 가장 좋았던 체크포인트의 채널별 평가 |
| `rate_quality_last_epoch_checkpoint.csv` | 마지막 epoch 체크포인트의 채널별 평가 |
| `per_class_iou_paper_like.csv` | 논문과 유사한 80채널 조건의 Cityscapes 클래스별 IoU |
| `confusion_matrix_paper_like.csv` | 80채널 조건의 의미분할 혼동행렬 |
| `qualitative_panel_paper_like.png` | 입력, 복원 RGB, 정답 의미지도, 예측 의미지도의 시각 비교 |
| `training_source_snapshot.py` | 해당 실험 당시 사용한 학습 코드 보존본. 새 실행은 `code/`의 원본을 사용 |

학습 완료 가중치는 원래 `best_checkpoint.pt`, `last_checkpoint.pt`, `final_checkpoint.pt`로 생성됩니다. 현재 Git 저장소에는 `.gitignore` 규칙 때문에 이 파일들이 포함되어 있지 않으므로, 결과 기록은 확인할 수 있지만 학습 완료 모델을 바로 배포하거나 추론할 수는 없습니다.

#### 기타 결과 폴더

| 폴더 | 역할 |
|---|---|
| `results/baseline_retrained_cpu_20260711_best_eval/` | CPU에서 다시 평가한 초기 기준 모델 결과 |
| `results/baseline_retrained_cpu_20260711_continued/` | 초기 기준 모델의 추가 학습 결과 |
| `results/paperlike_timed_latent20/` | 초기 논문 유사 설정과 처리시간 측정 결과 |
| `results/dataset_audit_20260711/` | Cityscapes 파일 목록, 해시와 클래스 분포 감사 결과 |
| `results/airtalking_neural_encoder_decoder_timed/` | 초기 모델 측정값을 사용한 AirTalking 시뮬레이션 결과 |

### `studies/airtalking_reproduction/`: 논문 재현

#### 실행 코드

| 파일 | 역할 |
|---|---|
| `code/airtalking_reproduction.py` | UAV 이동, 요청 발생, 무선 링크, 전송시간, 에너지와 정책 선택을 계산하는 핵심 시뮬레이터 |
| `code/calibrate_airtalking_params.py` | 시뮬레이션 가정값을 탐색하여 논문 수치와의 차이를 줄이는 보정 도구 |
| `code/measure_cityscapes_semantics.py` | Cityscapes 의미지도에서 실제 의미 데이터 크기와 압축 관련 통계를 측정 |
| `code/verify_against_paper.py` | 시뮬레이션 결과를 논문 그림에서 읽은 기준값과 비교하여 오차와 판정을 생성 |

`airtalking_reproduction.py`는 학습 코드나 `.pt` 가중치를 직접 실행하지 않습니다. 모델 학습 결과의 `airtalking_semantic_summary.json`에서 압축률과 처리시간을 읽어 통신 시뮬레이션 계산에 사용합니다.

#### 최종 재현 결과

대표 폴더는 `results/airtalking_enhanced_scalable_verified/`입니다.

| 파일·폴더 | 역할 |
|---|---|
| `launch_manifest.json` | 실행 명령과 입력 파일의 해시 등 재현 실행 기록 |
| `run_metadata.json` | 시뮬레이션 설정, 모델 측정값 출처와 적용값 |
| `summary_metrics.csv` | 전송 방식·영역 크기·정책별 평균 결과 |
| `repeat_metrics.csv` | 조건별 10회 반복의 개별 결과 |
| `statistical_summary.csv` | 반복 결과의 평균, 표준편차와 신뢰구간 |
| `verification_against_paper_enhanced_scalable_verified.csv` | 논문 기준값과 재현값의 항목별 차이 |
| `verification_against_paper_enhanced_scalable_verified.json` | 논문 비교 판정과 출처 정보를 담은 JSON 버전 |
| `simulation_source_snapshot.py` | 해당 실행 당시 시뮬레이터 코드 보존본 |
| `figures/` | 완료 요청 수, 처리시간, 이동거리, 에너지, SINR 등을 그린 PNG 그래프 |

`results/airtalking_cityscapes_calibrated_final_p012/`와 `results/airtalking_cityscapes_feature_paperhw/`는 최종 확장형 모델을 적용하기 전의 보정·중간 실험 결과입니다. `results/cityscapes_semantic_measurement/`에는 Cityscapes 의미 데이터 측정 결과가 있습니다.

### `studies/adaptive_semantic_compression/`: 후속 연구

#### 실행 코드

| 파일 | 역할 |
|---|---|
| `code/run_adaptive_probe.py` | 채널 적응형 압축 선택 규칙과 결과 생성 흐름을 작은 조건에서 확인하는 예비 실험 |
| `code/run_full_adaptive_research.py` | 동일한 시뮬레이션 조건에서 원본 전송, 고정 80채널, SINR 기반 적응형 압축을 비교하는 전체 후속 실험 |
| `code/validate_full_adaptive_results.py` | 예상한 75개 조합과 결과값의 누락·중복·비정상값을 검사하고 적응형 방식의 개선 여부를 판정 |

#### 최종 후속 연구 결과

대표 폴더는 `results/adaptive_enhanced_scalable_verified/`입니다.

| 파일·폴더 | 역할 |
|---|---|
| `run_metadata.json` | SINR 기반 채널 선택 규칙, 모델 측정값 출처와 실행 설정 |
| `summary_metrics.csv` | 3개 전송 방식 × 5개 영역 × 5개 정책의 평균 결과 75개 |
| `repeat_metrics.csv` | 각 조건의 반복 실행 원시 결과 |
| `statistical_summary.csv` | 반복 결과의 통계 요약 |
| `compression_mode_usage.csv` | 적응형 방식이 20·40·60·80·120 채널을 선택한 횟수와 비율 |
| `result_validation.json` | 결과 완전성 검사와 고정 방식 대비 적응형 방식의 조건별 성공·실패 판정 |
| `runner_source_snapshot.py` | 해당 실험 당시 전체 실행 코드 보존본 |
| `validator_source_snapshot.py` | 해당 실험 당시 검증 코드 보존본 |
| `figures/` | 채널 선택 비율, 완료 요청 수, 지연시간·품질 관계를 그린 PNG 그래프 |

`results/probe_outputs/`는 예비 실험 결과이며, `results/full_adaptive_results/`는 검증 절차를 강화하기 전의 전체 실험 결과입니다. 최종 확인에는 `adaptive_enhanced_scalable_verified/`를 사용합니다.

### `tools/`와 `tests/`: 검사 코드

| 파일 | 역할 |
|---|---|
| `tools/audit_cityscapes_dataset.py` | Cityscapes RGB·라벨 쌍, 파일 해시, 분할별 개수와 클래스 분포를 검사 |
| `tests/test_adaptive_enhancements.py` | 적응형 채널 선택과 후속 연구 결과 계산 테스트 |
| `tests/test_airtalking_reproduction_statistics.py` | 재현 시뮬레이션 반복 통계와 집계 테스트 |
| `tests/test_audit_cityscapes_dataset.py` | 데이터셋 감사 도구 테스트 |
| `tests/test_integrity_regressions.py` | 결과 파일 및 실행 기록의 무결성 회귀 테스트 |
| `tests/test_verify_against_paper_provenance.py` | 논문 비교 결과가 올바른 입력과 출처를 사용했는지 검사 |

## 대표 결과 위치

- 모델 학습·평가 결과(가중치 파일 제외): [`enhanced_scalable_full_256x128_verified`](studies/neural_encoder_decoder/results/enhanced_scalable_full_256x128_verified/)
- 논문 재현: [`airtalking_enhanced_scalable_verified`](studies/airtalking_reproduction/results/airtalking_enhanced_scalable_verified/)
- 후속 연구: [`adaptive_enhanced_scalable_verified`](studies/adaptive_semantic_compression/results/adaptive_enhanced_scalable_verified/)

## 환경과 테스트

검증 환경은 Windows, Python 3.12, PyTorch 2.12.1+CUDA 12.6, RTX 4060 Ti입니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s studies\neural_encoder_decoder\tests
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
