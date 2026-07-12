# Replication Experiment of AirTalking

AirTalking 논문의 UAV 의미통신 실험을 공개 정보로 다시 구현하고, 직접 학습한 인코더·디코더와 채널 적응형 압축을 검증한 프로젝트입니다.

## 먼저 볼 문서

- [통합 실험 보고서](reports/README.md): 인코더·디코더 학습, 논문 재현 방법과 결과, 후속 연구를 쉬운 표현으로 정리했습니다.
- 원시 수치와 실행 조건은 각 실험의 `results/` 폴더에 JSON·CSV로 보존되어 있습니다.

## 프로젝트 구조

| 경로 | 내용 |
|---|---|
| `base/` | 원 논문 PDF가 있는 로컬 폴더 |
| `dataset/` | Cityscapes 원본 데이터가 있는 로컬 폴더 |
| `reports/README.md` | 재현 및 후속 연구 통합 보고서 |
| `studies/neural_encoder_decoder/` | Cityscapes 인코더·디코더 학습 코드와 결과 |
| `studies/airtalking_reproduction/` | 논문 기반 UAV 시뮬레이터와 비교 결과 |
| `studies/adaptive_semantic_compression/` | 채널 적응형 압축 후속 연구 코드와 결과 |
| `tools/` | 데이터셋 감사 등 보조 도구 |
| `tests/` | 시뮬레이터·결과 무결성 테스트 |
| `requirements.txt` | Python 패키지 목록 |

`base/`와 `dataset/`은 용량과 배포 조건 때문에 Git에 포함되지 않는 로컬 폴더입니다.

## 대표 결과 위치

- 학습 모델: [`enhanced_scalable_full_256x128_verified`](studies/neural_encoder_decoder/results/enhanced_scalable_full_256x128_verified/)
- 논문 재현: [`airtalking_enhanced_scalable_verified`](studies/airtalking_reproduction/results/airtalking_enhanced_scalable_verified/)
- 후속 연구: [`adaptive_enhanced_scalable_verified`](studies/adaptive_semantic_compression/results/adaptive_enhanced_scalable_verified/)

## 환경과 테스트

검증 환경은 Windows, Python 3.12, PyTorch 2.12.1+CUDA 12.6, RTX 4060 Ti입니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s studies\neural_encoder_decoder\tests
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
