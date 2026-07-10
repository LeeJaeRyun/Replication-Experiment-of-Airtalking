# Replication Experiment of AirTalking

AirTalking 논문 재현과 후속 연구 실험을 연구 주제별로 정리한 저장소입니다.
원본 데이터셋은 저장소에 포함하지 않으며, 공개 데이터인 Cityscapes를 사용합니다.

## 전체 구조

```text
dataset/
  Cityscapes 원본 데이터. git에는 올리지 않음.

studies/
  airtalking_reproduction/
    AirTalking 논문 재현 실험

  adaptive_semantic_compression/
    채널 상태 기반 adaptive semantic compression 후속 실험

  neural_encoder_decoder/
    실제 학습 가능한 neural semantic encoder/decoder 후속 실험
```

## 폴더별 설명

| 폴더 | 무엇인지 | 실제로 한 작업 | 주요 산출물 |
|---|---|---|---|
| `studies/airtalking_reproduction` | AirTalking 논문 재현 실험 | 논문 공개 파라미터와 Cityscapes 기반 semantic profile을 사용해 UAV D2D 시뮬레이터를 구현하고 논문 Figure 값과 비교 | 재현 코드, 검증 CSV, 그래프 PNG, 재현 보고서 DOCX |
| `studies/adaptive_semantic_compression` | 후속 연구 1: adaptive semantic compression | 고정 압축률 대신 링크/SINR 상태에 따라 semantic compression level을 고르는 정책을 AirTalking 시뮬레이터에 넣어 비교 | adaptive 실험 코드, compression-quality 표, full adaptive 결과, 제안서 DOCX |
| `studies/neural_encoder_decoder` | 후속 연구 2: neural semantic encoder/decoder | Cityscapes RGB 이미지를 입력으로 받아 latent semantic feature를 만들고 segmentation map을 복원하는 작은 encoder/decoder를 실제 학습 | 학습 코드, encoder/decoder 성능 JSON/CSV, AirTalking timed simulation 결과, 최종 보고서 DOCX |

## 데이터 출처

사용한 공개 데이터는 Cityscapes입니다.

- 공식 사이트: https://www.cityscapes-dataset.com/
- 필요한 패키지:
  - `leftImg8bit_trainvaltest.zip`
  - `gtFine_trainvaltest.zip`

예상 로컬 경로:

```text
dataset/
  leftImg8bit_trainvaltest/leftImg8bit/{train,val,test}/
  gtFine_trainvaltest/gtFine/{train,val,test}/
```

`test` split은 공개 gtFine label이 실제 정답이 아니라 dummy/ignore 영역이므로 semantic 측정에는 사용하지 않았습니다.

## 주요 보고서

```text
studies/airtalking_reproduction/reports/
  AirTalking 재현 실험 보고서와 쉬운 설명 문서

studies/adaptive_semantic_compression/reports/
  adaptive semantic compression 연구 제안서

studies/neural_encoder_decoder/reports/
  AirTalking_Followup_Final_Report_KR.docx
```

## 실행 예시

```powershell
python studies\airtalking_reproduction\code\measure_cityscapes_semantics.py
python studies\airtalking_reproduction\code\airtalking_reproduction.py
python studies\airtalking_reproduction\code\verify_against_paper.py

python studies\adaptive_semantic_compression\code\run_adaptive_probe.py
python studies\adaptive_semantic_compression\code\run_full_adaptive_research.py
python studies\adaptive_semantic_compression\code\validate_full_adaptive_results.py

python studies\neural_encoder_decoder\code\train_semantic_encoder_decoder.py
```

각 연구 폴더의 `README.md`에 세부 설명과 결과 위치를 따로 적어두었습니다.
