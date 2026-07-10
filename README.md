# Replication Experiment of AirTalking

AirTalking 논문 재현 실험과 후속 연구 실험을 연구별로 정리한 저장소입니다.
원본 데이터셋은 저장소에 포함하지 않습니다. 현재 실제로 사용한 공개 데이터는 Cityscapes입니다.

## 전체 구조

```text
dataset/
  Cityscapes 원본 데이터 위치. git에는 포함하지 않음.

studies/
  airtalking_reproduction/
    AirTalking 논문 재현 실험

  adaptive_semantic_compression/
    채널 상태에 따라 semantic compression 비율을 바꾸는 후속 실험

  neural_encoder_decoder/
    Cityscapes로 직접 학습한 semantic encoder/decoder 실험

tools/
  build_easy_study_reports.py
    연구별 쉬운 DOCX 보고서 재생성 도구
```

## 폴더별 의미

| 폴더 | 한 줄 설명 | 주요 결과 |
|---|---|---|
| `studies/airtalking_reproduction` | AirTalking의 UAV D2D semantic communication 실험을 공개 정보와 Cityscapes 기반 측정값으로 재현 | 재현 코드, 검증 CSV/PNG, 쉬운 재현 보고서 |
| `studies/adaptive_semantic_compression` | 네트워크 상태가 나쁠 때는 더 세게 압축하고, 좋을 때는 덜 압축하는 adaptive compression 실험 | 상황별 압축률 실험, full adaptive 결과, 쉬운 후속 연구 보고서 |
| `studies/neural_encoder_decoder` | RGB 이미지를 작은 latent feature로 압축하고 semantic map으로 복원하는 encoder/decoder를 직접 학습 | 학습 코드, rho_c/정확도/시간 측정 결과, 쉬운 encoder/decoder 보고서 |

## Encoder/Decoder 반영 방식

`neural_encoder_decoder` 폴더는 따로 유지했습니다. 대신 그 결과를 다른 두 실험에도 반영했습니다.

- 재현 실험: neural encoder/decoder의 `rho_c`, mIoU, encode/decode 시간을 읽어 semantic 전송 profile로 쓸 수 있게 코드와 보고서에 반영했습니다.
- adaptive 실험: `paper_like` 압축 단계의 기준값을 직접 학습한 encoder/decoder 결과에 연결했습니다.
- 단, adaptive의 모든 압축 단계가 neural network로 학습된 것은 아닙니다. 현재 직접 학습된 모델은 논문값에 가까운 `paper_like` 1개 단계이고, 나머지 단계는 Cityscapes label 기반 proxy 품질을 사용합니다.

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

`test` split은 공개 gtFine 정답 라벨이 아니므로 semantic 품질 측정에는 사용하지 않습니다.

## 보고서

각 연구 폴더마다 DOCX는 하나만 남겼습니다.

```text
studies/airtalking_reproduction/reports/AirTalking_Reproduction_Easy_Final_KR.docx
studies/adaptive_semantic_compression/reports/Adaptive_Semantic_Compression_Easy_Final_KR.docx
studies/neural_encoder_decoder/reports/Neural_Encoder_Decoder_Easy_Final_KR.docx
```

보고서를 다시 만들려면:

```powershell
python tools\build_easy_study_reports.py
```

## 실행 예시

```powershell
python studies\airtalking_reproduction\code\measure_cityscapes_semantics.py
python studies\airtalking_reproduction\code\airtalking_reproduction.py --semantic-summary studies\neural_encoder_decoder\results\paperlike_timed_latent20\airtalking_semantic_summary.json
python studies\airtalking_reproduction\code\verify_against_paper.py

python studies\adaptive_semantic_compression\code\run_adaptive_probe.py --reuse-quality
python studies\adaptive_semantic_compression\code\run_full_adaptive_research.py
python studies\adaptive_semantic_compression\code\validate_full_adaptive_results.py

python studies\neural_encoder_decoder\code\train_semantic_encoder_decoder.py
```
