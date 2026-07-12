# Cityscapes 데이터셋 감사 요약

- 엄격 감사 결과: **통과**
- 데이터 루트: `C:\Users\firep\OneDrive\바탕 화면\Replication-Experiment-of-Airtalking\dataset`
- 실행 시간: 119.928초 (worker 1개, 순차 I/O)
- 오류/경고: 0 / 1

## 표본 수와 1:1 대응

| split | RGB | color GT | instanceIds GT | labelIds GT | polygons GT | 완전 대응 |
|---|---:|---:|---:|---:|---:|:---:|
| train | 2,975 | 2,975 | 2,975 | 2,975 | 2,975 | 예 |
| val | 500 | 500 | 500 | 500 | 500 | 예 |
| test | 1,525 | 1,525 | 1,525 | 1,525 | 1,525 | 예 |

## 도시와 split 누수

- **train**: 18개 — aachen, bochum, bremen, cologne, darmstadt, dusseldorf, erfurt, hamburg, hanover, jena, krefeld, monchengladbach, strasbourg, stuttgart, tubingen, ulm, weimar, zurich
- **val**: 3개 — frankfurt, lindau, munster
- **test**: 6개 — berlin, bielefeld, bonn, leverkusen, mainz, munich

- stem 누수: **없음**
- city 누수: **없음**

## PNG 및 polygon JSON 검사

- **train**
  - rgb: 2,975/2,975, dimensions [2048x1024: 2,975], modes [RGB: 2,975]
  - gt_color: 2,975/2,975, dimensions [2048x1024: 2,975], modes [RGBA: 2,975]
  - gt_instanceIds: 2,975/2,975, dimensions [2048x1024: 2,975], modes [I;16: 2,975]
  - gt_labelIds: 2,975/2,975, dimensions [2048x1024: 2,975], modes [L: 2,975]
  - polygons JSON: 2,975/2,975 파싱, 객체 235,690개, 파싱 오류 0개, 스키마 오류 0개
- **val**
  - rgb: 500/500, dimensions [2048x1024: 500], modes [RGB: 500]
  - gt_color: 500/500, dimensions [2048x1024: 500], modes [RGBA: 500]
  - gt_instanceIds: 500/500, dimensions [2048x1024: 500], modes [I;16: 500]
  - gt_labelIds: 500/500, dimensions [2048x1024: 500], modes [L: 500]
  - polygons JSON: 500/500 파싱, 객체 46,809개, 파싱 오류 0개, 스키마 오류 0개
- **test**
  - rgb: 1,525/1,525, dimensions [2048x1024: 1,525], modes [RGB: 1,525]
  - gt_color: 1,525/1,525, dimensions [2048x1024: 1,525], modes [RGBA: 1,525]
  - gt_instanceIds: 1,525/1,525, dimensions [2048x1024: 1,525], modes [I;16: 1,525]
  - gt_labelIds: 1,525/1,525, dimensions [2048x1024: 1,525], modes [L: 1,525]
  - polygons JSON: 1,525/1,525 파싱, 객체 4,966개, 파싱 오류 0개, 스키마 오류 0개

검사 범위: 인식된 RGB·GT PNG 전부의 크기와 모드를 검사했습니다. labelIds는 모든 픽셀을 완전히 디코딩했고, 나머지 PNG는 헤더를 검사했습니다. 모든 polygons JSON은 파싱 및 기본 스키마 검사를 했습니다.

## train/val 19-class 픽셀 분포

Cityscapes의 원본 `labelId`를 학습용 `trainId` 0~18로 매핑했습니다. 19개에 속하지 않는 픽셀은 `ignore`로 합쳤습니다.

| trainId | labelId | 클래스 | train 픽셀 | val 픽셀 | train+val 픽셀 |
|---:|---:|---|---:|---:|---:|
| 0 | 7 | road | 2,036,416,525 | 345,264,442 | 2,381,680,967 |
| 1 | 8 | sidewalk | 336,090,793 | 49,568,652 | 385,659,445 |
| 2 | 11 | building | 1,260,636,120 | 201,005,428 | 1,461,641,548 |
| 3 | 12 | wall | 36,199,498 | 6,718,315 | 42,917,813 |
| 4 | 13 | fence | 48,454,166 | 7,521,741 | 55,975,907 |
| 5 | 17 | pole | 67,789,506 | 13,565,658 | 81,355,164 |
| 6 | 19 | traffic light | 11,477,088 | 1,808,393 | 13,285,481 |
| 7 | 20 | traffic sign | 30,448,193 | 6,098,373 | 36,546,566 |
| 8 | 21 | vegetation | 879,783,988 | 158,868,008 | 1,038,651,996 |
| 9 | 22 | terrain | 63,949,536 | 7,625,026 | 71,574,562 |
| 10 | 23 | sky | 221,979,646 | 30,765,347 | 252,744,993 |
| 11 | 24 | person | 67,326,424 | 11,913,424 | 79,239,848 |
| 12 | 25 | rider | 7,463,162 | 1,975,596 | 9,438,758 |
| 13 | 26 | car | 386,328,286 | 59,731,217 | 446,059,503 |
| 14 | 27 | truck | 14,772,328 | 2,760,211 | 17,532,539 |
| 15 | 28 | bus | 12,990,290 | 3,563,120 | 16,553,410 |
| 16 | 31 | train | 12,863,955 | 1,031,648 | 13,895,603 |
| 17 | 32 | motorcycle | 5,449,152 | 729,415 | 6,178,567 |
| 18 | 33 | bicycle | 22,861,233 | 6,504,475 | 29,365,708 |
- train ignore 비율: **11.472098%** (715,747,311/6,239,027,200 픽셀)
- val ignore 비율: **12.546302%** (131,557,511/1,048,576,000 픽셀)
- train+val ignore 비율: **11.626660%** (847,304,822/7,287,603,200 픽셀)

## test GT 해석 주의

**test 픽셀 중 19개 평가 trainId에 매핑되는 픽셀이 하나도 없습니다. 따라서 동봉된 test gtFine은 무시 영역/placeholder일 뿐, 로컬 정확도나 mIoU를 계산할 수 있는 의미 분할 정답이 아닙니다. test 예측은 Cityscapes 공식 평가 서버로 평가해야 합니다.**

전수 검사 결과 test의 19-class 유효 픽셀은 0/3,198,156,800개이고, ignore 비율은 100.000000%입니다.

## 재현 가능한 데이터셋 지문

- SHA-256: `9c02129e4901ec77cb0e4ddc391f7c3484d4d5d2548a64af6539cd84048ab7de`
- inventory: 25,004개 파일, 12,399,642,484 bytes
- content SHA-256: 10,000개 파일, 11,661,992,359 bytes
- 정책: all RGB leftImg8bit and gtFine labelIds files

지문은 모든 파일의 상대 경로와 크기를 사용하며, 기본 정책에서는 모든 RGB와 labelIds 파일의 실제 바이트 SHA-256도 사용합니다. 정확한 정규화 방법과 전체 파일 inventory는 `dataset_manifest.json`에 기록되어 있습니다.

## 용어

- **stem**: 한 장면을 식별하는 공통 파일명 부분입니다. RGB와 네 GT 파일이 같은 stem을 가져야 합니다.
- **labelId**: Cityscapes 원본 의미 클래스 번호입니다.
- **trainId**: 학습·평가에 쓰도록 19개 클래스를 0~18로 다시 번호 붙인 값입니다.
- **ignore 비율**: 19개 평가 클래스에 속하지 않아 손실·mIoU 계산에서 제외해야 하는 픽셀 비율입니다.
- **fingerprint**: 파일 경로·크기·내용 해시를 하나의 SHA-256으로 합친 데이터셋 식별값입니다.
