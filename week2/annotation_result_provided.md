# ① 주석 결과

- 직접 폴리곤 주석 대신, 같은 방법으로 고른 영상 15장(클래스별 5장)의 **제공 라벨**을 주석 결과로 사용했다.
- 선정: 단계4 분할의 train 원본 중 증강본 · 0바이트 · 1주차 판단보류 영상을 빼고, 클래스마다 그 클래스가 든 영상 5장 (seed 42)
- 중복 확인: 15장끼리 64×36 흑백 상관 최댓값 **0.865** (1번 · 2번) — 기준 0.9 미만이라 비슷한 장면 반복 없음
- 형식: YOLOv8 Segmentation (클래스 ID + 정규화 폴리곤 좌표), 클래스 순서 0 crack · 1 efflorescence · 2 spall_detach
- 인스턴스 **24개**, 폴리곤 꼭짓점 최소 8 / 중앙 34 / 최대 396

| 클래스 | 인스턴스 수 | 들어 있는 영상 수 |
|---|---:|---:|
| crack | 10 | 5 |
| efflorescence | 8 | 5 |
| spall_detach | 6 | 5 |
| 합 | 24 | 15 |

영상별 개수: `annotation_result.csv` · 겹쳐 그린 그림: `annotation_result.png` · 라벨 사본: `annotate/provided_labels/`
