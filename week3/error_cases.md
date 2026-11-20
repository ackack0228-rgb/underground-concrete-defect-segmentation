# 단계5 테스트셋 오류 사례 (임계값 = 검증셋 선택값)

| 클래스 | 유형 | 영상 | 정답 클래스 | 임계값 | 그 클래스 최고 신뢰도 | 임계값 이상 개수 | 다른 클래스 예측 | 그림 |
|---|---|---|---|---:|---:|---:|---|---|
| crack | FP | `URT010_2022_02_05_001_14659.jpg` | spall_detach | 0.2 | 0.344 | 1 | spall_detach 0.06; spall_detach 0.76 | error_cases/crack/FP_1_URT010_2022_02_05_001_14659.jpg |
| crack | FP | `UWT015_2021_01_02_001_17710_jpg.rf` | efflorescence | 0.2 | 0.664 | 1 | efflorescence 0.02; efflorescence 0.04; efflorescence 0.78 | error_cases/crack/FP_2_UWT015_2021_01_02_001_17710_jpg.rf.7572a.jpg |
| crack | FN | `UCT011_2021_02_12_001_7770_jpg.rf.` | crack | 0.2 | 0.013 | 0 | - | error_cases/crack/FN_1_UCT011_2021_02_12_001_7770_jpg.rf.3b8c7b.jpg |
| crack | FN | `URT005_2021_01_02_001_17812_jpg.rf` | crack | 0.2 | 0.000 | 0 | - | error_cases/crack/FN_2_URT005_2021_01_02_001_17812_jpg.rf.73a26.jpg |
| crack | FN | `URT005_2021_01_03_001_13040.jpg` | crack | 0.2 | 0.194 | 0 | - | error_cases/crack/FN_3_URT005_2021_01_03_001_13040.jpg |
| efflorescence | FP | `URT017_2020_02_08_001_24512_jpg.rf` | spall_detach | 0.02 | 0.092 | 2 | spall_detach 0.82 | error_cases/efflorescence/FP_1_URT017_2020_02_08_001_24512_jpg.rf.c2fd9.jpg |
