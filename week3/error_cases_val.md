# 단계5 검증셋(보조) 오류 사례 (임계값 = 검증셋 선택값)

| 클래스 | 유형 | 영상 | 정답 클래스 | 임계값 | 그 클래스 최고 신뢰도 | 임계값 이상 개수 | 다른 클래스 예측 | 그림 |
|---|---|---|---|---:|---:|---:|---|---|
| crack | FP | `USU013_2022_03_16_002_483_jpg.rf.3` | efflorescence | 0.2 | 0.573 | 4 | - | error_cases_val/crack/FP_1_USU013_2022_03_16_002_483_jpg.rf.3187438.jpg |
| crack | FP | `USU013_2022_03_16_003_5020_jpg.rf.` | efflorescence | 0.2 | 0.906 | 1 | - | error_cases_val/crack/FP_2_USU013_2022_03_16_003_5020_jpg.rf.98cb3d.jpg |
| efflorescence | FP | `URT019_2022_03_07_001_13407.jpg` | spall_detach | 0.02 | 0.662 | 7 | spall_detach 0.67; spall_detach 0.72 | error_cases_val/efflorescence/FP_1_URT019_2022_03_07_001_13407.jpg |
| efflorescence | FP | `URT019_2022_04_15_001_9544_jpg.rf.` | spall_detach | 0.02 | 0.765 | 8 | spall_detach 0.13; spall_detach 0.22 | error_cases_val/efflorescence/FP_2_URT019_2022_04_15_001_9544_jpg.rf.bfe204.jpg |
| efflorescence | FN | `URT010_2022_02_08_001_2502_jpg.rf.` | efflorescence | 0.02 | 0.000 | 0 | spall_detach 0.08; spall_detach 0.91 | error_cases_val/efflorescence/FN_1_URT010_2022_02_08_001_2502_jpg.rf.8718a6.jpg |
| efflorescence | FN | `USU012_2021_02_07_005_2422_jpg.rf.` | efflorescence | 0.02 | 0.007 | 0 | spall_detach 0.89 | error_cases_val/efflorescence/FN_2_USU012_2021_02_07_005_2422_jpg.rf.19d626.jpg |
| efflorescence | FN | `USU013_2022_03_16_002_483_jpg.rf.3` | efflorescence | 0.02 | 0.000 | 0 | crack 0.24; crack 0.29; crack 0.55; crack 0.57 | error_cases_val/efflorescence/FN_3_USU013_2022_03_16_002_483_jpg.rf.3187438.jpg |
| spall_detach | FP | `URT007_2022_01_12_001_26564_jpg.rf` | efflorescence | 0.06 | 0.452 | 1 | efflorescence 0.31 | error_cases_val/spall_detach/FP_1_URT007_2022_01_12_001_26564_jpg.rf.70998.jpg |
| spall_detach | FP | `URT010_2022_02_08_001_2502_jpg.rf.` | efflorescence | 0.06 | 0.906 | 2 | - | error_cases_val/spall_detach/FP_2_URT010_2022_02_08_001_2502_jpg.rf.8718a6.jpg |
| spall_detach | FP | `USU012_2021_02_07_005_2422_jpg.rf.` | efflorescence | 0.06 | 0.886 | 1 | - | error_cases_val/spall_detach/FP_3_USU012_2021_02_07_005_2422_jpg.rf.19d626.jpg |
