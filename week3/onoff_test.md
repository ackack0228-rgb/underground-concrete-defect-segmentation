# 단계4 테스트셋 ON/OFF 결과 (임계값 = 검증셋에서 선택, 고정 적용)

- thresholds.json 생성 2026-11-21 02:33:33 (val 40장, sha 123d40a69aca) → 변경 없이 사용
- 테스트 영상 40장

| 지표 | crack | efflorescence | spall_detach | macro |
|---|---:|---:|---:|---:|
| 임계값 | 0.2 | 0.02 | 0.06 | — |
| True Positive | 10 | 13 | 14 | — |
| True Negative | 25 | 26 | 26 | — |
| False Positive | 2 | 1 | 0 | — |
| False Negative | 3 | 0 | 0 | — |
| Precision | 0.833 | 0.929 | 1.000 | 0.921 |
| Recall | 0.769 | 1.000 | 1.000 | 0.923 |
| F1-score | 0.800 | 0.963 | 1.000 | 0.921 |

논문 Table 5 (참고): crack P 0.917 R 0.846 F1 0.880 · efflorescence P 0.846 R 0.917 F1 0.880 · spall_detach P 1.000 R 0.867 F1 0.929 · macro F1 0.896
