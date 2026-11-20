# 학습 곡선 요약 (runs/step3, 41 epoch)

| 항목 | epoch 1 | 마지막 epoch | best.pt epoch |
|---|---:|---:|---:|
| epoch | 1 | 41 | 21 |
| train/seg_loss | 3.704 | 0.619 | 0.843 |
| val/seg_loss | 3.317 | 4.638 | 4.247 |
| train/box_loss | 1.843 | 0.368 | 0.512 |
| val/box_loss | 1.907 | 1.864 | 1.791 |
| train/cls_loss | 2.724 | 0.258 | 0.356 |
| val/cls_loss | 2.100 | 1.970 | 1.571 |
| metrics/mAP50(M) | 0.358 | 0.579 | 0.666 |
| metrics/mAP50-95(M) | 0.145 | 0.195 | 0.224 |
| metrics/mAP50(B) | 0.439 | 0.580 | 0.682 |
| metrics/mAP50-95(B) | 0.201 | 0.297 | 0.354 |

- 학습 시간 합: 2.31시간 (results.csv의 time 열, 검증 포함, 이어 학습 6구간 합산)
- val/seg_loss 최저 epoch: 2 · mask mAP50 최고 epoch: 33 (0.676)
