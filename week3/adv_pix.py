"""심화 ① · ② 공용: 인스턴스 목록(클래스 · 신뢰도 · 폴리곤) → 영상 단위 ON/OFF · 픽셀 IoU.

U-Net(의미 분할)은 인스턴스 · 신뢰도가 없어 mask mAP를 같은 방식으로 낼 수 없다. 그래서 두 모델을 비교할 때는
① 영상 단위 ON/OFF(검증셋 임계값 → 테스트 고정)와 ② 클래스별 픽셀 IoU(테스트 전체 합산 교집합 / 합집합)를 공통 지표로 쓴다.
픽셀 IoU는 960×540(원본의 1/2)에서 잰다.
"""
from pathlib import Path

import cv2
import numpy as np

from w3 import CLASS_LIST, SPLIT_ROOT

PW, PH = 960, 540                                         # 픽셀 IoU 계산 해상도 (가로 영상 기준)


def gt_mask(split, img_path, w, h):
    """정답 라벨 → (3, h, w) bool."""
    m = np.zeros((3, h, w), np.uint8)
    lab = SPLIT_ROOT / "labels" / split / (Path(img_path).stem + ".txt")
    for line in lab.read_text(encoding="utf-8").splitlines() if lab.exists() else []:
        v = line.split()
        if len(v) > 6:
            pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.fillPoly(m[int(v[0])], [pts], 1)
    return m.astype(bool)


def inst_mask(insts, thr, src_w, src_h, w, h):
    """인스턴스 [(cls, conf, poly(원본 좌표))] 중 클래스별 임계값 이상을 (3, h, w) bool로."""
    m = np.zeros((3, h, w), np.uint8)
    sx, sy = w / src_w, h / src_h
    for c, f, poly in insts:
        if f >= thr[CLASS_LIST[c]] and len(poly) >= 3:
            cv2.fillPoly(m[c], [(np.asarray(poly) * [sx, sy]).astype(np.int32)], 1)
    return m.astype(bool)


def onoff_from_scores(scores, gts, thr, cid):
    """scores[i] = 영상 i의 클래스 cid 최고 점수. 강의 10쪽 onoff와 같은 계산."""
    tp = fp = fn = tn = 0
    for s, gt in zip(scores, gts):
        pred_on, gt_on = s >= thr, cid in gt
        if gt_on and pred_on:
            tp += 1
        elif pred_on:
            fp += 1
        elif gt_on:
            fn += 1
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0}


def select_then_test(val_scores, val_gts, test_scores, test_gts, grid):
    """검증셋에서 클래스별 임계값(F1 최대, 동점이면 낮은 값) → 테스트에 고정 적용."""
    thr, curves, test = {}, {}, {}
    for cid, n in enumerate(CLASS_LIST):
        f1 = [onoff_from_scores([s[cid] for s in val_scores], val_gts, t, cid)["f1"] for t in grid]
        thr[n] = grid[f1.index(max(f1))]
        curves[n] = f1
        test[n] = onoff_from_scores([s[cid] for s in test_scores], test_gts, thr[n], cid)
    return thr, curves, test


class IoUAcc:
    def __init__(self):
        self.inter = np.zeros(3)
        self.union = np.zeros(3)

    def add(self, pred, gt):
        self.inter += (pred & gt).sum((1, 2))
        self.union += (pred | gt).sum((1, 2))

    def iou(self):
        return {n: float(self.inter[i] / self.union[i]) if self.union[i] else 0.0 for i, n in enumerate(CLASS_LIST)}
