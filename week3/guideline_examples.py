"""주석기준서 v0.3 예시 그림: 2 · 3주차 오류 분석에서 새로 찾은 제외 대상 · 포함 사례 · 경계 문제.

  python week3/guideline_examples.py    # → guideline_v03_examples.png · guideline_v03_width.png
각 칸은 기준 모델(runs/step3 best.pt)이 검증 임계값 이상으로 낸 예측(그 클래스)을 상자로 표시한다 (정답 라벨은 초록 테두리).
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from adv_pix import gt_mask
from w3 import CLASS_LIST, IMGSZ, PRED_CONF, SPLIT_ROOT, THR_FILE, WEEK3, load_model

HN = SPLIT_ROOT.parent / "crack-seg-hn" / "images" / "train"
TEST = SPLIT_ROOT / "images" / "test"
# (영상, 정답 split(없으면 None), 표시할 클래스, 제목)
CASES = [
    (TEST / "URT010_2022_02_05_001_14659.jpg", "test", 0, "제외 · 명암 띠 → crack 오탐 (테스트 FP)"),
    (next(TEST.glob("URT017_2020_02_08_001_24512_jpg.rf.*.jpg")), "test", 1, "제외 · 분필 표기 → efflorescence 오탐 (테스트 FP)"),
    # 2026-11-20: 처음 넣은 hn000 · hn002 · hn010은 예측 상자가 설명한 물체(트레이 · 케이블)가 아니라 옆 벽에 있어 틀린 예시였다
    # → 하드네거티브 사람 검토(adv3_hn_review.csv)에서 '남김'으로 판정한 조각으로 바꿈
    (HN / "hn013_r0.jpg", None, 2, "제외 · 배관 상단 브래킷 → spall_detach (train 하드네거티브)"),
    (HN / "hn073_r0.jpg", None, 2, "제외 · 케이블 클램프 → spall_detach (train 하드네거티브)"),
    (HN / "hn060_r0.jpg", None, 0, "제외 · 세로 배관 → crack (train 하드네거티브)"),
    (next(TEST.glob("URT005_2021_01_02_001_17812_jpg.rf.*.jpg")), "test", 0, "포함 · 거친 표면 위 가는 균열 (테스트 FN)"),
]


def main():
    thr = json.loads(THR_FILE.read_text(encoding="utf-8"))["thresholds"]
    model = load_model()
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.4))
    for ax, (p, split, cid, title) in zip(axes.flat, CASES):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        vis = img.copy()
        if split:
            g = gt_mask(split, p, w, h)
            for c in range(3):
                cs, _ = cv2.findContours(g[c].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cs, -1, (0, 220, 0), 4)
        r = model.predict(str(p), imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
        for c, f, b in zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
            if c == cid and f >= thr[CLASS_LIST[c]]:
                x1, y1, x2, y2 = b.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(vis, f"{CLASS_LIST[c]} {f:.2f}", (x1, max(y1 - 12, 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("주석기준서 v0.3 예시 — 빨간 상자 = 모델 예측(검증 임계값 이상), 초록 = 정답 라벨", fontsize=12)
    plt.tight_layout()
    plt.savefig(WEEK3 / "guideline_v03_examples.png", dpi=100)
    plt.close()

    # 경계 규칙 그림: 정답 띠 vs 실제 어두운 균열 픽셀 (심화 ⑥과 같은 영상)
    p = next(TEST.glob("UCT011_2021_02_12_001_13670_jpg.rf.*.jpg"))
    img = cv2.imread(str(p))
    g = gt_mask("test", p, img.shape[1], img.shape[0])[0]
    ys, xs = np.where(g)
    cy = int(np.median(ys))
    x0, x1 = max(xs[ys == cy].min() - 60, 0), xs[ys == cy].max() + 60
    crop = img[cy - 90:cy + 90, x0:x1].copy()
    band = g[cy - 90:cy + 90, x0:x1].astype(np.uint8)
    cs, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    vis = crop.copy()
    cv2.drawContours(vis, cs, -1, (0, 220, 0), 2)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    ax[0].imshow(cv2.cvtColor(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB))
    ax[0].set_title("원본 (3배 확대)")
    ax[1].imshow(cv2.cvtColor(cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB))
    ax[1].set_title("초록 = 제공 라벨 띠 (폭 약 45px, 실제 균열 약 3px)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(WEEK3 / "guideline_v03_width.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
