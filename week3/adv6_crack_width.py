"""3주차 심화 ⑥ (논문강의 47쪽 '균열폭 정량화'): 스켈레톤화 + 거리 변환으로 픽셀 단위 균열 폭 산출.

  python week3/adv6_crack_width.py     # → adv6_crack_width.md · adv6_crack_width.png · adv6_crack_width.csv
방법: 균열 마스크(원본 1920×1080 해상도)를 스켈레톤(skimage)으로 만들고, 마스크의 거리 변환(cv2, 경계까지 거리) 값을
  스켈레톤 위에서 읽어 폭 = 2 × 거리 − 1 로 본다 (스켈레톤 각 점에서 양쪽 경계까지).
세 가지 마스크의 폭을 비교한다 (테스트셋 중 정답 crack이 있는 영상).
  ① 정답 폴리곤(라벨 띠)  ② 예측 마스크(기준 best.pt, crack 임계값 = 검증셋 선택값)
  ③ 영상의 어두운 균열 픽셀: 정답 띠 안에서 black-hat(주변보다 어두운 가는 구조)으로 뽑은 픽셀 — 라벨 띠가 실제 균열보다 넓은지 보려고
★ 스케일 환산: 이 데이터에는 촬영 거리 · 초점거리 · 센서 크기(EXIF)가 없어 mm로 바꿀 수 없다. 공식만 적는다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

from adv_pix import gt_mask
from w3 import IMGSZ, PRED_CONF, THR_FILE, WEEK3, load_model, load_split

MIN_SKEL = 30                                         # 스켈레톤 픽셀이 이보다 적은 조각은 잡음으로 보고 뺀다


def widths(mask):
    """마스크(bool) → 스켈레톤 위 폭 배열(px)."""
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return np.array([])
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    sk = skeletonize(mask)
    n, lab = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(sk)
    for i in range(1, n):
        comp = lab == i
        if comp.sum() >= MIN_SKEL:
            keep |= comp
    return 2 * dist[keep] - 1                         # 중심 화소 자신이 거리 1이므로 -1 (폭 1px 선 → 1)


def dark_pixels(img, band):
    """정답 띠 안의 실제 균열 픽셀: black-hat(닫힘 - 원영상)이 띠 안 상위 값인 픽셀."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    if band.sum() == 0:
        return band
    t = max(np.percentile(bh[band], 75), 12)          # 띠 안에서 주변보다 확실히 어두운 픽셀
    m = (bh >= t) & band
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    return m


def stats(w):
    if len(w) == 0:
        return {"n": 0, "median": np.nan, "p90": np.nan, "max": np.nan}
    return {"n": int(len(w)), "median": float(np.median(w)), "p90": float(np.percentile(w, 90)), "max": float(w.max())}


def main():
    thr = json.loads(THR_FILE.read_text(encoding="utf-8"))["thresholds"]["crack"]
    model = load_model()
    imgs, gts = load_split("test")
    rows, examples = [], []
    for p, g in zip(imgs, gts):
        if 0 not in g:
            continue
        img = cv2.imread(p)
        h, w = img.shape[:2]
        gt = gt_mask("test", p, w, h)[0]
        r = model.predict(p, imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
        pred = np.zeros((h, w), np.uint8)
        if r.masks is not None:
            for c, f, poly in zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.masks.xy):
                if c == 0 and f >= thr and len(poly) >= 3:
                    cv2.fillPoly(pred, [poly.astype(np.int32)], 1)
        pred = pred.astype(bool)
        dark = dark_pixels(img, gt)
        wg, wp, wd = widths(gt), widths(pred), widths(dark)
        sg, sp, sd = stats(wg), stats(wp), stats(wd)
        rows.append({"image": Path(p).name, **{f"gt_{k}": v for k, v in sg.items()},
                     **{f"pred_{k}": v for k, v in sp.items()}, **{f"dark_{k}": v for k, v in sd.items()}})
        examples.append((p, img, gt, pred, dark))

    with open(WEEK3 / "adv6_crack_width.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    def fmt(v):
        return "-" if np.isnan(v) else f"{v:.1f}"

    L = ["# 심화 ⑥ 균열 폭 정량화 (픽셀 단위)", "",
         "- 방법: 마스크 → 스켈레톤(skimage.skeletonize) · 거리 변환(cv2.distanceTransform) → 스켈레톤 위 폭 = 2 × 경계까지 거리 − 1 (중심 화소 자신). 원본 해상도(1920×1080).",
         f"- 대상: 테스트셋 정답 crack 영상 {len(rows)}장. 예측 = 기준 best.pt, crack 임계값 {thr:g}(검증셋 선택값).",
         "- 어두운 균열 픽셀 = 정답 띠 안에서 black-hat(31px)이 띠 안 상위 25%이면서 12 이상인 픽셀 (라벨 띠가 실제 균열보다 얼마나 넓은지 보기 위한 참고값)", "",
         "| 영상 | 정답 띠 중앙값 / p90 | 예측 마스크 중앙값 / p90 | 어두운 균열 픽셀 중앙값 / p90 |", "|---|---:|---:|---:|"]
    for r in rows:
        L.append(f"| `{r['image'][:36]}` | {fmt(r['gt_median'])} / {fmt(r['gt_p90'])} | "
                 f"{fmt(r['pred_median'])} / {fmt(r['pred_p90'])} | {fmt(r['dark_median'])} / {fmt(r['dark_p90'])} |")
    med = {k: np.nanmedian([r[f"{k}_median"] for r in rows]) for k in ("gt", "pred", "dark")}
    both = [r for r in rows if r["pred_n"] > 0]
    L += ["", f"- 영상별 중앙값의 중앙값: 정답 띠 {med['gt']:.1f}px · 예측 {med['pred']:.1f}px · 어두운 균열 픽셀 {med['dark']:.1f}px",
          f"- 예측이 있는 {len(both)}장에서 예측 폭 / 정답 띠 폭 = "
          f"{np.median([r['pred_median'] / r['gt_median'] for r in both]):.2f} (중앙값)",
          "- 해석: 라벨 띠는 실제 어두운 균열보다 몇 배 넓게 그려져 있다(2주차 관찰과 같음). 그래서 이 라벨로 학습한 마스크의 폭은 '균열 폭'이 아니라 "
          "'라벨 띠 폭'을 따라간다. 폭을 재려면 주석 기준부터 '보이는 폭만큼'으로 맞춰야 한다.", "",
          "## 물리 단위 환산 (이 데이터로는 불가)", "",
          "- 폭(mm) = 폭(px) × 지상 화소 크기(mm/px), 지상 화소 크기 = 촬영 거리 × 센서 폭 / (초점거리 × 영상 가로 화소 수)",
          "- 이 데이터에는 EXIF(촬영 장비 · 초점거리)와 촬영 거리가 없다 → mm 환산 불가 (논문 한계 02와 같음). 현장에서는 기준 자(scale bar)나 거리 센서가 필요하다.",
          "- 또 예측 마스크는 YOLOv8 프로토타입 마스크(입력의 1/4 해상도)를 원본으로 키운 것이라 몇 px 단위의 폭은 정밀하지 않다."]
    (WEEK3 / "adv6_crack_width.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # 그림: 예측이 있는 영상 3장 — 원본 · 정답 띠 폭 · 예측 폭 · 어두운 균열 픽셀 폭 (스켈레톤 색 = 폭)
    pick, seen = [], set()
    for e in examples:                                    # 같은 장면(인접 프레임)은 한 장만
        key = Path(e[0]).name.split("_")[0] + Path(e[0]).name.split("_")[3]
        if e[3].sum() > 0 and key not in seen and len(pick) < 3:
            pick.append(e)
            seen.add(key)
    fig, axes = plt.subplots(len(pick), 4, figsize=(17, 3.3 * len(pick)))
    axes = np.atleast_2d(axes)
    vmax = 60
    for row, (p, img, gt, pred, dark) in zip(axes, pick):
        ys, xs = np.where(gt | pred)
        y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 40, img.shape[0])
        x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 40, img.shape[1])
        row[0].imshow(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        row[0].set_title(Path(p).name[:34], fontsize=8)
        for ax, m, lab in zip(row[1:], (gt, pred, dark), ("정답 띠", "예측 마스크", "어두운 균열 픽셀")):
            sub = m[y0:y1, x0:x1]
            dist = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 5)
            sk = skeletonize(sub)
            canvas = np.full(sub.shape, np.nan)
            canvas[sk] = 2 * dist[sk] - 1
            ax.imshow(sub, cmap="Greys", alpha=0.25)
            ys2, xs2 = np.where(sk)
            sc = ax.scatter(xs2, ys2, c=canvas[sk], s=2, cmap="turbo", vmin=0, vmax=vmax)
            w = widths(m)
            ax.set_title(f"{lab} — 폭 중앙값 {np.median(w):.1f}px" if len(w) else f"{lab} — 없음", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="폭 (px)")
    fig.suptitle("스켈레톤 위 균열 폭 (색 = 2 × 경계까지 거리 - 1, px)", fontsize=12)
    plt.savefig(WEEK3 / "adv6_crack_width.png", dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
