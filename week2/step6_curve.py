"""2주차 단계6 (강의 14쪽): 학습 곡선 확인.

강의 코드 그대로 curve.png (Segmentation loss · Mask mAP)를 그리고, 확인용 그림 curves_all.png
(손실 4종 train/val + mAP)를 덧붙인다. best.pt가 고른 epoch에 점선을 긋는다.
ultralytics 8.4는 best.pt를 fitness = mAP50-95(B) + mAP50-95(M) 이 가장 큰 epoch로 저장한다 (utils/metrics.py).

  python week2/step6_curve.py            # runs/step3/results.csv → week2/curve.png · curves_all.png · curve_summary.md
  python week2/step6_curve.py --run step3_quick
학습 중에도 실행할 수 있다 (results.csv는 epoch마다 한 줄씩 늘어난다).
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from paths import WEEK2, common  # noqa: F401  (common: 그림 한글 글꼴 설정)

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="step3")
args = ap.parse_args()
RUN = WEEK2.parent / "runs" / args.run

# ---------------------------------------------------------------- 강의 14쪽 코드 (경로만 바꿈)
# 학습 결과 불러오기
df = pd.read_csv(RUN / "results.csv")
df.columns = [c.strip() for c in df.columns]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# ① 학습 / 검증 Segmentation loss
for c in ["train/seg_loss", "val/seg_loss"]:
    if c in df.columns:
        ax[0].plot(df["epoch"], df[c], label=c)
ax[0].set_title("Segmentation Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

# ② Mask mAP
for c in ["metrics/mAP50(M)", "metrics/mAP50-95(M)"]:
    if c in df.columns:
        ax[1].plot(df["epoch"], df[c], label=c)
ax[1].set_title("Mask mAP")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("mAP")
ax[1].legend()

# (추가) best.pt가 고른 epoch 표시
df["fitness"] = df["metrics/mAP50-95(B)"] + df["metrics/mAP50-95(M)"]
best = df.loc[df["fitness"].idxmax()]
for a in ax:
    a.axvline(best["epoch"], color="gray", ls="--", lw=1)
ax[1].annotate(f"best.pt (epoch {int(best['epoch'])})", (best["epoch"], best["metrics/mAP50(M)"]),
               xytext=(5, 5), textcoords="offset points", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(WEEK2 / "curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 확인용: 손실 4종 + mAP
fig, axs = plt.subplots(1, 5, figsize=(22, 3.8))
for a, k in zip(axs, ["box_loss", "seg_loss", "cls_loss", "dfl_loss"]):
    a.plot(df["epoch"], df[f"train/{k}"], label="train")
    a.plot(df["epoch"], df[f"val/{k}"], label="val")
    a.set_title(k)
    a.set_xlabel("Epoch")
    a.legend()
for c, ls in [("metrics/mAP50(M)", "-"), ("metrics/mAP50-95(M)", "-"), ("metrics/mAP50(B)", ":"), ("metrics/mAP50-95(B)", ":")]:
    axs[4].plot(df["epoch"], df[c], ls, label=c.replace("metrics/", ""))
axs[4].set_title("mAP (M = mask, B = box)")
axs[4].set_xlabel("Epoch")
axs[4].legend(fontsize=8)
for a in axs:
    a.axvline(best["epoch"], color="gray", ls="--", lw=1)
plt.tight_layout()
plt.savefig(WEEK2 / "curves_all.png", dpi=110)
plt.close()

# ---------------------------------------------------------------- 요약
first, last = df.iloc[0], df.iloc[-1]
L = [f"# 학습 곡선 요약 (runs/{args.run}, {len(df)} epoch)", "",
     "| 항목 | epoch 1 | 마지막 epoch | best.pt epoch |", "|---|---:|---:|---:|",
     f"| epoch | 1 | {int(last['epoch'])} | {int(best['epoch'])} |"]
for c in ["train/seg_loss", "val/seg_loss", "train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss",
          "metrics/mAP50(M)", "metrics/mAP50-95(M)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
    L.append(f"| {c} | {first[c]:.3f} | {last[c]:.3f} | {best[c]:.3f} |")
# time 열은 --resume 할 때마다 0부터 다시 쌓인다 → 줄어드는 곳에서 구간을 나눠 구간별 마지막 값을 더한다
seg = (df["time"].diff() < 0).cumsum()
total_h = df.groupby(seg)["time"].last().sum() / 3600
L += ["", f"- 학습 시간 합: {total_h:.2f}시간 (results.csv의 time 열, 검증 포함, 이어 학습 {seg.max() + 1}구간 합산)",
      f"- val/seg_loss 최저 epoch: {int(df.loc[df['val/seg_loss'].idxmin(), 'epoch'])} · "
      f"mask mAP50 최고 epoch: {int(df.loc[df['metrics/mAP50(M)'].idxmax(), 'epoch'])} "
      f"({df['metrics/mAP50(M)'].max():.3f})"]
(WEEK2 / "curve_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
