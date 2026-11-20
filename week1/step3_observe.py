"""1주차 실습 단계3 — 영상 관찰 (강의 11쪽) + 제출물 ③ samples9.png

강의 코드대로 최대 9장을 3×3으로 보여 주고 samples9.png로 저장한다.
첫 번째 영상의 밝기 평균 · 표준편차를 출력하고, 영상 간 조도 차이를 비교할 수 있게
전체 영상의 밝기 통계도 brightness.md / brightness_hist.png 로 남긴다.
"""
import _use_venv  # noqa: F401  실행 버튼이 다른 가상환경을 골라도 crack-seg/.venv로 다시 실행

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import SEED, WEEK1, imread, is_augmented, list_images, readable, rel, require_images
import cv2

imgs = readable(list_images())   # 0바이트 영상 63장 제외 (data_summary.md 참고)
require_images(imgs)
random.seed(SEED)  # 다시 실행해도 같은 9장이 나오게

# ---- 강의 11쪽 코드 (cv2.imread → 한글 경로용 imread 로만 바꿈) ----
n = min(9, len(imgs))
sample = random.sample(imgs, n)

fig, ax = plt.subplots(3, 3, figsize=(10, 9))
axes = ax.ravel()

for a in axes:
    a.axis("off")

for a, p in zip(axes, sample):
    im = cv2.cvtColor(imread(p), cv2.COLOR_BGR2RGB)
    a.imshow(im)
    a.set_title(Path(p).name[:20], fontsize=7)

plt.tight_layout()
plt.savefig(WEEK1 / "samples9.png", dpi=110)
plt.close(fig)

# 첫 번째 영상의 밝기 통계 확인
if imgs:
    gray = imread(imgs[0], cv2.IMREAD_GRAYSCALE)
    print("밝기 평균:", round(gray.mean(), 1))
    print("밝기 표준편차:", round(gray.std(), 1))
# --------------------------------------------------------------------

# 추가: 전체 영상 밝기 통계 (조도가 낮은 / 대비가 낮은 영상 찾기)
stats = []
for p in imgs:
    g = imread(p, cv2.IMREAD_GRAYSCALE)
    stats.append((rel(p), float(g.mean()), float(g.std())))
means = np.array([s[1] for s in stats])
stds = np.array([s[2] for s in stats])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].hist(means, bins=30, color="#4a6fa5")
ax[0].set_xlabel("밝기 평균 (0-255)"); ax[0].set_ylabel("영상 수")
ax[1].hist(stds, bins=30, color="#a5694a")
ax[1].set_xlabel("밝기 표준편차 (대비)")
fig.suptitle(f"전체 {len(imgs)}장의 밝기 분포 (증강본 포함)")
plt.tight_layout()
plt.savefig(WEEK1 / "brightness_hist.png", dpi=110)
plt.close(fig)

# 어둡다 · 밝다 · 대비 낮음 목록은 원본 영상에서만 고른다 (증강본은 같은 장면이 반복돼 목록이 한 장면으로 채워짐)
orig = [s for s in stats if not is_augmented(s[0])]
order = sorted(orig, key=lambda s: s[1])
out = ["# 밝기 통계 (단계3 참고값)", "",
       "밝기 평균은 영상 간 상대적인 조도 차이를, 표준편차는 대비를 비교하는 참고값이다 (강의 11쪽).", "",
       f"- 첫 번째 영상 `{stats[0][0]}`: 평균 {stats[0][1]:.1f}, 표준편차 {stats[0][2]:.1f}",
       f"- 전체 {len(imgs)}장 평균: {means.mean():.1f} (최소 {means.min():.1f}, 최대 {means.max():.1f})",
       f"- 전체 표준편차 평균: {stds.mean():.1f} (최소 {stds.min():.1f}, 최대 {stds.max():.1f})",
       f"- 원본 {sum(not is_augmented(r) for r, _, _ in stats)}장만: 평균 {np.mean([m for r, m, _ in stats if not is_augmented(r)]):.1f}, "
       f"표준편차 평균 {np.mean([s for r, _, s in stats if not is_augmented(r)]):.1f} (증강본에는 밝기 · 대비를 일부러 바꾼 영상이 섞여 있다)",
       f"- samples9.png에 쓴 9장: " + ", ".join(f"`{rel(p)}`" for p in sample), "",
       f"아래 목록은 증강본을 뺀 원본 {len(orig)}장 기준이다.", "",
       "## 가장 어두운 영상 5장", "", "| 영상 | 평균 | 표준편차 |", "|---|---|---|"]
out += [f"| {r} | {m:.1f} | {s:.1f} |" for r, m, s in order[:5]]
out += ["", "## 가장 밝은 영상 5장", "", "| 영상 | 평균 | 표준편차 |", "|---|---|---|"]
out += [f"| {r} | {m:.1f} | {s:.1f} |" for r, m, s in order[-5:][::-1]]
out += ["", "## 대비가 가장 낮은 영상 5장", "", "| 영상 | 평균 | 표준편차 |", "|---|---|---|"]
out += [f"| {r} | {m:.1f} | {s:.1f} |" for r, m, s in sorted(orig, key=lambda s: s[2])[:5]]
(WEEK1 / "brightness.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("→ week1/samples9.png, brightness.md, brightness_hist.png 저장")
