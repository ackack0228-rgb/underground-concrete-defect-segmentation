"""2주차 단계1 결과 정리 (제출물 ①) — 직접 주석 대신 제공 라벨로 대체한 버전.

직접 주석(강의 9쪽)을 하지 않기로 해서(2026-11-10 사용자 결정), step1_prepare.py 로 고른 같은 15장의
제공 라벨을 ① 주석 결과로 정리한다. 제공 라벨은 이미 YOLOv8 Segmentation 포맷이라 export 결과와 형식이 같다.

  python week2/step1_provided.py → week2/annotate/provided_labels/*.txt (15장 라벨 사본, 단계2 --labels 확인용)
                                   week2/annotation_result.md (①) · annotation_result.png · annotation_result.csv

중복 확인(강의 9쪽): 15장끼리 64×36 흑백 상관을 모두 계산해 가장 큰 값을 적는다(선정 때 0.9 이상은 이미 뺐다).
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import csv
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from paths import CLASS_NAMES, SPLIT_ROOT, WEEK2, common

ANN = WEEK2 / "annotate"
OUT_LBL = ANN / "provided_labels"
COLORS = {0: (255, 60, 60), 1: (60, 200, 255), 2: (255, 200, 0)}     # RGB


def read_label(path):
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        v = ln.split()
        if len(v) >= 7:
            out.append((int(v[0]), np.array(v[1:], dtype=np.float64).reshape(-1, 2)))
    return out


def thumb(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((g - g.mean()) / (g.std() + 1e-6)).ravel()


def draw(img, polys):
    over = img.copy()
    for c, xy in polys:
        cv2.fillPoly(over, [np.round(xy * [img.shape[1], img.shape[0]]).astype(np.int32)], COLORS[c])
    out = cv2.addWeighted(img, 0.6, over, 0.4, 0)
    for c, xy in polys:
        cv2.polylines(out, [np.round(xy * [img.shape[1], img.shape[0]]).astype(np.int32)], True, COLORS[c], 4)
    return out


selected = list(csv.DictReader(open(ANN / "selected.csv", encoding="utf-8-sig")))
if OUT_LBL.exists():
    shutil.rmtree(OUT_LBL)
OUT_LBL.mkdir(parents=True)

rows, cnt, verts, panels, feats = [], Counter(), [], [], []
for r in selected:
    name, stem = r["file"], Path(r["file"]).stem
    src = SPLIT_ROOT / "labels" / "train" / f"{stem}.txt"
    shutil.copy2(src, OUT_LBL / src.name)
    img = common.imread(SPLIT_ROOT / "images" / "train" / name)[:, :, ::-1]
    polys = read_label(src)
    row = {"no": r["no"], "file": name}
    for cid, cname in CLASS_NAMES.items():
        n = sum(c == cid for c, _ in polys)
        cnt[cid] += n
        row[cname] = n
    verts += [len(xy) for _, xy in polys]
    rows.append(row)
    panels.append((r["no"], name, img, polys))
    feats.append(thumb(img))

sims = [(float(feats[i] @ feats[j]) / feats[i].size, i, j) for i, j in combinations(range(len(feats)), 2)]
best_sim, bi, bj = max(sims)

with open(WEEK2 / "annotation_result.csv", "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

# ---------------------------------------------------------------- 그림: 5행 × 3열 (열 = 선정 클래스)
fig, axes = plt.subplots(5, 3, figsize=(15, 9.6))
for k, (no, name, img, polys) in enumerate(panels):
    ax = axes[k % 5, k // 5]
    ax.imshow(draw(img, polys))
    ax.set_title(f"{no}. {name[:30]}", fontsize=8)
    ax.axis("off")
fig.legend(handles=[plt.Line2D([], [], color=np.array(COLORS[c]) / 255, lw=6, label=n) for c, n in CLASS_NAMES.items()],
           loc="upper center", ncol=3, fontsize=10, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(WEEK2 / "annotation_result.png", dpi=90)

# ---------------------------------------------------------------- ① 보고서
L = ["# ① 주석 결과", "",
     "- 직접 폴리곤 주석 대신, 같은 방법으로 고른 영상 15장(클래스별 5장)의 **제공 라벨**을 주석 결과로 사용했다.",
     "- 선정: 단계4 분할의 train 원본 중 증강본 · 0바이트 · 1주차 판단보류 영상을 빼고, 클래스마다 그 클래스가 든 영상 5장 (seed 42)",
     f"- 중복 확인: 15장끼리 64×36 흑백 상관 최댓값 **{best_sim:.3f}** ({panels[bi][0]}번 · {panels[bj][0]}번) — 기준 0.9 미만이라 비슷한 장면 반복 없음",
     "- 형식: YOLOv8 Segmentation (클래스 ID + 정규화 폴리곤 좌표), 클래스 순서 0 crack · 1 efflorescence · 2 spall_detach",
     f"- 인스턴스 **{sum(cnt.values())}개**, 폴리곤 꼭짓점 최소 {min(verts)} / 중앙 {int(np.median(verts))} / 최대 {max(verts)}", "",
     "| 클래스 | 인스턴스 수 | 들어 있는 영상 수 |", "|---|---:|---:|"]
for cid, cname in CLASS_NAMES.items():
    L.append(f"| {cname} | {cnt[cid]} | {sum(r[cname] > 0 for r in rows)} |")
L.append(f"| 합 | {sum(cnt.values())} | {len(rows)} |")
L += ["", "영상별 개수: `annotation_result.csv` · 겹쳐 그린 그림: `annotation_result.png` · 라벨 사본: `annotate/provided_labels/`"]
(WEEK2 / "annotation_result.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
