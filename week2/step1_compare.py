"""2주차 단계1 결과 정리 (제출물 ①): 직접 주석한 라벨을 세고, 같은 영상의 제공 라벨과 맞춰 본다.

(그림판 주석이면 step1_painted.py 가 export/ 에 써 준다. CVAT / Roboflow에서 내보낸 zip이면 export/ 에 풀어 둔 뒤 실행한다.)
(그 안의 **/*.txt 를 모두 찾는다. Roboflow는 파일명 뒤에 _jpg.rf.<해시>를 붙이므로 그 앞까지로 영상과 맞춘다.)

  python week2/step1_compare.py   → week2/annotation_result.md (①) · annotation_compare_N.png (4장씩)
                                    (+ annotation_compare.csv: 영상 · 클래스별 인스턴스 수와 마스크 IoU)

비교 기준: 영상마다 클래스별로 폴리곤을 모두 칠한 마스크(1920×1080)를 만들어 IoU = 겹친 넓이 / 합친 넓이.
둘 다 그 클래스를 안 그렸으면 IoU를 계산하지 않는다(빈칸). 제공 라벨은 '정답'이 아니라 비교 대상이다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import csv
import glob
import re
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from paths import CLASS_NAMES, SPLIT_ROOT, WEEK2, common

ANN = WEEK2 / "annotate"
EXPORT = ANN / "export"
COLORS = {0: (255, 60, 60), 1: (60, 200, 255), 2: (255, 200, 0)}     # RGB


def read_label(path):
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        v = ln.split()
        if len(v) >= 7:
            out.append((int(v[0]), np.array(v[1:], dtype=np.float64).reshape(-1, 2)))
    return out


def mask_of(polys, cid, w, h):
    m = np.zeros((h, w), np.uint8)
    for c, xy in polys:
        if c == cid:
            cv2.fillPoly(m, [np.round(xy * [w, h]).astype(np.int32)], 1)
    return m.astype(bool)


selected = [r["file"] for r in csv.DictReader(open(ANN / "selected.csv", encoding="utf-8-sig"))]
mine_files = {re.split(r"_jpg\.rf\.", Path(p).stem)[0]: p
              for p in glob.glob(str(EXPORT / "**" / "*.txt"), recursive=True)
              if Path(p).name not in ("classes.txt", "README.roboflow.txt", "README.dataset.txt")}
if not mine_files:
    raise SystemExit(f"내보낸 라벨이 없습니다. CVAT/Roboflow에서 내보낸 zip을 {EXPORT} 에 풀어 주세요.")

rows, cnt_mine, cnt_prov, ious = [], Counter(), Counter(), {c: [] for c in CLASS_NAMES}
not_done = []
panels = []
for name in selected:
    stem = Path(name).stem
    if stem not in mine_files:
        not_done.append(name)
        continue
    img = common.imread(SPLIT_ROOT / "images" / "train" / name)[:, :, ::-1]
    h, w = img.shape[:2]
    mine = read_label(mine_files[stem])
    prov = read_label(SPLIT_ROOT / "labels" / "train" / f"{stem}.txt")
    row = {"file": name}
    for cid, cname in CLASS_NAMES.items():
        a, b = mask_of(mine, cid, w, h), mask_of(prov, cid, w, h)
        nm, npv = sum(c == cid for c, _ in mine), sum(c == cid for c, _ in prov)
        cnt_mine[cid] += nm
        cnt_prov[cid] += npv
        iou = (a & b).sum() / (a | b).sum() if (a | b).any() else None
        if iou is not None:
            ious[cid].append(iou)
        row.update({f"{cname}_mine": nm, f"{cname}_provided": npv,
                    f"{cname}_iou": "" if iou is None else f"{iou:.3f}",
                    f"{cname}_px_mine": int(a.sum()), f"{cname}_px_provided": int(b.sum()),
                    f"{cname}_pts_mine": "/".join(str(len(xy)) for c, xy in mine if c == cid),
                    f"{cname}_pts_provided": "/".join(str(len(xy)) for c, xy in prov if c == cid)})
    rows.append(row)
    panels.append((name, img, mine, prov))

with open(WEEK2 / "annotation_compare.csv", "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

# ---------------------------------------------------------------- 그림: 영상마다 [직접 주석 | 제공 라벨]
def draw(img, polys):
    over = img.copy()
    for c, xy in polys:
        pts = np.round(xy * [img.shape[1], img.shape[0]]).astype(np.int32)
        cv2.fillPoly(over, [pts], COLORS[c])
    out = cv2.addWeighted(img, 0.6, over, 0.4, 0)
    for c, xy in polys:
        pts = np.round(xy * [img.shape[1], img.shape[0]]).astype(np.int32)
        cv2.polylines(out, [pts], True, COLORS[c], 4)
    return out


for old in WEEK2.glob("annotation_compare*.png"):       # 이전 그림 지우고 새로 (4장씩 나눠 저장 → 문서 한 쪽에 들어감)
    old.unlink()
PER = 4
for start in range(0, len(panels), PER):
    chunk = panels[start:start + PER]
    fig, axes = plt.subplots(len(chunk), 2, figsize=(10, 2.9 * len(chunk)))
    axes = np.atleast_2d(axes)
    for k, (name, img, mine, prov) in enumerate(chunk):
        for j, (polys, title) in enumerate([(mine, "직접 주석"), (prov, "제공 라벨")]):
            ax = axes[k, j]
            ax.imshow(draw(img, polys))
            ax.set_title(f"{start + k + 1}. {title} — {name[:32]}", fontsize=8)
            ax.axis("off")
    fig.legend(handles=[plt.Line2D([], [], color=np.array(COLORS[c]) / 255, lw=6, label=n) for c, n in CLASS_NAMES.items()],
               loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))   # 범례 자리 0.4인치
    fig.savefig(WEEK2 / f"annotation_compare_{start // PER + 1}.png", dpi=130)
    plt.close(fig)

# ---------------------------------------------------------------- ① 보고서
L = ["# ① 주석 결과", "",
     f"- 직접 주석한 영상: **{len(rows)}장** (선정 {len(selected)}장" + (f", 아직 안 한 영상 {len(not_done)}장" if not_done else "") + ")",
     f"- 직접 그린 인스턴스: **{sum(cnt_mine.values())}개**", "",
     "| 클래스 | 직접 주석 인스턴스 | 같은 영상의 제공 라벨 인스턴스 | 클래스 마스크 IoU 평균 (영상 수) |",
     "|---|---:|---:|---:|"]
for cid, cname in CLASS_NAMES.items():
    v = ious[cid]
    L.append(f"| {cname} | {cnt_mine[cid]} | {cnt_prov[cid]} | "
             + (f"{np.mean(v):.3f} ({len(v)})" if v else "-") + " |")
L.append(f"| 합 | {sum(cnt_mine.values())} | {sum(cnt_prov.values())} | |")
if not_done:
    L += ["", "아직 주석하지 않은 영상: " + ", ".join(not_done)]
L += ["", "영상별 수치: `annotation_compare.csv` · 겹쳐 그린 그림: `annotation_compare_*.png`"]
(WEEK2 / "annotation_result.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
