"""2주차 단계1 보조: 직접 주석할 때 옆에 두고 볼 참고 그림.

annotate/selected.csv 의 영상마다 [원본 | 제공 라벨 테두리]를 나란히 붙여 annotate/reference/ 에 저장한다.
결함이 어디쯤인지 · 경계를 어디까지 잡았는지 감을 잡는 용도다. 그대로 따라 그리는 게 아니라
주석기준서(week1/annotation_guideline.md) 기준으로 판단하고, 차이는 step1_compare.py 에서 확인한다.

  python week2/step1_reference.py   → week2/annotate/reference/NN_<파일명>.png
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import csv
from pathlib import Path

import cv2
import numpy as np

from paths import CLASS_NAMES, SPLIT_ROOT, WEEK2, common

ANN = WEEK2 / "annotate"
OUT = ANN / "reference"
COLORS = {0: (60, 60, 255), 1: (255, 200, 60), 2: (0, 200, 255)}     # BGR: crack 빨강 · efflorescence 하늘 · spall 노랑


def read_label(path):
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        v = ln.split()
        if len(v) >= 7:
            out.append((int(v[0]), np.array(v[1:], dtype=np.float64).reshape(-1, 2)))
    return out


OUT.mkdir(exist_ok=True)
for r in csv.DictReader(open(ANN / "selected.csv", encoding="utf-8-sig")):
    name, stem = r["file"], Path(r["file"]).stem
    img = common.imread(SPLIT_ROOT / "images" / "train" / name)
    h, w = img.shape[:2]
    polys = read_label(SPLIT_ROOT / "labels" / "train" / f"{stem}.txt")
    over = img.copy()
    for c, xy in polys:
        pts = np.round(xy * [w, h]).astype(np.int32)
        cv2.fillPoly(over, [pts], COLORS[c])
    over = cv2.addWeighted(img, 0.75, over, 0.25, 0)        # 옅게 칠해 아래 표면이 보이게
    for c, xy in polys:
        cv2.polylines(over, [np.round(xy * [w, h]).astype(np.int32)], True, COLORS[c], 3)
    counts = {CLASS_NAMES[c]: sum(k == c for k, _ in polys) for c in CLASS_NAMES}
    text = "provided: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
    cv2.putText(img, "original", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    cv2.putText(over, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    pair = np.hstack([img, over])
    ok, buf = cv2.imencode(".png", pair)                    # 한글 경로라 cv2.imwrite 대신
    buf.tofile(str(OUT / f"{int(r['no']):02d}_{stem}.png"))
    print(f"{r['no']:>2} {name}  {text}")
print(f"→ {common.rel(OUT)}")
