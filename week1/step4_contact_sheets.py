"""1주차 실습 단계4 준비 — 전체 영상 모아보기 (강의 12쪽)

하드네거티브 후보(시공이음부 · 케이블/배관 · 그림자 · 정반사 · 보수흔적 · 표면오염)를
눈으로 찾기 쉽도록 전체 영상을 번호를 붙여 5×6 시트로 저장한다.
시트에서 찾은 사례는 hard_negatives.csv 에 적고 step4_cases.py 로 그림을 만든다.

실행: python week1/step4_contact_sheets.py
결과: week1/contact_sheets/sheet_01.png ... + index.csv (번호 → 파일)
"""
import _use_venv  # noqa: F401  실행 버튼이 다른 가상환경을 골라도 crack-seg/.venv로 다시 실행

import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

from common import WEEK1, imread, is_augmented, list_images, readable, rel, require_images

ROWS, COLS = 5, 6
imgs = readable(list_images())
# 기본은 원본 영상만 (train의 증강본은 같은 장면의 회전 · 밝기 변형이라 후보 찾기에 중복). --all 이면 전부.
if "--all" not in sys.argv:
    imgs = [p for p in imgs if not is_augmented(p)]
require_images(imgs)

out_dir = WEEK1 / "contact_sheets"
out_dir.mkdir(exist_ok=True)
for old in out_dir.glob("sheet_*.png"):
    old.unlink()

with open(out_dir / "index.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["no", "file"])
    for i, p in enumerate(imgs, 1):
        w.writerow([i, rel(p)])

per = ROWS * COLS
for s in range(0, len(imgs), per):
    chunk = imgs[s:s + per]
    fig, ax = plt.subplots(ROWS, COLS, figsize=(COLS * 2.6, ROWS * 2.4))
    for a in ax.ravel():
        a.axis("off")
    for k, (a, p) in enumerate(zip(ax.ravel(), chunk)):
        im = imread(p)
        sc = 400 / max(im.shape[:2])
        if sc < 1:
            im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        a.imshow(im)
        a.set_title(f"#{s + k + 1}  {rel(p)[-22:]}", fontsize=6)
    plt.tight_layout()
    name = out_dir / f"sheet_{s // per + 1:02d}.png"
    plt.savefig(name, dpi=90)
    plt.close(fig)
    print("저장:", name.name, f"(#{s + 1}~#{s + len(chunk)})")
print(f"→ week1/contact_sheets/ 에 {(len(imgs) + per - 1) // per}장, index.csv 저장")
