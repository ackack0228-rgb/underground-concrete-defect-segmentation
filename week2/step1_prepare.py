"""2주차 단계1 (강의 9쪽) 준비: 직접 주석할 영상 15장을 골라 업로드 폴더에 모은다.

- 대상: 단계4 분할의 train 원본(증강본 · 0바이트 · 1주차 판단보류 3장 제외). 영상만 복사하고 라벨은 주지 않는다
  (제공 라벨을 보지 않고 주석기준서만으로 그린 뒤, 단계1 비교에서 제공 라벨과 맞춰 본다).
- 클래스마다 그 클래스가 들어 있는 영상 5장씩 (crack → efflorescence → spall_detach 순, 이미 뽑힌 영상은 건너뜀).
- 중복 확인(강의 9쪽): 뽑은 영상끼리 64×36 흑백 상관이 0.9 이상이면 비슷한 장면으로 보고 다음 후보로 넘어간다.

  python week2/step1_prepare.py  → week2/annotate/images/*.jpg (CVAT · Roboflow에 그대로 업로드) + annotate/selected.csv
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import csv
import random
import shutil

import cv2
import numpy as np

from paths import CLASS_NAMES, SPLIT_ROOT, WEEK2, common

N_PER_CLASS = 5
SEED = 42
OUT = WEEK2 / "annotate" / "images"
PENDING = {r["file"].split("/", 1)[1] for r in csv.DictReader(open(WEEK2.parent / "week1" / "pending.csv", encoding="utf-8-sig"))}

cands = [p for p in sorted((SPLIT_ROOT / "images" / "train").glob("*.jpg"))
         if not common.is_augmented(p) and p.name not in PENDING]
classes_in = {}
for p in cands:
    lines = (SPLIT_ROOT / "labels" / "train" / f"{p.stem}.txt").read_text(encoding="utf-8").split("\n")
    classes_in[p] = {int(ln.split()[0]) for ln in lines if ln.strip()}

rng = random.Random(SEED)
chosen, feats = [], []


def thumb(p):
    g = cv2.resize(common.imread(p, cv2.IMREAD_GRAYSCALE), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((g - g.mean()) / (g.std() + 1e-6)).ravel()


for cid in CLASS_NAMES:
    pool = [p for p in cands if cid in classes_in[p] and p not in chosen]
    rng.shuffle(pool)
    got = 0
    for p in pool:
        f = thumb(p)
        if any(float(f @ g) / f.size >= 0.9 for g in feats):     # 비슷한 장면은 건너뜀
            continue
        chosen.append(p)
        feats.append(f)
        got += 1
        if got == N_PER_CLASS:
            break

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
with open(WEEK2 / "annotate" / "selected.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["no", "file", "classes_in_provided_label"])
    for k, p in enumerate(chosen, 1):
        shutil.copy2(p, OUT / p.name)
        w.writerow([k, p.name, " ".join(CLASS_NAMES[c] for c in sorted(classes_in[p]))])
print(f"{len(chosen)}장 → {OUT}")
