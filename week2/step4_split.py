"""2주차 단계4 (강의 12쪽): 원본 ID 단위 분할과 data.yaml 작성.

강의 코드는 원본 ID를 무작위로 섞어 8:1:1로 나눈다. 이 데이터는 이미 논문과 같은 train 4,036 / val 40 / test 40으로
나뉘어 왔고, val·test는 증강 없는 원본만으로 구성되어 있다 (논문 Table 1, 강의 22쪽 "테스트셋 고정").
그래서 비율로 새로 나누지 않고, 강의 코드와 같이 **원본 ID로 묶은 그룹 단위**로 split을 정하되
그룹이 갈 split은 제공 val/test를 따른다. 그 과정에서 아래 두 가지를 걸러 낸다.

  1) 누수: 원본이 val/test에 있는데 그 원본의 영상(재인코딩 사본 · 증강본)이 train에 있으면 train에서 뺀다.
  2) 0바이트 영상 63장(단계3 무결성 점검) — 학습에서 제외.

영상은 복사하지 않고 하드링크로 만든다 (같은 C: 드라이브, 3.4GB를 또 쓰지 않음). 실패하면 복사한다.
  python week2/step4_split.py  → <datasets>/crack-seg-split/{images,labels}/{train,val,test} + data.yaml
                                 week2/split_report.md · split_manifest.csv
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from paths import CLASS_NAMES, DATA_ROOT, SPLIT_ROOT, SPLITS, WEEK2, common, get_origin_id

SRC_IMG_DIR = DATA_ROOT / "images"
SRC_LBL_DIR = DATA_ROOT / "labels"
OUT_ROOT = SPLIT_ROOT
PRIORITY = ("test", "val", "train")      # 한 원본이 여러 split에 있으면 평가셋(test → val)을 우선한다

# =========================
# 이미지 목록 확인 + 원본 ID별 그룹 생성  (강의 코드와 같은 구조)
# =========================
groups = defaultdict(list)              # origin_id -> [(제공 split, 영상 경로)]
for split in SPLITS:
    images = sorted((SRC_IMG_DIR / split).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"{SRC_IMG_DIR / split} 에 jpg 영상이 없습니다.")
    for img_path in images:
        groups[get_origin_id(img_path.stem)].append((split, img_path))

# =========================
# ★ 원본 ID 단위로 split 결정 — 한 그룹은 반드시 한 split에만 간다
# =========================
def assign_split(members):
    """그룹 안에 증강되지 않은 영상이 들어 있는 제공 split 중 우선순위가 가장 높은 곳."""
    present = {s for s, p in members if not common.is_augmented(p)} or {s for s, _ in members}
    return next(s for s in PRIORITY if s in present)


split_ids = {s: set() for s in SPLITS}
for origin_id, members in groups.items():
    split_ids[assign_split(members)].add(origin_id)

print("원본 ID 수")
for s in SPLITS:
    print(f"{s:5s}:", len(split_ids[s]))

# =========================
# 그룹 안의 영상별로 사용 / 제외 판정
# =========================
manifest = []                           # (origin_id, 제공 split, 파일명, 배정 split 또는 '', 사유)
for split, ids in split_ids.items():
    for origin_id in sorted(ids):
        for src_split, img_path in groups[origin_id]:
            if img_path.stat().st_size == 0:
                reason, dst = "제외: 0바이트 영상", ""
            elif src_split != split:
                reason, dst = f"제외: 누수 (원본이 {split}에 있음)", ""
            elif split != "train" and common.is_augmented(img_path):
                reason, dst = "제외: 평가셋 증강본", ""            # 이 데이터에는 없음 (확인용)
            else:
                reason, dst = "사용", split
            manifest.append((origin_id, src_split, img_path.name, dst, reason))

# =========================
# 출력 폴더 생성 + 이미지/라벨 배치 (하드링크, 안 되면 복사)
# =========================
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)             # 다시 실행해도 같은 결과가 나오게 매번 새로 만든다
for split in SPLITS:
    (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def place(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


missing = []
for origin_id, src_split, name, dst_split, reason in manifest:
    if not dst_split:
        continue
    img_path = SRC_IMG_DIR / src_split / name
    place(img_path, OUT_ROOT / "images" / dst_split / name)
    lbl_path = SRC_LBL_DIR / src_split / f"{img_path.stem}.txt"      # 대응 라벨
    if lbl_path.exists():
        place(lbl_path, OUT_ROOT / "labels" / dst_split / lbl_path.name)
    else:
        missing.append(name)
        print(f"[확인 필요] 라벨 없음: {name}")

# =========================
# data.yaml 생성
# =========================
data_yaml = {
    "path": OUT_ROOT.resolve().as_posix(),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": CLASS_NAMES,
}
with open(OUT_ROOT / "data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

# =========================
# 검증 1: 같은 원본 ID가 두 split에 있지 않은가 (분할 결과 폴더를 다시 읽어서 확인)
# =========================
out_ids = {s: {get_origin_id(p.stem) for p in (OUT_ROOT / "images" / s).glob("*.jpg")} for s in SPLITS}
overlap = {f"{a}∩{b}": len(out_ids[a] & out_ids[b])
           for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
aug_in_eval = sum(common.is_augmented(p) for s in ("val", "test") for p in (OUT_ROOT / "images" / s).glob("*.jpg"))
n_img = {s: len(list((OUT_ROOT / "images" / s).glob("*.jpg"))) for s in SPLITS}
n_lbl = {s: len(list((OUT_ROOT / "labels" / s).glob("*.txt"))) for s in SPLITS}
print("원본 ID 겹침:", overlap, "| val·test 증강본:", aug_in_eval)

# =========================
# 검증 2: 파일명은 달라도 같은 장면인 원본이 split을 넘나들지 않는가 (강의 22쪽 '중복 영상 검토')
#   원본(증강 아닌 영상)을 64×36 흑백으로 줄여 표준화한 뒤 상관계수를 잰다. 같은 영상의 재인코딩이면 ≈1.
# =========================
orig = []
for s, ids in split_ids.items():
    for origin_id in ids:
        for src_split, p in groups[origin_id]:
            if common.is_augmented(p) or p.stat().st_size == 0:
                continue
            g = cv2.resize(common.imread(p, cv2.IMREAD_GRAYSCALE), (64, 36), interpolation=cv2.INTER_AREA)
            g = g.astype(np.float32)
            orig.append((src_split, s, p.name, ((g - g.mean()) / (g.std() + 1e-6)).ravel()))
X = np.stack([o[3] for o in orig])
C = X @ X.T / X.shape[1]
pairs = []
for i in range(len(orig)):
    for j in range(i + 1, len(orig)):
        if C[i, j] > 0.9:
            pairs.append((float(C[i, j]), orig[i], orig[j]))
pairs.sort(key=lambda t: -t[0])
DUP = 0.99
dup_before = [p for p in pairs if p[0] >= DUP and p[1][0] != p[2][0]]      # 제공 split 기준
dup_after = [p for p in pairs if p[0] >= DUP and p[1][1] != p[2][1]                # 분할 후 (사용 영상만)
             and all(any(m[2] == o[2] and m[3] for m in manifest) for o in (p[1], p[2]))]

# =========================
# 기록
# =========================
with open(WEEK2 / "split_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["origin_id", "provided_split", "file", "assigned_split", "decision"])
    w.writerows(sorted(manifest))

dec = Counter((m[1], m[4]) for m in manifest)
leak = [m for m in manifest if m[4].startswith("제외: 누수")]
leak_ids = sorted({m[0] for m in leak})
short = lambda n: n.split("_jpg.rf.")[0] + ("_jpg.rf.…" if "_jpg.rf." in n else "")   # noqa: E731

L = ["# 원본 ID 단위 분할 결과", "",
     f"- 출력: `{OUT_ROOT}` (영상 · 라벨은 하드링크) · `data.yaml`",
     f"- 원본 ID 그룹: **{len(groups)}개** → train {len(split_ids['train'])} · val {len(split_ids['val'])} · "
     f"test {len(split_ids['test'])}",
     "", "## 영상 수", "",
     "| 구분 | train | val | test | 합 |", "|---|---:|---:|---:|---:|",
     "| 제공 데이터 | " + " | ".join(f"{sum(v for (s, _), v in dec.items() if s == sp):,}" for sp in SPLITS)
     + f" | {len(manifest):,} |"]
for r in ["제외: 0바이트 영상"] + sorted({m[4] for m in leak}):
    L.append(f"| {r} | " + " | ".join(f"{dec[(sp, r)]:,}" for sp in SPLITS)
             + f" | {sum(dec[(sp, r)] for sp in SPLITS):,} |")
L.append("| **분할 후 영상** | " + " | ".join(f"**{n_img[s]:,}**" for s in SPLITS) + f" | **{sum(n_img.values()):,}** |")
L.append("| 분할 후 라벨 | " + " | ".join(f"{n_lbl[s]:,}" for s in SPLITS) + f" | {sum(n_lbl.values()):,} |")
L += ["", "## 누수 제거", ""]
for oid in leak_ids:
    mem = groups[oid]
    L.append(f"- 원본 ID `{oid}`: " + ", ".join(f"{s} {c}장" for s, c in Counter(s for s, _ in mem).items())
             + f" → {next(s for s in SPLITS if oid in split_ids[s])}에 배정, train의 {sum(m[0] == oid for m in leak)}장 제외")
for c, a, b in dup_before:
    L.append(f"- 픽셀 비교: {a[0]} `{short(a[2])}` ↔ {b[0]} `{short(b[2])}` 상관 {c:.3f} (같은 영상의 재인코딩)")
L += ["", "## 검증", "",
      f"- 같은 원본 ID가 두 split에 있는 경우: " + ", ".join(f"{k} = {v}" for k, v in overlap.items()),
      f"- val · test에 들어간 증강본: {aug_in_eval}장",
      f"- 라벨 없는 영상: {len(missing)}장 · 영상 수 = 라벨 수: "
      + ("예" if n_img == n_lbl else f"아니오 {n_img} {n_lbl}"),
      f"- 원본 영상끼리 픽셀 상관 {DUP} 이상(같은 영상)이 split을 넘는 쌍: 분할 전 {len(dup_before)}쌍 → 분할 후 {len(dup_after)}쌍",
      f"- 참고: 상관 0.9 이상 쌍 {len(pairs)}개 중 가장 높은 6개 (0.99 미만은 같은 터널의 비슷한 장면)", ""]
L += ["| 상관 | 영상 A | 영상 B |", "|---:|---|---|"]
for c, a, b in pairs[:6]:
    L.append(f"| {c:.3f} | {a[0]} `{short(a[2])}` | {b[0]} `{short(b[2])}` |")   # 제공 split 표시
(WEEK2 / "split_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

print("\n분할 완료")
print("data.yaml:", OUT_ROOT / "data.yaml")
print("영상:", n_img, "라벨:", n_lbl)
