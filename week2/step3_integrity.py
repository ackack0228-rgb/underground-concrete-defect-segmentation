"""2주차 단계3 (강의 11쪽): 데이터 무결성 점검.

영상-라벨 대응 · 빈 파일 · 클래스 ID · 좌표 범위를 점검한다. 강의 코드(클래스 ID · 좌표 범위)를 그대로 먼저 돌리고,
강의 '확인 방법' 6개 항목을 모두 세는 확장 점검을 이어서 한다.

  python week2/step3_integrity.py            # 제공 데이터 점검  → week2/integrity.md, integrity_issues.csv
  python week2/step3_integrity.py --split    # 단계4 분할 결과 점검 → week2/integrity_split.md (수정 확인용)
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from paths import CLASS_NAMES, DATA_ROOT, SPLIT_ROOT, SPLITS, WEEK2

ap = argparse.ArgumentParser()
ap.add_argument("--split", action="store_true", help="단계4 분할 결과(crack-seg-split)를 점검")
args = ap.parse_args()
ROOT = SPLIT_ROOT if args.split else DATA_ROOT
OUT_MD = WEEK2 / ("integrity_split.md" if args.split else "integrity.md")
OUT_CSV = WEEK2 / ("integrity_split_issues.csv" if args.split else "integrity_issues.csv")


def G(pattern):
    return sorted(glob.glob(str(ROOT / pattern), recursive=True))


# ---------------------------------------------------------------- 강의 11쪽 코드 그대로
bad = set()
for p in G("labels/**/*.txt"):
    for ln in open(p):
        v = ln.split()
        if not v:
            continue
        if int(v[0]) not in (0, 1, 2):
            bad.add(p)
        if any(not 0 <= float(x) <= 1 for x in v[1:]):
            bad.add(p)
print("이상 파일:", len(bad))
lecture_bad = len(bad)

# ---------------------------------------------------------------- 확장 점검 (확인 방법 6항목)
issues = []          # (split, 파일, 종류, 내용)
stats = {s: Counter() for s in SPLITS}
inst = {s: Counter() for s in SPLITS}           # 클래스별 인스턴스 수
img_has = {s: Counter() for s in SPLITS}        # 클래스별 영상 수

for split in SPLITS:
    imgs = {Path(p).stem: Path(p) for p in G(f"images/{split}/*.*")}
    lbls = {Path(p).stem: Path(p) for p in G(f"labels/{split}/*.txt")}
    stats[split]["영상"] = len(imgs)
    stats[split]["라벨"] = len(lbls)

    for stem in sorted(imgs.keys() - lbls.keys()):            # 라벨 없는 영상
        issues.append((split, imgs[stem].name, "라벨 없는 영상", "배경 영상인지 라벨 누락인지 확인"))
    for stem in sorted(lbls.keys() - imgs.keys()):            # 영상 없는 라벨
        issues.append((split, lbls[stem].name, "영상 없는 라벨", "원본 영상 존재 여부 확인"))

    for stem, ip in imgs.items():                             # 영상 자체를 열 수 있는지
        size = ip.stat().st_size
        if size == 0:
            issues.append((split, ip.name, "0바이트 영상", "제공 zip에 빈 파일로 들어 있음"))
            continue
        buf = np.fromfile(str(ip), dtype=np.uint8)
        if cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_8) is None:
            issues.append((split, ip.name, "열 수 없는 영상", f"{size} bytes"))

    for stem, lp in lbls.items():
        lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            issues.append((split, lp.name, "빈 라벨", "의도된 배경 라벨인지 확인"))
            continue
        seen = set()
        for k, ln in enumerate(lines, 1):
            v = ln.split()
            n = len(v)
            if n == 5:
                issues.append((split, lp.name, "Detection 포맷 줄", f"{k}번째 줄 값 5개"))
                continue
            if n < 7 or (n - 1) % 2:
                issues.append((split, lp.name, "토큰 수 이상", f"{k}번째 줄 값 {n}개"))
                continue
            try:
                cid = int(v[0])
                xy = [float(x) for x in v[1:]]
            except ValueError:
                issues.append((split, lp.name, "숫자가 아닌 값", f"{k}번째 줄"))
                continue
            if cid not in CLASS_NAMES:
                issues.append((split, lp.name, "클래스 ID 이상", f"{k}번째 줄 클래스 {cid}"))
                continue
            out = [x for x in xy if not 0 <= x <= 1]
            if out:
                issues.append((split, lp.name, "좌표 범위 이상", f"{k}번째 줄 {len(out)}개 (예: {out[0]})"))
            if ln.strip() in seen:
                issues.append((split, lp.name, "중복 줄", f"{k}번째 줄이 앞 줄과 같음"))
            seen.add(ln.strip())
            inst[split][cid] += 1
            stats[split]["폴리곤 꼭짓점 합"] += len(xy) // 2
        for cid in {int(ln.split()[0]) for ln in lines if ln.split()[0].isdigit()}:
            img_has[split][cid] += 1

kinds = Counter(k for _, _, k, _ in issues)
print("확장 점검 이상 항목:", len(issues), dict(kinds))

with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["split", "file", "issue", "detail"])
    w.writerows(issues)

# ---------------------------------------------------------------- 보고서
L = [f"# 데이터 무결성 점검 ({'단계4 분할 결과' if args.split else '제공 데이터'})", "",
     f"- 점검 위치: `{ROOT}`",
     f"- 강의 11쪽 코드 결과 (클래스 ID · 좌표 범위): **이상 파일 {lecture_bad}개**",
     f"- 확장 점검 결과: **이상 항목 {len(issues)}개**" + (f" — {', '.join(f'{k} {v}' for k, v in kinds.items())}" if kinds else ""),
     "", "## 항목별 결과", "",
     "| 확인 항목 | train | val | test |", "|---|---:|---:|---:|"]
by = Counter((s, k) for s, _, k, _ in issues)
for k in ["라벨 없는 영상", "영상 없는 라벨", "0바이트 영상", "열 수 없는 영상", "빈 라벨",
          "Detection 포맷 줄", "토큰 수 이상", "숫자가 아닌 값", "클래스 ID 이상", "좌표 범위 이상", "중복 줄"]:
    L.append(f"| {k} | " + " | ".join(f"{by[(s, k)]:,}" for s in SPLITS) + " |")
L += ["", "## 구성", "", "| 구분 | train | val | test |", "|---|---:|---:|---:|"]
L.append("| 영상 파일 | " + " | ".join(f"{stats[s]['영상']:,}" for s in SPLITS) + " |")
L.append("| 라벨 파일 | " + " | ".join(f"{stats[s]['라벨']:,}" for s in SPLITS) + " |")
for cid, name in CLASS_NAMES.items():
    L.append(f"| {name} 인스턴스 (영상 수) | " +
             " | ".join(f"{inst[s][cid]:,} ({img_has[s][cid]:,})" for s in SPLITS) + " |")
L.append("| 인스턴스 합 | " + " | ".join(f"{sum(inst[s].values()):,}" for s in SPLITS) + " |")
if issues:
    L += ["", f"이상 항목 전체 목록: `{OUT_CSV.name}`"]
OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
print("저장:", OUT_MD.name, OUT_CSV.name)
