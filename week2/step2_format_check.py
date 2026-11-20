"""2주차 단계2 (강의 10쪽): 라벨이 Segmentation 포맷인지 확인.

강의 코드를 그대로 먼저 돌리고(첫 라벨 파일의 첫 줄), 같은 판정을 모든 라벨 파일의 모든 줄에 적용한 집계를 덧붙인다.
  python week2/step2_format_check.py                     # 단계4 분할 결과 → week2/format_check.txt
  python week2/step2_format_check.py --labels <폴더>      # 직접 주석해 내보낸 라벨(단계1) → week2/format_check_mine.txt
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import argparse
import glob
import io
import sys
from collections import Counter
from pathlib import Path

from paths import SPLIT_ROOT, WEEK2

ap = argparse.ArgumentParser()
ap.add_argument("--labels", help="라벨 폴더 (기본: 분할 결과 labels/)")
args = ap.parse_args()
LBL_DIR = Path(args.labels) if args.labels else SPLIT_ROOT / "labels"
OUT = WEEK2 / ("format_check_mine.txt" if args.labels else "format_check.txt")

buf = io.StringIO()


def say(*a):
    print(*a)
    print(*a, file=buf)


# ---------------------------------------------------------------- 강의 10쪽 코드 (경로만 바꿈)
lbls = sorted(
    glob.glob(str(LBL_DIR / "**" / "*.txt"), recursive=True)
)
say("라벨 폴더:", LBL_DIR)
say("라벨 파일 수:", len(lbls))
if not lbls:
    say("라벨 파일이 없습니다.")
    sys.exit(1)
else:
    line = open(lbls[0], encoding="utf-8").readline().split()
    say("첫 파일:", Path(lbls[0]).relative_to(LBL_DIR).as_posix())
    say("토큰 개수:", len(line))
    say("앞부분:", line[:9])
    if len(line) == 5:
        say("Detection 포맷입니다.")
    elif len(line) >= 7 and (len(line) - 1) % 2 == 0:
        say("Segmentation 포맷입니다.")
    else:
        say("라벨 형식을 다시 확인하세요.")


# ---------------------------------------------------------------- 같은 판정을 전체 줄에
def kind(v):
    if len(v) == 5:
        return "Detection(5개)"
    if len(v) >= 7 and (len(v) - 1) % 2 == 0:
        return "Segmentation"
    return "형식 이상"


say("")
say("[전체 집계] split별 줄(인스턴스) 판정 · 클래스 ID · 좌표 범위 · 꼭짓점 수")
for split in sorted({Path(p).relative_to(LBL_DIR).parts[0] if len(Path(p).relative_to(LBL_DIR).parts) > 1 else "."
                     for p in lbls}):
    files = [p for p in lbls if (Path(p).relative_to(LBL_DIR).parts[0] == split
                                 or (split == "." and len(Path(p).relative_to(LBL_DIR).parts) == 1))]
    kinds, cls, npts = Counter(), Counter(), []
    lo, hi = 1.0, 0.0
    for p in files:
        for ln in open(p, encoding="utf-8"):
            v = ln.split()
            if not v:
                continue
            kinds[kind(v)] += 1
            cls[v[0]] += 1
            xy = [float(x) for x in v[1:]]
            lo, hi = min(lo, *xy), max(hi, *xy)
            npts.append(len(xy) // 2)
    npts.sort()
    say(f"  {split:5s} 파일 {len(files):,} · 줄 {sum(kinds.values()):,} · {dict(kinds)} · 클래스 ID {dict(sorted(cls.items()))}"
        f" · 좌표 {lo:.4f}~{hi:.4f} · 꼭짓점 최소 {npts[0]} / 중앙 {npts[len(npts) // 2]} / 최대 {npts[-1]}")

OUT.write_text(buf.getvalue(), encoding="utf-8")
print("저장:", OUT.name)
