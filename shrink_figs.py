"""공개 저장소용: AI Hub 사진이 들어간 그림을 줄인다 (2026-11-20 사용자 결정 "원본 사진은 줄여서 올린다").

  python shrink_figs.py          # 가로 최대 MAX_W px로 (이미 작으면 그대로 — 여러 번 실행해도 같음)
처음 실행 때 원래 해상도 파일을 ~/datasets/crack-seg-figs-fullres/ 에 같은 구조로 백업한다 (이미 있으면 덮어쓰지 않음).
그래프(사진 없음)는 건드리지 않는다. 데이터셋 원본 복사본(week2/annotate/images · reference)은 .gitignore로 저장소에서 뺀다.
"""
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
BACKUP = Path.home() / "datasets" / "crack-seg-figs-fullres"
MAX_W = 1280
PHOTO_FIGS = [
    "week1/cases/*.png", "week1/contact_sheets/*.png", "week1/examples.png", "week1/hard_negatives.png", "week1/samples9.png",
    "week2/annotate/painted_check/*.png",   # painted/(그림판 원본)는 변환 입력이라 원래 크기 유지 → .gitignore로 제외
    "week2/annotation_compare_*.png", "week2/annotation_result.png",
    "week3/*_cases.png", "week3/adv3_hn_samples.png", "week3/guideline_v03_*.png", "week3/error_cases*/*/*.jpg",
]


def main():
    n = 0
    for pat in PHOTO_FIGS:
        for p in sorted(ROOT.glob(pat)):
            b = BACKUP / p.relative_to(ROOT)
            im = Image.open(p)
            if not b.exists() or im.width > MAX_W:         # 원래 크기 파일 = 새로 만든 그림 → 백업도 새로
                b.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, b)
            if im.width <= MAX_W:
                continue
            h = round(im.height * MAX_W / im.width)
            im = im.convert("RGB").resize((MAX_W, h), Image.LANCZOS)
            if p.suffix.lower() == ".png":
                im.save(p, optimize=True)
            else:
                im.save(p, quality=88)
            n += 1
            print(f"{p.relative_to(ROOT).as_posix()} → {MAX_W}×{h}")
    print(f"{n}개 줄임 (백업: {BACKUP})")


if __name__ == "__main__":
    main()
