"""1주차 실습 단계2 — 데이터 확인 및 확보 (강의 10쪽) + 제출물 ② 데이터 요약

제공된 실습용 데이터셋이 images 폴더(common.DATA) 아래에서 검색되는지 확인하고,
영상 수 · 해상도 · 폴더(데이터 구성)를 data_summary.md로 정리한다.
"""
import _use_venv  # noqa: F401  실행 버튼이 다른 가상환경을 골라도 crack-seg/.venv로 다시 실행

from collections import Counter
from pathlib import Path

from common import DATA, WEEK1, imread, is_augmented, list_images, rel, require_images

# ---- 강의 10쪽 코드 (경로만 common.DATA 로 바꿈 — 데이터는 OneDrive 밖에 둠) ----
imgs = list_images(jpg_only=True)
print("영상 수:", len(imgs))
print("예시:", imgs[0] if imgs else "없음")
# ----------------------------------------------------------------

all_imgs = list_images()
if len(all_imgs) != len(imgs):
    print(f"(jpg 외 확장자 포함 시 {len(all_imgs)}장 — 이후 단계는 전체를 사용)")
imgs = all_imgs
require_images(imgs)

# 해상도 · 확장자 · 폴더 구성
sizes, exts, folders, bad = Counter(), Counter(), Counter(), []
portrait_orig = 0   # 세로로 찍힌 원본 영상 수
for p in imgs:
    exts[Path(p).suffix.lower()] += 1
    parent = Path(rel(p)).parent.as_posix()
    folders[parent if parent != "." else "(images 바로 아래)"] += 1
    try:
        h, w = imread(p).shape[:2]
        sizes[(w, h)] += 1
        portrait_orig += h > w and not is_augmented(p)
    except IOError:
        bad.append(rel(p))

out = ["# 데이터 요약 (제출물 ②)", "", f"- 경로: `{DATA.as_posix()}` (검색 패턴 `**/*` 재귀)",
       f"- 영상 수: **{len(imgs)}장**", f"- 열리지 않는 영상: {len(bad)}장", ""]
out += ["## 해상도", "", "| 가로 × 세로 | 영상 수 |", "|---|---|"]
out += [f"| {w} × {h} | {n} |" for (w, h), n in sizes.most_common()]
out += ["", f"- 열리는 {sum(sizes.values())}장 기준 (열리지 않는 {len(bad)}장 제외).",
        (f"- 원본은 모두 가로 영상이고, 세로 영상은 원본을 90° 회전한 증강본(파일명 `rot90`)이다." if portrait_orig == 0
         else f"- 세로 영상 중 원본은 {portrait_orig}장이다.")]
out += ["", "## 데이터 구성 (폴더별)", "", "| 폴더 | 영상 수 |", "|---|---|"]
out += [f"| {f} | {n} |" for f, n in sorted(folders.items())]
# 원본 / 증강본 구성 (파일명 기준: _aug, _efflo_)
comp = Counter((Path(rel(p)).parent.as_posix(), "증강본" if is_augmented(p) else "원본") for p in imgs)
out += ["", "## 데이터 구성 (원본 · 증강본)", "", "| 폴더 | 원본 | 증강본 |", "|---|---|---|"]
out += [f"| {f} | {comp[(f, '원본')]} | {comp[(f, '증강본')]} |" for f in sorted(folders)]
out += ["", "- 파일명에 `_aug` · `_efflo_`가 붙은 영상은 원본을 회전 · 반전 · 밝기 · 대비 · 노이즈로 바꾼 증강본이다. val · test는 원본만 있다."]
out += ["", "## 확장자", "", "| 확장자 | 영상 수 |", "|---|---|"]
out += [f"| {e} | {n} |" for e, n in exts.most_common()]
out += ["", "## 데이터 출처", "",
        "- 교수님 제공 실습용 데이터셋 (지하 콘크리트 구조물 표면 영상, 기준 논문과 같은 4,116장 구성).",
        "- 기준 논문은 AI Hub 공개 데이터에서 고른 4,116장을 썼다. 추가 프로젝트에서는 AI Hub나 다른 공개 데이터셋을 쓸 수 있다 (강의 10쪽)."]
if bad:
    out += ["", "## 열리지 않는 영상", "",
            f"- 받은 압축 파일 안에 크기가 0바이트인 영상이 {len(bad)}장 있다 (다운로드 오류가 아니라 원본 압축 파일 상태). 이후 단계에서는 제외한다.",
            "- 모두 train이고, 원본 1장(`train/URT020_2021_04_03_001_2914.jpg`)과 나머지 증강본이다.", ""]
    out += [f"- {b}" for b in bad]

(WEEK1 / "data_summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print("\n→ week1/data_summary.md 저장")
