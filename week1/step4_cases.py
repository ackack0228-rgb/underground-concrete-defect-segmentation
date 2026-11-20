"""1주차 실습 단계4/5 — 사례 캡처 그림 만들기 (제출물 ④ 하드네거티브, ⑤-⑥ 예시 영상)

CSV에 적은 사례를 원본 + 확대(빨간 상자 부분) 그림과 표로 만든다.

  python week1/step4_cases.py                   # hard_negatives.csv → hard_negatives.png / .md
  python week1/step4_cases.py examples.csv      # examples.csv       → examples.png / .md

CSV 열 (첫 줄은 머리글):
  file          images 폴더 기준 경로 (contact_sheets/index.csv 의 file 열, 예: train/xxx.jpg)
  type          사례 종류 (예: 시공이음부, 케이블·배관, 그림자, 정반사, 보수흔적, 표면오염 / crack 등)
  confused_with 혼동되는 결함 (crack · efflorescence · spall_detach). 예시 영상이면 비워 둠
  box           확대할 영역 "x1 y1 x2 y2" (픽셀). 비우면 원본만
  note          한 문장 설명 — 왜 결함처럼 보이고, 왜 결함이 아닌지
"""
import _use_venv  # noqa: F401  실행 버튼이 다른 가상환경을 골라도 crack-seg/.venv로 다시 실행

import csv
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA, WEEK1, imread

csv_name = sys.argv[1] if len(sys.argv) > 1 else "hard_negatives.csv"
csv_path = WEEK1 / csv_name
stem = csv_path.stem
if not csv_path.exists():
    raise SystemExit(f"{csv_path.name} 이 없습니다. 위 docstring의 열 형식으로 만들어 주세요.")

with open(csv_path, encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r.get("file", "").strip()]
if not rows:
    raise SystemExit(f"{csv_path.name} 에 사례가 없습니다.")

fig, axes = plt.subplots(len(rows), 2, figsize=(9, 3.3 * len(rows)), squeeze=False)
has_conf = any(r.get("confused_with", "").strip() for r in rows)   # 예시 영상 표에는 혼동 열이 필요 없다
title = {"hard_negatives": "하드네거티브 (제출물 ④)", "examples": "클래스별 예시 영상 (제출물 ⑤-⑥)"}.get(stem, stem)
md = [f"# {title}", "", "| # | 종류 | 혼동되는 결함 | 영상 | 설명 |" if has_conf else "| # | 클래스 | 영상 | 설명 |",
      "|---|---|---|---|---|" if has_conf else "|---|---|---|---|"]
for i, r in enumerate(rows):
    im = cv2.cvtColor(imread(DATA / r["file"].strip()), cv2.COLOR_BGR2RGB)
    a0, a1 = axes[i]
    a0.imshow(im)
    short = r['file'].split('_jpg.rf.')[0] + '.jpg' if '_jpg.rf.' in r['file'] else r['file']   # Roboflow 해시가 붙은 긴 이름은 줄임
    a0.set_title(f"({i + 1}) {r['type']} — {short}", fontsize=8)
    box = r.get("box", "").split()
    if len(box) == 4:
        x1, y1, x2, y2 = map(int, box)
        a0.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=2))
        a1.imshow(im[y1:y2, x1:x2])
        a1.set_title("확대 (빨간 상자)", fontsize=8)
    conf = r.get("confused_with", "").strip()
    a1.text(0.0, -0.08, (f"혼동되는 결함: {conf}" if conf else ""), transform=a1.transAxes,
            fontsize=8, va="top")
    a0.axis("off"); a1.axis("off")
    md.append(f"| {i + 1} | {r['type']} | {conf or '-'} | `{r['file']}` | {r.get('note', '')} |" if has_conf
              else f"| {i + 1} | {r['type']} | `{r['file']}` | {r.get('note', '')} |")

plt.tight_layout()
plt.savefig(WEEK1 / f"{stem}.png", dpi=100)
plt.close(fig)

# 제출 문서용: 사례마다 한 장씩 (원본 | 확대) — 한 장에 다 넣으면 문서에서 글자가 너무 작아진다
one_dir = WEEK1 / "cases"
one_dir.mkdir(exist_ok=True)
for old in one_dir.glob(f"{stem}_*.png"):
    old.unlink()
for i, r in enumerate(rows):
    im = cv2.cvtColor(imread(DATA / r["file"].strip()), cv2.COLOR_BGR2RGB)
    box = r.get("box", "").split()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.78, 1]})
    a0.imshow(im); a0.axis("off"); a1.axis("off")
    if len(box) == 4:
        x1, y1, x2, y2 = map(int, box)
        a0.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=2.5))
        a1.imshow(im[y1:y2, x1:x2])
    a0.set_title("원본 (빨간 상자)", fontsize=10); a1.set_title("확대", fontsize=10)
    plt.tight_layout()
    plt.savefig(one_dir / f"{stem}_{i + 1}.png", dpi=130)
    plt.close(fig)
(WEEK1 / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
print(f"\n→ week1/{stem}.png, {stem}.md 저장")
