"""2주차 단계1 (강의 9쪽) — 그림판으로 직접 주석한 영상을 YOLOv8 Segmentation 라벨로 바꾼다.

노트북 환경이라 CVAT/Roboflow 대신 그림판으로 주석했다 (강의 9쪽 "수업 환경에 맞는 도구 선택").
annotate/images/ 의 영상을 그림판에서 열어 결함의 테두리를 그리거나 칠한다 (그림판 기본 팔레트 색이면 된다).
  crack = 빨강 계열 · efflorescence = 파랑/하늘 계열 · spall_detach = 노랑 계열
  - 결함 하나 = 테두리 하나(또는 칠한 덩어리 하나). 띠 모양은 양쪽 경계선 두 개로 그려도 된다.
  - 영상 가장자리에 걸친 결함은 테두리를 가장자리까지 그리면 된다 (40px 안에서 끝나면 가장자리에 닿은 것으로 본다).

어디에 저장해도 찾는다 — 파일 이름에 원래 영상 이름(예: URT002_2021_01_10_001_1636)이 들어 있으면 된다:
  annotate/painted/ · 바탕 화면 · annotate/images/ (png로 저장, 또는 jpg 위에 그대로 덮어써도 됨)
같은 영상의 그림이 여러 개면 가장 최근에 저장한 것을 쓴다. 원본은 분할 폴더(crack-seg-split)의 jpg를 읽는다.

처리: 원본과 달라진 픽셀 중 채도가 높은 픽셀 → 색상(hue)으로 클래스 판정 → 클래스별로 테두리 안쪽을 채움
  → 이어진 덩어리 하나 = 인스턴스 하나 → 외곽선 → approxPolyDP(1px) → 정규화 좌표
  python week2/step1_painted.py → annotate/painted/<이름>.png (모은 그림) · annotate/export/<이름>.txt (라벨)
                                  annotate/painted_check/<이름>.png (확인용) · annotate/painted_report.txt (결과 · 경고)
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import csv
from pathlib import Path

import cv2
import numpy as np

from paths import CLASS_NAMES, SPLIT_ROOT, WEEK2

ANN = WEEK2 / "annotate"
PAINTED, IMAGES, EXPORT, CHECK = ANN / "painted", ANN / "images", ANN / "export", ANN / "painted_check"
ORIG_DIR = SPLIT_ROOT / "images" / "train"
DESKTOPS = [Path.home() / "OneDrive" / "바탕 화면", Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop"]
DRAW_BGR = {0: (60, 60, 255), 1: (255, 200, 60), 2: (0, 200, 255)}
DIFF_MIN = 40      # 원본과 이만큼 이상 달라진 픽셀만 (저장할 때 생기는 미세한 차이 무시)
SAT_MIN = 70       # 채도(0~255)가 이 이상인 픽셀만 그림으로 본다 — 콘크리트 · 백태는 회색/흰색이라 채도가 낮다
HUE = {0: [(0, 12), (165, 180)], 2: [(18, 40)], 1: [(85, 135)]}   # OpenCV hue(0~180): 빨강 · 노랑 · 파랑/하늘
MIN_AREA = 50      # 이보다 작은 덩어리는 실수로 찍힌 점으로 보고 버림
EDGE_GAP = 40      # 가장자리에서 이만큼 안에서 끝난 테두리 선은 가장자리에 닿은 것으로 본다


def imread(p):
    a = np.fromfile(str(p), np.uint8)
    return cv2.imdecode(a, cv2.IMREAD_COLOR) if a.size else None


def paint_mask(img, orig):
    changed = np.abs(img.astype(int) - orig.astype(int)).max(2) >= DIFF_MIN
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = changed & (s >= SAT_MIN) & (v >= 60)
    out = {}
    for cls, ranges in HUE.items():
        m = np.zeros(h.shape, bool)
        for lo, hi in ranges:
            m |= (h >= lo) & (h <= hi)
        out[cls] = (vivid & m).astype(np.uint8)
    other = vivid & ~np.any(np.stack(list(out.values())), 0).astype(bool)
    return out, int(other.sum())


def fill_outline(comp):
    """테두리 안쪽을 채운다. bbox 안에서 선이 아닌 영역 중, 영상 가장자리가 아닌 bbox 변에 닿지 않는 영역 = 안쪽.
    칠해서 채운 덩어리는 안쪽 영역이 없어 그대로 나온다."""
    H, W = comp.shape
    comp = comp.copy()
    ys, xs = np.nonzero(comp)
    for y, x in zip(ys, xs):          # 가장자리 가까이에서 끝난 선은 가장자리까지 이어 붙인다
        if x < EDGE_GAP:
            comp[y, :x] = 1
        if x >= W - EDGE_GAP:
            comp[y, x:] = 1
        if y < EDGE_GAP:
            comp[:y, x] = 1
        if y >= H - EDGE_GAP:
            comp[y:, x] = 1
    ys, xs = np.nonzero(comp)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sub = comp[y0:y1 + 1, x0:x1 + 1]
    n, lab = cv2.connectedComponents((sub == 0).astype(np.uint8), connectivity=4)
    out = sub.copy()
    for k in range(1, n):
        r, c = np.nonzero(lab == k)
        open_edge = ((r.min() == 0 and y0 > 0) or (r.max() == sub.shape[0] - 1 and y1 < H - 1)
                     or (c.min() == 0 and x0 > 0) or (c.max() == sub.shape[1] - 1 and x1 < W - 1))
        if not open_edge:
            out[lab == k] = 1
    full = np.zeros_like(comp)
    full[y0:y1 + 1, x0:x1 + 1] = out
    return full


def instances(img, orig):
    """→ ([(cls, poly(N,2), 넓이, 선 픽셀 수)], 인식 못한 색 픽셀 수)"""
    masks, other = paint_mask(img, orig)
    res = []
    for cls, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # 끊긴 선 이어 붙임
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < MIN_AREA:
                mask[lab == k] = 0
        if not mask.any():
            continue
        filled_all = fill_outline(mask)     # 같은 색 선 전체를 함께 채움 (양쪽 경계선 두 개로 그린 띠도 채워짐)
        n, lab = cv2.connectedComponents(filled_all, connectivity=8)
        for k in range(1, n):
            filled = (lab == k).astype(np.uint8)
            if filled.sum() < MIN_AREA:
                continue
            cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
            if len(c) >= 3:
                res.append((cls, c, int(filled.sum()), int((mask & filled).sum())))
    return res, other


def find_sources(stem):
    """이 영상의 그림 후보 (경로, 수정 시각). 이름에 stem이 들어 있는 png/jpg."""
    cands = []
    for d in [PAINTED, IMAGES] + [d for d in DESKTOPS if d.exists()]:
        for p in d.glob(f"*{stem}*"):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp") and p.is_file():
                cands.append(p)
    return sorted(set(cands), key=lambda p: p.stat().st_mtime, reverse=True)


def main():
    for d in (PAINTED, EXPORT, CHECK):
        d.mkdir(exist_ok=True)
    selected = [Path(r["file"]).stem for r in csv.DictReader(open(ANN / "selected.csv", encoding="utf-8-sig"))]
    for old in EXPORT.glob("*.txt"):       # 이전 실행 결과를 지우고 새로 쓴다 (그림을 지우면 라벨도 빠짐)
        old.unlink()
    for old in CHECK.glob("*.png"):
        old.unlink()
    R, done = [], 0
    for stem in selected:
        orig = imread(ORIG_DIR / f"{stem}.jpg")
        assert orig is not None, f"원본 없음: {ORIG_DIR / (stem + '.jpg')}"
        chosen, notes = None, []
        for p in find_sources(stem):
            img = imread(p)
            if img is None:
                continue
            if img.shape != orig.shape:
                notes.append(f"  · 건너뜀 {p.name}: 크기 {img.shape[1]}×{img.shape[0]} ≠ 원본 {orig.shape[1]}×{orig.shape[0]}")
                continue
            masks, other = paint_mask(img, orig)
            if sum(int(m.sum()) for m in masks.values()) + other < MIN_AREA:
                continue                                    # 그린 흔적 없음 (원본 그대로)
            chosen = (p, img)
            break
        if chosen is None:
            R += [f"- {stem}: 그림 없음"] + notes
            continue
        p, img = chosen
        if p.parent != PAINTED or p.name != f"{stem}.png":
            cv2.imencode(".png", img)[1].tofile(str(PAINTED / f"{stem}.png"))   # 모아 두기 (원본 파일은 그대로)
        H, W = orig.shape[:2]
        inst, other = instances(img, orig)
        lines, vis, over = [], orig.copy(), orig.copy()
        R.append(f"- {stem}: 인스턴스 {len(inst)}개  (사용한 그림: {p})")
        for cls, poly, area, line_px in inst:
            lines.append(f"{cls} " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in poly))
            cv2.fillPoly(over, [poly], DRAW_BGR[cls])
            cv2.polylines(vis, [poly], True, DRAW_BGR[cls], 3)
            R.append(f"    {CLASS_NAMES[cls]}: 꼭짓점 {len(poly)} · 넓이 {area:,}px")
            width = area / max(cv2.arcLength(poly.reshape(-1, 1, 2).astype(np.int32), True) / 2, 1)   # 평균 두께(px)
            if cls != 0 and area < 1.5 * line_px and width < 15:   # 안쪽이 안 채워지고 가는 선만 남음
                R.append(f"    ⚠ 확인 필요: {CLASS_NAMES[cls]} 테두리가 닫히지 않아 안쪽이 안 채워졌을 수 있음 (선만 남음)")
        if other >= 200:
            R.append(f"    ⚠ 확인 필요: 빨강 · 노랑 · 파랑 계열이 아닌 색으로 그린 픽셀 {other:,}개는 무시함")
        if not inst:
            R.append("    ⚠ 확인 필요: 그린 흔적은 있지만 인스턴스를 만들지 못함")
        R += notes
        (EXPORT / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        vis = cv2.addWeighted(over, 0.35, vis, 0.65, 0)
        cv2.imencode(".png", cv2.resize(vis, (W // 2, H // 2)))[1].tofile(str(CHECK / f"{stem}.png"))
        done += 1
    head = [f"직접 주석 변환 결과: {done}/{len(selected)}장", ""]
    text = "\n".join(head + R) + "\n"
    (ANN / "painted_report.txt").write_text(text, encoding="utf-8")
    print(text)
    if done == 0:
        raise SystemExit("그린 영상을 하나도 찾지 못했습니다.")


if __name__ == "__main__":
    main()
