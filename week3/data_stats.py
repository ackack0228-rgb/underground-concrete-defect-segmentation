"""클래스별 데이터 통계 (강의 3주차 9쪽 '클래스별 성능 차이를 데이터 수 · 결함 크기 · 경계 명확성으로'): → data_stats.md

  python week3/data_stats.py
분할(2주차, crack-seg-split) 라벨 폴리곤으로 split · 클래스마다 영상 수 · 인스턴스 수 · 인스턴스 면적(영상 대비 %) ·
'가늘기'(면적 / 둘레² × 4π = 원 1, 가는 띠일수록 0에 가까움)를 센다. train은 증강본 포함 / 원본만 두 줄.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import numpy as np

from w3 import CLASS_LIST, SPLIT_ROOT, WEEK3, common


def poly_stats(v):
    pts = np.array(v, float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))       # 정규화 좌표 → 영상 대비 비율
    per = np.sqrt((np.diff(np.r_[x, x[0]]) * 16) ** 2 + (np.diff(np.r_[y, y[0]]) * 9) ** 2).sum()   # 16:9 화면 비율로 둘레
    return area, 4 * np.pi * (area * 144) / per ** 2 if per else 0


def collect(split, originals_only=False):
    S = {n: {"imgs": 0, "inst": 0, "area": [], "round": []} for n in CLASS_LIST}
    n_img = 0
    for lab in sorted((SPLIT_ROOT / "labels" / split).glob("*.txt")):
        if originals_only and common.is_augmented(lab.stem):
            continue
        n_img += 1
        seen = set()
        for line in lab.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) < 7:
                continue
            n = CLASS_LIST[int(v[0])]
            a, r = poly_stats(v[1:])
            S[n]["inst"] += 1
            S[n]["area"].append(a * 100)
            S[n]["round"].append(r)
            seen.add(n)
        for n in seen:
            S[n]["imgs"] += 1
    return n_img, S


def main():
    L = ["# 클래스별 데이터 통계 (분할 후, 라벨 폴리곤 기준)", "",
         "- 면적 = 인스턴스 폴리곤 넓이 / 영상 넓이(%) · 둥근 정도 = 4π·면적/둘레² (원 = 1, 가늘고 긴 띠일수록 0에 가까움)", "",
         "| split | 영상 수 | 클래스 | 정답 영상 수 | 인스턴스 수 | 면적 중앙값(%) | 면적 범위(%) | 둥근 정도 중앙값 |",
         "|---|---:|---|---:|---:|---:|---|---:|"]
    for split, orig in (("train", False), ("train", True), ("val", False), ("test", False)):
        n_img, S = collect(split, orig)
        for n in CLASS_LIST:
            a = np.array(S[n]["area"])
            L.append(f"| {split}{' (원본만)' if orig else ''} | {n_img} | {n} | {S[n]['imgs']} | {S[n]['inst']} | "
                     f"{np.median(a):.2f} | {a.min():.2f} \\~ {a.max():.1f} | {np.median(S[n]['round']):.2f} |")
    L += ["", "- 읽는 법: crack은 면적이 작고 둥근 정도가 가장 낮아(가는 띠) 경계 몇 픽셀 차이가 IoU에 크게 영향을 준다. "
          "spall_detach는 덩어리 모양이라 경계가 비교적 분명하다. 해석은 README '성능 분석'에 씀."]
    (WEEK3 / "data_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
