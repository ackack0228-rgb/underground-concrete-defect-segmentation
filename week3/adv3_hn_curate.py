"""3주차 심화 ③ 2차: 마이닝한 하드네거티브 108장을 사람이 검토해 '조각 전체가 결함이 아닌 것'만 남긴다.

  python week3/adv3_hn_build.py     # (먼저) 마이닝 → crack-seg-hn (108장)
  python week3/adv3_hn_curate.py    # 검토 결과 반영 → crack-seg-hn 의 train 배경 조각을 바꿈 · adv3_hn_review.csv · adv3_hn_curate.md
검토 이유: 기준 모델이 결함으로 본 '라벨 없는 조각' 가운데 상당수가 실제로는 라벨이 빠진 진짜 결함이었다
  (거북등 균열 · 박락 덩어리 등). 그런 조각을 배경으로 넣으면 진짜 결함을 무시하도록 가르치게 된다.
검토 기준 (2026-11-16, 조각마다 예측 상자를 그려 놓고 눈으로 판정):
  남김 = 조각 전체에 결함이 안 보이고, 예측이 걸린 곳의 원인(분필 · 시설물 · 모서리 · 그림자)이 분명함
  뺌(실제 결함) = 조각 안에 라벨 안 된 균열 · 박락이 보임 / 뺌(판단 어려움) = 조각 가장자리에 잘린 예측 · 희미한 선 · 무늬
남은 조각 + 1주차에 직접 확인한 train 하드네거티브 4건(상자 영역, 라벨 겹침 0%)을 3번씩 넣는다(오버샘플링).
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import csv
import json

import cv2

from w3 import SPLIT_ROOT, WEEK3, common  # noqa: F401

HN_ROOT = SPLIT_ROOT.parent / "crack-seg-hn"
REPEAT = 3
KEEP = {  # 번호: 원인
    3: "분필 표기 → efflorescence", 7: "분필 표기 → efflorescence", 22: "분필 곡선 → crack · efflorescence",
    27: "분필 곡선 → crack", 5: "이음부 · 금속 사다리 모서리 → spall_detach", 13: "배관 상단 브래킷 → spall_detach",
    31: "케이블 고정구 → spall_detach", 48: "케이블 고정 볼트 → spall_detach", 54: "앵커 구멍 · 케이블 → efflorescence",
    60: "세로 배관 → crack · spall_detach", 71: "앵커 구멍 → efflorescence", 73: "케이블 클램프 → spall_detach",
    90: "케이블 클램프 → spall_detach", 91: "난간 그림자 → spall_detach", 95: "금속 고정구 · 배관 → spall_detach",
}
REAL = {8: "박락 덩어리", 17: "박락 덩어리", 18: "거북등 균열", 29: "균열 · 박락", 35: "박락(들뜸)", 42: "거북등 균열",
        45: "망상 균열", 47: "균열", 51: "망상 균열", 52: "망상 균열", 56: "망상 균열", 57: "균열", 59: "망상 균열"}
WEEK1_TRAIN = [  # 1주차 hard_negatives.csv 중 train 영상 (상자 = x1 y1 x2 y2, 라벨 겹침 0% 확인)
    ("URT007_2022_01_03_001_13166.jpg", (900, 400, 1920, 540), "케이블 · 배관 → crack (1주차)"),
    ("URT002_2021_03_16_001_7052.jpg", (200, 0, 700, 1080), "보수흔적(녹색 보수재) → crack (1주차)"),
    ("URT008_2022_02_04_001_13752.jpg", (150, 0, 1150, 1000), "표면오염(얼룩) → efflorescence (1주차)"),
    # URT017_2020_02_08_001_24512 (분필 표기)는 2주차 누수 처리로 train에서 빠져 있어(테스트에 같은 영상) 쓰지 않는다
]


def main():
    mined = json.loads((WEEK3 / "adv3_hn_list.json").read_text(encoding="utf-8"))["chosen"]
    img_dir, lab_dir = HN_ROOT / "images" / "train", HN_ROOT / "labels" / "train"
    rows = []
    # 이전 배경 조각 모두 지우고 다시 쓴다 (분할 원본 하드링크는 그대로)
    for f in list(img_dir.glob("hn*.jpg")) + list(lab_dir.glob("hn*.txt")):
        f.unlink()
    src_dir = SPLIT_ROOT / "images" / "train"
    n_files = 0
    for i, d in enumerate(mined):
        if i in KEEP:
            decision, reason = "남김", KEEP[i]
        elif i in REAL:
            decision, reason = "뺌(실제 결함)", REAL[i]
        else:
            decision, reason = "뺌(판단 어려움)", "조각 가장자리 예측 · 희미한 선 · 무늬 등"
        rows.append({"no": f"hn{i:03d}", "src": d["src"], "fp_class": d["fp_class"], "fp_conf": d["fp_conf"],
                     "decision": decision, "reason": reason})
        if decision == "남김":
            img = cv2.imread(str(src_dir / d["src"]))
            x0, y0, cw, ch = d["box"]
            for k in range(REPEAT):
                name = f"hn{i:03d}_r{k}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), img[y0:y0 + ch, x0:x0 + cw], [cv2.IMWRITE_JPEG_QUALITY, 95])
                (lab_dir / f"{name}.txt").write_text("", encoding="utf-8")
                n_files += 1
    for j, (src, (x1, y1, x2, y2), reason) in enumerate(WEEK1_TRAIN):
        img = cv2.imread(str(src_dir / src))
        assert img is not None, src
        rows.append({"no": f"hnw1_{j}", "src": src, "fp_class": "-", "fp_conf": "-", "decision": "남김(1주차)", "reason": reason})
        for k in range(REPEAT):
            name = f"hnw1_{j}_r{k}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (lab_dir / f"{name}.txt").write_text("", encoding="utf-8")
            n_files += 1

    with open(WEEK3 / "adv3_hn_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cnt = {k: sum(r["decision"] == k for r in rows) for k in ("남김", "뺌(실제 결함)", "뺌(판단 어려움)", "남김(1주차)")}
    n_train = len(list(img_dir.glob("*.jpg")))
    L = ["# 심화 ③ 하드네거티브 검토 (2차)", "",
         f"- 마이닝 108장을 조각마다 예측 상자를 그려 눈으로 판정: 남김 {cnt['남김']} · 뺌(실제 결함) {cnt['뺌(실제 결함)']} · "
         f"뺌(판단 어려움) {cnt['뺌(판단 어려움)']}",
         "- ★ 기준 모델이 '라벨 없는 곳'에서 높은 신뢰도로 찾은 것 중 적어도 13장은 라벨이 빠진 실제 결함(거북등 · 망상 균열, 박락 덩어리)이었다 "
         "→ 자동 마이닝만으로 배경을 만들면 진짜 결함을 배경으로 가르치게 된다. 주석 누락 점검 도구로도 쓸 수 있다(주석기준서 v0.3 ⑤).",
         f"- 남긴 {cnt['남김']}장 + 1주차 직접 확인 {cnt['남김(1주차)']}건 = {cnt['남김'] + cnt['남김(1주차)']}개 영역 × {REPEAT}번(오버샘플링) "
         f"= 배경 영상 {n_files}장 → train {n_train}장",
         "- 남긴 조각의 원인: " + " / ".join(sorted({r['reason'] for r in rows if r['decision'].startswith('남김')})),
         "- 전체 판정표: adv3_hn_review.csv"]
    (WEEK3 / "adv3_hn_curate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
