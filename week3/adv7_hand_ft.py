"""3주차 심화 ⑦ (추가): 2주차에 직접 그린 주석 2장으로 기준 모델(best.pt)을 짧게 미세조정해 본다.

  python week3/adv7_hand_ft.py --variant hand       # 2장 = 직접 주석 라벨  → runs/ft_hand/
  python week3/adv7_hand_ft.py --variant provided   # 2장 = 제공 라벨(대조군) → runs/ft_provided/
  python week3/adv7_hand_ft.py --variant hand --seed 1   # 반복 확인 → runs/ft_hand_s1/
  그 뒤 adv_eval_run.py --run ft_hand --tag ft_hand / --run ft_provided --tag ft_provided 로 본 실험과 같은 절차로 평가.

설계 (두 변형은 '2장의 라벨 파일'만 다르고 나머지는 전부 같다 → 차이 = 직접 주석의 효과):
  · 시작 가중치 = runs/step3/weights/best.pt (2주차 epoch 21, 제공 라벨로 학습한 기준 모델)
  · 학습 영상 = 대상 2장 × COPIES장(하드링크 복사, 과대 표집) + train에서 무작위 REPLAY장(seed 0, 두 변형 동일 · 잊어버림 방지)
  · val = 2주차 분할의 val 40장 그대로, test는 학습 · 선택에 쓰지 않음
  · 설정 = 2주차 단계5와 같은 imgsz 960 · batch 4 · AdamW lr0 0.00015 · Mosaic/MixUp 끔, 다만 짧게 EPOCHS epoch · warmup 없음
대상 2장은 train에 있던 영상이라, 이 2장 자체의 맞춤 정도는 성능 근거가 아니다(평가는 test 40장).
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import argparse
import os
import random
import shutil
import time

from ultralytics import YOLO

from w3 import BEST_PT, ROOT, SPLIT_ROOT

FT_ROOT = SPLIT_ROOT.parent / "crack-seg-ft"
HAND_DIR = ROOT / "week2" / "annotate" / "export"          # 2주차 직접 주석(step1_painted.py 결과)
TARGETS = ["URT002_2021_01_10_001_1636", "URT004_2021_02_05_091_19626"]
COPIES, REPLAY, EPOCHS = 10, 98, 10


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(variant):
    d = FT_ROOT / variant
    if d.exists():
        shutil.rmtree(d)
    (d / "images" / "train").mkdir(parents=True)
    (d / "labels" / "train").mkdir(parents=True)
    tr_img, tr_lab = SPLIT_ROOT / "images" / "train", SPLIT_ROOT / "labels" / "train"

    pool = sorted(p.stem for p in tr_img.glob("*.jpg") if p.stem not in TARGETS)
    replay = random.Random(0).sample(pool, REPLAY)
    for s in replay:
        link(tr_img / f"{s}.jpg", d / "images" / "train" / f"{s}.jpg")
        lab = tr_lab / f"{s}.txt"
        if lab.exists():
            link(lab, d / "labels" / "train" / f"{s}.txt")
    for s in TARGETS:
        lab = (HAND_DIR if variant == "hand" else tr_lab) / f"{s}.txt"
        for k in range(COPIES):
            link(tr_img / f"{s}.jpg", d / "images" / "train" / f"{s}_c{k}.jpg")
            shutil.copy2(lab, d / "labels" / "train" / f"{s}_c{k}.txt")

    (d / "data.yaml").write_text(
        f"path: {d.as_posix()}\ntrain: images/train\nval: {(SPLIT_ROOT / 'images' / 'val').as_posix()}\n"
        "names:\n  0: crack\n  1: efflorescence\n  2: spall_detach\n", encoding="utf-8")
    (d / "replay.txt").write_text("\n".join(replay) + "\n", encoding="utf-8")
    return d / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["hand", "provided"], required=True)
    ap.add_argument("--seed", type=int, default=42)   # 학습 시드만 바꿈(재생 98장 선택은 seed 0 고정)
    args = ap.parse_args()
    name = f"ft_{args.variant}" + ("" if args.seed == 42 else f"_s{args.seed}")
    data = build(args.variant)
    t0 = time.time()
    YOLO(BEST_PT).train(
        data=str(data), epochs=EPOCHS, imgsz=960, batch=4, optimizer="AdamW", lr0=0.00015, warmup_epochs=0,
        mosaic=0.0, mixup=0.0, patience=EPOCHS, seed=args.seed, deterministic=True,
        project=str(ROOT / "runs"), name=name, exist_ok=True,
    )
    with open(ROOT / "runs" / name / "train_time.txt", "a", encoding="utf-8") as f:
        f.write(f"{(time.time() - t0) / 3600:.2f} h\n")


if __name__ == "__main__":
    main()
