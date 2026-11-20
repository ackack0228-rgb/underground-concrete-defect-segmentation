"""3주차 심화 ③ · ⑤ 공용: 2주차 단계5와 같은 설정으로 YOLOv8s-Seg를 다시 학습한다 (seed · 데이터만 바꿈).

  python week3/adv_yolo_train.py --seed 0                         # ⑤ 반복학습 → runs/step3_seed0/
  python week3/adv_yolo_train.py --seed 42 --data hn --name step3_hn   # ③ 하드네거티브 추가 데이터 → runs/step3_hn/
  python week3/adv_yolo_train.py --name step3_seed0 --resume      # 끊겼을 때 이어서
설정은 week2/step5_train.py(= 강의 13쪽 = 논문 Table 1)와 같다: imgsz 960 · batch 4 · AdamW lr0 0.00015 · Mosaic/MixUp 끔
· 60 epoch · patience 20. 기준 실험(seed 42)은 runs/step3/.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import argparse
import time

from ultralytics import YOLO

from w3 import ROOT, SPLIT_ROOT

HN_ROOT = SPLIT_ROOT.parent / "crack-seg-hn"          # 심화 ③ 데이터 (adv3_hn_build.py가 만든다)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", choices=["split", "hn"], default="split")
    ap.add_argument("--name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    name = args.name or f"step3_seed{args.seed}"
    t0 = time.time()

    if args.resume:
        YOLO(ROOT / "runs" / name / "weights" / "last.pt").train(resume=True)
    else:
        data = (HN_ROOT if args.data == "hn" else SPLIT_ROOT) / "data.yaml"
        YOLO(ROOT / "yolov8s-seg.pt").train(
            data=str(data), epochs=60, imgsz=960, batch=args.batch, optimizer="AdamW", lr0=0.00015,
            mosaic=0.0, mixup=0.0, patience=20, seed=args.seed,
            project=str(ROOT / "runs"), name=name, exist_ok=True,
        )
    with open(ROOT / "runs" / name / "train_time.txt", "a", encoding="utf-8") as f:
        f.write(f"{'resume ' if args.resume else ''}{(time.time() - t0) / 3600:.2f} h\n")


if __name__ == "__main__":
    main()
