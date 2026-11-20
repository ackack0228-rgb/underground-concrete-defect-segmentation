"""2주차 단계5 (강의 13쪽): YOLOv8s-Seg 미세조정. 설정은 강의 코드 = 논문 Table 1 그대로.

  python week2/step5_train.py            # 60 epoch 전체 학습 → runs/step3/ (weights/best.pt, results.csv, results.png)
  python week2/step5_train.py --quick    # 수업 시간용: 3 epoch만 돌려 loss가 출력되는지 확인 → runs/step3_quick/
  python week2/step5_train.py --resume   # 중간에 끊겼을 때 runs/step3/weights/last.pt에서 이어서
  python week2/step5_train.py --batch 2  # CUDA out of memory가 나면 (강의 16쪽)
학습에 걸린 시간은 runs/<name>/train_time.txt 에 기록한다 (강의 18쪽 '학습에 걸린 시간을 기록').

★ Windows에서는 데이터 로더 작업 프로세스가 이 파일을 다시 import 한다. 학습 코드를 if __name__ == "__main__": 안에
  두지 않으면 "An attempt has been made to start a new process before ..." 오류로 멈춘다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import argparse
import time
from datetime import datetime

import torch
from ultralytics import YOLO

from paths import SPLIT_ROOT, WEEK2

ROOT = WEEK2.parent                                  # crack-seg/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    name = "step3_quick" if args.quick else "step3"

    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    t0 = time.time()
    start = datetime.now()

    if args.resume:
        model = YOLO(ROOT / "runs" / name / "weights" / "last.pt")
        model.train(resume=True)
    else:
        # 사전학습된 YOLOv8s-Seg 모델 불러오기
        model = YOLO(ROOT / "yolov8s-seg.pt")
        # 모델 학습
        model.train(
            data=str(SPLIT_ROOT / "data.yaml"),   # 데이터셋 설정 파일 (단계4가 만든 원본 ID 단위 분할)
            epochs=3 if args.quick else 60,       # 전체 학습 횟수
            imgsz=960,                            # 입력 이미지 크기 (letterbox)
            batch=args.batch,                     # GPU 메모리 부족 시 2로 조정
            optimizer="AdamW",                    # 옵티마이저
            lr0=0.00015,                          # 초기 학습률
            mosaic=0.0,                           # Mosaic 증강 비활성화
            mixup=0.0,                            # MixUp 증강 비활성화
            patience=20,                          # Early Stopping 기준
            seed=42,                              # 재현성을 위한 시드 고정
            project=str(ROOT / "runs"),
            name=name,
            exist_ok=True,                        # 다시 돌리면 같은 폴더에 덮어쓴다 (step32 · step33 … 이 생기지 않게)
        )

    sec = time.time() - t0
    out = ROOT / "runs" / name / "train_time.txt"
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"{'resume' if args.resume else 'start'} {start:%Y-%m-%d %H:%M} → {datetime.now():%Y-%m-%d %H:%M}, "
                f"{sec / 3600:.2f}시간 ({sec / 60:.1f}분), batch={args.batch}, GPU={torch.cuda.get_device_name(0)}\n")
    print("학습 시간 기록:", out)


if __name__ == "__main__":
    main()
