"""1주차 실습 단계1 — 환경 구축 (강의 9쪽)

YOLOv8s-Seg 실행 환경을 확인하고 모델이 정상 로드되는지 본다.
처음 실행하면 yolov8s-seg.pt 가중치가 자동으로 내려받아진다(crack-seg/ 폴더에 저장).
결과는 화면과 env_check.txt에 같이 남긴다 (제출물 ① 환경 확인).
"""
import _use_venv  # noqa: F401  실행 버튼이 다른 가상환경을 골라도 crack-seg/.venv로 다시 실행

import os
import platform
import sys

from common import ROOT, WEEK1

os.chdir(ROOT)  # 가중치를 crack-seg/ 에 받도록

lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


# ---- 강의 9쪽 코드 그대로 ----
import torch
from ultralytics import YOLO

log("torch:", torch.__version__)
log("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    log("device:", torch.cuda.get_device_name(0))

model = YOLO("yolov8s-seg.pt")
log("model loaded")
# ---------------------------

# 추가 기록: 버전과 모델 정보 (재현성)
import cv2
import matplotlib
import ultralytics

log("")
log("python:", sys.version.split()[0], "|", platform.platform())
log("ultralytics:", ultralytics.__version__)
log("opencv-python:", cv2.__version__)
log("matplotlib:", matplotlib.__version__)
if torch.cuda.is_available():
    log("torch CUDA build:", torch.version.cuda)
    log("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
log("task:", model.task)
log("classes in pretrained weights:", len(model.names), "(COCO 사전학습 가중치)")
n_params = sum(p.numel() for p in model.model.parameters())
log("parameters:", f"{n_params / 1e6:.1f} M")

(WEEK1 / "env_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n→ week1/env_check.txt 저장")
