"""연산 복잡도 (논문강의 43쪽 한계 04 '연산복잡도 보고'): 파라미터 수 · GFLOPs → flops.md

  python week3/flops.py
torch.utils.flop_counter로 같은 입력 크기에서 잰다 (곱셈 + 덧셈 = 2 FLOPs). 추론 시간은 adv2_unet_eval.py(GPU)와 adv1_tile.py.
YOLO 학습 로그의 39.9 GFLOPs는 640×640 기준이고, 960×960으로 바꾸면 (960/640)² 배 = 89.8 로 이 스크립트 값과 같다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json

import torch
from torch.utils.flop_counter import FlopCounterMode
from ultralytics import YOLO

from adv2_unet_train import make_model
from w3 import ROOT, WEEK3


def gflops(m, shape):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(torch.zeros(*shape))
    return fc.get_total_flops() / 1e9


def main():
    yolo = YOLO(ROOT / "runs" / "step3" / "weights" / "best.pt").model.float().eval()
    unet = make_model(pretrained=False).eval()
    ms = {"unet": json.loads((WEEK3 / "adv2_unet_aug.json").read_text(encoding="utf-8"))}
    rows = [("YOLOv8s-Seg", yolo, (1, 3, 544, 960), f"{ms['unet']['yolo']['ms']:.0f}"),
            ("YOLOv8s-Seg (학습 입력 960×960 letterbox)", yolo, (1, 3, 960, 960), "-"),
            ("U-Net (resnet34)", unet, (1, 3, 544, 960), f"{ms['unet']['unet']['ms']:.0f}")]
    L = ["# 연산 복잡도 · 추론 시간", "",
         "| 모델 | 입력 | 파라미터 | GFLOPs | 1장 추론 시간 (GPU, 전처리 포함) |", "|---|---|---:|---:|---:|"]
    for name, m, shape, t in rows:
        L.append(f"| {name} | {shape[3]}×{shape[2]} | {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M | {gflops(m, shape):.1f} | {t + ' ms' if t != '-' else '-'} |")
    import re
    t = re.search(r"전체 (\d+) ms · 타일 9장 (\d+) ms", (WEEK3 / "adv1_tile.md").read_text(encoding="utf-8"))
    L.append(f"| YOLOv8s-Seg 타일 9장 (심화 ①) | 960×540 × 9 | 11.8M | 약 9배 | {t.group(2)} ms (같은 측정의 전체 영상 {t.group(1)} ms) |")
    L += ["", "- GFLOPs = torch.utils.flop_counter (곱셈 + 덧셈 = 2). 시간 = RTX 3060 Laptop, 테스트 영상 20장 평균(워밍업 3장), 학습이 끝난 뒤 GPU가 한가할 때 잼."]
    (WEEK3 / "flops.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
