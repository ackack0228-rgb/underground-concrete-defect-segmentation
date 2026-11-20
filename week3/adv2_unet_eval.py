"""3주차 심화 ② 평가: U-Net(runs/unet) vs YOLOv8s-Seg(runs/step3) — 같은 테스트셋 · 같은 절차.

  python week3/adv2_unet_eval.py      # adv1_tile.py를 먼저 실행해 둘 것 (YOLO 예측 · 픽셀 IoU를 거기서 가져온다)
→ adv2_unet.md · adv2_unet.png · adv2_unet_cases.png
공통 지표 (adv_pix.py 설명 참고):
  ① 영상 단위 ON/OFF — U-Net은 인스턴스 신뢰도가 없어서 영상 점수를 따로 만든다. 임계값은 검증셋에서 고르고 테스트에 고정 적용.
     ★ 주 지표 = '확률 0.5 이상 픽셀의 면적 비율'(AGRID). 2026-11-20 검수: 처음 쓴 '확률 지도 최댓값(5×5 평균)'은 포화되어
     (val 정답 영상 최소 0.9992, 결함 없는 영상도 0.999 ~ 1.0) 0.1 ~ 0.99 임계값이 사실상 같은 판정 → 비교용 행으로만 남김(UTHRS).
     YOLO는 본 실험 결과 그대로.
  ② 클래스별 픽셀 IoU (960×540, 테스트 전체 합산). U-Net 마스크 = 확률 0.5 이상(학습 때 검증 mIoU를 잰 기준).
  ③ 파라미터 수 · 1장 추론 시간(GPU, 같은 영상, 각각 워밍업 후 평균)
mask mAP(인스턴스 단위)는 U-Net이 인스턴스를 내지 않아 같은 방식으로 비교할 수 없다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from adv2_unet_train import H, W, SegData, make_model
from adv_pix import PH, PW, IoUAcc, gt_mask, select_then_test
from w3 import CLASS_LIST, IMGSZ, PRED_CONF, ROOT, WEEK3, load_model, load_split

UTHRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]          # 최댓값 점수(비교용)
AGRID = [1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]              # 면적 비율 점수(주 지표). 1e-6 = 픽셀 하나라도


def unet_probs(model, path):
    img = cv2.imread(path)
    x = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    x = (x - SegData.MEAN) / SegData.STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        p = torch.sigmoid(model(x).float())[0]
    return p.cpu().numpy()                                  # (3, 544, 960)


def main():
    import sys
    run = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "unet"   # 2026-11-19: unet_aug 비교용
    sfx = "" if run == "unet" else "_" + run.removeprefix("unet_")
    dev = "cuda"
    unet = make_model(pretrained=False).to(dev).eval()
    ck = torch.load(ROOT / "runs" / run / "weights" / "best.pt", map_location=dev, weights_only=False)
    unet.load_state_dict(ck["model"])
    best_ep = ck["epoch"] + 1
    hist = pd.read_csv(ROOT / "runs" / run / "results.csv")

    val_imgs, val_gts = load_split("val")
    test_imgs, test_gts = load_split("test")
    probs = {}
    for p in val_imgs + test_imgs:
        pr = unet_probs(unet, p)
        sm = np.stack([cv2.blur(c, (5, 5)) for c in pr])
        probs[p] = (pr, sm.reshape(3, -1).max(1), (pr >= 0.5).reshape(3, -1).mean(1))
    thr_max, _, test_max = select_then_test([probs[p][1] for p in val_imgs], val_gts,
                                            [probs[p][1] for p in test_imgs], test_gts, UTHRS)
    thr, curves, test = select_then_test([probs[p][2] for p in val_imgs], val_gts,
                                         [probs[p][2] for p in test_imgs], test_gts, AGRID)
    acc = IoUAcc()
    for p in test_imgs:
        m = np.stack([cv2.resize((c >= 0.5).astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST) for c in probs[p][0]])
        acc.add(m.astype(bool), gt_mask("test", p, PW, PH))
    u_iou = acc.iou()

    tile = json.loads((WEEK3 / "adv1_tile.json").read_text(encoding="utf-8"))["full"]
    base = json.loads((WEEK3 / "adv_runs" / "seed42.json").read_text(encoding="utf-8"))
    y_iou = tile["pixel_iou"]

    # 파라미터 · 속도 (테스트 영상 20장, 워밍업 3장)
    yolo = load_model()
    n_y = sum(p.numel() for p in yolo.model.parameters())
    n_u = sum(p.numel() for p in unet.parameters())

    def timeit(fn):
        for p in test_imgs[:3]:
            fn(p)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for p in test_imgs[:20]:
            fn(p)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 20 * 1000
    t_y = timeit(lambda p: yolo.predict(p, imgsz=IMGSZ, conf=PRED_CONF, verbose=False))
    t_u = timeit(lambda p: unet_probs(unet, p))

    out = {"unet": {"thresholds": thr, "test": test, "macro_f1": float(np.mean([test[n]["f1"] for n in CLASS_LIST])),
                    "pixel_iou": u_iou, "best_epoch": best_ep, "params": n_u, "ms": t_u, "score": "mask area fraction (prob>=0.5)",
                    "val_curves": curves, "grid": AGRID,
                    "maxprob": {"thresholds": thr_max, "test": test_max,
                                "macro_f1": float(np.mean([test_max[n]["f1"] for n in CLASS_LIST]))}},
           "yolo": {"thresholds": base["thresholds"], "test": base["test"], "macro_f1": base["test_macro_f1"],
                    "pixel_iou": y_iou, "params": n_y, "ms": t_y}}
    (WEEK3 / f"adv2_unet{sfx}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# 심화 ② 모델 비교: YOLOv8s-Seg vs U-Net", "",
         f"- U-Net: resnet34 인코더(ImageNet), 960×544, BCE + Dice, {len(hist)} epoch 학습(조기 종료 patience 10), "
         f"best = epoch {best_ep} (검증 mIoU {hist['mIoU'].max():.3f})",
         f"- 학습 설정: AdamW lr 1e-4 + 기울기 클리핑 · 배치 합산 Dice · 증강 {'강함(scale · translate · HSV)' if run == 'unet_aug' else '약함(좌우 뒤집기 · 밝기)'}",
         "- 같은 분할(2주차) · 같은 테스트셋. 임계값은 두 모델 모두 검증셋에서 고르고 테스트에 고정 적용",
         "- U-Net 영상 점수 = 확률 0.5 이상 픽셀의 면적 비율 (확률 최댓값은 포화돼 임계값이 의미 없어 비교용 행으로만 둠)", "",
         "| 지표 | YOLOv8s-Seg | U-Net |", "|---|---:|---:|",
         f"| 파라미터 수 | {n_y / 1e6:.1f}M | {n_u / 1e6:.1f}M |",
         f"| 1장 추론 시간 (GPU, 전처리 포함) | {t_y:.0f} ms | {t_u:.0f} ms |",
         "| 출력 | 인스턴스(마스크 + 신뢰도) | 픽셀별 클래스 확률 |",
         "| 선택 임계값 (crack / efflo / spall; U-Net = 면적 비율) | " + " / ".join(f"{base['thresholds'][n]:g}" for n in CLASS_LIST) +
         " | " + " / ".join(f"{thr[n]:g}" for n in CLASS_LIST) + " |"]
    for n in CLASS_LIST:
        L.append(f"| 테스트 ON/OFF F1 {n} (FP / FN) | {base['test'][n]['f1']:.3f} ({base['test'][n]['fp']} / {base['test'][n]['fn']}) | "
                 f"{test[n]['f1']:.3f} ({test[n]['fp']} / {test[n]['fn']}) |")
    L.append(f"| 테스트 macro F1 | {base['test_macro_f1']:.3f} | {out['unet']['macro_f1']:.3f} |")
    L.append(f"| (비교용) U-Net 확률 최댓값 점수: 임계값 · macro F1 | - | "
             + " / ".join(f"{thr_max[n]:g}" for n in CLASS_LIST) + f" · {out['unet']['maxprob']['macro_f1']:.3f} |")
    for n in CLASS_LIST:
        L.append(f"| 테스트 픽셀 IoU {n} | {y_iou[n]:.3f} | {u_iou[n]:.3f} |")
    L.append(f"| 테스트 픽셀 IoU 평균 | {np.mean(list(y_iou.values())):.3f} | {np.mean(list(u_iou.values())):.3f} |")
    L.append(f"| 테스트 mask mAP50 | {base['seg_test']['mask_mAP50']:.3f} | (인스턴스 없음 — 계산 불가) |")
    L += ["", "- 한 번씩만 학습한 비교라 시드 변동(⑤)보다 작은 차이는 의미를 두기 어렵다. U-Net은 발산 · 붕괴 때문에 손실 · 학습률을 두 번 고쳤고(runs/_unet_*), 증강 두 가지(unet · unet_aug) 중 최종은 검증 mIoU로 고른다."]
    (WEEK3 / f"adv2_unet{sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # 그림 1: 학습 곡선 + 테스트 비교
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.8))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train loss")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val loss")
    a2 = ax[0].twinx()
    a2.plot(hist["epoch"], hist["mIoU"], color="k", ls="--", label="val mIoU")
    a2.set_ylabel("val mIoU")
    ax[0].axvline(best_ep, color="gray", ls=":")
    ax[0].set_xlabel("epoch")
    ax[0].set_title("U-Net 학습 곡선 (점선 = best epoch)")
    ax[0].legend(loc="upper center", fontsize=8)
    a2.legend(loc="upper right", fontsize=8)
    x = np.arange(4)
    for j, (lab, f1, iou) in enumerate([("YOLOv8s-Seg", base["test"], y_iou), ("U-Net", test, u_iou)]):
        v = [f1[n]["f1"] for n in CLASS_LIST] + [np.mean([f1[n]["f1"] for n in CLASS_LIST])]
        b = ax[1].bar(x + (j - 0.5) * 0.2 - 0.2, v, 0.2, label=f"{lab} F1")
        ax[1].bar_label(b, fmt="%.2f", fontsize=7)
        v = [iou[n] for n in CLASS_LIST] + [np.mean(list(iou.values()))]
        b = ax[1].bar(x + (j - 0.5) * 0.2 + 0.2, v, 0.2, label=f"{lab} 픽셀 IoU", alpha=0.6)
        ax[1].bar_label(b, fmt="%.2f", fontsize=7)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(CLASS_LIST + ["평균"])
    ax[1].set_ylim(0, 1.12)
    ax[1].set_title("테스트: ON/OFF F1 (진한 색) · 픽셀 IoU (옅은 색)")
    ax[1].legend(fontsize=7, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(WEEK3 / f"adv2_unet{sfx}.png", dpi=140)
    plt.close()

    # 그림 2: 클래스별 예시 1장씩 — 정답 / YOLO / U-Net
    colors = {0: (0, 0, 255), 1: (255, 120, 0), 2: (0, 210, 255)}
    pick = [next(p for p, g in zip(test_imgs, test_gts) if c in g) for c in range(3)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 8.6))
    for row, p in zip(axes, pick):
        img = cv2.imread(p)
        small = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_AREA)
        g = gt_mask("test", p, PW, PH)
        r = yolo.predict(p, imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
        ym = np.zeros((3, PH, PW), np.uint8)
        if r.masks is not None:
            for c, f, poly in zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.masks.xy):
                if f >= base["thresholds"][CLASS_LIST[c]] and len(poly) >= 3:
                    cv2.fillPoly(ym[c], [(poly * [PW / img.shape[1], PH / img.shape[0]]).astype(np.int32)], 1)
        um = np.stack([cv2.resize((c >= 0.5).astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST) for c in probs[p][0]])
        for ax, m, lab in zip(row, (g.astype(np.uint8), ym, um), ("정답", "YOLOv8s-Seg", "U-Net")):
            vis = small.copy()
            for c in range(3):
                over = vis.copy()
                over[m[c] > 0] = colors[c]
                vis = cv2.addWeighted(over, 0.45, vis, 0.55, 0)
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{lab} — {Path(p).name[:28]}", fontsize=9)
            ax.axis("off")
    fig.suptitle("빨강 = crack · 파랑 = efflorescence · 노랑 = spall_detach", fontsize=11)
    plt.tight_layout()
    plt.savefig(WEEK3 / f"adv2_unet{sfx}_cases.png", dpi=95)
    plt.close()


if __name__ == "__main__":
    main()
