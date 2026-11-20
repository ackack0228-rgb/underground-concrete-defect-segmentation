"""3주차 심화 ② (논문강의 47쪽 '모델 비교'): U-Net을 같은 데이터(2주차 분할)로 학습해 YOLOv8s-Seg와 비교.

  python week3/adv2_unet_train.py                  # → runs/unet/ (weights/best.pt · last.pt · results.csv)
  python week3/adv2_unet_train.py --resume         # 끊겼을 때 (train_queue.py가 자동으로 붙임)
- 모델: segmentation_models_pytorch Unet(resnet34, ImageNet 사전학습), 출력 3채널(클래스별 sigmoid — 한 픽셀에 두 결함이 겹칠 수 있음)
- 입력: 960×544 (YOLO imgsz 960과 비슷한 해상도). 세로 영상(rot90 증강본)은 가로로 돌려 넣는다. 라벨 폴리곤 → 클래스별 이진 마스크.
- 손실 BCE + Dice, AdamW lr 1e-4 + 기울기 클리핑 1.0, batch 4, AMP, 최대 40 epoch, patience 10. 모델 선택 = 검증셋 mIoU(세 클래스 평균) 최고 epoch
  (YOLO와 같이 학습 손실이 아니라 검증 성능으로 고름). 증강: 기본 = 좌우 뒤집기 + 밝기 · 대비 조금 (YOLO 기본 증강보다 약함),
  --aug strong = + scale 0.5 ~ 1.5 · translate 0.1 · HSV (YOLO 기본과 비슷, 2026-11-19 과적합 대응 → runs/unet_aug).
- 처음 한 번 영상을 960×544로 줄여 ~/datasets/crack-seg-unet/ 에 저장해 두고 쓴다 (매번 1920×1080 디코딩 방지).
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from w3 import ROOT, SPLIT_ROOT

CACHE = SPLIT_ROOT.parent / "crack-seg-unet"
W, H = 960, 544
SEED = 42


def rasterize(lab, w, h):
    m = np.zeros((h, w, 3), np.uint8)
    if lab.exists():
        for line in lab.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) > 6:
                c = int(v[0])
                pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                ch = np.ascontiguousarray(m[:, :, c])
                cv2.fillPoly(ch, [pts], 255)
                m[:, :, c] = ch
    return m


def build_cache():
    for split in ("train", "val", "test"):
        d = CACHE / split
        d.mkdir(parents=True, exist_ok=True)
        imgs = sorted((SPLIT_ROOT / "images" / split).glob("*.jpg"))
        for p in imgs:
            if (d / (p.stem + ".png")).exists():
                continue
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            m = rasterize(SPLIT_ROOT / "labels" / split / (p.stem + ".txt"), w, h)
            if h > w:                                        # 세로 영상 → 가로로
                img, m = np.rot90(img), np.rot90(m)
            img = cv2.resize(np.ascontiguousarray(img), (W, H), interpolation=cv2.INTER_AREA)
            m = cv2.resize(np.ascontiguousarray(m), (W, H), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(d / (p.stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(d / (p.stem + ".png")), m)       # 마스크: B=crack · G=efflo · R=spall 채널
        print(split, len(imgs), "장 캐시 완료", flush=True)


class SegData(Dataset):
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, split, aug):
        self.items = sorted((CACHE / split).glob("*.jpg"))
        self.aug = aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        m = (cv2.imread(str(p.with_suffix(".png"))) > 127).astype(np.float32)
        if self.aug:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if self.aug == "strong":                         # YOLO 기본 증강과 비슷하게: scale 0.5 · translate 0.1 · HSV
                s = rng.uniform(0.5, 1.5)
                tx, ty = rng.uniform(-0.1, 0.1) * W, rng.uniform(-0.1, 0.1) * H
                M = np.float32([[s, 0, (1 - s) * W / 2 + tx], [0, s, (1 - s) * H / 2 + ty]])
                img = cv2.warpAffine(np.ascontiguousarray(img), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0.447,) * 3)
                m = cv2.warpAffine(np.ascontiguousarray(m), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)       # float: H 0~360, S · V 0~1
                hsv[..., 0] = (hsv[..., 0] + rng.uniform(-0.015, 0.015) * 360) % 360
                hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.3, 1.7), 0, 1)
                hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.6, 1.4), 0, 1)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            else:
                img = np.clip(img * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08), 0, 1)
        img = (img - self.MEAN) / self.STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(m.transpose(2, 0, 1))))


def make_model(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, classes=3)


def dice_loss(logits, target, eps=1.0):
    """클래스별로 배치 전체를 합쳐 Dice (smp DiceLoss 기본과 같은 방식).
    ★ 2026-11-19 수정: 처음엔 영상마다 Dice를 따로 냈는데, crack이 있는 영상이 19%뿐이라 빈 영상에서
    '아무것도 예측 안 함'이 Dice 1(손실 0)이 되어 epoch 3에 crack · efflo IoU가 0으로 무너졌다 → 배치 합산으로 바꾸고 처음부터 다시 학습."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((0, 2, 3)) + target.sum((0, 2, 3)) + eps)).mean()


@torch.no_grad()
def val_iou(model, loader, dev):
    model.eval()
    inter = torch.zeros(3, device=dev)
    union = torch.zeros(3, device=dev)
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lo = model(x)
        loss_sum += (F.binary_cross_entropy_with_logits(lo.float(), y) + dice_loss(lo.float(), y)).item() * len(x)
        n += len(x)
        pr = torch.sigmoid(lo.float()) >= 0.5
        yt = y > 0.5
        inter += (pr & yt).sum((0, 2, 3))
        union += (pr | yt).sum((0, 2, 3))
    iou = (inter / union.clamp(min=1)).cpu().numpy()
    return loss_sum / n, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="unet")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--aug", choices=["weak", "strong"], default="weak")   # strong = 2026-11-19 추가(과적합 대응, runs/unet_aug)
    ap.add_argument("--lr", type=float, default=1e-4)   # 2026-11-19: 3e-4는 epoch 6~7에 발산(crack 출력이 0으로 죽음) → 1e-4 + 클리핑
    args = ap.parse_args()
    out = ROOT / "runs" / args.name
    (out / "weights").mkdir(parents=True, exist_ok=True)

    build_cache()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda"
    model = make_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler()
    start, best, bad = 0, -1.0, 0
    if args.resume and (out / "weights" / "last.pt").exists():
        ck = torch.load(out / "weights" / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        args.aug = ck.get("aug", args.aug)                  # --resume만 붙여도 처음 증강 설정 그대로
        print(f"이어서: epoch {start + 1}부터 (best mIoU {best:.4f})", flush=True)

    tr = DataLoader(SegData("train", "strong" if args.aug == "strong" else True), batch_size=args.batch, shuffle=True, num_workers=4,
                    persistent_workers=True, pin_memory=True, drop_last=True)
    va = DataLoader(SegData("val", False), batch_size=args.batch, num_workers=2)
    log = out / "results.csv"
    if start == 0:
        with open(log, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "time", "train_loss", "val_loss", "iou_crack", "iou_efflorescence",
                                    "iou_spall_detach", "mIoU", "lr"])

    t0 = time.time()
    for ep in range(start, args.epochs):
        model.train()
        tl, k = 0.0, 0
        for i, (x, y) in enumerate(tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(x)
            loss = F.binary_cross_entropy_with_logits(lo.float(), y) + dice_loss(lo.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
            k += 1
            if i % 200 == 0:
                print(f"epoch {ep + 1}/{args.epochs}  {i}/{len(tr)}  loss {tl / k:.4f}", flush=True)
        sched.step()
        vl, iou = val_iou(model, va, dev)
        miou = float(iou.mean())
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, round(time.time() - t0, 1), round(tl / k, 5), round(vl, 5),
                                    *[round(float(v), 4) for v in iou], round(miou, 4), opt.param_groups[0]["lr"]])
        print(f"epoch {ep + 1}: train {tl / k:.4f} val {vl:.4f} IoU {np.round(iou, 3)} mIoU {miou:.4f}", flush=True)
        if miou > best:
            best, bad = miou, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "mIoU": miou}, out / "weights" / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "bad": bad, "aug": args.aug}, out / "weights" / "last.pt")
        if bad >= args.patience:
            print(f"조기 종료: {args.patience} epoch 동안 검증 mIoU 개선 없음", flush=True)
            break
    with open(out / "train_time.txt", "a", encoding="utf-8") as f:
        f.write(f"{'resume ' if args.resume else ''}{(time.time() - t0) / 3600:.2f} h\n")


if __name__ == "__main__":
    main()
