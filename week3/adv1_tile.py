"""3주차 심화 ① (논문강의 47쪽 '타일 기반 추론'): 원본을 겹치는 타일로 나눠 추론하고 중복 제거 → 가는 균열 검출이 나아지는가.

  python week3/adv1_tile.py      # → adv1_tile.md · adv1_tile.png · adv1_tile_cases.png
- 기본(full): 1920×1080 전체를 imgsz 960으로 (가로 1/2로 줄어듦 — 논문 한계 03 '입력 960 리사이즈')
- 타일(tile): 원본의 1/2 크기 타일(960×540)을 가로 · 세로 50% 겹쳐 3×3 = 9장 → 각각 imgsz 960 (= 원본 해상도 그대로)
- 합침(full+tile): 위 둘의 예측을 모두 모은 것
타일 예측은 원본 좌표로 옮긴 뒤 클래스별 NMS(box IoU 0.5)로 겹친 타일의 중복을 지운다.
세 방식 모두 임계값은 검증셋에서 따로 고르고(같은 구간 · 같은 동점 규칙) 테스트셋에 그대로 적용한다. 모델은 기준 best.pt 하나.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import batched_nms

from adv_pix import PH, PW, IoUAcc, gt_mask, inst_mask, select_then_test
from w3 import CLASS_LIST, IMGSZ, PRED_CONF, THR_FILE, WEEK3, load_model, load_split

THRS = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]
BUSY = "--busy" in __import__("sys").argv                  # 학습과 동시에 실행했으면 --busy (시간 해석 주의 표시)
CACHE = WEEK3 / "cache" / "tile_preds.json"


def run(model, img, x0=0, y0=0):
    """예측 → [(cls, conf, box[x1,y1,x2,y2], poly)] (원본 좌표)."""
    r = model.predict(img, imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
    out = []
    if r.masks is None:
        return out
    for c, f, b, poly in zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(),
                             r.boxes.xyxy.cpu().numpy(), r.masks.xy):
        out.append((int(c), float(f), (b + [x0, y0, x0, y0]).tolist(), (poly + [x0, y0]).round(1).tolist()))
    return out


def nms(insts, iou=0.5):
    if not insts:
        return []
    b = torch.tensor([i[2] for i in insts], dtype=torch.float32)
    s = torch.tensor([i[1] for i in insts])
    c = torch.tensor([i[0] for i in insts])
    return [insts[k] for k in batched_nms(b, s, c, iou).tolist()]


def predict_all(model, paths):
    res = {}
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        full = run(model, img)
        t1 = time.perf_counter()
        tw, th = w // 2, h // 2
        tiles = []
        for iy in range(3):
            for ix in range(3):
                x0, y0 = ix * tw // 2, iy * th // 2
                tiles += run(model, img[y0:y0 + th, x0:x0 + tw], x0, y0)
        tiles = nms(tiles)
        t2 = time.perf_counter()
        res[Path(p).name] = {"w": w, "h": h, "full": full, "tile": tiles, "both": nms(full + tiles),
                             "t_full": t1 - t0, "t_tile": t2 - t1}
    return res


def scores(res, paths, key):
    out = []
    for p in paths:
        s = [0.0, 0.0, 0.0]
        for c, f, _, _ in res[Path(p).name][key]:
            s[c] = max(s[c], f)
        out.append(s)
    return out


def main():
    model = load_model()
    val_imgs, val_gts = load_split("val")
    test_imgs, test_gts = load_split("test")
    if CACHE.exists():
        res = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        res = predict_all(model, val_imgs + test_imgs)
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(res), encoding="utf-8")

    keys = {"full": "전체 영상(기본)", "tile": "타일 9장", "both": "전체 + 타일"}
    out = {}
    for k in keys:
        thr, curves, test = select_then_test(scores(res, val_imgs, k), val_gts, scores(res, test_imgs, k), test_gts, THRS)
        acc = IoUAcc()
        for p in test_imgs:
            d = res[Path(p).name]
            w, h = (PW, PH) if d["w"] >= d["h"] else (PH, PW)
            acc.add(inst_mask([(c, f, poly) for c, f, _, poly in d[k]], thr, d["w"], d["h"], w, h), gt_mask("test", p, w, h))
        out[k] = {"thresholds": thr, "test": test, "macro_f1": float(np.mean([test[n]["f1"] for n in CLASS_LIST])),
                  "pixel_iou": acc.iou(), "val_curves": curves}
    t_full = np.mean([res[Path(p).name]["t_full"] for p in test_imgs])
    t_tile = np.mean([res[Path(p).name]["t_tile"] for p in test_imgs])
    (WEEK3 / "adv1_tile.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # 기본 실험 crack FN 3장이 타일에서는 어떻게 되는가 (분석용 — 임계값은 위에서 검증셋으로 정한 값)
    base_thr = json.loads(THR_FILE.read_text(encoding="utf-8"))["thresholds"]
    fn_cases = []
    for p, g in zip(test_imgs, test_gts):
        d = res[Path(p).name]
        sf = max([f for c, f, _, _ in d["full"] if c == 0], default=0)
        if 0 in g and sf < base_thr["crack"]:
            st = max([f for c, f, _, _ in d["tile"] if c == 0], default=0)
            fn_cases.append((Path(p).name, sf, st))

    L = ["# 심화 ① 타일 기반 추론", "",
         "- 타일: 원본 1/2 크기(960×540) 3×3 = 9장, 50% 겹침, 각 imgsz 960 → 원본 해상도 그대로. 클래스별 NMS(box IoU 0.5)로 중복 제거",
         "- 세 방식 모두 검증셋에서 클래스별 임계값을 따로 고르고 테스트셋에 고정 적용 (모델은 기준 best.pt)",
         f"- 추론 시간(테스트 평균, 1장): 전체 {t_full * 1000:.0f} ms · 타일 9장 {t_tile * 1000:.0f} ms (약 {t_tile / t_full:.1f}배)"
         + (" — ★ 다른 학습과 GPU를 같이 쓰며 잰 값 (절대값은 부풀려짐, 비율만 참고)" if BUSY else ""), "",
         "## 테스트 ON/OFF (F1, 괄호 = FP / FN)", "",
         "| 방식 | 임계값 (crack / efflo / spall) | " + " | ".join(CLASS_LIST) + " | macro F1 |", "|---|---|" + "---:|" * 4]
    for k, lab in keys.items():
        o = out[k]
        L.append(f"| {lab} | {' / '.join(f'{o['thresholds'][n]:g}' for n in CLASS_LIST)} | " +
                 " | ".join(f"{o['test'][n]['f1']:.3f} ({o['test'][n]['fp']} / {o['test'][n]['fn']})" for n in CLASS_LIST) +
                 f" | {o['macro_f1']:.3f} |")
    L += ["", "## 테스트 픽셀 IoU (선택 임계값, 960×540, 테스트 전체 합산)", "",
          "| 방식 | " + " | ".join(CLASS_LIST) + " | 평균 |", "|---|" + "---:|" * 4]
    for k, lab in keys.items():
        iou = out[k]["pixel_iou"]
        L.append(f"| {lab} | " + " | ".join(f"{iou[n]:.3f}" for n in CLASS_LIST) + f" | {np.mean(list(iou.values())):.3f} |")
    L += ["", "## 기본 실험의 crack 미탐 영상 — crack 최고 신뢰도 (전체 → 타일)", "",
          "| 영상 | 전체 영상 | 타일 |", "|---|---:|---:|"]
    for n, sf, st in fn_cases:
        L.append(f"| `{n[:40]}` | {sf:.3f} | {st:.3f} |")
    (WEEK3 / "adv1_tile.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # 그림 1: 테스트 F1 막대
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(4)
    for j, (k, lab) in enumerate(keys.items()):
        v = [out[k]["test"][n]["f1"] for n in CLASS_LIST] + [out[k]["macro_f1"]]
        b = ax.bar(x + (j - 1) * 0.27, v, 0.27, label=lab, color=["#9e9e9e", "#2c7fb8", "#7fcdbb"][j])
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LIST + ["macro"])
    ax.set_ylim(0.4, 1.1)
    ax.set_ylabel("테스트 F1")
    ax.set_title("전체 영상 vs 타일 추론 (임계값은 각각 검증셋에서 선택)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(WEEK3 / "adv1_tile.png", dpi=150)
    plt.close()

    # 그림 2: crack 미탐 영상에서 전체 vs 타일 crack 예측 (각 방식의 검증 임계값 적용)
    if fn_cases:
        fig, axes = plt.subplots(len(fn_cases), 2, figsize=(12, 3.6 * len(fn_cases)))
        axes = np.atleast_2d(axes)
        for row, (n, sf, st) in zip(axes, fn_cases):
            p = next(q for q in test_imgs if Path(q).name == n)
            img = cv2.imread(p)
            d = res[n]
            for ax, k in zip(row, ("full", "tile")):
                vis = img.copy()
                t = out[k]["thresholds"]["crack"]
                for c, f, b, poly in d[k]:
                    if c == 0 and f >= t:
                        cv2.polylines(vis, [np.asarray(poly, np.int32)], True, (0, 0, 255), 5)
                        cv2.putText(vis, f"crack {f:.2f}", (int(b[0]), max(int(b[1]) - 10, 40)), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.6, (0, 0, 255), 4)
                g = gt_mask("test", p, d["w"], d["h"])[0].astype(np.uint8)
                cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cs, -1, (0, 255, 0), 3)
                ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                ax.set_title(f"{keys[k]} (crack 임계값 {t:g}) — {n[:30]}", fontsize=9)
                ax.axis("off")
        fig.suptitle("기본 실험의 crack 미탐 영상: 초록 = 정답, 빨강 = 예측 crack", fontsize=12)
        plt.tight_layout()
        plt.savefig(WEEK3 / "adv1_tile_cases.png", dpi=90)
        plt.close()


if __name__ == "__main__":
    main()
