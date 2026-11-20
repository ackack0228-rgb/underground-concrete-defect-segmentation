"""3주차 심화 ③ (논문강의 47쪽 '하드네거티브 수집'): 결함이 아닌데 모델이 결함으로 보는 배경을 모아 학습에 추가.

  python week3/adv3_hn_build.py
이 데이터에는 결함 없는 영상(빈 라벨)이 한 장도 없다 → train 영상에서 라벨 영역과 겹치지 않는 조각을 잘라 배경 영상으로 쓴다.
  1. 후보: train 영상마다 2/3 크기(1920×1080이면 1280×720) 조각 9개(3×3 격자) 중 라벨 폴리곤(+16px)과 한 픽셀도 안 겹치는 것
  2. 마이닝: 기준 모델(runs/step3 best.pt)이 검증셋 임계값(thresholds.json) 이상으로 무엇이든 예측한 조각 = 하드네거티브
     (원본 ID당 최대 2장, 신뢰도 높은 순 최대 200장)
  3. ~/datasets/crack-seg-hn/ = 2주차 분할(하드링크) + train에 하드네거티브 조각(빈 라벨 txt) 추가
★ train에서만 뽑는다 (val · test는 그대로). 평가용으로 val · test의 라벨 없는 조각도 저장하지만(neg_eval/) 학습 · 선택에는 쓰지 않는다.
결과: adv3_hn_build.md · adv3_hn_samples.png
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from w3 import CLASS_LIST, IMGSZ, PRED_CONF, SPLIT_ROOT, THR_FILE, WEEK3, load_model

from paths import get_origin_id  # noqa: E402  (w3가 week2를 sys.path에 넣어 둠)

HN_ROOT = SPLIT_ROOT.parent / "crack-seg-hn"
NEG_EVAL = SPLIT_ROOT.parent / "crack-seg-neg-eval"   # val · test 라벨 없는 조각 (평가 전용)
MAX_HN, PER_ORIGIN, MARGIN = 200, 2, 16


def label_mask(lab, w, h):
    m = np.zeros((h, w), np.uint8)
    if lab.exists():
        for line in lab.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) > 6:
                pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
    return cv2.dilate(m, np.ones((2 * MARGIN + 1, 2 * MARGIN + 1), np.uint8))


def free_crops(img_path, split):
    img = cv2.imread(str(img_path))                         # datasets 경로는 한글 없음
    if img is None:
        return None, []
    h, w = img.shape[:2]
    cw, ch = w * 2 // 3, h * 2 // 3
    m = label_mask(SPLIT_ROOT / "labels" / split / (img_path.stem + ".txt"), w, h)
    out = []
    for iy in range(3):
        for ix in range(3):
            x0, y0 = (w - cw) * ix // 2, (h - ch) * iy // 2
            if m[y0:y0 + ch, x0:x0 + cw].sum() == 0:
                out.append((x0, y0, cw, ch))
    return img, out


def main():
    thr = json.loads(THR_FILE.read_text(encoding="utf-8"))["thresholds"]
    model = load_model()                                    # 기준 모델 (runs/step3 best.pt)

    # ---------------------------------------------------------- 1 · 2 train 조각 마이닝
    train = sorted((SPLIT_ROOT / "images" / "train").glob("*.jpg"))
    stats = {"train_images": len(train), "candidates": 0}
    hard = []
    for k, p in enumerate(train):
        img, crops = free_crops(p, "train")
        for (x0, y0, cw, ch) in crops:
            stats["candidates"] += 1
            crop = img[y0:y0 + ch, x0:x0 + cw]
            r = model.predict(crop, imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            over = [(CLASS_LIST[c], float(f)) for c, f in zip(cls, cf) if f >= thr[CLASS_LIST[c]]]
            if over:
                top = max(over, key=lambda x: x[1])
                hard.append({"src": p.name, "origin": get_origin_id(p.stem), "box": [x0, y0, cw, ch],
                             "fp_class": top[0], "fp_conf": round(top[1], 4), "n_fp": len(over)})
        if k % 500 == 0:
            print(f"{k}/{len(train)}  후보 {stats['candidates']}  하드 {len(hard)}", flush=True)
    stats["hard_found"] = len(hard)

    # 원본 ID당 최대 PER_ORIGIN장, 신뢰도 높은 순 MAX_HN장
    hard.sort(key=lambda d: -d["fp_conf"])
    per = defaultdict(int)
    chosen = []
    for d in hard:
        if per[d["origin"]] < PER_ORIGIN and len(chosen) < MAX_HN:
            per[d["origin"]] += 1
            chosen.append(d)

    # ---------------------------------------------------------- 3 데이터 폴더 (2주차 분할 하드링크 + 조각)
    if HN_ROOT.exists():
        shutil.rmtree(HN_ROOT)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (HN_ROOT / sub / split).mkdir(parents=True)
            for f in (SPLIT_ROOT / sub / split).iterdir():
                if f.suffix in (".jpg", ".txt"):
                    os.link(f, HN_ROOT / sub / split / f.name)
    for i, d in enumerate(chosen):
        img = cv2.imread(str(SPLIT_ROOT / "images" / "train" / d["src"]))
        x0, y0, cw, ch = d["box"]
        name = f"hn{i:03d}_{Path(d['src']).stem[:60]}"
        d["file"] = name + ".jpg"
        cv2.imwrite(str(HN_ROOT / "images" / "train" / d["file"]), img[y0:y0 + ch, x0:x0 + cw], [cv2.IMWRITE_JPEG_QUALITY, 95])
        (HN_ROOT / "labels" / "train" / (name + ".txt")).write_text("", encoding="utf-8")   # 빈 라벨 = 배경
    (HN_ROOT / "data.yaml").write_text(
        f"path: {HN_ROOT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_LIST)), encoding="utf-8")

    # ---------------------------------------------------------- 평가 전용: val · test 라벨 없는 조각 (선택에 안 씀)
    if NEG_EVAL.exists():
        shutil.rmtree(NEG_EVAL)
    neg_count = {}
    for split in ("val", "test"):
        (NEG_EVAL / split).mkdir(parents=True)
        n = 0
        for p in sorted((SPLIT_ROOT / "images" / split).glob("*.jpg")):
            img, crops = free_crops(p, split)
            for j, (x0, y0, cw, ch) in enumerate(crops):
                cv2.imwrite(str(NEG_EVAL / split / f"{p.stem[:60]}_c{j}.jpg"), img[y0:y0 + ch, x0:x0 + cw])
                n += 1
        neg_count[split] = n
    stats.update(chosen=len(chosen), origins=len(per), neg_eval=neg_count,
                 by_class={c: sum(d["fp_class"] == c for d in chosen) for c in CLASS_LIST})
    (WEEK3 / "adv3_hn_list.json").write_text(json.dumps({"stats": stats, "chosen": chosen}, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    # 예시 그림 (클래스별로 섞어 12장, 예측 표시)
    random.seed(0)
    show = []
    for c in CLASS_LIST:
        pool = [d for d in chosen if d["fp_class"] == c]
        show += random.sample(pool, min(4, len(pool)))
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, d in zip(axes.flat, show):
        img = cv2.imread(str(HN_ROOT / "images" / "train" / d["file"]))
        r = model.predict(img, imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
        keep = np.array([f >= thr[CLASS_LIST[int(c)]] for c, f in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())])
        vis = r[keep].plot(line_width=3, font_size=26) if keep.any() else img
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{d['fp_class']} {d['fp_conf']:.2f} · {d['origin'][:22]}", fontsize=9)
    fig.suptitle("하드네거티브 예시 — 라벨 없는 train 조각인데 기준 모델이 결함으로 예측", fontsize=13)
    plt.tight_layout()
    plt.savefig(WEEK3 / "adv3_hn_samples.png", dpi=110)
    plt.close()

    L = ["# 심화 ③ 하드네거티브 수집", "",
         f"- train 영상 {stats['train_images']}장 → 라벨과 겹치지 않는 조각 후보 {stats['candidates']}개 "
         f"(영상의 2/3 크기, 3×3 격자, 라벨 +{MARGIN}px 여유)",
         f"- 기준 모델이 검증 임계값 이상으로 예측한 조각(하드네거티브) {stats['hard_found']}개 "
         f"→ 원본 ID당 최대 {PER_ORIGIN}장 · 최대 {MAX_HN}장 = **{len(chosen)}장** (원본 ID {len(per)}개)",
         "- 조각이 유발한 오탐 클래스: " + " · ".join(f"{c} {stats['by_class'][c]}" for c in CLASS_LIST),
         f"- 새 데이터: `{HN_ROOT}` (train {stats['train_images']} + {len(chosen)} 배경 = {stats['train_images'] + len(chosen)}장, val · test 그대로)",
         f"- 평가 전용 음성 조각(선택 · 학습에 안 씀): val {neg_count['val']}개 · test {neg_count['test']}개 → `{NEG_EVAL}`"]
    (WEEK3 / "adv3_hn_build.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
