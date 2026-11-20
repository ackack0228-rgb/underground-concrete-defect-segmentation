"""심화 ③ · ⑤ 공용: 학습된 YOLO 가중치 하나를 본 실험과 똑같은 절차로 평가한다.

  python week3/adv_eval_run.py --run step3_seed0 --tag seed0
절차 = 단계1(val · test mask mAP) → 단계3(검증셋에서만 클래스별 임계값, 같은 구간 · 같은 동점 규칙) → 단계4(테스트 고정 적용).
결과 → week3/adv_runs/<tag>.json. 기본 실험(runs/step3)도 --run step3 --tag seed42 로 같은 함수로 다시 잰다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import argparse
import json
from datetime import datetime

import numpy as np

from w3 import (CLASS_LIST, DATA_YAML, IMGSZ, ROOT, WEEK3, load_cache, load_model, load_split, onoff, save_cache,
                use_weights)

THRS = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]     # 단계3과 같은 구간


def seg_metrics(model, split, name):
    r = model.val(data=str(DATA_YAML), split=split, imgsz=IMGSZ, plots=False, verbose=False,
                  project=str(ROOT / "runs" / "_adv_eval"), name=f"{name}_{split}", exist_ok=True)
    idx = list(r.seg.ap_class_index)
    return {"mask_mAP50": float(r.seg.map50), "mask_mAP50-95": float(r.seg.map), "box_mAP50": float(r.box.map50),
            "class_mask_mAP50": {n: float(r.seg.ap50[idx.index(i)]) for i, n in enumerate(CLASS_LIST)},
            "class_mask_mAP50-95": {n: float(r.seg.ap[idx.index(i)]) for i, n in enumerate(CLASS_LIST)}}


def select_and_test(model, tag):
    """검증셋에서 임계값을 고른 뒤(먼저) 테스트셋에 그대로 적용(나중)."""
    val_imgs, val_gts = load_split("val")
    load_cache("val")
    thr, val_f1, curves = {}, {}, {}
    for cid, n in enumerate(CLASS_LIST):
        scores = [onoff(model, val_imgs, val_gts, t, cid)["f1"] for t in THRS]
        i = scores.index(max(scores))                       # 동점이면 낮은 임계값
        thr[n], val_f1[n], curves[n] = THRS[i], scores[i], scores
    save_cache("val", val_imgs)

    test_imgs, test_gts = load_split("test")
    load_cache("test")
    test = {n: onoff(model, test_imgs, test_gts, thr[n], cid) for cid, n in enumerate(CLASS_LIST)}
    save_cache("test", test_imgs)
    return {"thresholds": thr, "val_f1": val_f1, "val_macro_f1": float(np.mean(list(val_f1.values()))),
            "val_curves": curves, "test": test, "test_macro_f1": float(np.mean([test[n]["f1"] for n in CLASS_LIST]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", default="best.pt")
    args = ap.parse_args()
    pt = ROOT / "runs" / args.run / "weights" / args.weights
    use_weights(pt, args.tag)
    model = load_model()

    out = {"run": args.run, "weights": str(pt.relative_to(ROOT)), "evaluated": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    # best.pt가 몇 번째 epoch인지 (ultralytics 체크포인트에 저장된 값 + 1)
    try:
        import torch
        out["best_epoch"] = int(torch.load(pt, map_location="cpu", weights_only=False)["epoch"]) + 1
    except Exception:
        pass
    if out.get("best_epoch", 0) <= 0 and args.weights == "best.pt":
        # 학습이 끝난 ultralytics best.pt는 epoch = -1로 저장된다 (2026-11-20 발견) → results.csv에서 fitness 최고 epoch
        # (ultralytics 8.4 fitness = mAP50-95(B) + mAP50-95(M); 기준 실험 step3에서 epoch 21로 기록과 일치 확인)
        import csv
        rows = list(csv.DictReader(open(ROOT / "runs" / args.run / "results.csv", encoding="utf-8")))
        out["best_epoch"] = int(max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]) + float(r["metrics/mAP50-95(M)"]))["epoch"])
    out["seg_test"] = seg_metrics(model, "test", args.tag)
    out["seg_val"] = seg_metrics(model, "val", args.tag)
    out.update(select_and_test(model, args.tag))
    d = WEEK3 / "adv_runs"
    d.mkdir(exist_ok=True)
    (d / f"{args.tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("thresholds", "val_macro_f1", "test_macro_f1")}, ensure_ascii=False))
    print("test mask mAP50", round(out["seg_test"]["mask_mAP50"], 3), "mAP50-95", round(out["seg_test"]["mask_mAP50-95"], 3))


if __name__ == "__main__":
    main()
