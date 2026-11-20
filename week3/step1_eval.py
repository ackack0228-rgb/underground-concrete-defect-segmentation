"""3주차 단계1 (강의 9쪽): 세그멘테이션 성능 측정 — best.pt로 테스트셋 mask mAP50 · mAP50-95.

  python week3/step1_eval.py     # → runs/step3_eval_test/ (PR 곡선 · 혼동행렬) · week3/seg_metrics.md · seg_metrics.json
비교용으로 같은 best.pt의 검증셋 성능도 함께 구한다 (runs/step3_eval_val/). 임계값과는 관계없는 단계다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json

import numpy as np

from w3 import BEST_PT, CLASS_LIST, DATA_YAML, IMGSZ, ROOT, WEEK3, load_model, load_split, log_run


def evaluate(model, split):
    # ---------------------------------------------------------- 강의 9쪽 코드 (저장 위치만 지정)
    results = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=IMGSZ,
        plots=True,
        project=str(ROOT / "runs"),
        name=f"step3_eval_{split}",
        exist_ok=True,
    )
    print(f"[{split}] mask mAP50   :", round(results.seg.map50, 4))
    print(f"[{split}] mask mAP50-95:", round(results.seg.map, 4))
    print(f"[{split}] box mAP50    :", round(results.box.map50, 4))
    for i, name in enumerate(CLASS_LIST):
        print(f"[{split}]", name, "mask mAP50-95:", round(results.seg.maps[i], 4))

    # ---------------------------------------------------------- 리포트용으로 더 꺼내 둠
    idx = list(results.seg.ap_class_index)                # 정답이 있는 클래스만 들어 있다
    per = {}
    for i, name in enumerate(CLASS_LIST):
        if i in idx:
            k = idx.index(i)
            per[name] = {"P": float(results.seg.p[k]), "R": float(results.seg.r[k]),
                         "mAP50": float(results.seg.ap50[k]), "mAP50-95": float(results.seg.ap[k]),
                         "box_mAP50": float(results.box.ap50[k])}
        else:
            per[name] = None
    return {"mask_mAP50": float(results.seg.map50), "mask_mAP50-95": float(results.seg.map),
            "box_mAP50": float(results.box.map50), "box_mAP50-95": float(results.box.map),
            "mask_P": float(results.seg.mp), "mask_R": float(results.seg.mr), "per_class": per}


def count(split):
    paths, gts = load_split(split)
    inst = np.zeros(3, int)
    imgs = np.zeros(3, int)
    for g in gts:
        for c in g:
            inst[c] += 1
        for c in set(g):
            imgs[c] += 1
    return len(paths), inst.tolist(), imgs.tolist(), sum(1 for g in gts if not g)


def main():
    model = load_model()
    out = {"weights": str(BEST_PT.relative_to(ROOT)), "imgsz": IMGSZ}
    for split in ("test", "val"):
        out[split] = evaluate(model, split)
        n, inst, imgs, empty = count(split)
        out[split].update({"n_images": n, "instances": inst, "images_with_class": imgs, "images_without_label": empty})
    (WEEK3 / "seg_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    t, v = out["test"], out["val"]
    L = ["# 단계1 세그멘테이션 성능 (best.pt = 2주차 epoch 21, imgsz 960)", "",
         f"- 테스트 {t['n_images']}장 · 검증 {v['n_images']}장 (2주차 원본 ID 단위 분할, 증강본 없음)",
         "- 논문(Table 4) 테스트: mask mAP50 0.443 · mask mAP50-95 0.222", "",
         "## 전체", "", "| 지표 | 테스트 | 검증(참고) |", "|---|---:|---:|"]
    for k, lab in [("mask_mAP50", "mask mAP50"), ("mask_mAP50-95", "mask mAP50-95"), ("box_mAP50", "box mAP50"),
                   ("box_mAP50-95", "box mAP50-95"), ("mask_P", "mask Precision"), ("mask_R", "mask Recall")]:
        L.append(f"| {lab} | {t[k]:.3f} | {v[k]:.3f} |")
    L += ["", "## 클래스별 (테스트)", "",
          "| 클래스 | 인스턴스 | 영상 | mask P | mask R | mask mAP50 | mask mAP50-95 | box mAP50 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, name in enumerate(CLASS_LIST):
        c = t["per_class"][name]
        if c:
            L.append(f"| {name} | {t['instances'][i]} | {t['images_with_class'][i]} | {c['P']:.3f} | {c['R']:.3f} | "
                     f"{c['mAP50']:.3f} | {c['mAP50-95']:.3f} | {c['box_mAP50']:.3f} |")
    L += ["", "## 클래스별 (검증, 참고)", "",
          "| 클래스 | 인스턴스 | 영상 | mask mAP50 | mask mAP50-95 |", "|---|---:|---:|---:|---:|"]
    for i, name in enumerate(CLASS_LIST):
        c = v["per_class"][name]
        if c:
            L.append(f"| {name} | {v['instances'][i]} | {v['images_with_class'][i]} | {c['mAP50']:.3f} | {c['mAP50-95']:.3f} |")
    (WEEK3 / "seg_metrics.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    log_run("step1_eval", f"test mask mAP50 {t['mask_mAP50']:.3f} / mAP50-95 {t['mask_mAP50-95']:.3f}")


if __name__ == "__main__":
    main()
