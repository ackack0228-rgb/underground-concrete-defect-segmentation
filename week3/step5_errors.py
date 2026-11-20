"""3주차 단계5 (강의 13쪽): 테스트셋 클래스별 FP · FN 영상 추출 → error_cases/<클래스>/ · error_cases.md.

  python week3/step5_errors.py
임계값은 thresholds.json(검증셋)에서 읽는다. 강의 코드는 r.plot()만 저장하는데, 원인을 보려면 정답과 나란히 봐야 해서
그림마다 [정답 테두리 | 예측(클래스별 임계값 적용)] 두 칸으로 저장한다. 강의대로 클래스별 FP · FN 최대 3장.
  python week3/step5_errors.py --split val   # (2026-11-20 추가) 보조 사례: 테스트에 spall 오류 · efflo FN이 없어서
                                              검증셋 오류를 같은 방식으로 → error_cases_val/ · error_cases_val.md (임계값은 같은 val 선택값)
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json
from pathlib import Path

import cv2
import numpy as np

from w3 import (CLASS_LIST, IMGSZ, PRED_CONF, SPLIT_ROOT, THR_FILE, WEEK3, common, load_cache, load_model,
                load_split, log_run, predict_conf_cls)

import sys
SPLIT = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "test"
SFX = "" if SPLIT == "test" else f"_{SPLIT}"
SAVE_DIR = WEEK3 / f"error_cases{SFX}"
COLORS = {0: (40, 40, 230), 1: (230, 120, 30), 2: (0, 200, 255)}      # BGR: crack 빨강 · efflo 파랑 · spall 노랑


def imwrite(path, img):
    """한글 경로(OneDrive 바탕 화면)에서도 저장되게."""
    ok, buf = cv2.imencode(Path(path).suffix, img)
    buf.tofile(str(path))


def draw_gt(img, split, p):
    out = img.copy()
    h, w = img.shape[:2]
    lab = SPLIT_ROOT / "labels" / split / (Path(p).stem + ".txt")
    for line in lab.read_text(encoding="utf-8").splitlines() if lab.exists() else []:
        v = line.split()
        if not v:
            continue
        c = int(v[0])
        pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.polylines(out, [pts], True, COLORS[c], 4)
        x, y = pts[:, 0].min(), max(pts[:, 1].min() - 8, 30)
        cv2.putText(out, CLASS_LIST[c], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLORS[c], 3)
    return out


def banner(img, text):
    bar = np.full((60, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    return np.vstack([bar, img])


def main():
    best = json.loads(THR_FILE.read_text(encoding="utf-8"))["thresholds"]   # 검증셋에서 정한 값
    model = load_model()
    test_imgs, test_gts = load_split(SPLIT)
    load_cache(SPLIT)
    SAVE_DIR.mkdir(exist_ok=True)
    rows = []

    # ---------------------------------------------------------- 강의 13쪽 코드 (예측은 캐시 · conf 0.001)
    for cid, name in enumerate(CLASS_LIST):
        fp_imgs, fn_imgs = [], []
        for p, gt in zip(test_imgs, test_gts):
            conf, pred_cls = predict_conf_cls(model, p)
            # 검증셋에서 정한 클래스별 임계값 사용
            pred_on = bool(((pred_cls == cid) & (conf >= best[name])).any())
            gt_on = cid in gt
            if pred_on and not gt_on:
                fp_imgs.append(p)
            if not pred_on and gt_on:
                fn_imgs.append(p)
        print(f"{name} | FP={len(fp_imgs)} | FN={len(fn_imgs)}")

        class_dir = SAVE_DIR / name
        class_dir.mkdir(exist_ok=True)
        for f in class_dir.glob("*.jpg"):                                   # 이전 실행 결과 정리
            f.unlink()
        # 대표 FP / FN 최대 3장씩 저장
        for error_type, paths in [("FP", fp_imgs), ("FN", fn_imgs)]:
            for i, p in enumerate(paths[:3]):
                r = model.predict(p, imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                thr = np.array([best[CLASS_LIST[c]] for c in cls])
                keep = cf >= thr                                            # 세 클래스 모두 각자의 임계값 적용
                pred_img = r[keep].plot(line_width=3, font_size=28) if keep.any() else r.orig_img.copy()
                gt_img = draw_gt(r.orig_img, SPLIT, p)
                gt_txt = ", ".join(sorted(CLASS_LIST[c] for c in set(test_gts[test_imgs.index(p)]))) or "none"
                both = np.hstack([banner(gt_img, f"GT: {gt_txt}"),
                                  banner(pred_img, f"Pred (class thr) - {error_type} {name}")])
                both = cv2.resize(both, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                out_path = class_dir / f"{error_type}_{i + 1}_{Path(p).stem[:40]}.jpg"
                imwrite(out_path, both)

                m = cls == cid
                top = float(cf[m].max()) if m.any() else 0.0
                others = sorted({f"{CLASS_LIST[c]} {cf[j]:.2f}" for j, c in enumerate(cls) if keep[j] and c != cid})
                rows.append({"class": name, "type": error_type, "n": i + 1, "image": Path(p).name,
                             "gt": gt_txt, "thr": best[name], "top_conf": round(top, 4),
                             "n_over_thr": int((m & keep).sum()), "other_pred": "; ".join(others),
                             "file": str(out_path.relative_to(WEEK3)).replace("\\", "/")})

    (WEEK3 / f"error_cases{SFX}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    L = [f"# 단계5 {'테스트셋' if SPLIT == 'test' else '검증셋(보조)'} 오류 사례 (임계값 = 검증셋 선택값)", "",
         "| 클래스 | 유형 | 영상 | 정답 클래스 | 임계값 | 그 클래스 최고 신뢰도 | 임계값 이상 개수 | 다른 클래스 예측 | 그림 |",
         "|---|---|---|---|---:|---:|---:|---|---|"]
    for r in rows:
        L.append(f"| {r['class']} | {r['type']} | `{r['image'][:34]}` | {r['gt']} | {r['thr']:g} | {r['top_conf']:.3f} | "
                 f"{r['n_over_thr']} | {r['other_pred'] or '-'} | {r['file']} |")
    (WEEK3 / f"error_cases{SFX}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if SPLIT == "test":
        log_run("step5_errors", f"{len(rows)} cases")


if __name__ == "__main__":
    main()
