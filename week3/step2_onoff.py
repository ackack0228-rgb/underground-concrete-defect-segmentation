"""3주차 단계2 (강의 10쪽): ON/OFF 평가 함수 동작 확인 — 검증셋에서 thr=0.5.

  python week3/step2_onoff.py    # → week3/onoff_check.md
함수 본체는 w3.onoff (강의 코드). 여기서는 ① 영상별 정답 클래스 목록(gts)이 제대로 만들어졌는지
② thr=0.5에서 TP+FP+FN+TN = 영상 수인지 확인한다. 테스트셋은 쓰지 않는다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

from collections import Counter
from pathlib import Path

from w3 import CLASS_LIST, WEEK3, load_cache, load_model, load_split, log_run, onoff, save_cache


def main():
    model = load_model()
    val_imgs, val_gts = load_split("val")
    load_cache("val")

    L = ["# 단계2 ON/OFF 함수 동작 확인 (검증셋, thr = 0.5)", "",
         f"- 검증 영상 {len(val_imgs)}장. 정답 ON = 라벨 txt에 그 클래스 줄이 하나 이상.", ""]
    # 정답 클래스 목록 예시 (강의 10쪽 '한 영상에 여러 클래스가 동시에 존재할 수 있음')
    combo = Counter(tuple(sorted(set(g))) for g in val_gts)
    L += ["## 영상별 정답 클래스 조합", "", "| 조합 | 영상 수 |", "|---|---:|"]
    for k, n in sorted(combo.items(), key=lambda x: -x[1]):
        L.append(f"| {' + '.join(CLASS_LIST[c] for c in k) if k else '(결함 없음)'} | {n} |")
    L += ["", "예시 (앞 3장):", ""]
    for p, g in list(zip(val_imgs, val_gts))[:3]:
        L.append(f"- `{Path(p).name[:40]}` → gts = {g}")

    L += ["", "## thr = 0.5 결과", "", "| 클래스 | TP | FP | FN | TN | 합 | Precision | Recall | F1 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cid, name in enumerate(CLASS_LIST):
        r = onoff(model, val_imgs, val_gts, 0.5, cid)
        s = r["tp"] + r["fp"] + r["fn"] + r["tn"]
        assert s == len(val_imgs), "TP+FP+FN+TN이 영상 수와 다르다"
        print(name, r)
        L.append(f"| {name} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {s} | "
                 f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    save_cache("val", val_imgs)
    L += ["", "세 클래스 모두 TP+FP+FN+TN = 40 → 함수 정상 동작. 임계값 탐색은 단계3에서 검증셋으로만 한다."]
    (WEEK3 / "onoff_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    log_run("step2_onoff", "val thr=0.5 check")


if __name__ == "__main__":
    main()
