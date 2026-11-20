"""3주차 단계4 (강의 12쪽): 검증셋에서 정한 임계값을 테스트셋에 그대로 적용 → onoff_test.csv · onoff_test.md.

  python week3/step4_test.py
★ 임계값은 thresholds.json(단계3이 val로 만든 파일)에서 읽기만 한다. 이 스크립트에는 임계값을 고르는 코드가 없다.
  thresholds.json이 없으면 멈춘다 (단계3을 먼저 실행해야 함).
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json

import pandas as pd

from w3 import CLASS_LIST, THR_FILE, WEEK3, load_cache, load_model, load_split, log_run, onoff, save_cache


def main():
    if not THR_FILE.exists():
        raise FileNotFoundError("thresholds.json이 없습니다 — 단계3(step3_threshold.py, 검증셋)을 먼저 실행하세요.")
    meta = json.loads(THR_FILE.read_text(encoding="utf-8"))
    assert meta["selected_on"] == "val"
    best = meta["thresholds"]                          # 검증셋에서 선택한 임계값 (변경하지 않음)
    print("검증셋에서 정한 임계값:", best, "(", meta["created"], ")")

    model = load_model()
    test_imgs, test_gts = load_split("test")
    load_cache("test")

    # ---------------------------------------------------------- 강의 12쪽 코드
    class_names = CLASS_LIST
    rows = []
    for cid, name in enumerate(class_names):
        r = onoff(model, test_imgs, test_gts, best[name], cid)   # 검증셋에서 선택한 임계값 그대로 사용
        rows.append([name, best[name], r["tp"], r["fp"], r["fn"], r["tn"], r["precision"], r["recall"], r["f1"]])
    df = pd.DataFrame(rows, columns=["class", "threshold", "TP", "FP", "FN", "TN", "precision", "recall", "f1"])
    print(df.round(3))
    macro = df[["precision", "recall", "f1"]].mean()
    print("macro F1:", round(df["f1"].mean(), 3))
    save_cache("test", test_imgs)

    df.to_csv(WEEK3 / "onoff_test.csv", index=False, encoding="utf-8-sig")
    paper = {"crack": (0.917, 0.846, 0.880), "efflorescence": (0.846, 0.917, 0.880), "spall_detach": (1.0, 0.867, 0.929)}
    L = ["# 단계4 테스트셋 ON/OFF 결과 (임계값 = 검증셋에서 선택, 고정 적용)", "",
         f"- thresholds.json 생성 {meta['created']} (val {meta['n_val_images']}장, sha {meta['val_list_sha']}) → 변경 없이 사용",
         f"- 테스트 영상 {len(test_imgs)}장", "",
         "| 지표 | " + " | ".join(class_names) + " | macro |", "|---|" + "---:|" * 4,
         "| 임계값 | " + " | ".join(f"{best[n]:g}" for n in class_names) + " | — |"]
    for k, lab in [("TP", "True Positive"), ("TN", "True Negative"), ("FP", "False Positive"), ("FN", "False Negative")]:
        L.append(f"| {lab} | " + " | ".join(str(int(v)) for v in df[k]) + " | — |")
    for k, lab in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]:
        L.append(f"| {lab} | " + " | ".join(f"{v:.3f}" for v in df[k]) + f" | {macro[k]:.3f} |")
    L += ["", "논문 Table 5 (참고): " + " · ".join(
        f"{n} P {paper[n][0]:.3f} R {paper[n][1]:.3f} F1 {paper[n][2]:.3f}" for n in class_names) + " · macro F1 0.896"]
    (WEEK3 / "onoff_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    log_run("step4_test", f"thresholds from json {best} → test macro F1 {df['f1'].mean():.3f}")


if __name__ == "__main__":
    main()
