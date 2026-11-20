"""3주차 심화 ④ (논문강의 47쪽 '공통 임계값 대조'): 단일 공통 임계값 대비 클래스별 임계값의 이득.

  python week3/adv4_common_threshold.py    # 단계3 · 4 다음에 실행 → adv4_common_threshold.md · .png
절차는 본 실험과 같다: 공통 임계값도 검증셋에서만 고르고(세 클래스 macro F1 최대, 동점이면 낮은 값)
테스트셋에는 그대로 적용한다. 클래스별 임계값은 thresholds.json(단계3)에서 읽는다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json

import matplotlib.pyplot as plt
import numpy as np

from w3 import CLASS_LIST, THR_FILE, WEEK3, load_cache, load_model, load_split, log_run, onoff

THRS = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]     # 단계3과 같은 구간


def evaluate(model, imgs, gts, thr_by_class):
    return {n: onoff(model, imgs, gts, thr_by_class[n], c) for c, n in enumerate(CLASS_LIST)}


def main():
    per_class = json.loads(THR_FILE.read_text(encoding="utf-8"))["thresholds"]
    model = load_model()
    val_imgs, val_gts = load_split("val")
    load_cache("val")

    # ① 검증셋에서 공통 임계값 선택
    val_macro = []
    for t in THRS:
        r = evaluate(model, val_imgs, val_gts, {n: t for n in CLASS_LIST})
        val_macro.append(np.mean([r[n]["f1"] for n in CLASS_LIST]))
    i = int(np.argmax(val_macro))                            # argmax = 동점이면 앞(낮은 값)
    common_thr = THRS[i]
    print("검증 macro F1:", dict(zip(THRS, np.round(val_macro, 3))), "→ 공통", common_thr)

    # ② 두 방식을 검증 · 테스트에 적용
    test_imgs, test_gts = load_split("test")
    load_cache("test")
    res = {}
    for split, imgs, gts in [("val", val_imgs, val_gts), ("test", test_imgs, test_gts)]:
        res[split] = {"class": evaluate(model, imgs, gts, per_class),
                      "common": evaluate(model, imgs, gts, {n: common_thr for n in CLASS_LIST})}

    L = ["# 심화 ④: 공통 임계값 vs 클래스별 임계값", "",
         f"- 공통 임계값: 검증셋 macro F1이 가장 큰 값 = **{common_thr:g}** (동점이면 낮은 값)",
         "- 클래스별 임계값: " + " · ".join(f"{n} {per_class[n]:g}" for n in CLASS_LIST) + " (단계3, 검증셋)",
         "- 둘 다 검증셋에서 정하고 테스트셋에 고정 적용", "",
         "## 공통 임계값 후보별 검증 macro F1", "", "| 임계값 | " + " | ".join(f"{t:g}" for t in THRS) + " |",
         "|---|" + "---:|" * len(THRS),
         "| 검증 macro F1 | " + " | ".join(f"**{v:.3f}**" if k == i else f"{v:.3f}" for k, v in enumerate(val_macro)) + " |",
         "", "## 결과 (F1, 괄호 = FP / FN)", "",
         "| 셋 | 방식 | " + " | ".join(CLASS_LIST) + " | macro F1 |", "|---|---|" + "---:|" * 4]
    for split in ("val", "test"):
        for k, lab in [("common", f"공통 {common_thr:g}"), ("class", "클래스별")]:
            r = res[split][k]
            L.append(f"| {'검증' if split == 'val' else '테스트'} | {lab} | " +
                     " | ".join(f"{r[n]['f1']:.3f} ({r[n]['fp']} / {r[n]['fn']})" for n in CLASS_LIST) +
                     f" | {np.mean([r[n]['f1'] for n in CLASS_LIST]):.3f} |")
    gain = {s: np.mean([res[s]["class"][n]["f1"] for n in CLASS_LIST]) -
            np.mean([res[s]["common"][n]["f1"] for n in CLASS_LIST]) for s in ("val", "test")}
    L += ["", f"- 클래스별 임계값의 macro F1 이득: 검증 {gain['val']:+.3f} · 테스트 {gain['test']:+.3f}"]
    (WEEK3 / "adv4_common_threshold.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # 그림: 테스트셋 클래스별 F1 막대 (공통 vs 클래스별)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(4)
    for j, (k, lab, col) in enumerate([("common", f"공통 {common_thr:g}", "#9e9e9e"), ("class", "클래스별", "#2c7fb8")]):
        r = res["test"][k]
        v = [r[n]["f1"] for n in CLASS_LIST]
        v.append(np.mean(v))
        b = ax.bar(x + (j - 0.5) * 0.36, v, 0.36, label=lab, color=col)
        ax.bar_label(b, fmt="%.3f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LIST + ["macro"])
    ax.set_ylim(0.5, 1.08)
    ax.set_ylabel("테스트 F1")
    ax.set_title("공통 임계값 vs 클래스별 임계값 (둘 다 검증셋에서 선택)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(WEEK3 / "adv4_common_threshold.png", dpi=150)
    plt.close()
    print("\n".join(L))
    log_run("adv4_common_threshold", f"common {common_thr} (val) → gain val {gain['val']:+.3f} test {gain['test']:+.3f}")


if __name__ == "__main__":
    main()
