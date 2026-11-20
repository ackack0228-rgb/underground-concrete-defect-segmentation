"""3주차 심화 ⑤ (논문강의 47쪽 '반복학습'): 시드 3개(42 · 0 · 1)로 같은 학습을 반복해 평균 ± 표준편차.

  python week3/adv_eval_run.py --run step3 --tag seed42        # (각 시드를 같은 절차로 평가 — 먼저)
  python week3/adv_eval_run.py --run step3_seed0 --tag seed0
  python week3/adv_eval_run.py --run step3_seed1 --tag seed1
  python week3/adv5_seed_summary.py                            # → adv5_seeds.md · adv5_seeds.png
시드마다 검증셋에서 임계값을 따로 고르고 테스트에 고정 적용한다 (본 실험과 같은 절차). 표준편차는 표본 표준편차(n-1).
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json

import matplotlib.pyplot as plt
import numpy as np

from w3 import CLASS_LIST, WEEK3

TAGS = ["seed42", "seed0", "seed1"]


def main():
    R = {t: json.loads((WEEK3 / "adv_runs" / f"{t}.json").read_text(encoding="utf-8")) for t in TAGS}
    metrics = {
        "best.pt epoch": lambda r: r.get("best_epoch", np.nan),
        "테스트 mask mAP50": lambda r: r["seg_test"]["mask_mAP50"],
        "테스트 mask mAP50-95": lambda r: r["seg_test"]["mask_mAP50-95"],
        "검증 mask mAP50": lambda r: r["seg_val"]["mask_mAP50"],
        **{f"테스트 mask mAP50 {n}": (lambda n: lambda r: r["seg_test"]["class_mask_mAP50"][n])(n) for n in CLASS_LIST},
        "검증 macro F1 (선택 임계값)": lambda r: r["val_macro_f1"],
        **{f"테스트 F1 {n}": (lambda n: lambda r: r["test"][n]["f1"])(n) for n in CLASS_LIST},
        "테스트 macro F1": lambda r: r["test_macro_f1"],
    }
    L = ["# 심화 ⑤ 반복학습 (시드 3개)", "",
         "- 설정은 모두 같다(2주차 단계5 = 논문 Table 1). 시드만 42(기준 실험) · 0 · 1. best.pt = 검증 fitness 최고 epoch.",
         "- 시드마다 검증셋에서 클래스별 임계값을 고른 뒤 테스트셋에 고정 적용.", "",
         "| 지표 | " + " | ".join(f"seed {t[4:]}" for t in TAGS) + " | 평균 ± 표준편차 |", "|---|" + "---:|" * (len(TAGS) + 1)]
    summary = {}
    for lab, f in metrics.items():
        v = np.array([f(R[t]) for t in TAGS], float)
        summary[lab] = (float(v.mean()), float(v.std(ddof=1)))
        dig = 0 if "epoch" in lab else 3
        L.append(f"| {lab} | " + " | ".join(f"{x:.{dig}f}" for x in v) + f" | {v.mean():.{dig}f} ± {v.std(ddof=1):.{dig}f} |")
    L += ["", "## 시드별 선택 임계값 (검증셋)", "", "| 시드 | " + " | ".join(CLASS_LIST) + " |", "|---|" + "---:|" * 3]
    for t in TAGS:
        L.append(f"| {t[4:]} | " + " | ".join(f"{R[t]['thresholds'][n]:g}" for n in CLASS_LIST) + " |")
    m, s = summary["테스트 macro F1"]
    L += ["", f"- 테스트 macro F1 = {m:.3f} ± {s:.3f}. 테스트 40장에서 영상 1장 = 클래스별 F1 약 0.03 ~ 0.04 차이라 "
          "시드 간 차이의 상당 부분은 영상 1 ~ 2장 차이다."]
    (WEEK3 / "adv5_seeds.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (WEEK3 / "adv5_seeds.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    labs = ["mask mAP50", "mask mAP50-95"]
    for i, t in enumerate(TAGS):
        ax[0].scatter(range(2), [R[t]["seg_test"]["mask_mAP50"], R[t]["seg_test"]["mask_mAP50-95"]], label=f"seed {t[4:]}", s=50, zorder=3)
        ax[1].scatter(range(4), [R[t]["test"][n]["f1"] for n in CLASS_LIST] + [R[t]["test_macro_f1"]], s=50, zorder=3, label=f"seed {t[4:]}")
    mm = [summary["테스트 mask mAP50"], summary["테스트 mask mAP50-95"]]
    ax[0].errorbar(range(2), [a for a, _ in mm], [b for _, b in mm], fmt="_", color="k", capsize=8, ms=30, label="평균 ± SD")
    ff = [summary[f"테스트 F1 {n}"] for n in CLASS_LIST] + [summary["테스트 macro F1"]]
    ax[1].errorbar(range(4), [a for a, _ in ff], [b for _, b in ff], fmt="_", color="k", capsize=8, ms=30, label="평균 ± SD")
    ax[0].set_xticks(range(2))
    ax[0].set_xticklabels(labs)
    ax[0].set_title("테스트 세그멘테이션 (mask)")
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(CLASS_LIST + ["macro"])
    ax[1].set_title("테스트 ON/OFF F1 (시드별 검증 임계값)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(WEEK3 / "adv5_seeds.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
