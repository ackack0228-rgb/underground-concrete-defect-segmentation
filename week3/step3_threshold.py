"""3주차 단계3 (강의 11쪽): 검증셋에서만 클래스별 임계값 탐색 → thresholds.json · val_threshold_f1.png.

  python week3/step3_threshold.py
★ 이 스크립트는 테스트셋을 읽지 않는다 (load_split("val")만). 선택한 임계값은 thresholds.json에 저장하고,
  단계4 · 5는 그 파일을 읽기만 한다. 실행 시각은 run_log.txt에 남는다.
덧붙임 (강의 16쪽 '임계값을 바꿔도 F1 동일 → confidence 분포 확인'): 클래스별 최고 신뢰도 분포를 val_conf_dist.png로 그린다.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import hashlib
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from w3 import (BEST_PT, CLASS_LIST, PRED_CONF, ROOT, THR_FILE, WEEK3, load_cache, load_model, load_split,
                log_run, onoff, predict_conf_cls, save_cache)

# ---------------------------------------------------------------- 강의 11쪽 코드
THRS = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]
class_names = CLASS_LIST


def main():
    model = load_model()
    val_imgs, val_gts = load_split("val")             # ★ 검증셋만
    load_cache("val")

    best = {}
    all_scores = {}
    detail = {}
    for cid, name in enumerate(class_names):
        scores = []
        detail[name] = []
        for t in THRS:
            r = onoff(model, val_imgs, val_gts, t, cid)
            scores.append(r["f1"])
            detail[name].append({"thr": t, **r})
            print(f"{name} | thr={t:.3f} | F1={r['f1']:.3f} | FP={r['fp']} | FN={r['fn']}")
        all_scores[name] = scores

        # 최고 F1
        max_f1 = max(scores)
        # 동점이면 더 낮은 threshold 선택 (Recall 우선)
        best_idx = next(i for i, s in enumerate(scores) if s == max_f1)
        best[name] = THRS[best_idx]
        print(f">> {name} 선택 임계값: {THRS[best_idx]} (F1={max_f1:.3f})\n")
    print("최종 임계값:", best)
    save_cache("val", val_imgs)

    # F1 곡선 시각화 (강의 코드 + 선택점 표시 · 논문 값 비교를 위해 x축 로그 눈금)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = {"crack": "#d62728", "efflorescence": "#1f77b4", "spall_detach": "#e6a700"}
    for name in class_names:
        ax.plot(THRS, all_scores[name], marker="o", label=name, color=colors[name])
        i = THRS.index(best[name])
        ax.scatter([THRS[i]], [all_scores[name][i]], s=180, facecolors="none", edgecolors=colors[name], linewidths=2)
        ax.annotate(f"{best[name]:g}", (THRS[i], all_scores[name][i]), textcoords="offset points", xytext=(6, 8),
                    color=colors[name], fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(THRS)
    ax.set_xticklabels([f"{t:g}" for t in THRS])
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1-score")
    ax.set_title("Validation F1 by Threshold (val 40장, ○ = 선택)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(WEEK3 / "val_threshold_f1.png", dpi=150)
    plt.close()

    # confidence 분포: 영상별로 그 클래스 예측의 최고 신뢰도 (정답 ON/OFF로 나눠서)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    bins = np.logspace(np.log10(PRED_CONF), 0, 25)
    for cid, name in enumerate(class_names):
        on, off = [], []
        for p, g in zip(val_imgs, val_gts):
            conf, cls = predict_conf_cls(model, p)
            m = conf[cls == cid].max() if (cls == cid).any() else PRED_CONF * 0.999
            (on if cid in g else off).append(max(m, PRED_CONF))
        ax = axes[cid]
        ax.hist([on, off], bins=bins, label=["정답 ON 영상", "정답 OFF 영상"], color=[colors[name], "#999999"])
        ax.axvline(best[name], color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks([0.001, 0.01, 0.1, 1])
        ax.set_xticklabels(["0.001\n(예측 없음 포함)", "0.01", "0.1", "1"])     # mathtext 10^-3 대신 (한글 글꼴에서 깨짐)
        ax.minorticks_off()
        ax.set_title(f"{name} (선택 {best[name]:g})")
        ax.set_xlabel("영상별 최고 신뢰도")
    axes[0].set_ylabel("영상 수")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(WEEK3 / "val_conf_dist.png", dpi=130)
    plt.close()

    # 저장: 단계4 · 5는 이 파일만 읽는다
    h = hashlib.sha256("".join(sorted(val_imgs)).encode()).hexdigest()[:12]
    THR_FILE.write_text(json.dumps({
        "thresholds": best, "selected_on": "val", "n_val_images": len(val_imgs), "val_list_sha": h,
        "thr_grid": THRS, "tie_rule": "최고 F1 동점이면 낮은 임계값", "weights": str(BEST_PT.relative_to(ROOT)),
        "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, indent=2, ensure_ascii=False), encoding="utf-8")

    # 표 (리포트 ④)
    L = ["# 단계3 검증셋 임계값 탐색 (val 40장만 사용)", "",
         f"- 탐색 구간: {', '.join(f'{t:g}' for t in THRS)} (강의 11쪽 = 논문 Table 3)",
         "- 규칙: 클래스마다 F1이 가장 큰 임계값, 동점이면 낮은 쪽(Recall 우선)",
         f"- 선택: " + " · ".join(f"{n} {best[n]:g}" for n in class_names) +
         " (논문: crack 0.010 · efflorescence 0.004 · spall_detach 0.060)", ""]
    L += ["## 임계값별 F1", "", "| 임계값 | " + " | ".join(class_names) + " |", "|---:|" + "---:|" * 3]
    for i, t in enumerate(THRS):
        cells = []
        for n in class_names:
            s = f"{all_scores[n][i]:.3f}"
            cells.append(f"**{s}**" if t == best[n] else s)
        L.append(f"| {t:g} | " + " | ".join(cells) + " |")
    L += ["", "## 선택 임계값에서의 검증 성능 (논문 Table 2와 같은 형식)", "",
          "| 지표 | " + " | ".join(class_names) + " |", "|---|" + "---:|" * 3]
    sel = {n: next(d for d in detail[n] if d["thr"] == best[n]) for n in class_names}
    L.append("| 선택 임계값 | " + " | ".join(f"{best[n]:g}" for n in class_names) + " |")
    for k, lab in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]:
        L.append(f"| {lab} | " + " | ".join(f"{sel[n][k]:.3f}" for n in class_names) + " |")
    for k, lab in [("tp", "TP"), ("fp", "False Positive"), ("fn", "False Negative"), ("tn", "TN")]:
        L.append(f"| {lab} | " + " | ".join(str(sel[n][k]) for n in class_names) + " |")
    L += ["", f"- 검증 macro F1 (선택 임계값): {np.mean([sel[n]['f1'] for n in class_names]):.3f} (논문 0.878)", "",
          "## 클래스·임계값별 세부 (TP / FP / FN / TN)", "", "| 임계값 | " + " | ".join(class_names) + " |",
          "|---:|" + "---|" * 3]
    for i, t in enumerate(THRS):
        L.append(f"| {t:g} | " + " | ".join(
            f"{detail[n][i]['tp']} / {detail[n][i]['fp']} / {detail[n][i]['fn']} / {detail[n][i]['tn']}"
            for n in class_names) + " |")
    (WEEK3 / "val_threshold.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (WEEK3 / "val_threshold_detail.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    print("\n".join(L))
    log_run("step3_threshold", f"val only → {best} (sha {h})")


if __name__ == "__main__":
    main()
