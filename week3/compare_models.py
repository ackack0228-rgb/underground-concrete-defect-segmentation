"""최종 리포트(README) '모델 성능 비교' · 'Confusion Matrix' 절: 모델별 영상 단위 ON/OFF 지표 + 2×2 혼동행렬 + bootstrap 95% CI.

  python week3/compare_models.py        # → compare_models.md · compare_models.json · compare_confusion.png · compare_bootstrap.png
- 대상 모델(모두 같은 테스트 40장, 임계값은 각 모델이 검증셋에서 고른 값 그대로 — 여기서 새로 고르지 않는다):
  YOLOv8s-Seg 기준(runs/step3, thresholds = adv_runs/seed42.json) · YOLOv8s-Seg + 하드네거티브(runs/step3_hn, adv_runs/hn.json)
  · U-Net 최종(runs/unet_aug, adv2_unet_aug.json — 검증 mIoU로 고른 쪽) · U-Net 약한 증강(runs/unet, adv2_unet.json, 참고)
- 클래스마다 영상 단위 ON/OFF 2×2 표(TP · FP · FN · TN) → Accuracy = (TP+TN)/40 · Precision · Recall · F1.
  '완전 일치 Accuracy' = 세 클래스 판정이 모두 맞은 영상 비율 (다중 라벨이라 하나의 Accuracy로 줄일 때).
- bootstrap: 테스트 40장을 복원추출(2,000번, seed 0)해 macro F1을 다시 계산 → 2.5 · 97.5 백분위수 = 95% CI.
  기준 YOLO와의 차이도 같은 표본으로 짝지어 계산 (차이의 CI가 0을 포함하면 40장으로는 차이를 말하기 어렵다).
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from adv2_unet_eval import unet_probs
from adv2_unet_train import make_model
from w3 import CLASS_LIST, ROOT, WEEK3, load_cache, load_model, load_split, predict_conf_cls, save_cache, use_weights

N_BOOT, SEED = 2000, 0


def yolo_on(run, tag, thr_json, imgs):
    thr = json.loads((WEEK3 / thr_json).read_text(encoding="utf-8"))["thresholds"]
    use_weights(ROOT / "runs" / run / "weights" / "best.pt", tag)
    load_cache("test")
    model = None
    on = []
    for p in imgs:
        try:
            conf, cls = predict_conf_cls(None, p)              # 캐시에 있으면 모델 없이
        except AttributeError:
            model = model or load_model()
            conf, cls = predict_conf_cls(model, p)
        on.append([bool(((cls == c) & (conf >= thr[n])).any()) for c, n in enumerate(CLASS_LIST)])
    save_cache("test", imgs)
    return np.array(on), thr


def unet_on(run, thr_json, imgs):
    thr = json.loads((WEEK3 / thr_json).read_text(encoding="utf-8"))["unet"]["thresholds"]
    m = make_model(pretrained=False).cuda().eval()
    m.load_state_dict(torch.load(ROOT / "runs" / run / "weights" / "best.pt", map_location="cuda", weights_only=False)["model"])
    on = []
    for p in imgs:
        area = (unet_probs(m, p) >= 0.5).reshape(3, -1).mean(1)   # adv2_unet_eval의 주 지표(면적 비율)와 같은 점수
        on.append([bool(area[c] >= thr[n]) for c, n in enumerate(CLASS_LIST)])
    return np.array(on), thr


def stats(on, gt):
    out = {}
    for c, n in enumerate(CLASS_LIST):
        p, g = on[:, c], gt[:, c]
        tp, fp, fn, tn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()), int((~p & ~g).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[n] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / len(g), "precision": pr, "recall": rc,
                  "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0}
    out["macro"] = {k: float(np.mean([out[n][k] for n in CLASS_LIST])) for k in ("accuracy", "precision", "recall", "f1")}
    out["macro"]["exact_match"] = float((on == gt).all(1).mean())
    return out


def macro_f1(on, gt):
    return stats(on, gt)["macro"]["f1"]


def main():
    imgs, gts = load_split("test")
    gt = np.array([[c in g for c in range(3)] for g in gts])
    models = {
        "YOLOv8s-Seg (기준)": yolo_on("step3", "seed42", "adv_runs/seed42.json", imgs),
        "YOLOv8s-Seg + 하드네거티브": yolo_on("step3_hn", "hn", "adv_runs/hn.json", imgs),
        "U-Net (최종 · 강한 증강)": unet_on("unet_aug", "adv2_unet_aug.json", imgs),
        "U-Net (약한 증강 · 참고)": unet_on("unet", "adv2_unet.json", imgs),
    }
    base = "YOLOv8s-Seg (기준)"
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(imgs), (N_BOOT, len(imgs)))
    R = {}
    for name, (on, thr) in models.items():
        s = stats(on, gt)
        boot = np.array([macro_f1(on[i], gt[i]) for i in idx])
        diff = boot - np.array([macro_f1(models[base][0][i], gt[i]) for i in idx])
        s["thresholds"] = thr
        s["boot_macro_f1_ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        s["boot_diff_vs_base_ci"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        s["per_image_on"] = {Path(p).name: o.tolist() for p, o in zip(imgs, on)}
        R[name] = s
    (WEEK3 / "compare_models.json").write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")

    # 확인: 각 모델 결과가 원래 평가 파일과 같은가 (같은 임계값 · 같은 점수라 같아야 한다)
    ref = {"YOLOv8s-Seg (기준)": json.loads((WEEK3 / "adv_runs/seed42.json").read_text(encoding="utf-8"))["test"],
           "YOLOv8s-Seg + 하드네거티브": json.loads((WEEK3 / "adv_runs/hn.json").read_text(encoding="utf-8"))["test"],
           "U-Net (최종 · 강한 증강)": json.loads((WEEK3 / "adv2_unet_aug.json").read_text(encoding="utf-8"))["unet"]["test"],
           "U-Net (약한 증강 · 참고)": json.loads((WEEK3 / "adv2_unet.json").read_text(encoding="utf-8"))["unet"]["test"]}
    for name in R:
        for n in CLASS_LIST:
            assert all(R[name][n][k] == ref[name][n][k] for k in ("tp", "fp", "fn", "tn")), (name, n)

    L = ["# 모델 성능 비교 (테스트 40장, 영상 단위 ON/OFF)", "",
         "- 임계값은 각 모델이 **검증셋에서 고른 값 그대로**(여기서 새로 고르지 않음). U-Net 점수 = 확률 0.5 이상 면적 비율(adv2_unet_eval).",
         "- Accuracy = (TP + TN) / 40. 완전 일치 = 세 클래스 판정이 모두 맞은 영상 비율. 95% CI = 테스트 영상 bootstrap 2,000번.",
         "- 각 모델의 TP · FP · FN · TN이 원래 평가 파일(adv_runs/*.json · adv2_unet*.json)과 같음을 스크립트에서 확인함.", "",
         "## 표 1. macro 평균 (세 클래스)", "",
         "| 모델 | Accuracy | Precision | Recall | F1 | 완전 일치 | macro F1 95% CI | 기준과 차이 95% CI |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for name, s in R.items():
        m = s["macro"]
        ci, d = s["boot_macro_f1_ci"], s["boot_diff_vs_base_ci"]
        L.append(f"| {name} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['exact_match']:.3f} "
                 f"| {ci[0]:.3f} \\~ {ci[1]:.3f} | {'-' if name == base else f'{d[0]:+.3f} \\~ {d[1]:+.3f}'} |")
    L += ["", "## 표 2. 클래스별 (Accuracy / Precision / Recall / F1, 괄호 = TP · FP · FN · TN)", "",
          "| 모델 | " + " | ".join(CLASS_LIST) + " |", "|---|" + "---|" * 3]
    for name, s in R.items():
        L.append(f"| {name} | " + " | ".join(
            f"{s[n]['accuracy']:.3f} / {s[n]['precision']:.3f} / {s[n]['recall']:.3f} / {s[n]['f1']:.3f} "
            f"({s[n]['tp']} · {s[n]['fp']} · {s[n]['fn']} · {s[n]['tn']})" for n in CLASS_LIST) + " |")
    L += ["", "## 모델별 가장 잘 맞힌 · 가장 많이 틀린 클래스 (F1 기준, 동점이면 오류 수)", ""]
    for name, s in R.items():
        key = lambda n: (s[n]["f1"], -(s[n]["fp"] + s[n]["fn"]))
        best, worst = max(CLASS_LIST, key=key), min(CLASS_LIST, key=key)
        L.append(f"- {name}: 가장 잘 = {best} (F1 {s[best]['f1']:.3f}) · 가장 많이 틀림 = {worst} "
                 f"(F1 {s[worst]['f1']:.3f}, FP {s[worst]['fp']} · FN {s[worst]['fn']})")
    (WEEK3 / "compare_models.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # 그림 1: 모델 × 클래스 2×2 혼동행렬
    fig, axes = plt.subplots(len(R), 3, figsize=(9.5, 2.55 * len(R)))
    for r, (name, s) in enumerate(R.items()):
        for c, n in enumerate(CLASS_LIST):
            ax = axes[r, c]
            cm = np.array([[s[n]["tp"], s[n]["fn"]], [s[n]["fp"], s[n]["tn"]]])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=40)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{['TP', 'FN'][j] if i == 0 else ['FP', 'TN'][j]}\n{cm[i, j]}", ha="center", va="center",
                            fontsize=10, color="white" if cm[i, j] > 20 else "black")
            ax.set_xticks([0, 1], ["예측 ON", "예측 OFF"], fontsize=8)
            ax.set_yticks([0, 1], ["정답 ON", "정답 OFF"], fontsize=8)
            ax.set_title(f"{n}  F1 {s[n]['f1']:.3f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=9, labelpad=12)
    fig.suptitle("테스트 40장 영상 단위 ON/OFF 혼동행렬 (임계값 = 각 모델의 검증셋 선택값)", fontsize=11)
    plt.tight_layout()
    plt.savefig(WEEK3 / "compare_confusion.png", dpi=120)
    plt.close()

    # 그림 2: macro F1 + 95% CI
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    names = list(R)
    v = [R[n]["macro"]["f1"] for n in names]
    lo = [R[n]["macro"]["f1"] - R[n]["boot_macro_f1_ci"][0] for n in names]
    hi = [R[n]["boot_macro_f1_ci"][1] - R[n]["macro"]["f1"] for n in names]
    ax.barh(range(len(names)), v, xerr=[lo, hi], capsize=4, color=["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"])
    for i, x in enumerate(v):
        ax.text(0.505, i, f"{x:.3f}", va="center", fontsize=9, color="white")
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("테스트 macro F1 (막대 = 95% bootstrap CI)")
    plt.tight_layout()
    plt.savefig(WEEK3 / "compare_bootstrap.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
