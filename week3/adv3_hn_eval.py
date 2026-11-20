"""3주차 심화 ③ 평가: 하드네거티브를 넣고 다시 학습한 모델(runs/step3_hn)이 오탐을 줄였는가.

  python week3/adv_eval_run.py --run step3_hn --tag hn      # (먼저 — 본 실험과 같은 절차, 검증셋 임계값)
  python week3/adv3_hn_eval.py                               # → adv3_hn_eval.md · adv3_hn_eval.png
비교 대상은 기준 실험(seed 42, 같은 설정 · 같은 시드, 데이터만 다름)과 시드 3개 평균(⑤ — 시드 차이만으로 생기는 변동 폭 참고).
오탐 측정 ① 테스트 ON/OFF의 FP  ② 평가 전용 음성 조각(val · test 영상에서 라벨과 겹치지 않는 조각, 학습 · 선택에 안 씀)
  중 각 모델의 검증 임계값 이상 예측이 나온 조각 비율.
"""
import _use_venv  # noqa: F401  (반드시 첫 줄: Step 3 전용 가상환경으로 다시 실행)

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from w3 import CLASS_LIST, IMGSZ, PRED_CONF, ROOT, SPLIT_ROOT, WEEK3, load_model, use_weights

NEG_EVAL = SPLIT_ROOT.parent / "crack-seg-neg-eval"


def neg_fp(run, thr):
    use_weights(ROOT / "runs" / run / "weights" / "best.pt", f"neg_{run}")
    model = load_model()
    out = {}
    for split in ("val", "test"):
        files = sorted((NEG_EVAL / split).glob("*.jpg"))
        hit = np.zeros(3, int)
        anyhit = 0
        for f in files:
            r = model.predict(str(f), imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
            on = set(int(c) for c, s in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())
                     if s >= thr[CLASS_LIST[int(c)]])
            for c in on:
                hit[c] += 1
            anyhit += bool(on)
        out[split] = {"n": len(files), "any": int(anyhit), **{n: int(hit[i]) for i, n in enumerate(CLASS_LIST)}}
    return out


def main():
    base = json.loads((WEEK3 / "adv_runs" / "seed42.json").read_text(encoding="utf-8"))
    hn = json.loads((WEEK3 / "adv_runs" / "hn.json").read_text(encoding="utf-8"))
    seeds = json.loads((WEEK3 / "adv5_seeds.json").read_text(encoding="utf-8")) if (WEEK3 / "adv5_seeds.json").exists() else {}
    neg = {"기준": neg_fp("step3", base["thresholds"]), "하드네거티브": neg_fp("step3_hn", hn["thresholds"])}

    L = ["# 심화 ③ 하드네거티브 추가 학습 결과", "",
         "- 기준: runs/step3 (seed 42) · 하드네거티브: runs/step3_hn (같은 설정 · seed 42, train에 배경 영상 54장 추가 — 마이닝 108장 중 사람이 검토해 남긴 15장 + 1주차 3건 = 18개 × 3, adv3_hn_curate.md)",
         "- 두 모델 모두 검증셋에서 임계값을 따로 고르고 테스트셋에 고정 적용", "",
         "| 지표 | 기준 | 하드네거티브 |" + (" 시드 3개 평균 ± SD (참고) |" if seeds else ""), "|---|---:|---:|" + ("---:|" if seeds else "")]

    def row(lab, a, b, key=None, dig=3):
        s = f" {seeds[key][0]:.{dig}f} ± {seeds[key][1]:.{dig}f} |" if seeds and key in seeds else (" - |" if seeds else "")
        L.append(f"| {lab} | {a:.{dig}f} | {b:.{dig}f} |" + s)

    row("best.pt epoch", base.get("best_epoch", 0), hn.get("best_epoch", 0), "best.pt epoch", 0)
    row("테스트 mask mAP50", base["seg_test"]["mask_mAP50"], hn["seg_test"]["mask_mAP50"], "테스트 mask mAP50")
    row("테스트 mask mAP50-95", base["seg_test"]["mask_mAP50-95"], hn["seg_test"]["mask_mAP50-95"], "테스트 mask mAP50-95")
    L.append("| 선택 임계값 (crack / efflo / spall) | " + " / ".join(f"{base['thresholds'][n]:g}" for n in CLASS_LIST) + " | "
             + " / ".join(f"{hn['thresholds'][n]:g}" for n in CLASS_LIST) + " |" + (" - |" if seeds else ""))
    for n in CLASS_LIST:
        L.append(f"| 테스트 {n} F1 (FP / FN) | {base['test'][n]['f1']:.3f} ({base['test'][n]['fp']} / {base['test'][n]['fn']}) | "
                 f"{hn['test'][n]['f1']:.3f} ({hn['test'][n]['fp']} / {hn['test'][n]['fn']}) |" + (" - |" if seeds else ""))
    row("테스트 macro F1", base["test_macro_f1"], hn["test_macro_f1"], "테스트 macro F1")
    L.append(f"| 테스트 FP 합 (세 클래스) | {sum(base['test'][n]['fp'] for n in CLASS_LIST)} | "
             f"{sum(hn['test'][n]['fp'] for n in CLASS_LIST)} |" + (" - |" if seeds else ""))
    L += ["", "## 평가 전용 음성 조각에서의 오탐 (결함 없는 조각인데 결함으로 예측한 비율)", "",
          "| 셋 | 조각 수 | 모델 | 하나라도 | crack | efflorescence | spall_detach |", "|---|---:|---|---:|---:|---:|---:|"]
    for split in ("val", "test"):
        for k in neg:
            d = neg[k][split]
            L.append(f"| {split} | {d['n']} | {k} | {d['any']} ({d['any'] / d['n']:.0%}) | " +
                     " | ".join(str(d[n]) for n in CLASS_LIST) + " |")
    (WEEK3 / "adv3_hn_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (WEEK3 / "adv3_hn_eval.json").write_text(json.dumps(neg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    x = np.arange(3)
    for j, (k, r) in enumerate([("기준", base), ("하드네거티브", hn)]):
        b = ax[0].bar(x + (j - 0.5) * 0.36, [r["test"][n]["fp"] for n in CLASS_LIST], 0.36, label=k)
        ax[0].bar_label(b)
        tot = [neg[k][s]["any"] / neg[k][s]["n"] for s in ("val", "test")]
        b2 = ax[1].bar(np.arange(2) + (j - 0.5) * 0.36, tot, 0.36, label=k)
        ax[1].bar_label(b2, fmt="%.2f")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(CLASS_LIST)
    ax[0].set_title("테스트 ON/OFF 오탐(FP) 영상 수")
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels([f"val 음성 조각 {neg['기준']['val']['n']}개", f"test 음성 조각 {neg['기준']['test']['n']}개"])
    ax[1].set_title("음성 조각 중 결함으로 예측한 비율")
    for a in ax:
        a.legend()
    plt.tight_layout()
    plt.savefig(WEEK3 / "adv3_hn_eval.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
