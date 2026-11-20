"""심화 ⑦ 요약: 직접 주석 2장 미세조정(hand) vs 대조군(provided), 시드 42 · 1 · 2 → adv7_hand_ft.md

  python week3/adv7_summary.py      (adv_runs/ft_<변형>[_s<시드>]_last.json 을 읽기만 한다)
"""
import json

import numpy as np

from w3 import CLASS_LIST, WEEK3

RUNS = WEEK3 / "adv_runs"
SEEDS = [42, 1, 2]
KEYS = [("test mask mAP50", lambda d: d["seg_test"]["mask_mAP50"]),
        ("test mask mAP50-95", lambda d: d["seg_test"]["mask_mAP50-95"]),
        *[(f"test {n} mAP50", (lambda n: lambda d: d["seg_test"]["class_mask_mAP50"][n])(n)) for n in CLASS_LIST],
        ("val mask mAP50", lambda d: d["seg_val"]["mask_mAP50"]),
        ("test macro F1", lambda d: d["test_macro_f1"]),
        *[(f"test {n} F1", (lambda n: lambda d: d["test"][n]["f1"])(n)) for n in CLASS_LIST]]


def load(tag):
    p = RUNS / f"{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    base = load("seed42")
    runs = {v: {s: load(f"ft_{v}" + ("" if s == 42 else f"_s{s}") + "_last") for s in SEEDS} for v in ("hand", "provided")}
    lines = ["# 심화 ⑦ 직접 주석 2장 미세조정 — 시드 반복 (last.pt, epoch 10)", "",
             "두 변형은 대상 2장의 라벨만 다르다(hand = 직접 주석, provided = 제공 라벨). 재생 98장 · 설정 동일, 학습 시드만 42 · 1 · 2.", "",
             "| 지표 | 기준 모델 | hand 평균 ± 표준편차 | provided 평균 ± 표준편차 | hand − provided (시드별) |",
             "|---|---|---|---|---|"]
    for name, f in KEYS:
        h = [f(runs["hand"][s]) for s in SEEDS if runs["hand"][s] and runs["provided"][s]]
        p = [f(runs["provided"][s]) for s in SEEDS if runs["hand"][s] and runs["provided"][s]]
        diff = " · ".join(f"{a - b:+.3f}" for a, b in zip(h, p))
        lines.append(f"| {name} | {f(base):.3f} | {np.mean(h):.3f} ± {np.std(h):.3f} | {np.mean(p):.3f} ± {np.std(p):.3f} | {diff} |")
    lines += ["", "시드별 원자료", "", "| 변형 | 시드 | test mAP50 | mAP50-95 | macro F1 | 임계값 |", "|---|---|---|---|---|---|"]
    for v in runs:
        for s in SEEDS:
            d = runs[v][s]
            if d:
                lines.append(f"| {v} | {s} | {d['seg_test']['mask_mAP50']:.3f} | {d['seg_test']['mask_mAP50-95']:.3f} | "
                             f"{d['test_macro_f1']:.3f} | {d['thresholds']} |")
    (WEEK3 / "adv7_hand_ft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
