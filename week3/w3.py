"""3주차 공통: 경로 · 영상별 정답 클래스 목록 · 예측 캐시 · ON/OFF 판정 함수(강의 10쪽).

★ 예측 신뢰도 하한: 강의 코드의 model.predict(p, imgsz=960)는 ultralytics 기본값 conf=0.25를 쓴다.
  그러면 0.25보다 낮은 임계값(0.001 ~ 0.2)은 모두 같은 예측 집합을 보게 되어 F1이 변하지 않는다
  (강의 16쪽 '임계값을 바꿔도 F1 동일'). 그래서 conf=0.001로 한 번 예측해 두고 임계값은 그 뒤에 적용한다.
  같은 영상을 임계값 · 클래스마다 다시 예측하지 않도록 결과를 preds_<split>.json 에 저장해 다시 쓴다
  (best.pt가 바뀌면 자동으로 다시 예측).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

WEEK3 = Path(__file__).resolve().parent              # crack-seg/week3/
ROOT = WEEK3.parent                                   # crack-seg/
sys.path.insert(0, str(ROOT / "week2"))
from paths import SPLIT_ROOT, CLASS_NAMES, common     # noqa: E402,F401  (common: 한글 글꼴 · imread)
import matplotlib                                     # noqa: E402

matplotlib.rcParams["axes.unicode_minus"] = False     # 한글 글꼴에 '−' 기호가 없어 로그 눈금 10^-3이 깨지는 것 방지

DATA_YAML = SPLIT_ROOT / "data.yaml"                  # 2주차 단계4가 만든 원본 ID 단위 분할
BEST_PT = ROOT / "runs" / "step3" / "weights" / "best.pt"
CLASS_LIST = [CLASS_NAMES[i] for i in range(3)]       # ["crack", "efflorescence", "spall_detach"]
IMGSZ = 960
PRED_CONF = 0.001                                     # 예측을 남기는 최소 신뢰도 (탐색 구간 최솟값과 같게)
THR_FILE = WEEK3 / "thresholds.json"                  # 단계3(val)이 쓰고 단계4 · 5(test)는 읽기만 한다
RUN_LOG = WEEK3 / "run_log.txt"                       # 실행 순서 기록 (임계값을 val에서 먼저 정했다는 증거)


def check_files():
    """강의 9쪽: 파일 존재 확인."""
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {DATA_YAML}")
    if not BEST_PT.exists():
        raise FileNotFoundError(f"best.pt를 찾을 수 없습니다: {BEST_PT}")


def log_run(step, msg=""):
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {step}  {msg}\n")


def load_split(split):
    """split 영상 경로 목록과 영상별 정답 클래스 ID 목록 (라벨 txt 각 줄의 첫 숫자).

    라벨 파일이 없거나 비어 있으면 정답 클래스 없음 = 세 클래스 모두 정답 OFF.
    """
    img_dir = SPLIT_ROOT / "images" / split
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    gts = []
    for p in paths:
        lab = SPLIT_ROOT / "labels" / split / (p.stem + ".txt")
        ids = []
        if lab.exists():
            for line in lab.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.append(int(line.split()[0]))
        gts.append(ids)
    return [str(p) for p in paths], gts


# ---------------------------------------------------------------- 예측 캐시
_CACHE = {}
_CUR = {"pt": BEST_PT, "tag": ""}                       # 심화(다른 가중치)는 use_weights()로 바꾼다


def use_weights(pt, tag):
    """평가할 가중치를 바꾼다. 캐시는 preds_<tag>_<split>.json 으로 따로 둔다 (기본 실험은 tag 없음)."""
    _CUR.update(pt=Path(pt), tag=tag)
    _CACHE.clear()


def _weights_key():
    st = _CUR["pt"].stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def _cache_file(split):
    return WEEK3 / "cache" / f"preds_{_CUR['tag']}_{split}.json" if _CUR["tag"] else WEEK3 / f"preds_{split}.json"


def load_cache(split):
    f = _cache_file(split)
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("weights") == _weights_key() and d.get("conf") == PRED_CONF and d.get("imgsz") == IMGSZ:
            _CACHE.update(d["preds"])


def save_cache(split, paths):
    preds = {Path(p).name: _CACHE[Path(p).name] for p in paths if Path(p).name in _CACHE}
    _cache_file(split).parent.mkdir(exist_ok=True)
    _cache_file(split).write_text(json.dumps(
        {"weights": _weights_key(), "conf": PRED_CONF, "imgsz": IMGSZ, "preds": preds}, ensure_ascii=False),
        encoding="utf-8")


def predict_conf_cls(model, p):
    """영상 1장의 예측 (conf 배열, cls 배열). 캐시에 있으면 다시 예측하지 않는다."""
    import numpy as np
    key = Path(p).name
    if key not in _CACHE:
        r = model.predict(p, imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
        _CACHE[key] = {"conf": r.boxes.conf.cpu().numpy().round(6).tolist(),
                       "cls": r.boxes.cls.cpu().numpy().astype(int).tolist()}
    c = _CACHE[key]
    return np.array(c["conf"], dtype=float), np.array(c["cls"], dtype=int)


# ---------------------------------------------------------------- 강의 10쪽 ON/OFF 함수 (예측 부분만 캐시로)
def onoff(model, paths, gts, thr, cid_target):
    tp = fp = fn = tn = 0
    for p, gt in zip(paths, gts):
        # 예측 수행 (conf=0.001로 한 번 예측한 결과를 다시 씀 — 위 설명 참고)
        conf, cid = predict_conf_cls(model, p)

        # 해당 클래스의 예측 중 threshold 이상이 하나라도 있으면 ON
        pred_on = bool(((cid == cid_target) & (conf >= thr)).any())
        # 정답 라벨에 해당 클래스가 있으면 ON
        gt_on = cid_target in gt

        # TP / FP / FN / TN 계산
        if gt_on and pred_on:
            tp += 1
        elif not gt_on and pred_on:
            fp += 1
        elif gt_on and not pred_on:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def load_model():
    from ultralytics import YOLO
    check_files()
    return YOLO(str(_CUR["pt"]))
