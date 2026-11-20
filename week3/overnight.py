"""밤새 무인 실행 (2026-11-19 사용자 "자고 일어날 거니까 멈추지 말고 끝까지"):
train_queue.txt 작업을 차례로 학습하고, 하나 끝날 때마다 그 모델의 평가를 바로 돌린다.

  python week3/overnight.py        # 백그라운드(분리 프로세스)로. 다시 실행하면 끝난 학습 · 평가는 건너뛴다.
train_queue.py와 다른 점:
  · 학습이 오류로 끝나면 --resume으로 2번까지 다시 시도하고, 그래도 안 되면 그 작업만 건너뛰고 다음으로 간다(큐 전체가 멈추지 않음).
    U-Net이 CUDA 메모리 부족이면 batch 2로 다시.
  · 실행 중 Windows 절전 진입을 막는다(SetThreadExecutionState).
  · 평가: unet → adv2_unet_eval / step3_hn → adv_eval_run(hn) → adv3_hn_eval / seed0 · seed1 → adv_eval_run → adv5_seed_summary
    → (seed 요약이 생기면) adv3_hn_eval 다시. 마지막에 GPU가 한가할 때 타일 추론 시간 다시 재기(adv1_tile) → adv2_unet_eval 다시.
기록: week3/overnight.log (+ 각 학습 runs/<이름>.log, 각 평가 week3/overnight_logs/<단계>.log), 끝난 평가는 week3/overnight_done.txt.
"""
import ctypes
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WEEK3 = Path(__file__).resolve().parent
ROOT = WEEK3.parent
sys.path.insert(0, str(WEEK3))
import train_queue as tq  # noqa: E402  (jobs() · battery() 재사용)

PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = WEEK3 / "overnight.log"
EVAL_DONE = WEEK3 / "overnight_done.txt"
ELOG = WEEK3 / "overnight_logs"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake():
    try:  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED — 화면은 꺼져도 시스템은 절전 안 함
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:
        log(f"절전 방지 설정 실패: {e}")


def done_evals():
    return set(EVAL_DONE.read_text(encoding="utf-8").split()) if EVAL_DONE.exists() else set()


def run_eval(key, script, *args, force=False):
    if key in done_evals() and not force:
        return True
    ELOG.mkdir(exist_ok=True)
    log(f"  평가 {key}: {script} {' '.join(args)}")
    with open(ELOG / f"{key}.log", "a", encoding="utf-8") as lf:
        rc = subprocess.call([str(PY), str(WEEK3 / script), *args], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=ENV)
    if rc == 0:
        with open(EVAL_DONE, "a", encoding="utf-8") as f:
            f.write(key + "\n")
        log(f"  평가 완료 {key}")
        return True
    log(f"  평가 오류 {key} (코드 {rc}) → overnight_logs/{key}.log, 다음으로 진행")
    return False


def train(name, script, args):
    """성공하면 True. 실패 시 --resume으로 2번까지 재시도."""
    for attempt in range(3):
        a = list(args)
        if (ROOT / "runs" / name / "weights" / "last.pt").exists():
            a = ["--name", name, "--resume"]
            if attempt > 0 and name == "unet" and "out of memory" in tail(name):
                a += ["--batch", "2"]
        cmd = [str(PY), str(WEEK3 / script), *a]
        log(f"학습 시작 {name} (시도 {attempt + 1}): {' '.join(cmd[1:])}")
        with open(ROOT / "runs" / f"{name}.log", "a", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=ENV)
            while p.poll() is None:
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pct, ac = tq.battery()
                    if not ac and pct < tq.MIN_BATTERY:
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                        log(f"배터리 {pct}% (충전기 없음) → 중단. 충전 후 overnight.py 다시 실행하면 이어짐")
                        sys.exit(2)
        if p.returncode == 0:
            with open(tq.DONE, "a", encoding="utf-8") as f:
                f.write(name + "\n")
            log(f"학습 완료 {name}")
            return True
        log(f"학습 오류 {name} (코드 {p.returncode}) 마지막 줄: {tail(name)[-300:]!r}")
    log(f"학습 {name} 3번 실패 → 건너뜀")
    return False


def tail(name, n=4000):
    p = ROOT / "runs" / f"{name}.log"
    return p.read_text(encoding="utf-8", errors="replace")[-n:] if p.exists() else ""


def evals_after(name):
    if name == "unet":
        run_eval("unet", "adv2_unet_eval.py")
    elif name == "unet_aug":
        run_eval("unet_aug", "adv2_unet_eval.py", "--run", "unet_aug")
    elif name == "step3_hn":
        if run_eval("eval_hn", "adv_eval_run.py", "--run", "step3_hn", "--tag", "hn"):
            run_eval("hn", "adv3_hn_eval.py")
    elif name in ("step3_seed0", "step3_seed1"):
        tag = name.replace("step3_", "")
        run_eval(f"eval_{tag}", "adv_eval_run.py", "--run", name, "--tag", tag)
        if {"eval_seed0", "eval_seed1"} <= done_evals():
            if run_eval("seeds", "adv5_seed_summary.py") and "eval_hn" in done_evals():
                run_eval("hn_with_seeds", "adv3_hn_eval.py")


def main():
    keep_awake()
    log("=== overnight 시작")
    trained = set(tq.DONE.read_text(encoding="utf-8").split()) if tq.DONE.exists() else set()
    for name in trained:                                   # 이미 끝난 학습의 평가가 빠졌으면 먼저
        evals_after(name)
    failed = set()
    while True:
        nxt = next((j for j in tq.jobs() if j[0] not in failed), None)
        if nxt is None:
            break
        name, script, args = nxt
        if train(name, script, args):
            evals_after(name)
        else:
            failed.add(name)
    # GPU가 한가할 때 추론 시간 다시 재기 (학습과 동시에 잰 타일 시간은 부정확했음)
    if "tile_retime" not in done_evals():
        c = WEEK3 / "cache" / "tile_preds.json"
        if c.exists():
            shutil.move(c, c.with_name("tile_preds_busy_backup.json"))
        if run_eval("tile_retime", "adv1_tile.py") and (ROOT / "runs" / "unet" / "weights" / "best.pt").exists():
            run_eval("unet_retime", "adv2_unet_eval.py")
            if (ROOT / "runs" / "unet_aug" / "weights" / "best.pt").exists():
                run_eval("unet_aug_retime", "adv2_unet_eval.py", "--run", "unet_aug")
    log(f"=== overnight 끝 (건너뛴 학습: {sorted(failed) or '없음'})")


if __name__ == "__main__":
    main()
