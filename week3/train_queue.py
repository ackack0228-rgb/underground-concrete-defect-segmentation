"""심화 재학습 큐: train_queue.txt의 작업을 위에서부터 하나씩 학습한다 (끝난 작업은 train_queue_done.txt에 기록).

  python week3/train_queue.py         # 백그라운드로 돌려 둔다. 다시 실행하면 끝나지 않은 작업부터 이어서 (--resume 자동)
train_queue.txt 한 줄 = "이름 | 스크립트 | 인자"  (# 은 주석). 학습 중에도 줄을 추가할 수 있다.
배터리 보호: 충전기가 빠진 상태에서 배터리가 20% 미만이면 학습 프로세스를 멈추고 큐를 끝낸다
(매 epoch last.pt가 저장되므로 충전 후 다시 실행하면 이어진다).
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WEEK3 = Path(__file__).resolve().parent
ROOT = WEEK3.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
QUEUE = WEEK3 / "train_queue.txt"
DONE = WEEK3 / "train_queue_done.txt"
LOG = WEEK3 / "train_queue.log"
MIN_BATTERY = 20


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def battery():
    """(잔량 %, 충전기 연결 여부). 읽지 못하면 (100, True)."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "$b=Get-CimInstance Win32_Battery; \"$($b.EstimatedChargeRemaining) $($b.BatteryStatus)\""],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return int(out[0]), int(out[1]) != 1                      # BatteryStatus 1 = 방전 중(충전기 없음)
    except Exception:
        return 100, True


def jobs():
    done = set(DONE.read_text(encoding="utf-8").split()) if DONE.exists() else set()
    for line in QUEUE.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, script, argstr = [s.strip() for s in line.split("|")]
            if name not in done:
                yield name, script, argstr.split()


def main():
    while True:
        nxt = next(jobs(), None)
        if nxt is None:
            log("큐 완료")
            return
        name, script, args = nxt
        if (ROOT / "runs" / name / "weights" / "last.pt").exists():
            args = ["--name", name, "--resume"]
        cmd = [str(PY), str(WEEK3 / script), *args]
        log(f"시작 {name}: {' '.join(cmd[1:])}")
        with open(ROOT / "runs" / f"{name}.log", "a", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                 env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
            while p.poll() is None:
                time.sleep(60)
                pct, ac = battery()
                if not ac and pct < MIN_BATTERY:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                    log(f"배터리 {pct}% (충전기 없음) → {name} 중단. 충전 후 train_queue.py 다시 실행하면 이어짐")
                    sys.exit(2)
        if p.returncode != 0:
            log(f"오류 {name} (종료 코드 {p.returncode}) → 큐 멈춤, runs/{name}.log 확인")
            sys.exit(1)
        with open(DONE, "a", encoding="utf-8") as f:
            f.write(name + "\n")
        log(f"완료 {name}")


if __name__ == "__main__":
    main()
