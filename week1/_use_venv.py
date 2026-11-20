"""VS Code 실행 버튼이 다른 가상환경(예: Step 1의 .venv)을 골라도 Step 3 전용 crack-seg/.venv로 다시 실행한다.

각 단계 스크립트의 맨 첫 import로 둔다 (cv2 · ultralytics를 불러오기 전에 환경을 바꿔야 하므로).
"""
import subprocess
import sys
from pathlib import Path

VENV_PY = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"

if VENV_PY.exists() and Path(sys.prefix).resolve() != VENV_PY.parent.parent.resolve():
    sys.exit(subprocess.call([str(VENV_PY), *sys.argv]))
