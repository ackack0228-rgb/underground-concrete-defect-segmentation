"""1주차 공통 함수: 경로, 영상 목록, 한글 경로에서도 열리는 imread."""
import glob
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"   # 그림 제목의 한글이 □로 깨지지 않게 (Windows 기본 글꼴)
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent      # crack-seg/
WEEK1 = Path(__file__).resolve().parent             # crack-seg/week1/
# 데이터(3.4GB)는 OneDrive 동기화를 피하려고 OneDrive 밖에 둔다. 환경변수 CRACK_DATA로 바꿀 수 있다.
# 그 안의 구조는 강의 10쪽과 같다: <DATA_ROOT>/images/**/*.jpg, <DATA_ROOT>/labels/**/*.txt
DATA_ROOT = Path(os.environ.get("CRACK_DATA", Path.home() / "datasets" / "crack-seg"))   # 기본 ~/datasets/crack-seg
DATA = DATA_ROOT / "images"

CLASSES = ["crack", "efflorescence", "spall_detach"]
EXTS = ("jpg", "jpeg", "png", "bmp")
SEED = 0                                            # 샘플 선택을 다시 해도 같은 결과가 나오게 고정


def list_images(jpg_only=False):
    """data/images 아래 영상을 정렬해서 돌려준다.

    강의 코드는 *.jpg 만 찾는다. 제공 데이터가 png 등일 수도 있어서 기본은 확장자 여러 개를 찾고,
    jpg_only=True 이면 강의 코드와 똑같이 jpg만 찾는다.
    """
    exts = ("jpg",) if jpg_only else EXTS
    files = set()
    for e in exts:
        for pat in (f"**/*.{e}", f"**/*.{e.upper()}"):
            files.update(glob.glob(str(DATA / pat), recursive=True))
    return sorted(files)


def readable(imgs):
    """0바이트(빈) 파일을 뺀 목록. 몇 장이 빠졌는지는 단계2 data_summary.md에 기록된다."""
    return [p for p in imgs if Path(p).stat().st_size > 0]


def is_augmented(path):
    """제공 데이터의 train에는 증강본이 섞여 있다 (파일명에 _aug 3,901장, _efflo_bright/contrast/hflip 15장).
    val·test는 모두 원본."""
    name = Path(path).name
    return "_aug" in name or "_efflo_" in name


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread는 Windows에서 한글이 들어간 경로(예: '바탕 화면')를 못 연다 → None.
    파일을 바이트로 읽은 뒤 imdecode로 풀어서 경로 문제를 피한다."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:                      # 제공 zip에 0바이트 영상이 63개 들어 있다
        raise IOError(f"빈 파일(0바이트): {path}")
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"영상을 열 수 없음: {path}")
    return img


def rel(path):
    """보고서에 쓸 짧은 경로 (DATA = <DATA_ROOT>/images 기준, 예: train/xxx.jpg)."""
    try:
        return Path(path).resolve().relative_to(DATA).as_posix()
    except ValueError:
        return Path(path).name


def require_images(imgs):
    if not imgs:
        raise SystemExit(
            f"영상이 0장입니다. 제공 데이터를 {DATA} 아래에 넣은 뒤 다시 실행하세요.\n"
            "(강의 15쪽: data/images 경로와 glob 패턴 확인)"
        )
