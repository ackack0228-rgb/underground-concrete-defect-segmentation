"""2주차 공통: 1주차 common.py(경로 · imread · 한글 글꼴)를 그대로 쓰고, 2주차 경로만 더한다."""
import re
import sys
from pathlib import Path

WEEK2 = Path(__file__).resolve().parent             # crack-seg/week2/
sys.path.insert(0, str(WEEK2.parent / "week1"))
import common                                       # noqa: E402  (1주차 공통 함수)

DATA_ROOT = common.DATA_ROOT                        # 제공 데이터 (~/datasets/crack-seg, 환경변수 CRACK_DATA로 변경)
# 분할 결과도 OneDrive 밖에 둔다. 영상은 복사하지 않고 하드링크로 만들어 디스크를 더 쓰지 않는다.
SPLIT_ROOT = DATA_ROOT.parent / "crack-seg-split"
CLASS_NAMES = {0: "crack", 1: "efflorescence", 2: "spall_detach"}
SPLITS = ("train", "val", "test")


def get_origin_id(stem):
    """원본 ID 추출 (강의 11쪽 함수를 이 데이터 파일명 규칙에 맞게 고침).

    이 데이터의 파일명은 세 가지가 섞여 있다.
      증강본      URT002_..._1636_aug_1_00064(_augfill_01382_brightness)  → '_aug' 앞
      백태 증강본 URT..._efflo_bright / _efflo_contrast / _efflo_hflip    → '_efflo_' 앞
      Roboflow 내보내기  URT003_..._2478_jpg.rf.<해시>                    → '_jpg.rf.' 앞
    """
    return re.split(r"_aug|_efflo_|_jpg\.rf\.", stem)[0]
