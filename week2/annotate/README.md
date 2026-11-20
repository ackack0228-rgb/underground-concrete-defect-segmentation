# 단계1 직접 주석 안내 (강의 9쪽)

`images/` 의 15장(클래스별 5장, `selected.csv`)을 **주석기준서(week1/annotation_guideline.md)만 보고** 폴리곤으로 그린다.
제공 라벨은 보지 않는다 — 다 그린 뒤 `step1_compare.py`가 같은 영상의 제공 라벨과 맞춰 본다.

## Roboflow (웹, 추천)
1. roboflow.com 가입 → **New Project** → Project Type **Instance Segmentation**, 이름 아무거나
2. 클래스는 **crack · efflorescence · spall_detach 순서로** 만든다 (Classes 탭에서 미리 추가해 두면 순서가 고정됨)
3. **Upload** → `images/` 폴더의 15장을 끌어다 놓기 → Save and Continue → 직접 주석(Annotate)으로 배정
4. 영상마다 **Polygon 도구(P)** 로 결함 경계를 따라 클릭 → 클래스 선택
   - Smart Polygon(SAM)을 써도 되지만, 가는 균열은 손으로 경계를 고쳐 기준서의 "보이는 폭만큼 좁게"에 맞출 것
   - 제외 대상(시공이음부 · 케이블 · 그림자 · 보수재 띠 · 분필 · 정반사 · 얼룩)은 그리지 않는다
   - 결함이 없다고 판단되면 그대로 두고 "Null"로 표시
5. 다 그리면 **Add to Dataset** → Generate New Version
   - Preprocessing의 **Resize는 지운다**(원본 1920×1080 유지), Augmentation은 **넣지 않는다**
6. **Export Dataset → Format "YOLOv8"** (Instance Segmentation 프로젝트라 폴리곤 좌표로 나옴) → zip 다운로드

## CVAT (app.cvat.ai)
1. Projects → + → Labels에 crack · efflorescence · spall_detach 순서로 추가 → Task 만들고 15장 업로드
2. 도구 모음의 **Polygon** 으로 그리기 (N으로 시작, 끝점에서 다시 N)
3. Task → Actions → **Export task dataset → "Ultralytics YOLO Segmentation 1.0"** (영상 저장 체크 해제) → zip 다운로드

## 내보낸 뒤
zip을 **`week2/annotate/export/`** 에 풀고 알려 주면 된다. 그 다음은 아래 두 줄 (crack-seg 폴더에서):
```
.venv\Scripts\python.exe week2\step2_format_check.py --labels week2\annotate\export
.venv\Scripts\python.exe week2\step1_compare.py
```
