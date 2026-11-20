@echo off
chcp 65001 > nul
rem 3주차 실습 + 심화 (crack-seg 폴더 기준). 순서가 곧 절차: 단계3(검증셋)에서 임계값을 정해 thresholds.json에 저장한 뒤
rem 단계4 · 5 · 심화는 그 파일을 읽기만 한다. 실행 시각은 week3\run_log.txt에 쌓인다.
rem 학습(각 1 ~ 2시간)은 여기서 하지 않는다 — 학습 명령은 맨 아래 주석. 아래는 학습된 가중치(runs\...)로 평가만 다시 한다.
rem 기준 best.pt(Release v1.0 → runs\step3\weights\best.pt)만 있으면 실습 6단계 + 심화 ①④⑥ + 요약 표가 돌고,
rem 심화 ②③⑤ 평가와 모델 비교표는 그 가중치(runs\step3_seed0 · step3_seed1 · step3_hn · unet · unet_aug)가 있을 때만 돈다.
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
rem ---- 실습 6단계 (강의 9 ~ 14쪽)
%PY% week3\step1_eval.py || goto :err
%PY% week3\step2_onoff.py || goto :err
%PY% week3\step3_threshold.py || goto :err
%PY% week3\step4_test.py || goto :err
%PY% week3\step5_errors.py || goto :err
%PY% week3\step5_errors.py --split val || goto :err
rem ---- 심화 (논문강의 47쪽 순서) — 각 모델도 검증셋에서 임계값을 고르고 테스트에 고정 적용
%PY% week3\adv1_tile.py || goto :err
%PY% week3\adv_eval_run.py --run step3 --tag seed42 || goto :err
if exist runs\step3_seed0\weights\best.pt (%PY% week3\adv_eval_run.py --run step3_seed0 --tag seed0 || goto :err) else (echo [건너뜀] runs\step3_seed0 가중치 없음 - 저장소의 adv_runs\seed0.json 사용)
if exist runs\step3_seed1\weights\best.pt (%PY% week3\adv_eval_run.py --run step3_seed1 --tag seed1 || goto :err) else (echo [건너뜀] runs\step3_seed1 가중치 없음 - 저장소의 adv_runs\seed1.json 사용)
if exist runs\step3_hn\weights\best.pt (%PY% week3\adv_eval_run.py --run step3_hn --tag hn || goto :err) else (echo [건너뜀] runs\step3_hn 가중치 없음 - 저장소의 adv_runs\hn.json 사용)
if exist runs\unet\weights\best.pt (%PY% week3\adv2_unet_eval.py --run unet || goto :err) else (echo [건너뜀] runs\unet 가중치 없음 - 저장소의 adv2_unet.md 사용)
if exist runs\unet_aug\weights\best.pt (%PY% week3\adv2_unet_eval.py --run unet_aug || goto :err) else (echo [건너뜀] runs\unet_aug 가중치 없음 - 저장소의 adv2_unet_aug.md 사용)
if exist runs\step3_hn\weights\best.pt (%PY% week3\adv3_hn_eval.py || goto :err) else (echo [건너뜀] 하드네거티브 평가 - 저장소의 adv3_hn_eval.md 사용)
%PY% week3\adv4_common_threshold.py || goto :err
%PY% week3\adv5_seed_summary.py || goto :err
%PY% week3\adv6_crack_width.py || goto :err
%PY% week3\adv7_summary.py || goto :err
rem ---- 최종 리포트용 표 · 그림
set ALLW=1
for %%W in (step3_hn unet unet_aug) do if not exist runs\%%W\weights\best.pt set ALLW=0
if "%ALLW%"=="1" (%PY% week3\compare_models.py || goto :err) else (echo [건너뜀] 모델 비교표 - 저장소의 compare_models.md 사용)
%PY% week3\data_stats.py || goto :err
%PY% week3\flops.py || goto :err
%PY% week3\guideline_examples.py || goto :err
%PY% shrink_figs.py || goto :err
echo.
echo 완료
pause
exit /b 0
:err
echo 오류로 중단됨
pause
exit /b 1

rem ---- 학습 명령 (참고, 순서대로 실행하면 runs\ 아래 가중치가 다시 만들어진다)
rem %PY% week2\step5_train.py                                        기준 YOLOv8s-Seg (seed 42) → runs\step3
rem %PY% week3\adv_yolo_train.py --seed 0                             ⑤ → runs\step3_seed0   (--seed 1 → runs\step3_seed1)
rem %PY% week3\adv3_hn_build.py  다음  %PY% week3\adv3_hn_curate.py  ③ 데이터 → %PY% week3\adv_yolo_train.py --seed 42 --data hn --name step3_hn
rem %PY% week3\adv2_unet_train.py --name unet                         ② U-Net 약한 증강   (--name unet_aug --aug strong → 강한 증강)
rem %PY% week3\adv7_hand_ft.py --variant hand 또는 provided --seed 42 / 1 / 2   ⑦ 직접 주석 2장 미세조정 (adv_eval_run --weights last.pt)
rem 또는 week3\overnight.py 가 train_queue.txt 순서로 학습 + 평가를 이어서 한다.
