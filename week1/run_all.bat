@echo off
chcp 65001 > nul
rem 1주차 실습 단계 1~4 한 번에 실행 (crack-seg 폴더에서 실행)
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
%PY% week1\step1_env_check.py || goto :err
%PY% week1\step2_data_check.py || goto :err
%PY% week1\step3_observe.py || goto :err
%PY% week1\step4_contact_sheets.py || goto :err
echo.
echo 완료. 하드네거티브는 hard_negatives.csv 작성 후 step4_cases.py 실행
pause
exit /b 0
:err
echo 오류로 중단됨
pause
exit /b 1
