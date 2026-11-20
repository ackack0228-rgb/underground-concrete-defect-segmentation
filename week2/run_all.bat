@echo off
chcp 65001 > nul
rem 2주차 실습: 단계3 점검 → 단계4 분할 → 단계2 포맷 확인 → 단계5 학습(60 epoch, 약 4.5시간) → 단계6 곡선 (crack-seg 폴더 기준)
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
%PY% week2\step3_integrity.py || goto :err
%PY% week2\step4_split.py || goto :err
%PY% week2\step3_integrity.py --split || goto :err
%PY% week2\step2_format_check.py || goto :err
%PY% week2\step5_train.py || goto :err
%PY% week2\step6_curve.py || goto :err
echo.
echo 완료. 직접 주석(단계1)은 annotate\README.md 참고
pause
exit /b 0
:err
echo 오류로 중단됨
pause
exit /b 1
