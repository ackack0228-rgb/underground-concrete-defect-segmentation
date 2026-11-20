@echo off
chcp 65001 > nul
rem 그림판으로 그린 주석을 찾아 라벨로 바꾸고, 제출 문서 ①과 PDF까지 다시 만든다 (더블클릭).
rem 그림은 바탕 화면 · week2\annotate\painted · week2\annotate\images 어디에 저장해도 된다 (파일 이름에 원래 영상 이름이 들어 있으면 됨).
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
echo [1/5] 그린 영상 찾기 · 라벨로 변환
%PY% week2\step1_painted.py || goto :err
echo [2/5] 라벨 포맷 확인
%PY% week2\step2_format_check.py --labels week2\annotate\export > nul || goto :err
echo [3/5] 제공 라벨과 비교
%PY% week2\step1_compare.py > nul || goto :err
echo [4/5] 제출 문서 다시 만들기
python week2\_docgen\gen_week2_docx.py --auto || goto :err
echo [5/5] PDF 변환
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_pdf.ps1" || goto :err
echo.
echo 완료. 위에 "확인 필요"가 있으면 week2\annotate\painted_check 의 그림을 확인하세요.
pause
exit /b 0
:err
echo.
echo 오류로 중단됨 - 위 메시지를 Claude에게 보여 주세요.
pause
exit /b 1
