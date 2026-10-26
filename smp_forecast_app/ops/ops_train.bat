@echo off
rem SMP 과제2 운영 작업 (train). 작업 스케줄러가 run_hidden.vbs로 창 없이 실행. 기록: data\processed\smp_forecast_app\ops\run_log.txt
rem 이 파일은 cp949(ANSI)·CRLF로 저장한다. UTF-8로 저장하면 cmd가 한글 주석을 잘못 읽어 명령으로 실행한다
cd /d C:\developer\SMP
set PYTHONIOENCODING=utf-8
C:\Python312\python.exe -W ignore -m smp_forecast_app.src.ops train >> data\processed\smp_forecast_app\ops\console_train.log 2>&1
exit /b %ERRORLEVEL%
