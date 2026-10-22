@echo off
rem SMP 과제2 운영 작업 (daily). 작업 스케줄러가 숨김 창으로 실행. 기록: data\processed\smp_forecast_app\ops\run_log.txt
cd /d C:\developer\SMP
set PYTHONIOENCODING=utf-8
C:\Python312\python.exe -W ignore -m smp_forecast_app.src.ops daily >> data\processed\smp_forecast_app\ops\console_daily.log 2>&1
