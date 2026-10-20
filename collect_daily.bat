@echo off
rem SMP supply-side data daily collection. Run daily 00:50 by Task Scheduler task "SMP_daily_collect".
rem When everything is collected, this task deletes itself (check_done returns 0).
cd /d C:\developer\SMP
set PYTHONIOENCODING=utf-8
set PY=C:\Python312\python.exe
set LOG=data\raw\api\collect_log.txt
echo ===== %date% %time% ===== >> %LOG%
%PY% -m common.collect.smp_fuel 20240101 20241231 >> %LOG% 2>&1
%PY% -m common.collect.smp_fuel 20260101 >> %LOG% 2>&1
set /p DATES=<data\raw\api\gen_by_fuel\target_dates.txt
%PY% -m common.collect.gen_by_fuel %DATES% >> %LOG% 2>&1
%PY% -m common.collect.check_done >> %LOG% 2>&1
if %errorlevel%==0 (
  echo [DONE] all collected - deleting task SMP_daily_collect >> %LOG%
  schtasks /delete /tn "SMP_daily_collect" /f >> %LOG% 2>&1
)
