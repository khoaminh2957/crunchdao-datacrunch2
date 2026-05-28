@echo off
REM CrunchDAO daily pipeline wrapper — invoked by Windows Task Scheduler.
REM Sets UTF-8 for safe console output then delegates to cron_daily.ps1.
REM All output (stdout + stderr) is appended to submission.log next to this file.

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Add user-site Scripts dir to PATH so `crunch.exe` is callable from cron_daily.ps1
set "PATH=%PATH%;C:\Users\Admin\AppData\Roaming\Python\Python314\Scripts"

cd /d "%~dp0"

echo. >> "%~dp0submission.log"
echo === run_pipeline.bat start %DATE% %TIME% === >> "%~dp0submission.log"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0cron_daily.ps1" >> "%~dp0submission.log" 2>&1
set RC=%ERRORLEVEL%

echo === run_pipeline.bat end rc=%RC% %DATE% %TIME% === >> "%~dp0submission.log"
exit /b %RC%
