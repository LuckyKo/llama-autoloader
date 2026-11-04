@echo off
REM Graceful engine backend restart (preserves KV state)
REM Usage: reload_model.bat [port]
REM   Use in sync/async mode (models must be loaded).
REM   In safe mode the models are already evacuated - no restart needed.
set "PORT=%~1"
if "%PORT%"=="" set "PORT=1234"
cd /d "%~dp0"
python restart.py --port %PORT%
