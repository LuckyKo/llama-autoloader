@echo off
REM In-place autoloader SERVER restart WITH KV state preservation.
REM
REM Flow (restart_server.py):
REM   1. save each loaded model's KV state to disk
REM   2. POST /v1/restart  -> the server unloads models and re-execs in place
REM   3. wait for the server to come back up
REM   4. restore each model's KV state (JIT-loads + KV <- disk)
REM
REM The terminal that started the server stays the running server (the re-exec
REM is in-place, nothing is detached).
REM
REM Usage: restart_server.bat [port]
REM   [port] = main client-facing port (default 1234).
set "PORT=%~1"
if "%PORT%"=="" set "PORT=1234"
cd /d "%~dp0"
python restart_server.py --port %PORT%
