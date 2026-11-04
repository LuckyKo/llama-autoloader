@echo off
REM In-place autoloader SERVER restart via the REST endpoint (POST /v1/restart).
REM The server process re-execs in place, so the terminal that started it stays
REM the running server (nothing is detached). The endpoint unloads loaded models
REM first so their engine subprocesses are not orphaned by the hard re-exec.
REM
REM Usage: restart_server.bat [port]
REM   [port] = main client-facing port (default 1234); the forwarder falls back
REM   management traffic to the internal FastAPI port.
set "PORT=%~1"
if "%PORT%"=="" set "PORT=1234"
set "URL=http://127.0.0.1:%PORT%"

echo [restart_server] POST %URL%/v1/restart
curl -s -X POST "%URL%/v1/restart"
echo.
echo [restart_server] waiting for the server to come back up...
set /a N=0
:wait
timeout /t 1 /nobreak >nul
curl -s "%URL%/health" >nul 2>nul
if errorlevel 1 (
    set /a N+=1
    if %N% LSS 30 (
        echo   ...not up yet (%N%s)
        goto wait
    )
    echo [restart_server] server did not come back within 30s
    exit /b 1
)
echo [restart_server] server is back up on port %PORT%.
