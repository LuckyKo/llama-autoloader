@echo off
rem ============================================================
rem  start.bat - fresh launch of llama-autoloader.
rem
rem  1. Kills stale python processes LISTENING on 1234 (forwarder)
rem     and 1235 (internal FastAPI). Refuses to kill anything that
rem     is NOT a python process (safety guard).
rem  2. Waits until both ports are actually free before relaunching.
rem     taskkill is async, and uvicorn binds 1235 FIRST - launching
rem     too early dies with "Address already in use" and the
rem     forwarder on 1234 never comes up.
rem  3. Deletes __pycache__ (stale .pyc protection).
rem  4. Launches:  python server.py --port 1234
rem
rem  Port match = netstat local-address column compared EXACTLY
rem  against "127.0.0.1:<port>". A plain findstr ":1234 " ALSO
rem  matches :12345 / :12340 - do not "simplify" this back.
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "FOREIGN="
echo [start] Killing old instance (ports 1234, 1235) ...
call :killport 1234
call :killport 1235
if defined FOREIGN (
    echo [start] ERROR: a NON-python process is holding an autoloader port.
    echo [start] Refusing to kill it. Close it manually, then rerun start.bat.
    pause
    exit /b 1
)

echo [start] Waiting for ports 1234 and 1235 to be free ...
set /a TRIES=0
:waitports
call :portbusy
if not defined BUSY goto portsfree
set /a TRIES+=1
if %TRIES% GEQ 20 (
    echo [start] ERROR: port %BUSY% still busy after ~20s. Aborting.
    pause
    exit /b 1
)
echo [start]   port %BUSY% busy, retrying... (%TRIES%/20)
rem ping-sleep instead of timeout: timeout errors when stdin is redirected.
ping -n 2 127.0.0.1 >nul
goto waitports

:portsfree
echo [start] Ports free.
echo [start] Clearing stale bytecode (__pycache__) ...
if exist __pycache__ rmdir /s /q __pycache__

echo [start] Starting autoloader on port 1234 ...
python server.py --port 1234

rem Keep the window open if the server dies immediately, so the
rem error is readable instead of the console vanishing.
if errorlevel 1 (
    echo.
    echo [start] Server exited with an error - see above. Press a key to close.
    pause
)
endlocal
exit /b 0

rem ---------------- subroutines ----------------

:killport
rem %1 = port. Kills the python process listening on 127.0.0.1:%1.
set "KP_PID="
for /f "tokens=2,5" %%B in ('netstat -ano ^| findstr "LISTENING"') do (
    if /i "%%B"=="127.0.0.1:%1" set "KP_PID=%%C"
)
if not defined KP_PID exit /b 0
set "KP_IMG="
for /f "tokens=1" %%I in ('tasklist /FI "PID eq %KP_PID%" /NH 2^>nul ^| findstr /i "exe"') do set "KP_IMG=%%I"
if not defined KP_IMG (
    echo [start]   port %1 - PID %KP_PID% already gone.
    exit /b 0
)
echo [start]   port %1 held by PID %KP_PID% [%KP_IMG%]
if /i "%KP_IMG:~0,6%"=="python" (
    taskkill /F /PID %KP_PID% >nul 2>&1
    echo [start]   killed.
) else (
    echo [start]   NOT python - refusing to kill.
    set "FOREIGN=1"
)
exit /b 0

:portbusy
rem Sets BUSY=1234/1235 if either autoloader port is still LISTENING.
set "BUSY="
for /f "tokens=2,5" %%B in ('netstat -ano ^| findstr "LISTENING"') do (
    if /i "%%B"=="127.0.0.1:1234" set "BUSY=1234"
    if /i "%%B"=="127.0.0.1:1235" set "BUSY=1235"
)
exit /b 0
