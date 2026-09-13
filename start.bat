@echo off
rem ============================================================
rem  start.bat - fresh launch of llama-autoloader.
rem
rem  MEANT TO BE RUN INTERACTIVELY (double-click, or from a manual cmd
rem  window). It launches the server in the FOREGROUND and blocks until
rem  you close it. Do NOT run it from an automated pipeline / async shell
rem  with a timeout - it will be killed mid-run and leave orphaned
rem  llama-server.exe children behind (which this script then has to
rem  clean up on the next launch).
rem
rem  1. Kills stale python processes LISTENING on 1234 (forwarder)
rem     and 1235 (internal FastAPI). Refuses to kill anything that
rem     is NOT a python process (safety guard).
rem  2. Kills any orphaned llama-server.exe child processes. When the
rem     parent python is killed, its llama-server.exe children become
rem     orphans and keep holding base_port (9001) - this blocks clean
rem     restarts. We kill by image name: on this dedicated dev box the
rem     only llama-server.exe instances are the autoloader's own children
rem     (spawned from ./backends/), so a blanket kill is safe here.
rem  3. Waits until BOTH the python ports (1234/1235) AND the base-port
rem     range (9001-9010) are actually free before relaunching. taskkill
rem     is async, and uvicorn binds 1235 FIRST - launching too early dies
rem     with "Address already in use" and the forwarder on 1234 never
rem     comes up.
rem  4. Deletes __pycache__ (stale .pyc protection).
rem  5. Launches:  python server.py --port 1234
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

rem Kill orphaned llama-server.exe children. If the parent python was killed,
rem its llama-server.exe subprocesses become orphans and keep holding base_port
rem (9001). We kill by image name: on this dedicated dev box the only
rem llama-server.exe instances are the autoloader's own children (spawned from
rem ./backends/), so a blanket kill is safe here. If you ever run another
rem llama-server.exe for unrelated work, scope this down to base-port PIDs.
echo [start] Killing orphaned llama-server.exe processes ...
taskkill /F /IM llama-server.exe >nul 2>&1

echo [start] Waiting for ports 1234, 1235 and base port range (9001-9010) to be free ...
set /a TRIES=0
:waitports
call :portbusy
if not defined BUSY goto portsfree
set /a TRIES+=1
if %TRIES% GEQ 20 (
    echo [start] ERROR: port %BUSY% still busy after ~40s (20 retries). Aborting.
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
rem Sets BUSY=<port> if an autoloader port is still LISTENING.
rem Covers the python ports (1234/1235) AND the llama-server base-port
rem range (9001-9010). Exact match on "127.0.0.1:<port>" - do NOT relax
rem to a substring findstr, that would also match :90011 etc.
set "BUSY="
for /f "tokens=2,5" %%B in ('netstat -ano ^| findstr "LISTENING"') do (
    if /i "%%B"=="127.0.0.1:1234" set "BUSY=1234"
    if /i "%%B"=="127.0.0.1:1235" set "BUSY=1235"
    if /i "%%B"=="127.0.0.1:9001" set "BUSY=9001"
    if /i "%%B"=="127.0.0.1:9002" set "BUSY=9002"
    if /i "%%B"=="127.0.0.1:9003" set "BUSY=9003"
    if /i "%%B"=="127.0.0.1:9004" set "BUSY=9004"
    if /i "%%B"=="127.0.0.1:9005" set "BUSY=9005"
    if /i "%%B"=="127.0.0.1:9006" set "BUSY=9006"
    if /i "%%B"=="127.0.0.1:9007" set "BUSY=9007"
    if /i "%%B"=="127.0.0.1:9008" set "BUSY=9008"
    if /i "%%B"=="127.0.0.1:9009" set "BUSY=9009"
    if /i "%%B"=="127.0.0.1:9010" set "BUSY=9010"
)
exit /b 0
