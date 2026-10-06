---
tags: [start-bat, windows, port-reuse, race-condition, taskkill]
aliases: [start-bat-port-wait, killed-but-not-restarted, address-already-in-use-on-relaunch]
related: [[raw-tcp-forwarder-architecture]], [[restart-bat-vs-start-bat]]
confidence: verified
---

# start.bat must WAIT for port release before relaunching (killed-but-not-restarted root cause)

The autoloader's "kill old instance then restart" flow fails if it launches `python server.py`
immediately after `taskkill /F`. Root cause, verified on this host:

- **uvicorn binds the INTERNAL port 1235 FIRST** (`server.py:2014`, `uvicorn.run(..., port=INTERNAL_API_PORT)`).
  The main forwarder on 1234 only starts later, inside FastAPI's `_startup` hook.
- `taskkill /F` is **asynchronous** — the OS keeps the socket held (TIME_WAIT / not-yet-released)
  after it returns. Observed case: port 1235 stayed LISTENING for **>20s** after the kill.
- If 1235 is still held when the new process starts, uvicorn raises `OSError: Address already in use`
  and dies *before* the forwarder ever binds 1234 → symptom = "server was killed but not brought back up".

**Fix (in `start.bat`):** after killing, poll `netstat -ano | findstr LISTENING` until BOTH 1234 and 1235
are free (up to ~20s), THEN launch. Do NOT rely on `SO_REUSEADDR` alone — it does not reliably allow binding
over a just-killed process's still-held socket on Windows.

## Related gotchas found while writing start.bat
- **`timeout /t N` errors with "Input redirection is not supported"** when stdin is redirected (background/CI).
  Use `ping -n <N+1> 127.0.0.1 >nul` for a sleep that needs no keyboard input.
- **findstr substring trap:** `findstr ":1234 "` also matches `:12345` / `:12340` (findstr splits on whitespace,
  so the trailing space is ignored). Match the netstat local-address column EXACTLY instead:
  `for /f "tokens=2,5" %%B in ('netstat -ano ^| findstr LISTENING') do if /i "%%B"=="127.0.0.1:1234"`
  (token 2 = local address, token 5 = PID). Verified to exclude `:12345`/`:12340` traps.
