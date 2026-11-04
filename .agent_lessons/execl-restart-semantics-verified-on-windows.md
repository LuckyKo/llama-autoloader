---
name: execl-restart-semantics-verified-on-windows
description: "Empirically verified Windows semantics of the os.execl in-place restart: old process exits before atexit runs, the new process can rebind the main/internal ports, but Popen-spawned engine children survive and become unreachable orphans."
metadata:
  type: project
  verified: 2026-11-04
---

Measured on this host (Python 3.12, Windows) with a parent/child probe pair, for the new
`server_restart.restart_server_process()` helper:

| Claim | Result |
|---|---|
| `os.execl` on Windows = spawn new, then exit old | Confirmed: `psutil.pid_exists(old_pid)` is False in the child; the parent never reaches the statement after `execl` |
| `atexit` handlers do NOT run | Confirmed: an `atexit`-registered marker file was never written |
| Python signal handlers are skipped | Same mechanism (hard exit), no handler fired |
| New process does NOT inherit the bound listening sockets | Confirmed: the child successfully re-bound the same port (`dup(3)` on the old fd returned EBADF) — no EADDRINUSE race |
| Engine processes survive the re-exec | **Confirmed and this is the hazard**: a `subprocess.Popen` child of the server was still alive and heartbeating 3 s after the parent `execl`'d, with its parent pid gone |

**Rule:** an in-place `execl` restart is clean with respect to *itself* (no duplicate server, no
port race, threads such as the raw-TCP forwarder die with the process). It is NOT clean with
respect to the engines the `ModelManager` spawned — those become unreachable orphans holding GPU
committed memory and their `base_port` range, exactly the `BUG_0012` / `kill-proc-tree-complete-drain`
failure mode. A restart endpoint must save state and `await manager.unload_all()` (mirroring
`restart.py`) before re-exec, or refuse to restart while models are loaded.

**How to re-verify in 2 minutes:** write a parent script that (1) `Popen`s a long-looping child
writing a heartbeat file, (2) binds a port, (3) registers an `atexit` marker, (4) `os.execl`s
itself into a child script that checks `psutil.pid_exists(old)`, tries to re-bind the port, and
checks the child's liveness. Never run the "broken variant" discriminator without mocking
`subprocess.Popen` and `os._exit` — otherwise it actually forks a copy of itself.

Related: [[route-decorator-inside-another-endpoint-body]], [[kill-proc-tree-complete-drain]],
[[autoloader-restart-py-graceful-backend-restart]]
