"""Server-restart helper — in-place re-exec on all platforms.

POST /v1/restart calls :func:`restart_server_process`. On POSIX the process
image is replaced (same PID, console, env). On Windows a new process image is
spawned inheriting the current console (no DETACHED_PROCESS / CREATE_NEW_CONSOLE)
with the same interpreter, argv, env and CWD, then the old process exits — the
terminal that started the server stays the running server (nothing is detached,
no hidden child).

The re-exec is deliberately NOT a detached ``subprocess.Popen``: a detached child
would outlive the console and break the "restart in place" contract.
"""

import os
import sys
from typing import NoReturn


def restart_server_process() -> NoReturn:
    """Re-launch the autoloader server in place via ``os.execl`` (same interpreter + argv + env).

    Does NOT return on success: ``os.execl`` replaces the process image (POSIX) or
    spawns a replacement and exits the old process (Windows). The ``os._exit(1)``
    below is only reached if ``execl`` returns, which happens on failure.

    Windows caveat (os-exit-skips-shutdown-cleanup-orphan): on Windows ``os.execl``
    is implemented as spawn-new-process-then-exit-old, so the OLD process's
    ``atexit`` handlers and Python-level signal handlers are NOT run — the old
    image leaves via a hard transition, not a graceful shutdown. Concretely the
    FastAPI ``_lifespan`` finally block (``_forwarder.stop()`` +
    ``await manager.stop()``) is bypassed entirely. The raw-TCP forwarder is a
    set of daemon threads, so it dies with the old process by construction (no
    orphan). The engine subprocesses (llama-server / bridge launchers, spawned
    via ``subprocess.Popen``) are the real orphan risk: a hard exit leaves them
    running, so the ``/v1/restart`` endpoint explicitly unloads all models
    (``manager.unload_all()``) BEFORE calling this function. The re-exec'd child
    does NOT inherit the old bound sockets (main/internal ports rebind cleanly,
    no EADDRINUSE race). If any other ``atexit``/lifespan cleanup is added later
    that must run on restart, invoke it in the endpoint before the re-exec,
    because the hard exit will skip it.
    """
    os.execl(sys.executable, sys.executable, *sys.argv)
    os._exit(1)
