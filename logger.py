"""Console-log-to-file logging for llama-autoloader.

Ported from Agent Cascade's ``agent_cascade/log.py`` and adapted for a single
process (no instance_id). It provides:

- A Windows-safe rotating file handler (shutil.move instead of os.rename).
- A capturing stdout/stderr stream that tees every write through the logger so
  uvicorn's console output lands in the log file without duplication.
- ``init_logging()`` which wires a dedicated "autoloader" logger with BOTH a
  StreamHandler(original_stdout) and a rotating FileHandler, then replaces
  sys.stdout/sys.stderr with capturing streams before uvicorn.run() so that
  uvicorn's dictConfig handlers bind to our streams.

This module is self-contained: it does NOT import server.py or read config.yaml.
All settings are passed as parameters so it can be tested standalone.
"""

import logging
import logging.handlers
import os
import shutil
import sys
import threading
from pathlib import Path


class _WindowsSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    RotatingFileHandler that uses shutil.move() instead of os.rename() for file rotation.

    On Windows, os.rename() fails with PermissionError when the target file is open
    in another process (e.g., editors, file watchers, antivirus). shutil.move() handles
    this more gracefully by falling back to copy+delete.

    Also suppresses errors during rotation rather than letting them cascade into
    infinite logging recursion.
    """

    def rotate(self, source, dest):
        try:
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(source, dest)
        except OSError as e:
            # Don't use logger here - we're inside emit() already, calling logger.warning()
            # would cause recursion/nested emit on the same handler -> "Logging error" messages.
            # Write directly to stderr for observability without recursion risk.
            try:
                sys.__stderr__.write(f"[LOG ROTATION FAILED] {source} -> {dest}: {e}\n")
                sys.__stderr__.flush()
            except Exception:
                pass  # Silently ignore - rotation will retry next time file size threshold is hit


class _CapturingStream:
    """
    Replacement for sys.stdout / sys.stderr that routes every write()
    through the logger so it ends up in console.log.

    Writes are logged at INFO level (stdout) or WARNING level (stderr).
    A per-instance threading.RLock allows re-entrant calls (e.g., when logging
    system writes error messages back to stderr during handler failures).
    """
    def __init__(self, stream_type, original_stream):
        self._stream_type = stream_type  # 'stdout' or 'stderr'
        self._original = original_stream
        self._lock = threading.RLock()

    def write(self, msg):
        # Type guard: ensure msg is a string (handles non-string writes gracefully)
        msg = str(msg)
        with self._lock:
            if msg and msg.strip():
                level = logging.INFO if self._stream_type == 'stdout' else logging.WARNING
                try:
                    logger.log(level, msg.rstrip('\n\r'))
                except Exception:
                    # Fallback to original stream if logging fails (avoids recursion)
                    try:
                        self._original.write(f"[LOGGING FAILED] {msg}")
                        self._original.flush()
                    except Exception:
                        pass
            # Also write to the original stream so it still appears on screen
            try:
                self._original.write(msg)
                self._original.flush()
            except (OSError, ValueError):
                # Broken pipe, closed stream, etc. — acceptable failures during I/O
                pass

    def flush(self):
        try:
            self._original.flush()
        except (OSError, ValueError):
            pass

    def isatty(self):
        """Delegate to the original stream's isatty, falling back to False."""
        try:
            return self._original.isatty()
        except Exception:
            return False


def setup_logger(level=None, *, log_dir=None, filename="console.log",
                 max_bytes=10 * 1024 * 1024, backup_count=5):
    """
    Build the "autoloader" logger with a console StreamHandler and a rotating
    FileHandler. Returns (logger, original_stdout, original_stderr).

    The original stdout/stderr are captured BEFORE creating the StreamHandler so
    the handler writes to the real terminal and NOT our capturing stream (which
    would cause infinite recursion: write → log → handler → captured stdout → log → ...).
    """
    if level is None:
        level = logging.INFO

    # Capture original stdout/stderr BEFORE creating the StreamHandler.
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

    handler = logging.StreamHandler(stream=_original_stdout)
    # Do not run handler.setLevel(level) so that users can change the level via logger.setLevel later.
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)

    _logger = logging.getLogger("autoloader")
    _logger.setLevel(level)
    # Prevent duplicate output if any root-level handler is ever added.
    _logger.propagate = False

    # Only add handlers once (prevent duplicates on restart/reload).
    if not _logger.handlers:
        _logger.addHandler(handler)

        # File handler — rotating console log.
        if log_dir is None:
            log_dir = Path("logs")
        log_dir = Path(log_dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Cannot create log directory {log_dir}: {e}") from e

        try:
            file_handler = _WindowsSafeRotatingFileHandler(
                log_dir / filename,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8',
                errors='replace',
                delay=True  # Open lazily to avoid locking issues on startup
            )
        except OSError as e:
            raise RuntimeError(f"Cannot open console log file {log_dir / filename}: {e}") from e

        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    return _logger, _original_stdout, _original_stderr


# ─── Module-level State ───────────────────────────────────────────────────
_initialized = False
_original_stdout = None
_original_stderr = None
_init_lock = threading.Lock()  # Protects check-then-act in init_logging()

# Public logger reference — assigned by init_logging(). Safe to import before
# calling init_logging(); calls will use a basic logger until initialized.
logger = logging.getLogger("autoloader")


def _logging_excepthook(exc_type, exc_value, exc_tb):
    """Replace sys.excepthook to log uncaught main-thread exceptions."""
    try:
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    except Exception:
        # During interpreter shutdown logger may be unavailable; fall back to sys.__excepthook__
        sys.__excepthook__(exc_type, exc_value, exc_tb)


def _threading_excepthook(args):
    """Replace threading.excepthook to log uncaught thread exceptions."""
    try:
        thread_name = args.thread.name if args.thread else "unknown"
        # Python 3.12+ renamed exc_tb -> exc_traceback in _thread._ExceptHookArgs
        if hasattr(args, 'exc_traceback'):
            tb = args.exc_traceback
        elif hasattr(args, 'exc_tb'):
            tb = args.exc_tb
        else:
            tb = None
        logger.error(
            "Uncaught exception in thread %s", thread_name,
            exc_info=(args.exc_type, args.exc_value, tb)
        )
        if tb is None:
            logger.warning("Uncaught exception in thread %s — no traceback available", thread_name)
    except Exception:
        # During interpreter shutdown logger may be unavailable; fall back to sys.__excepthook__
        try:
            exc_tb = getattr(args, 'exc_traceback', None) or getattr(args, 'exc_tb', None)
            sys.__excepthook__(args.exc_type, args.exc_value, exc_tb)
        except Exception:
            pass  # Final safety net during interpreter shutdown


def init_logging(level=None, *, log_dir=None, filename="console.log",
                 max_bytes=10 * 1024 * 1024, backup_count=5) -> None:
    """
    Initialize the logging system.

    Sets up handlers, replaces sys.stdout/sys.stderr with capturing streams,
    and installs exception hooks. Must be called once at application startup,
    BEFORE uvicorn.run() so uvicorn's dictConfig handlers bind to our streams.

    Idempotent: if already initialized this is a no-op (the server may be imported
    and later run, or init_logging called more than once).

    Args:
        level: Logging level (default: INFO).
        log_dir: Directory for the rotating console log file.
        filename: Name of the console log file inside log_dir.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of rotated backups to keep.
    """
    global _initialized, logger, _original_stdout, _original_stderr

    with _init_lock:
        if _initialized:
            return  # Idempotent — server may be imported then run.

        _logger, _original_stdout, _original_stderr = setup_logger(
            level, log_dir=log_dir, filename=filename,
            max_bytes=max_bytes, backup_count=backup_count,
        )
        logger = _logger

        # Redirect sys.stdout and sys.stderr so that ALL print() calls and uncaught
        # thread exceptions are routed through the logging system → console.log.
        # The StreamHandler above already writes to _original_stdout, so no recursion.
        sys.stdout = _CapturingStream('stdout', _original_stdout)
        sys.stderr = _CapturingStream('stderr', _original_stderr)

        # Install global exception hooks for main thread and daemon threads
        sys.excepthook = _logging_excepthook
        threading.excepthook = _threading_excepthook

        _initialized = True  # Set inside lock to prevent concurrent init


def reset_logging() -> None:
    """
    Reset the logging system to its default state.

    Restores original sys.stdout/sys.stderr and uninstalls exception hooks.
    Useful for testing or clean shutdown.
    """
    global _initialized, logger, _original_stdout, _original_stderr

    if not _initialized:
        return

    # Restore original streams
    if _original_stdout is not None:
        sys.stdout = _original_stdout
    if _original_stderr is not None:
        sys.stderr = _original_stderr

    # Uninstall exception hooks only if they are our custom ones
    if hasattr(sys, '__excepthook__') and sys.excepthook is _logging_excepthook:
        sys.excepthook = sys.__excepthook__

    try:
        if hasattr(threading, 'excepthook') and threading.excepthook is _threading_excepthook:
            # Python 3.8+: restore default excepthook
            if hasattr(threading, '__standard_excepthook__'):
                threading.excepthook = threading.__standard_excepthook__
            else:
                threading.excepthook = None  # Triggers Python to use default
    except Exception:
        pass

    _initialized = False
