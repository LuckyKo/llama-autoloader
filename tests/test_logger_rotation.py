"""Regression tests for console.log rotation (BUG_0104 audit follow-up).

The reported BUG_0104 symptom (a fresh console.log staying empty after
rotation) was disproved by the preserved log.  While auditing the handler we
did find a real failure mode: a backup file locked by another process made
stdlib ``doRollover()`` raise before the active file was touched, which dropped
the record and could recurse through the tee'd stderr.  These tests pin the
invariants that matter:

- a successful rotation leaves a writable, non-empty new console.log;
- a locked backup must not drop records or stop file logging;
- the capturing stdout/stderr tee must survive that same failure.
"""

import logging
import os
import shutil

import logger as logger_mod
from logger import _WindowsSafeRotatingFileHandler


def _make_handler(tmp_path, *, max_bytes=200, backup_count=2):
    handler = _WindowsSafeRotatingFileHandler(
        tmp_path / "console.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        errors="replace",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger(f"test.rotation.{id(handler)}")
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False
    return log, handler


def _close(log, handler):
    log.handlers[:] = []
    handler.close()


def _prime_locked_backup(tmp_path):
    """Create a full active log and a backup that cannot be moved/removed."""
    base = tmp_path / "console.log"
    backup = tmp_path / "console.log.1"
    base.write_text("y" * 400, encoding="utf-8")
    backup.write_text("old backup\n", encoding="utf-8")
    return base, backup


def _simulate_locked_backup(monkeypatch):
    """Make every operation that touches console.log.1 fail with WinError 32."""
    real_replace = os.replace
    real_remove = os.remove
    real_move = shutil.move

    def is_backup(path):
        return os.fspath(path).endswith("console.log.1")

    def locked_replace(src, dst):
        if is_backup(src) or is_backup(dst):
            raise PermissionError(32, "file is locked", os.fspath(src))
        return real_replace(src, dst)

    def locked_remove(path):
        if is_backup(path):
            raise PermissionError(32, "file is locked", os.fspath(path))
        return real_remove(path)

    def locked_move(src, dst):
        if is_backup(dst):
            raise PermissionError(32, "file is locked", os.fspath(src))
        return real_move(src, dst)

    monkeypatch.setattr(logger_mod.os, "replace", locked_replace)
    monkeypatch.setattr(logger_mod.os, "remove", locked_remove)
    monkeypatch.setattr(logger_mod.shutil, "move", locked_move)


def test_rotation_creates_new_file_and_keeps_writing(tmp_path):
    log, handler = _make_handler(tmp_path, max_bytes=200, backup_count=3)
    try:
        for i in range(20):
            log.info("before-%02d-%s", i, "x" * 40)
        log.info("AFTER_ROTATION_SENTINEL")
    finally:
        _close(log, handler)

    base = tmp_path / "console.log"
    assert base.exists()
    assert base.stat().st_size > 0
    assert "AFTER_ROTATION_SENTINEL" in base.read_text(encoding="utf-8")
    assert (tmp_path / "console.log.1").exists()
    assert (tmp_path / "console.log.2").exists()


def test_locked_backup_is_skipped_and_record_is_not_dropped(tmp_path, monkeypatch):
    base, _backup = _prime_locked_backup(tmp_path)
    _simulate_locked_backup(monkeypatch)

    log, handler = _make_handler(tmp_path, max_bytes=100, backup_count=2)
    errors = []
    monkeypatch.setattr(handler, "handleError", lambda record: errors.append(record))
    try:
        log.info("AFTER_FAILED_ROTATION_%s", "z" * 100)
        log.info("SECOND_AFTER_FAILED_ROTATION")
    finally:
        _close(log, handler)

    text = base.read_text(encoding="utf-8")
    assert "AFTER_FAILED_ROTATION" in text
    assert "SECOND_AFTER_FAILED_ROTATION" in text
    # A dropped record would have gone through handleError().
    assert errors == []


def test_capturing_stream_survives_locked_backup(tmp_path, monkeypatch):
    base, _backup = _prime_locked_backup(tmp_path)
    _simulate_locked_backup(monkeypatch)

    log, handler = _make_handler(tmp_path, max_bytes=100, backup_count=2)
    errors = []
    monkeypatch.setattr(handler, "handleError", lambda record: errors.append(record))
    monkeypatch.setattr(logger_mod, "logger", log)

    original = open(os.devnull, "w", encoding="utf-8")
    stream = logger_mod._CapturingStream("stdout", original)
    try:
        stream.write("CAPTURING_SENTINEL\n")
    finally:
        original.close()
        _close(log, handler)

    assert "CAPTURING_SENTINEL" in base.read_text(encoding="utf-8")
    assert errors == []


def test_locked_backup_keeps_active_log_writable(tmp_path):
    """Windows-specific sanity check with a real open handle on the backup."""
    base, backup = _prime_locked_backup(tmp_path)
    log, handler = _make_handler(tmp_path, max_bytes=100, backup_count=2)
    errors = []
    handler.handleError = lambda record: errors.append(record)
    try:
        with backup.open("r", encoding="utf-8"):
            log.info("REAL_LOCKED_SENTINEL")
    finally:
        _close(log, handler)

    assert "REAL_LOCKED_SENTINEL" in base.read_text(encoding="utf-8")
    assert errors == []
