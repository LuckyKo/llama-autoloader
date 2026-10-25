"""Regression tests for BUG_0013: an explicit unload that races an in-flight load.

The 2026-10-25 incident: a client's ``POST /v1/unload_all`` (200, cards really freed) landed
while the loader was reading a model's weights. The load task learned about the kill only
through ``proc.poll()``, so the loader logged::

    llama-server process exited prematurely with code 15
    Model ... failed to become ready
    Dead-backend recovery reload of '...' failed
    Dead-backend recovery for '...' failed (attempt 1); suppressing retries for 5.0s

Nothing there failed: the release was requested and it worked. These tests pin the
distinction — a load aborted by an unload ends as ``LoadAbortedByUnload`` (INFO), never as a
failed start, and never arms the BUG_0007 recovery cooldown.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest

import server
from server import LoadAbortedByUnload


class _FakeProc:
    """Minimal Popen stand-in: alive until the fake unload kills it."""

    pid = 987654
    stdout = None
    stderr = None

    def __init__(self):
        self.exit_code = None

    def poll(self):
        return self.exit_code

    def kill(self):
        self.exit_code = 15

    def terminate(self):
        self.exit_code = 15

    def wait(self, timeout=None):
        return self.exit_code or 0


class _ExitedProc(_FakeProc):
    """A process that is already gone (what the readiness poll sees after a kill)."""

    def __init__(self, exit_code=15):
        super().__init__()
        self.exit_code = exit_code


@pytest.mark.asyncio
async def test_unload_during_load_ends_the_load_aborted(seeded_manager, caplog):
    """The whole race: load in flight -> unload_all -> the load must not look like a crash."""
    mgr = seeded_manager
    mgr.scan()  # register the seeded GGUFs (house pattern: caller scans after seeding)
    mid = "model-a.gguf"
    proc = _FakeProc()
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def _wait_until_ready(self_, port, proc=None, timeout=120.0, abort_probe=None):
        entered.set()
        await gate.wait()
        # The unload killed the engine; same signal the real poll loop would observe.
        proc.exit_code = 15
        assert abort_probe is not None
        assert abort_probe() is True, "the load must be able to see the requested release"
        return False

    caplog.set_level(logging.INFO)
    with patch("server.subprocess.Popen", side_effect=lambda *a, **k: proc), \
         patch.object(server.ModelManager, "_wait_until_ready", new=_wait_until_ready):
        task = asyncio.create_task(mgr.load_model(mid))
        await asyncio.wait_for(entered.wait(), timeout=10)
        assert mid in mgr.loaded, "the model is registered before the readiness wait"

        result = await mgr.unload_all()
        gate.set()
        with pytest.raises(LoadAbortedByUnload):
            await task

    assert result["unloaded_all"] is True
    assert mid not in mgr.loaded
    assert mid not in mgr._load_aborts, "the marker must not leak past the aborted load"
    assert mgr.recovery_retry_after(mid) is None, "an aborted load must not arm the cooldown"

    messages = [r.getMessage() for r in caplog.records]
    assert not [m for m in messages if "failed to become ready" in m], messages
    assert not [m for m in messages if "exited prematurely" in m], messages
    assert [m for m in messages if "aborted by an unload" in m], messages


@pytest.mark.asyncio
async def test_abort_marker_is_only_set_for_an_outside_release(isolated_manager):
    """The marker must never fire for the load task's own cleanup (force/failed reload)."""
    mgr = isolated_manager
    mid = "model-a.gguf"

    # Nothing loading: a plain unload has nothing to abort.
    assert mgr._mark_load_aborted_locked(mid) is False
    assert mid not in mgr._load_aborts

    # A finished task is not a running load.
    finished = asyncio.get_running_loop().create_future()
    finished.set_result(None)
    mgr._loading_tasks[mid] = finished
    assert mgr._mark_load_aborted_locked(mid) is False
    assert mid not in mgr._load_aborts

    # The load's own cleanup runs INSIDE the load task: not an outside release.
    mgr._loading_tasks[mid] = asyncio.current_task()
    assert mgr._mark_load_aborted_locked(mid) is False
    assert mid not in mgr._load_aborts

    # Another task driving the load IS an outside release.
    other = asyncio.create_task(asyncio.sleep(30))
    try:
        mgr._loading_tasks[mid] = other
        assert mgr._mark_load_aborted_locked(mid) is True
        assert mid in mgr._load_aborts
    finally:
        other.cancel()
        with pytest.raises(asyncio.CancelledError):
            await other
        mgr._loading_tasks.pop(mid, None)
        mgr._load_aborts.discard(mid)


@pytest.mark.asyncio
async def test_requested_unload_exit_is_logged_as_info_not_error(isolated_manager, caplog):
    mgr = isolated_manager
    caplog.set_level(logging.INFO)
    ok = await mgr._wait_until_ready(19555, proc=_ExitedProc(15), timeout=1.0,
                                     abort_probe=lambda: True)

    assert ok is False
    messages = [r.getMessage() for r in caplog.records]
    assert not [m for m in messages if "exited prematurely" in m], messages
    assert [m for m in messages if "exited after a requested unload" in m], messages


@pytest.mark.asyncio
async def test_genuine_crash_is_still_an_error(isolated_manager, caplog):
    """The abort path must not mute real crashes (BUG_0007 depends on those being visible)."""
    mgr = isolated_manager
    caplog.set_level(logging.INFO)
    ok = await mgr._wait_until_ready(19555, proc=_ExitedProc(1), timeout=1.0,
                                     abort_probe=lambda: False)

    assert ok is False
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert [m for m in errors if "exited prematurely with code 1" in m], errors


def _add_unready_resident(mgr, model_id="dead.gguf", pid=424242, port=19555):
    from pathlib import Path
    cfg = server.ModelConfig.default_for(Path(model_id))
    lm = server.LoadedModel(
        model_id=model_id,
        gguf_path=Path(model_id),
        config=cfg,
        port=port,
        process=None,
        pid=pid,
    )
    lm.ready = False  # what a failed/dead engine leaves behind
    mgr.models[model_id] = cfg
    mgr.loaded[model_id] = lm
    return lm


@pytest.mark.asyncio
async def test_recovery_does_not_book_an_aborted_load_as_a_failure(isolated_manager, caplog):
    """Dead-backend recovery must not arm the cooldown when an unload aborted its reload."""
    mgr = isolated_manager
    lm = _add_unready_resident(mgr)

    async def _aborted(mid, force=False):  # noqa: ARG001 - signature parity
        raise LoadAbortedByUnload(f"Load of '{mid}' was aborted by an unload of the same model")

    caplog.set_level(logging.INFO)
    with patch.object(mgr, "_load_model_serialized", new=_aborted):
        recovered = await mgr.recover_dead_model(lm.model_id, lm.port, lm.generation)

    assert recovered is None
    assert mgr.recovery_retry_after(lm.model_id) is None, "must not arm the recovery cooldown"
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert not [m for m in errors if "recovery reload" in m], errors
    assert [m for m in caplog.records if "aborted by an unload" in m.getMessage()]


@pytest.mark.asyncio
async def test_recovery_still_books_a_real_reload_failure(isolated_manager, caplog):
    """Control: a genuine reload failure still arms the cooldown and logs an ERROR."""
    mgr = isolated_manager
    lm = _add_unready_resident(mgr)

    async def _boom(mid, force=False):  # noqa: ARG001 - signature parity
        raise RuntimeError("engine died during launch")

    caplog.set_level(logging.INFO)
    with patch.object(mgr, "_load_model_serialized", new=_boom):
        recovered = await mgr.recover_dead_model(lm.model_id, lm.port, lm.generation)

    assert recovered is None
    assert mgr.recovery_retry_after(lm.model_id) is not None, "a real failure must arm the cooldown"
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert [m for m in errors if "recovery reload" in m], errors
