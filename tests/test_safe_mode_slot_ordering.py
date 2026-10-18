"""Safe-mode slot-ordering fixes + the orphan state-temp sweep.

Mirrors tests/test_regression.py's style: an isolated ModelManager via the
shared ``isolated_manager`` fixture, engine HTTP mocked, behaviour asserted
directly. Covers:

  * an unload DRAINS a slot action holding a granted ticket before dropping
    the FIFO queue and killing the engine (without the drain the save is cut
    mid-write — the orphan ``*.bin.<hash>.tmp`` symptom),
  * the drain is bounded: past the cap the kill proceeds anyway,
  * a state/load landing right after a deliberate unload is refused with 409
    instead of silently cold-reloading (and only inside that window),
  * _perform_slot_action stops retrying other endpoint formats when the
    engine is unreachable, but still retries them on an HTTP error,
  * the temp sweep removes only stale ``*.bin.<hash>.tmp`` files and never
    touches ``.bin`` state files.

Constants (SLOT_ACTION_DRAIN_TIMEOUT_S / RECENT_UNLOAD_REFUSE_WINDOW_S /
STATE_TEMP_MAX_AGE_S) are read from the live `server` module — never hardcoded —
so a change to a bound is visible to these tests.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import server
from server import (
    HTTPException,
    ModelConfig,
)
import httpx

# Assert against the live values, never hardcode: this one is imported, the others
# are read as server.X or patched inside the test that cares about them.
RECENT_UNLOAD_REFUSE_WINDOW_S = server.RECENT_UNLOAD_REFUSE_WINDOW_S


@pytest.fixture()
def autoloader_logs():
    """Capture records emitted on the "autoloader" logger for one test.

    Production sets the level via init_logging(); tests never call it, so the
    logger still defaults to WARNING and would drop the INFO lines we assert
    on. Raise the level for the duration of the test and restore it after.
    """
    recs: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            recs.append(record.getMessage())

    logger = logging.getLogger("autoloader")
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # keep pytest's caplog/root handlers out of the way
    handler = _Cap()
    logger.addHandler(handler)
    try:
        yield recs
    finally:
        logger.removeHandler(handler)
        logger.level = old_level
        logger.propagate = old_propagate


@pytest.fixture()
def mgr(isolated_manager):
    """Isolated manager with one loaded (fake) model, mirroring tests/test_regression.py:TestState."""
    cfg = ModelConfig.default_for(Path("m.gguf"))
    lm = MagicMock()
    lm.port = 19500
    lm.ready = True
    lm.config = cfg
    lm.state_path = None
    lm.pid = 4242  # a pid that does not exist -> the kill path no-ops cleanly
    isolated_manager.models["m.gguf"] = cfg
    isolated_manager.loaded["m.gguf"] = lm
    return isolated_manager


def _watch_queue_removals(mgr_, order):
    """Record when an unload drops a port's FIFO queue (the observable boundary)."""
    real_remove = mgr_.queue_registry.remove

    def watched_remove(port):
        order.append("queue-dropped")
        real_remove(port)

    mgr_.queue_registry.remove = watched_remove


async def wait_until(pred, what: str, deadline: float = 2.0):
    """Bounded wait for a test-setup race (fixed sleeps go red on a loaded host)."""
    limit = time.monotonic() + deadline
    while not pred():
        if time.monotonic() > limit:
            raise AssertionError(f"setup race: {what}")
        await asyncio.sleep(0.01)


# ===========================================================================
# Unload drains an in-flight slot action before killing the engine
# ===========================================================================


class TestUnloadDrain:
    async def test_unload_waits_for_granted_slot_action(self, mgr, autoloader_logs):
        """A save holding a granted ticket must finish before the queue is dropped."""
        order: list[str] = []

        async def slow_save(port, path, timeout=None):
            await asyncio.sleep(0.2)
            order.append("slot-action-finished")

        with patch.object(mgr, "_perform_slot_save", new=AsyncMock(side_effect=slow_save)):
            save_task = asyncio.create_task(mgr.save_state("m.gguf", "lbl"))
            await wait_until(lambda: mgr._inflight_slot_actions.get(19500, 0) == 1, "save in flight")

            _watch_queue_removals(mgr, order)
            await mgr.unload_model("m.gguf")
            await save_task

        assert mgr._inflight_slot_actions == {}
        assert order == ["slot-action-finished", "queue-dropped"], (
            f"save cut mid-write: {order}"
        )
        assert any("waiting for in-flight slot action" in m for m in autoloader_logs)

    async def test_drain_cap_proceeds_with_kill(self, mgr, autoloader_logs):
        """Past the bounded cap the kill still happens; memory safety wins."""
        order: list[str] = []

        async def very_slow_save(port, path, timeout=None):
            await asyncio.sleep(1.0)
            order.append("slot-action-finished")

        with patch.object(mgr, "_perform_slot_save", new=AsyncMock(side_effect=very_slow_save)), \
             patch.object(server, "SLOT_ACTION_DRAIN_TIMEOUT_S", 0.2):
            save_task = asyncio.create_task(mgr.save_state("m.gguf", "lbl"))
            await wait_until(lambda: mgr._inflight_slot_actions.get(19500, 0) == 1, "save in flight")
            _watch_queue_removals(mgr, order)
            await mgr.unload_model("m.gguf")
            assert order == ["queue-dropped"], (
                f"drain exceeded its cap: {order}"
            )
            await save_task

        assert any("did not finish within" in m for m in autoloader_logs)

    async def test_drain_covers_queue_handoff(self, mgr, autoloader_logs):
        """A second save that gets GRANTED when the first releases is still drained.

        This is why _begin_slot_action runs before acquire_ticket: counting only
        granted holders would let the unload kill the engine between save1 and
        save2, re-opening exactly the truncation symptom the drain exists to close.
        """
        order: list[str] = []
        call_no = iter(range(1, 100))

        async def save(port, path, timeout=None):
            i = next(call_no)
            order.append(f"save{i}-start")
            await asyncio.sleep(0.1)
            order.append(f"save{i}-end")

        with patch.object(mgr, "_perform_slot_save", new=AsyncMock(side_effect=save)):
            first = asyncio.create_task(mgr.save_state("m.gguf", "l1"))
            await wait_until(lambda: bool(order) and order[0] == "save1-start", "save1 writing")
            second = asyncio.create_task(mgr.save_state("m.gguf", "l2"))
            await wait_until(lambda: mgr._inflight_slot_actions.get(19500, 0) == 2, "save2 queued")

            _watch_queue_removals(mgr, order)
            await mgr.unload_model("m.gguf")
            await asyncio.gather(first, second)

        assert order == ["save1-start", "save1-end", "save2-start", "save2-end", "queue-dropped"], (
            f"save2 was granted only after save1 released and was still cut: {order}"
        )

    async def test_no_drain_when_no_slot_action_is_in_flight(self, mgr, autoloader_logs):
        """Nothing counted against the port -> no wait, no INFO line."""
        mgr.queue_registry.get_or_create(19500, "m.gguf").acquire_ticket(timeout=5.0)  # a queue holder, not a slot action
        assert mgr._inflight_slot_actions == {}
        await mgr.unload_model("m.gguf")
        assert not any("waiting for in-flight slot action" in m for m in autoloader_logs)


# ===========================================================================
# load_state refuses to auto-load a deliberately released endpoint
# ===========================================================================


class TestRecentUnloadRefusal:
    async def test_load_state_refused_after_recent_unload(self, mgr):
        (mgr.save_state_dir / "m.gguf.lbl.bin").write_bytes(b"state")
        await mgr.unload_model("m.gguf")
        with pytest.raises(HTTPException) as exc:
            await mgr.load_state("m.gguf", "lbl")
        assert exc.value.status_code == 409
        assert "deliberately released" in exc.value.detail

    async def test_refusal_window_is_bounded(self, mgr):
        """Outside the window the old auto-load-before-restore behavior stands."""

        async def fake_load(model_id):
            lm = MagicMock()
            lm.port = 19500
            lm.ready = True
            mgr.loaded["m.gguf"] = lm
            return lm

        (mgr.save_state_dir / "m.gguf.lbl.bin").write_bytes(b"state")
        await mgr.unload_model("m.gguf")  # model out of self.loaded, marker armed
        mgr._recent_unload["m.gguf"] = time.monotonic() - (RECENT_UNLOAD_REFUSE_WINDOW_S + 1)

        load_model = AsyncMock(side_effect=fake_load)
        with patch.object(mgr, "load_model", new=load_model), \
             patch.object(mgr, "_perform_slot_restore", new=AsyncMock()):
            path = await mgr.load_state("m.gguf", "lbl")
        assert load_model.call_count == 1, "never took the auto-load path"
        assert path.name == "m.gguf.lbl.bin"


# ===========================================================================
# Stop retrying against a dead engine; make the failure legible
# ===========================================================================


class TestSlotActionRetry:
    async def test_unreachable_engine_stops_retrying(self, mgr):
        """An empty-message ConnectError must break the loop and stay legible."""
        post = AsyncMock(side_effect=httpx.ConnectError(""))
        with patch.object(mgr.client, "post", new=post):
            with pytest.raises(RuntimeError) as exc:
                await mgr._perform_slot_action("save", 19500, mgr.save_state_dir / "x.bin")
        assert post.call_count == 1, f"retried against a dead engine: {post.call_count} attempts"
        # repr() fallback: str(ConnectError("")) is empty, the log/exception must not be.
        assert "ConnectError" in str(exc.value)

    async def test_dead_mid_response_stops_retrying(self, mgr):
        """httpx.ReadError('') (peer EOF after headers) must break the loop too."""
        post = AsyncMock(side_effect=httpx.ReadError(""))
        with patch.object(mgr.client, "post", new=post):
            with pytest.raises(RuntimeError) as exc:
                await mgr._perform_slot_action("save", 19500, mgr.save_state_dir / "x.bin")
        assert post.call_count == 1, f"retried against a dead engine: {post.call_count} attempts"
        # Same repr() trap: str(ReadError("")) is empty, so the log/exception
        # must carry the class name via the repr(e) fallback.
        assert "ReadError" in str(exc.value)

    async def test_read_timeout_still_tries_other_formats(self, mgr):
        """A reachable-but-slow engine must NOT take the unreachable abort."""
        post = AsyncMock(side_effect=httpx.ReadTimeout(""))
        with patch.object(mgr.client, "post", new=post):
            with pytest.raises(RuntimeError) as exc:
                await mgr._perform_slot_action("save", 19500, mgr.save_state_dir / "x.bin")
        assert post.call_count == 3, f"treated a slow engine as unreachable: {post.call_count} attempts"
        assert "ReadTimeout" in str(exc.value)

    async def test_http_error_still_tries_other_formats(self, mgr):
        """A reachable engine returning 4xx must still try the other formats."""
        post = AsyncMock(return_value=httpx.Response(404, text="no such endpoint"))
        with patch.object(mgr.client, "post", new=post):
            with pytest.raises(RuntimeError) as exc:
                await mgr._perform_slot_action("save", 19500, mgr.save_state_dir / "x.bin")
        assert post.call_count == 3
        assert "HTTP 404" in str(exc.value)
        assert "no such endpoint" in str(exc.value)


# ===========================================================================
# Orphan state-temp cleanup (interrupted-save *.bin.<hash>.tmp scratch files)
# ===========================================================================


class TestStateTempSweep:
    def test_sweep_removes_only_old_temps(self, mgr):
        d = mgr.save_state_dir
        stale = d / ".m.gguf.lbl.bin.08a653e446816a81.tmp"
        fresh = d / ".m.gguf.lbl.bin.deadbeefcafebabe.tmp"
        completed = d / "m.gguf.lbl.bin"
        unrelated = d / "scratch.tmp"
        for p in (stale, fresh, completed, unrelated):
            p.write_bytes(b"x")
        old = time.time() - (server.STATE_TEMP_MAX_AGE_S + 60)
        os.utime(stale, (old, old))
        os.utime(unrelated, (old, old))

        assert mgr._cleanup_state_temps() == 1
        assert not stale.exists()
        assert fresh.exists()
        assert completed.exists()
        assert unrelated.exists(), "must only match the .bin.<hash>.tmp pattern"

    def test_sweep_never_touches_fresh_temps(self, mgr):
        """An in-flight save's own scratch file is never reclaimed."""
        d = mgr.save_state_dir
        in_flight = d / ".m.gguf.now.bin.1111222233334444.tmp"
        in_flight.write_bytes(b"x")
        assert mgr._cleanup_state_temps() == 0
        assert in_flight.exists()
