"""Regression tests for BUG_0001: dead llama-server recovery in the TCP fast path.

The forwarder trusts ``loaded[mid].ready`` for already-loaded models. When a backend
dies silently (GPU TDR / OOM / silent exit), every subsequent LLM request used to get
a permanent 502 "llama-server unreachable" and the registry kept claiming the model was
healthy. These tests pin the two halves of the fix:

  * ``ModelManager.recover_dead_model`` evicts (only after a verified kill), reloads,
    dedups a thundering herd, and never starts a duplicate engine over a live process.
  * the forwarder turns a connection-refused into one recovery attempt and retries the
    same request on the recovered port instead of returning a bare 502.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from server import LoadedModel, ModelConfig, UnloadResult
from tcp_forwarder import (
    RawTCPForwarder,
    _BackendUnreachable,
    _LLAMA_SERVER_LABEL,
)


# --------------------------------------------------------------------------- helpers
def _add_resident(mgr, model_id: str = "dead.gguf", port: int = 19555,
                  ready: bool = True, pid: int = 0,
                  generation: int = 0) -> LoadedModel:
    cfg = ModelConfig.default_for(Path(model_id))
    lm = LoadedModel(
        model_id=model_id,
        gguf_path=Path(model_id),
        config=cfg,
        port=port,
        process=None,
        pid=pid,
        generation=generation,
    )
    lm.ready = ready
    mgr.models[model_id] = cfg
    mgr.loaded[model_id] = lm
    return lm


def _new_resident(mgr, model_id: str, port: int,
                  generation: int = 0) -> LoadedModel:
    lm = LoadedModel(
        model_id=model_id,
        gguf_path=Path(model_id),
        config=mgr.models[model_id],
        port=port,
        process=None,
        pid=0,
        generation=generation,
    )
    lm.ready = True
    return lm


class _FakeQueue:
    """Minimal ticket queue whose context manager grants immediately."""

    def __init__(self):
        self.acquired = 0

    @contextmanager
    def ticket(self, timeout=300.0):  # noqa: ARG002 - API parity
        self.acquired += 1
        yield self.acquired


class _FakeRegistry:
    def __init__(self):
        self.calls = []
        self.queues = {}

    def get_or_create(self, port: int, model_id: str) -> _FakeQueue:
        self.calls.append((port, model_id))
        self.queues.setdefault(port, _FakeQueue())
        return self.queues[port]


class _StoppedQueue:
    """Queue already dropped by a concurrent unload/recovery."""

    @contextmanager
    def ticket(self, timeout=300.0):  # noqa: ARG002 - API parity
        raise RuntimeError("FIFO queue stopped")
        yield  # pragma: no cover - makes this a generator for @contextmanager


class _FakeManager:
    def __init__(self):
        self.queue_registry = _FakeRegistry()
        self.loaded = {}


class _RecoveringManager(_FakeManager):
    def recovery_in_progress(self, dead_port: int,
                             expected_generation=None) -> bool:  # noqa: ARG002
        return dead_port == 9001


def _fwd(manager=None) -> RawTCPForwarder:
    return RawTCPForwarder(listen_port=19999, internal_api_port=19998,
                           model_manager=manager or _FakeManager())


# --------------------------------------------------------------------------- manager recovery
@pytest.mark.asyncio
async def test_recover_dead_model_evicts_and_reloads(isolated_manager, monkeypatch):
    mgr = isolated_manager
    old = _add_resident(mgr)
    loads = []

    async def _fake_load(model_id):
        loads.append(model_id)
        new = _new_resident(mgr, model_id, old.port + 1)
        mgr.loaded[model_id] = new
        return new

    monkeypatch.setattr(mgr, "_load_model_serialized", _fake_load)

    result = await mgr.recover_dead_model("dead.gguf", old.port)

    assert result is not None and result.ready
    assert result.port == old.port + 1
    assert loads == ["dead.gguf"]
    assert old.ready is False
    assert mgr.loaded["dead.gguf"] is result


@pytest.mark.asyncio
async def test_recover_dead_model_dedups_concurrent_requests(isolated_manager, monkeypatch):
    """A burst hitting the same dead port must run ONE unload/reload cycle."""
    mgr = isolated_manager
    old = _add_resident(mgr)
    loads = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_load(model_id):
        loads.append(model_id)
        started.set()
        await release.wait()
        new = _new_resident(mgr, model_id, old.port + 1)
        mgr.loaded[model_id] = new
        return new

    monkeypatch.setattr(mgr, "_load_model_serialized", _slow_load)

    first = asyncio.create_task(mgr.recover_dead_model("dead.gguf", old.port))
    await asyncio.wait_for(started.wait(), timeout=5)

    followers = [
        asyncio.create_task(mgr.recover_dead_model("dead.gguf", old.port))
        for _ in range(4)
    ]
    # Let the followers see the existing recovery task and join it.
    for _ in range(5):
        await asyncio.sleep(0)

    release.set()
    results = await asyncio.gather(first, *followers)

    assert loads == ["dead.gguf"]
    assert all(r is results[0] for r in results)
    assert results[0].port == old.port + 1


@pytest.mark.asyncio
async def test_recover_dead_model_refuses_reload_when_kill_unverified(
        isolated_manager, monkeypatch):
    """Never start a second engine on top of a process that may still be alive."""
    mgr = isolated_manager
    old = _add_resident(mgr, pid=424242)

    async def _failed_unload_finish(lm, model_id):  # noqa: ARG001
        return False, "still running"

    async def _must_not_load(model_id):  # pragma: no cover - asserted
        raise AssertionError(f"must not reload {model_id} after a failed kill")

    monkeypatch.setattr(mgr, "_unload_finish", _failed_unload_finish)
    monkeypatch.setattr(mgr, "_load_model_serialized", _must_not_load)

    result = await mgr.recover_dead_model("dead.gguf", old.port)

    assert result is None
    assert mgr.loaded["dead.gguf"] is old
    assert old.ready is False  # truthful: the engine is not serving


@pytest.mark.asyncio
async def test_recover_dead_model_reuses_replacement_on_new_port(
        isolated_manager, monkeypatch):
    mgr = isolated_manager
    replacement = _add_resident(mgr, port=19556)

    async def _must_not_load(model_id):  # pragma: no cover - asserted
        raise AssertionError(f"must not reload {model_id} after another recovery won")

    monkeypatch.setattr(mgr, "_load_model_serialized", _must_not_load)

    result = await mgr.recover_dead_model("dead.gguf", 19555)

    assert result is replacement


@pytest.mark.asyncio
async def test_recover_dead_model_unknown_alias_reuses_ready_fallback(
        isolated_manager, monkeypatch):
    """A dead alias request already evicted by another recovery still gets served."""
    mgr = isolated_manager
    fallback = _add_resident(mgr, model_id="other.gguf", port=19557)

    async def _must_not_load(model_id):  # pragma: no cover - asserted
        raise AssertionError(f"must not load unknown alias {model_id}")

    monkeypatch.setattr(mgr, "_load_model_serialized", _must_not_load)

    result = await mgr.recover_dead_model("unknown-provider-alias", 19555)

    assert result is fallback


@pytest.mark.asyncio
async def test_recover_dead_model_preserves_newer_ready_generation(
        isolated_manager, monkeypatch):
    """BUG_0003: a replacement that reused the dead port must not be evicted."""
    mgr = isolated_manager
    old = _add_resident(mgr, ready=False, generation=10)
    replacement = _new_resident(mgr, "dead.gguf", old.port, generation=11)
    mgr.loaded["dead.gguf"] = replacement
    unloads = []
    loads = []

    async def _must_not_unload(model_id):
        unloads.append(model_id)
        raise AssertionError("healthy replacement must not be unloaded")

    async def _must_not_load(model_id):
        loads.append(model_id)
        raise AssertionError("healthy replacement must not be reloaded")

    monkeypatch.setattr(mgr, "_unload_unlocked", _must_not_unload)
    monkeypatch.setattr(mgr, "_load_model_serialized", _must_not_load)

    result = await mgr.recover_dead_model(
        "dead.gguf", old.port, expected_generation=old.generation)

    assert result is replacement
    assert replacement.ready is True
    assert mgr.loaded["dead.gguf"] is replacement
    assert unloads == [] and loads == []


@pytest.mark.asyncio
async def test_recover_dead_model_still_evicts_exact_dead_generation(
        isolated_manager, monkeypatch):
    """BUG_0003: the instance that actually refused must still be recovered."""
    mgr = isolated_manager
    old = _add_resident(mgr, generation=20)
    loads = []

    async def _fake_load(model_id):
        loads.append(model_id)
        new = _new_resident(mgr, model_id, old.port, generation=21)
        mgr.loaded[model_id] = new
        return new

    monkeypatch.setattr(mgr, "_load_model_serialized", _fake_load)

    result = await mgr.recover_dead_model(
        "dead.gguf", old.port, expected_generation=old.generation)

    assert result is not None and result.ready
    assert result.generation == 21
    assert old.ready is False
    assert loads == ["dead.gguf"]


@pytest.mark.asyncio
async def test_recover_dead_model_joins_unready_newer_generation(
        isolated_manager, monkeypatch):
    """A same-port replacement still loading is joined, never unloaded."""
    mgr = isolated_manager
    old = _add_resident(mgr, ready=False, generation=30)
    replacement = _new_resident(mgr, "dead.gguf", old.port, generation=31)
    replacement.ready = False
    mgr.loaded["dead.gguf"] = replacement
    loads = []

    async def _join_load(model_id):
        loads.append(model_id)
        replacement.ready = True
        return replacement

    async def _must_not_unload(model_id):
        raise AssertionError("in-flight replacement must not be unloaded")

    monkeypatch.setattr(mgr, "_unload_unlocked", _must_not_unload)
    monkeypatch.setattr(mgr, "_load_model_serialized", _join_load)

    result = await mgr.recover_dead_model(
        "dead.gguf", old.port, expected_generation=old.generation)

    assert result is replacement
    assert result.ready is True
    assert loads == ["dead.gguf"]


@pytest.mark.asyncio
async def test_recovery_teardown_does_not_hold_manager_lock(
        isolated_manager, monkeypatch):
    """BUG_0005: routing/status must stay responsive during slow drain/kill."""
    mgr = isolated_manager
    old = _add_resident(mgr, generation=40)
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_failed_unload(lm, model_id):  # noqa: ARG001
        started.set()
        await release.wait()
        return False, "still running"

    monkeypatch.setattr(mgr, "_unload_finish", _slow_failed_unload)

    recovery = asyncio.create_task(
        mgr.recover_dead_model("dead.gguf", old.port, old.generation))
    await asyncio.wait_for(started.wait(), timeout=5)

    # The old implementation held _lock across this wait, so this call blocked
    # until the teardown finished. It must now complete while teardown is paused.
    resolved = await asyncio.wait_for(
        mgr.resolve_model_id_strict("dead.gguf"), timeout=1.0)
    assert resolved == "dead.gguf"
    assert mgr.recovery_in_progress(old.port, old.generation) is True
    assert "dead.gguf" in mgr._unload_in_progress

    release.set()
    assert await asyncio.wait_for(recovery, timeout=5) is None
    assert mgr.loaded["dead.gguf"] is old
    assert "dead.gguf" not in mgr._unload_in_progress


@pytest.mark.asyncio
async def test_recoveries_for_different_models_run_concurrently(
        isolated_manager, monkeypatch):
    """BUG_0005: a per-model lock, not the global lock, serializes teardown."""
    mgr = isolated_manager
    old_a = _add_resident(mgr, model_id="a.gguf", port=19601, generation=50)
    old_b = _add_resident(mgr, model_id="b.gguf", port=19602, generation=51)
    started = set()
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_unload(lm, model_id):
        started.add(model_id)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return True, ""

    monkeypatch.setattr(mgr, "_unload_finish", _slow_unload)

    async def _fake_load(model_id):
        lm = _new_resident(mgr, model_id, 19600 + (1 if model_id == "a.gguf" else 2),
                           generation=60 + (1 if model_id == "a.gguf" else 2))
        mgr.loaded[model_id] = lm
        return lm

    monkeypatch.setattr(mgr, "_load_model_serialized", _fake_load)

    first = asyncio.create_task(
        mgr.recover_dead_model("a.gguf", old_a.port, old_a.generation))
    second = asyncio.create_task(
        mgr.recover_dead_model("b.gguf", old_b.port, old_b.generation))

    await asyncio.wait_for(both_started.wait(), timeout=5)
    release.set()
    result_a, result_b = await asyncio.gather(first, second)

    assert result_a is not None and result_b is not None
    assert started == {"a.gguf", "b.gguf"}
    assert mgr.loaded["a.gguf"] is result_a
    assert mgr.loaded["b.gguf"] is result_b


@pytest.mark.asyncio
async def test_explicit_unload_waits_for_inflight_recovery(
        isolated_manager, monkeypatch):
    """An unload must not report success while recovery is about to reload."""
    mgr = isolated_manager
    old = _add_resident(mgr, generation=70)
    started = asyncio.Event()
    release = asyncio.Event()
    new_lm = _new_resident(mgr, "dead.gguf", old.port, generation=71)

    async def _slow_unload(lm, model_id):  # noqa: ARG001
        started.set()
        await release.wait()
        return True, ""

    async def _fake_load(model_id, force=False):  # noqa: ARG001
        mgr.loaded[model_id] = new_lm
        return new_lm

    monkeypatch.setattr(mgr, "_unload_finish", _slow_unload)
    monkeypatch.setattr(mgr, "_load_model_serialized", _fake_load)

    recovery = asyncio.create_task(
        mgr.recover_dead_model("dead.gguf", old.port, old.generation))
    await asyncio.wait_for(started.wait(), timeout=5)

    unload = asyncio.create_task(mgr.unload_model("dead.gguf"))
    await asyncio.sleep(0.05)
    assert not unload.done(), "unload returned during the recovery teardown"

    release.set()
    assert await asyncio.wait_for(recovery, timeout=5) is new_lm
    result = await asyncio.wait_for(unload, timeout=5)

    assert result is not None and result.unloaded
    assert result.model_id == "dead.gguf"
    assert "dead.gguf" not in mgr.loaded


def test_note_request_end_releases_original_generation_only(isolated_manager):
    """BUG_0004: a stale completion must not decrement a replacement engine."""
    mgr = isolated_manager
    old = _add_resident(mgr, port=19560, generation=100)
    old_token = mgr.note_request_start(old.port, old.generation)

    replacement = _new_resident(mgr, "dead.gguf", old.port, generation=101)
    mgr.loaded["dead.gguf"] = replacement
    new_token = mgr.note_request_start(replacement.port, replacement.generation)

    assert old_token is old
    assert new_token is replacement
    assert old.active_requests == 1
    assert replacement.active_requests == 1

    mgr.note_request_end(old_token)
    assert old.active_requests == 0
    assert replacement.active_requests == 1, \
        "stale end decremented the replacement engine"

    mgr.note_request_end(new_token)
    assert replacement.active_requests == 0


def test_note_request_start_rejects_stale_generation(isolated_manager):
    """A request routed to a dead generation must not attach to its replacement."""
    mgr = isolated_manager
    old = _add_resident(mgr, port=19561, generation=110)
    replacement = _new_resident(mgr, "dead.gguf", old.port, generation=111)
    mgr.loaded["dead.gguf"] = replacement

    assert mgr.note_request_start(old.port, old.generation) is None
    assert replacement.active_requests == 0


@pytest.mark.asyncio
async def test_do_load_model_cleans_stale_not_ready_entry(isolated_manager, monkeypatch):
    mgr = isolated_manager
    _add_resident(mgr, ready=False)
    unloads = []

    async def _unload(model_id):
        unloads.append(model_id)
        mgr.loaded.pop(model_id, None)
        return UnloadResult(model_id, True, 0)

    def _stop_after_cleanup(model_id=None):  # noqa: ARG001
        raise RuntimeError("stop after stale cleanup")

    monkeypatch.setattr(mgr, "_unload_unlocked", _unload)
    monkeypatch.setattr(mgr, "_allocate_port", _stop_after_cleanup)

    with pytest.raises(RuntimeError, match="stop after stale cleanup"):
        await mgr._do_load_model("dead.gguf")

    assert unloads == ["dead.gguf"]


@pytest.mark.asyncio
async def test_do_load_model_refuses_stale_entry_when_kill_unverified(
        isolated_manager, monkeypatch):
    mgr = isolated_manager
    _add_resident(mgr, ready=False)

    async def _failed_unload(model_id):
        return UnloadResult(model_id, False, 123, "still running")

    monkeypatch.setattr(mgr, "_unload_unlocked", _failed_unload)

    with pytest.raises(RuntimeError, match="still running"):
        await mgr._do_load_model("dead.gguf")


# --------------------------------------------------------------------------- forwarder
def test_llm_connect_refused_raises_recoverable(monkeypatch):
    f = _fwd()

    def _refuse(port):  # noqa: ARG001
        raise ConnectionRefusedError(10061, "actively refused")

    sent = []
    monkeypatch.setattr(f, "_connect_backend", _refuse)
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: sent.append((status, message)))

    with pytest.raises(_BackendUnreachable):
        f._pipe_to_backend_inner(object(), b"", b"", 9001, _LLAMA_SERVER_LABEL)

    assert sent == []


def test_internal_api_connect_refused_still_returns_502(monkeypatch):
    f = _fwd()

    def _refuse(port):  # noqa: ARG001
        raise ConnectionRefusedError(10061, "actively refused")

    sent = []
    monkeypatch.setattr(f, "_connect_backend", _refuse)
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: sent.append((status, message)))

    f._pipe_to_backend_inner(object(), b"", b"", 19998, "internal API")

    assert sent and sent[0][0] == 502


def test_pipe_with_ticket_recovers_and_retries_same_request(monkeypatch):
    mgr = _FakeManager()
    f = _fwd(mgr)
    calls = []

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label,
              engine_generation=0):
        calls.append((target_port, label))
        if target_port == 9001:
            raise _BackendUnreachable(target_port, ConnectionRefusedError(10061, "refused"))

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend",
                        lambda model_id, port, expected_generation=0: (9002, 7))
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: errors.append((status, message)))

    f._pipe_with_ticket(object(), b"", b"", 9001, "dead.gguf", _FakeQueue())

    assert [port for port, _ in calls] == [9001, 9002]
    assert all(label == _LLAMA_SERVER_LABEL for _, label in calls)
    assert errors == []
    # Retry must use the recovered port's queue.
    assert mgr.queue_registry.calls == [(9002, "dead.gguf")]


def test_pipe_with_ticket_returns_503_when_recovery_fails(monkeypatch):
    f = _fwd()
    calls = []

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label,
              engine_generation=0):
        calls.append(target_port)
        raise _BackendUnreachable(target_port, ConnectionRefusedError(10061, "refused"))

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend",
                        lambda model_id, port, expected_generation=0: None)
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: errors.append((status, message)))

    f._pipe_with_ticket(object(), b"", b"", 9001, "dead.gguf", _FakeQueue())

    assert calls == [9001]
    assert errors and errors[0][0] == 503
    assert "could not be restarted" in errors[0][1]


def test_pipe_with_ticket_joins_recovery_after_queue_stop(monkeypatch):
    """Requests woken by the dead port's queue drop must join, not get a spurious 503."""
    mgr = _RecoveringManager()
    f = _fwd(mgr)
    calls = []

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label,
              engine_generation=0):
        calls.append(target_port)

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend",
                        lambda model_id, port, expected_generation=0: (9002, 7))
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: errors.append((status, message)))

    f._pipe_with_ticket(object(), b"", b"", 9001, "dead.gguf", _StoppedQueue())

    assert calls == [9002]
    assert errors == []
    assert mgr.queue_registry.calls == [(9002, "dead.gguf")]


def test_pipe_with_ticket_only_retries_once(monkeypatch):
    f = _fwd()
    calls = []

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label,
              engine_generation=0):
        calls.append(target_port)
        raise _BackendUnreachable(target_port, ConnectionRefusedError(10061, "refused"))

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend",
                        lambda model_id, port, expected_generation=0: (9002, 7))
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: errors.append((status, message)))

    f._pipe_with_ticket(object(), b"", b"", 9001, "dead.gguf", _FakeQueue())

    assert calls == [9001, 9002]
    assert errors and errors[0][0] == 503
    assert "reload attempt" in errors[0][1]


def test_resolve_target_carries_engine_generation(monkeypatch):
    mgr = _FakeManager()
    lm = SimpleNamespace(port=9001, generation=42, ready=True, touch=lambda: None)
    mgr.loaded["m.gguf"] = lm
    f = _fwd(mgr)
    monkeypatch.setattr(f, "_resolve_request_model", lambda model_id: "m.gguf")

    assert f._resolve_target("alias") == (9001, 42)
    # The old port-only helper must keep working and agree with the new target.
    assert f._resolve_target_port("alias") == 9001


def test_pipe_with_ticket_carries_generation_through_recovery(monkeypatch):
    """BUG_0003/BUG_0004: recovery and retry stay bound to one engine identity."""
    mgr = _FakeManager()
    f = _fwd(mgr)
    calls = []
    recoveries = []

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label,
              engine_generation=0):
        calls.append((target_port, engine_generation))
        if target_port == 9001:
            raise _BackendUnreachable(
                target_port, ConnectionRefusedError(10061, "refused"))

    def _recover(model_id, port, expected_generation=0):
        recoveries.append((model_id, port, expected_generation))
        return 9002, 77

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend", _recover)
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: errors.append((status, message)))

    f._pipe_with_ticket(object(), b"", b"", 9001, "dead.gguf", _FakeQueue(),
                        expected_generation=76)

    assert calls == [(9001, 76), (9002, 77)]
    assert recoveries == [("dead.gguf", 9001, 76)]
    assert errors == []


class _CancelSock:
    def __init__(self):
        self.shutdown_calls = 0
        self.closed = False

    def shutdown(self, how):  # noqa: ARG002
        self.shutdown_calls += 1

    def close(self):
        self.closed = True


def test_cancel_active_targets_one_engine_generation():
    """BUG_0004: cancelling E1 must not close E2's sockets on the same port."""
    f = _fwd()
    old_client, old_target = _CancelSock(), _CancelSock()
    new_client, new_target = _CancelSock(), _CancelSock()
    f._register_active(9001, 1, old_client, old_target)
    f._register_active(9001, 2, new_client, new_target)

    assert f.cancel_active(9001, 1) == 1
    assert old_client.shutdown_calls == 1 and old_client.closed
    assert old_target.shutdown_calls == 1 and old_target.closed
    assert new_client.shutdown_calls == 0 and not new_client.closed
    assert new_target.shutdown_calls == 0 and not new_target.closed

    assert f.cancel_active(9001, 2) == 1
    assert new_client.shutdown_calls == 1 and new_client.closed
    assert new_target.shutdown_calls == 1 and new_target.closed


@pytest.mark.asyncio
async def test_recovery_tasks_are_scoped_by_generation(isolated_manager, monkeypatch):
    """A delayed recovery for generation E1 must not attach to E2's task."""
    mgr = isolated_manager
    release = asyncio.Event()
    started = []

    async def _blocked(model_id, dead_port, expected_generation=0):
        started.append((model_id, dead_port, expected_generation))
        await release.wait()
        return None

    monkeypatch.setattr(mgr, "_recover_dead_model_impl", _blocked)

    first = asyncio.create_task(mgr.recover_dead_model("m.gguf", 19570, 1))
    second = asyncio.create_task(mgr.recover_dead_model("m.gguf", 19570, 2))
    for _ in range(10):
        await asyncio.sleep(0)

    assert sorted(started) == [("m.gguf", 19570, 1), ("m.gguf", 19570, 2)]
    assert mgr.recovery_in_progress(19570, 1) is True
    assert mgr.recovery_in_progress(19570, 2) is True
    assert mgr.recovery_in_progress(19570, 3) is False

    release.set()
    await asyncio.gather(first, second)


def test_pipe_with_ticket_rejects_when_recovery_waiters_saturated(monkeypatch):
    """BUG_0005: one dead model must not monopolize every forwarder worker."""
    import tcp_forwarder as tf

    monkeypatch.setattr(tf, "_MAX_RECOVERY_WAITERS", 1)
    f = _fwd()
    assert f._reserve_recovery_waiter(9001, 11) is True

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label,
              engine_generation=0):  # noqa: ARG001
        raise _BackendUnreachable(
            target_port, ConnectionRefusedError(10061, "refused"))

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: errors.append((status, message)))

    try:
        f._pipe_with_ticket(object(), b"", b"", 9001, "dead.gguf", _FakeQueue(),
                            expected_generation=11)
    finally:
        f._release_recovery_waiter(9001, 11)

    assert errors and errors[0][0] == 503
    assert "saturated" in errors[0][1]
    assert f._recovery_waiters == {}


def test_recover_dead_backend_runs_on_manager_main_loop():
    """The async recovery must be scheduled on the captured main loop, never a new one."""
    loop_holder = {}

    class _Mgr:
        def __init__(self):
            self.loaded = {}
            self.models = {}
            self.calls = []

        async def resolve_request_model(self, model_id):  # noqa: ARG002
            return None

        async def recover_dead_model(self, model_id, dead_port,
                                     expected_generation=0):
            self.calls.append((model_id, dead_port, expected_generation,
                               threading.current_thread().name))
            return SimpleNamespace(model_id=model_id, port=9002, ready=True,
                                   generation=7)

    mgr = _Mgr()
    f = _fwd(mgr)

    def _run_loop():
        loop = asyncio.new_event_loop()
        loop_holder["loop"] = loop
        mgr._main_loop = loop
        try:
            loop.run_forever()
        finally:
            loop.close()

    t = threading.Thread(target=_run_loop, daemon=True, name="manager-loop")
    t.start()
    while "loop" not in loop_holder:
        time.sleep(0.01)

    try:
        target = f._recover_dead_backend("dead.gguf", 9001, 5)
    finally:
        loop_holder["loop"].call_soon_threadsafe(loop_holder["loop"].stop)
        t.join(timeout=5)

    assert target == (9002, 7)
    assert len(mgr.calls) == 1
    assert mgr.calls[0][0:3] == ("dead.gguf", 9001, 5)
    assert mgr.calls[0][3] != threading.current_thread().name
