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
                  ready: bool = True, pid: int = 0) -> LoadedModel:
    cfg = ModelConfig.default_for(Path(model_id))
    lm = LoadedModel(
        model_id=model_id,
        gguf_path=Path(model_id),
        config=cfg,
        port=port,
        process=None,
        pid=pid,
    )
    lm.ready = ready
    mgr.models[model_id] = cfg
    mgr.loaded[model_id] = lm
    return lm


def _new_resident(mgr, model_id: str, port: int) -> LoadedModel:
    lm = LoadedModel(
        model_id=model_id,
        gguf_path=Path(model_id),
        config=mgr.models[model_id],
        port=port,
        process=None,
        pid=0,
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


class _RecoveringManager(_FakeManager):
    def recovery_in_progress(self, dead_port: int) -> bool:
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

    monkeypatch.setattr(mgr, "load_model", _fake_load)

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

    monkeypatch.setattr(mgr, "load_model", _slow_load)

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

    async def _failed_unload(model_id):
        return UnloadResult(model_id, False, 424242, "still running")

    async def _must_not_load(model_id):  # pragma: no cover - asserted
        raise AssertionError(f"must not reload {model_id} after a failed kill")

    monkeypatch.setattr(mgr, "_unload_unlocked", _failed_unload)
    monkeypatch.setattr(mgr, "load_model", _must_not_load)

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

    monkeypatch.setattr(mgr, "load_model", _must_not_load)

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

    monkeypatch.setattr(mgr, "load_model", _must_not_load)

    result = await mgr.recover_dead_model("unknown-provider-alias", 19555)

    assert result is fallback


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

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label):
        calls.append((target_port, label))
        if target_port == 9001:
            raise _BackendUnreachable(target_port, ConnectionRefusedError(10061, "refused"))

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend", lambda model_id, port: 9002)
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

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label):
        calls.append(target_port)
        raise _BackendUnreachable(target_port, ConnectionRefusedError(10061, "refused"))

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend", lambda model_id, port: None)
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

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label):
        calls.append(target_port)

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend", lambda model_id, port: 9002)
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: errors.append((status, message)))

    f._pipe_with_ticket(object(), b"", b"", 9001, "dead.gguf", _StoppedQueue())

    assert calls == [9002]
    assert errors == []
    assert mgr.queue_registry.calls == [(9002, "dead.gguf")]


def test_pipe_with_ticket_only_retries_once(monkeypatch):
    f = _fwd()
    calls = []

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label):
        calls.append(target_port)
        raise _BackendUnreachable(target_port, ConnectionRefusedError(10061, "refused"))

    errors = []
    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend", lambda model_id, port: 9002)
    monkeypatch.setattr(f, "_send_error_response",
                        lambda sock, status, message: errors.append((status, message)))

    f._pipe_with_ticket(object(), b"", b"", 9001, "dead.gguf", _FakeQueue())

    assert calls == [9001, 9002]
    assert errors and errors[0][0] == 503
    assert "reload attempt" in errors[0][1]


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

        async def recover_dead_model(self, model_id, dead_port):
            self.calls.append((model_id, dead_port, threading.current_thread().name))
            return SimpleNamespace(model_id=model_id, port=9002, ready=True)

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
        port = f._recover_dead_backend("dead.gguf", 9001)
    finally:
        loop_holder["loop"].call_soon_threadsafe(loop_holder["loop"].stop)
        t.join(timeout=5)

    assert port == 9002
    assert len(mgr.calls) == 1
    assert mgr.calls[0][0:2] == ("dead.gguf", 9001)
    assert mgr.calls[0][2] != threading.current_thread().name
