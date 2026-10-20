"""Regression tests for BUG_0002, BUG_0006, and BUG_0007.

BUG_0002: an established backend connection that stops answering must be detected by a
bounded /health probe, not by a raw data-silence timeout (long prompt eval is healthy).
BUG_0006: post-connect send failures and ambiguous connect failures need a stage-aware
dead-backend policy; a partially delivered request must never be retried.
BUG_0007: a failed dead-backend reload must arm a per-model cooldown instead of spawning
a doomed load on every retry.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
from types import SimpleNamespace

import pytest

import tcp_forwarder as tf
from server import RecoveryCooldownError, RecoveryFailure
from tcp_forwarder import _BackendUnreachable, _LLAMA_SERVER_LABEL

from tests.test_dead_backend_recovery import (
    _FakeManager,
    _FakeQueue,
    _add_resident,
    _fwd,
    _new_resident,
)


# --------------------------------------------------------------------------- helpers
def _loopback_pair():
    """Return two connected blocking loopback sockets (never socketpair on Windows)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname(), timeout=2)
    server, _addr = listener.accept()
    listener.close()
    client.settimeout(3)
    server.settimeout(3)
    return client, server


def _silent_pipe(f, *, alive: bool, calls: list | None = None):
    """Run _bidirectional_pipe against a backend that accepts then stays silent."""
    client, client_fwd = _loopback_pair()
    backend, backend_fwd = _loopback_pair()
    f._running = True

    def _liveness():
        if calls is not None:
            calls.append(time.monotonic())
        return alive

    result = {}

    def _run():
        try:
            f._bidirectional_pipe(
                client_fwd, backend_fwd, liveness_check=_liveness)
        except Exception as e:  # pragma: no cover - surfaced in the assertion below
            result["error"] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return client, backend, thread, result


# --------------------------------------------------------------------------- BUG_0002
def test_silent_but_connected_backend_returns_bounded_504(monkeypatch):
    monkeypatch.setattr(tf, "_PIPE_POLL_INTERVAL", 0.05)
    monkeypatch.setattr(tf, "_LIVENESS_SILENCE_BEFORE_PROBE", 0.0)

    f = _fwd()
    client, backend, thread, result = _silent_pipe(f, alive=False)
    try:
        thread.join(timeout=3)
        assert not thread.is_alive(), "pipe did not finish after a failed liveness probe"
        assert "error" not in result
        response = client.recv(65536)
        assert b"504 Gateway Timeout" in response
        assert b"health probe failed" in response
    finally:
        client.close()
        backend.close()


def test_healthy_but_silent_backend_is_not_evicted(monkeypatch):
    """Long prompt eval: no response bytes for a while is not death."""
    monkeypatch.setattr(tf, "_PIPE_POLL_INTERVAL", 0.05)
    monkeypatch.setattr(tf, "_LIVENESS_SILENCE_BEFORE_PROBE", 0.0)

    calls = []
    f = _fwd()
    client, backend, thread, result = _silent_pipe(f, alive=True, calls=calls)
    try:
        time.sleep(0.2)
        assert thread.is_alive(), "healthy silent backend was killed"
        assert calls, "liveness probe was never consulted"
        f._running = False  # simulate forwarder shutdown, not an eviction
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert "error" not in result
        data = client.recv(65536)
        assert b"504" not in data
    finally:
        client.close()
        backend.close()


def test_liveness_requires_consecutive_failures_and_marks_engine_unready(monkeypatch):
    mgr = _FakeManager()
    mgr.models = {}
    lm = _add_resident(mgr, model_id="m.gguf", port=19555, generation=7)
    mgr.loaded["m.gguf"] = lm
    f = _fwd(mgr)
    scheduled = []

    monkeypatch.setattr(tf, "_LIVENESS_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(tf, "_LIVENESS_PROBE_INTERVAL", 0.0)
    monkeypatch.setattr(f, "_probe_engine_health", lambda port: False)
    monkeypatch.setattr(
        f, "_schedule_recovery",
        lambda model_id, port, generation: scheduled.append(
            (model_id, port, generation)))

    assert f._check_engine_liveness(lm.port, lm.generation) is True
    assert lm.ready is True, "one transient probe failure must not evict"
    assert f._check_engine_liveness(lm.port, lm.generation) is False
    assert lm.ready is False
    assert scheduled == [("m.gguf", 19555, 7)]
    # The generation is retired here, so its liveness bookkeeping must not survive:
    # these dicts are keyed by a monotonically increasing generation, so anything
    # left behind leaks one entry (including a Lock) per reload cycle.
    assert not f._liveness_probe_locks
    assert not f._liveness_last_probe
    assert not f._liveness_failures


def test_watchdog_retires_liveness_state_for_unloaded_generation(monkeypatch):
    mgr = _FakeManager()
    mgr.models = {}
    gone = _add_resident(mgr, model_id="gone.gguf", port=19560, generation=12)
    kept = _add_resident(mgr, model_id="kept.gguf", port=19561, generation=13)
    mgr.loaded["gone.gguf"] = gone
    mgr.loaded["kept.gguf"] = kept
    f = _fwd(mgr)

    monkeypatch.setattr(tf, "_LIVENESS_PROBE_INTERVAL", 0.0)
    monkeypatch.setattr(f, "_probe_engine_health", lambda port: False)
    for lm in (gone, kept):
        assert f._check_engine_liveness(lm.port, lm.generation) is True
    assert (gone.port, gone.generation) in f._liveness_probe_locks

    # A plain unload / LRU eviction drops the registry entry without ever
    # confirming a hang, so the registry sweep is what bounds these dicts.
    del mgr.loaded["gone.gguf"]
    f._retire_stale_liveness_state(list(mgr.loaded.items()))

    assert (gone.port, gone.generation) not in f._liveness_probe_locks
    assert (gone.port, gone.generation) not in f._liveness_last_probe
    assert (gone.port, gone.generation) not in f._liveness_failures
    # A still-registered engine must never lose its accumulated probe state.
    assert (kept.port, kept.generation) in f._liveness_probe_locks


def test_successful_liveness_probe_resets_failures(monkeypatch):
    mgr = _FakeManager()
    mgr.models = {}
    lm = _add_resident(mgr, model_id="m.gguf", port=19556, generation=8)
    mgr.loaded["m.gguf"] = lm
    f = _fwd(mgr)
    results = iter([False, True])

    monkeypatch.setattr(tf, "_LIVENESS_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(tf, "_LIVENESS_PROBE_INTERVAL", 0.0)
    monkeypatch.setattr(f, "_probe_engine_health", lambda port: next(results))
    monkeypatch.setattr(f, "_schedule_recovery", lambda *args: None)

    assert f._check_engine_liveness(lm.port, lm.generation) is True  # failure 1
    assert f._check_engine_liveness(lm.port, lm.generation) is True  # success resets
    assert lm.ready is True
    assert f._liveness_failures.get((lm.port, lm.generation)) is None


def test_exited_engine_process_is_detected_without_health_probe(monkeypatch):
    mgr = _FakeManager()
    mgr.models = {}
    lm = _add_resident(mgr, model_id="m.gguf", port=19557, generation=9)
    lm.process = SimpleNamespace(poll=lambda: 3)
    mgr.loaded["m.gguf"] = lm
    f = _fwd(mgr)
    scheduled = []
    monkeypatch.setattr(
        f, "_probe_engine_health",
        lambda port: pytest.fail("must not probe a process that already exited"))
    monkeypatch.setattr(
        f, "_schedule_recovery",
        lambda model_id, port, generation: scheduled.append(model_id))

    assert f._check_engine_liveness(lm.port, lm.generation) is False
    assert lm.ready is False
    assert scheduled == ["m.gguf"]


class _CaptureSock:
    def __init__(self):
        self.sent = []
        self.closed = False

    def sendall(self, data):
        self.sent.append(data)

    def shutdown(self, how):  # noqa: ARG002
        pass

    def close(self):
        self.closed = True


def test_watchdog_cancel_sends_504_only_before_response_started():
    f = _fwd()
    waiting_client, waiting_target = _CaptureSock(), _CaptureSock()
    streaming_client, streaming_target = _CaptureSock(), _CaptureSock()
    f._register_active(9001, 7, waiting_client, waiting_target,
                       response_started=False)
    f._register_active(9001, 7, streaming_client, streaming_target,
                       response_started=True)

    assert f._cancel_unresponsive_active(9001, 7) == 2
    assert waiting_client.sent and b"504 Gateway Timeout" in waiting_client.sent[0]
    assert streaming_client.sent == [], "injected a second response into a live stream"
    for sock in (waiting_client, waiting_target, streaming_client, streaming_target):
        assert sock.closed


def test_one_probe_per_engine_interval_under_concurrency(monkeypatch):
    mgr = _FakeManager()
    mgr.models = {}
    lm = _add_resident(mgr, model_id="m.gguf", port=19558, generation=10)
    mgr.loaded["m.gguf"] = lm
    f = _fwd(mgr)
    probes = []

    def _slow_probe(port):  # noqa: ARG001
        probes.append(port)
        time.sleep(0.1)
        return True

    monkeypatch.setattr(tf, "_LIVENESS_PROBE_INTERVAL", 60.0)
    monkeypatch.setattr(f, "_probe_engine_health", _slow_probe)

    threads = [
        threading.Thread(target=f._check_engine_liveness,
                         args=(lm.port, lm.generation))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert probes == [lm.port], "health probes stampeded for one engine"


# --------------------------------------------------------------------------- BUG_0006
class _SendFailSocket:
    def __init__(self):
        self.closed = False

    def sendall(self, _data):
        raise BrokenPipeError(10054, "connection reset during send")

    def close(self):
        self.closed = True


def test_send_failure_marks_unready_and_never_retries(monkeypatch):
    mgr = _FakeManager()
    mgr.models = {}
    lm = _add_resident(mgr, model_id="m.gguf", port=19560, generation=20)
    mgr.loaded["m.gguf"] = lm
    f = _fwd(mgr)
    sent = []
    scheduled = []
    target = _SendFailSocket()

    monkeypatch.setattr(f, "_connect_backend", lambda port: target)
    monkeypatch.setattr(
        f, "_send_error_response",
        lambda sock, status, message: sent.append((status, message)))
    monkeypatch.setattr(
        f, "_schedule_recovery",
        lambda model_id, port, generation: scheduled.append(
            (model_id, port, generation)))

    f._pipe_to_backend_inner(
        object(), b"POST /v1/chat/completions HTTP/1.1\r\n\r\n", b"{}",
        lm.port, _LLAMA_SERVER_LABEL, engine_generation=lm.generation)

    assert sent and sent[0][0] == 502
    assert target.closed
    assert lm.ready is False
    assert scheduled == [("m.gguf", lm.port, lm.generation)]


def test_send_failure_on_internal_api_does_not_touch_models(monkeypatch):
    mgr = _FakeManager()
    mgr.models = {}
    lm = _add_resident(mgr, model_id="m.gguf", port=19561, generation=21)
    mgr.loaded["m.gguf"] = lm
    f = _fwd(mgr)
    sent = []
    scheduled = []

    monkeypatch.setattr(f, "_connect_backend", lambda port: _SendFailSocket())
    monkeypatch.setattr(
        f, "_send_error_response",
        lambda sock, status, message: sent.append((status, message)))
    monkeypatch.setattr(
        f, "_schedule_recovery",
        lambda *args: scheduled.append(args))

    f._pipe_to_backend_inner(
        object(), b"POST /v1/x HTTP/1.1\r\n\r\n", b"", 19998, "internal API")

    assert sent and sent[0][0] == 502
    assert lm.ready is True
    assert scheduled == []


def _connect_raises(exc):
    def _connect(port):  # noqa: ARG001
        raise exc
    return _connect


def test_connect_timeout_confirmed_dead_is_safe_to_recover(monkeypatch):
    mgr = _FakeManager()
    mgr.models = {}
    lm = _add_resident(mgr, model_id="m.gguf", port=19562, generation=22)
    mgr.loaded["m.gguf"] = lm
    f = _fwd(mgr)

    monkeypatch.setattr(f, "_connect_backend", _connect_raises(TimeoutError("timed out")))
    monkeypatch.setattr(f, "_probe_engine_health", lambda port: False)

    with pytest.raises(_BackendUnreachable):
        f._pipe_to_backend_inner(
            object(), b"", b"", lm.port, _LLAMA_SERVER_LABEL,
            engine_generation=lm.generation)


def test_connect_timeout_against_live_engine_keeps_normal_502(monkeypatch):
    mgr = _FakeManager()
    mgr.models = {}
    lm = _add_resident(mgr, model_id="m.gguf", port=19563, generation=23)
    mgr.loaded["m.gguf"] = lm
    f = _fwd(mgr)
    sent = []

    monkeypatch.setattr(f, "_connect_backend", _connect_raises(TimeoutError("timed out")))
    monkeypatch.setattr(f, "_probe_engine_health", lambda port: True)
    monkeypatch.setattr(
        f, "_send_error_response",
        lambda sock, status, message: sent.append((status, message)))

    f._pipe_to_backend_inner(
        object(), b"", b"", lm.port, _LLAMA_SERVER_LABEL,
        engine_generation=lm.generation)

    assert sent and sent[0][0] == 502
    assert lm.ready is True, "a slow but alive engine must not be evicted"


def test_connection_refused_does_not_need_health_probe(monkeypatch):
    f = _fwd()
    monkeypatch.setattr(
        f, "_connect_backend",
        _connect_raises(ConnectionRefusedError(10061, "refused")))
    monkeypatch.setattr(
        f, "_probe_engine_health",
        lambda port: pytest.fail("refusal is already conclusive"))

    with pytest.raises(_BackendUnreachable):
        f._pipe_to_backend_inner(
            object(), b"", b"", 9001, _LLAMA_SERVER_LABEL)


# --------------------------------------------------------------------------- BUG_0007
@pytest.mark.asyncio
async def test_failed_recovery_arms_exponential_cooldown(isolated_manager, monkeypatch):
    mgr = isolated_manager
    old = _add_resident(mgr, generation=30)
    loads = []

    async def _fail(model_id):
        loads.append(model_id)
        raise RuntimeError("bad engine config")

    monkeypatch.setattr(mgr, "_load_model_serialized", _fail)

    assert await mgr.recover_dead_model(
        "dead.gguf", old.port, old.generation) is None
    assert loads == ["dead.gguf"]
    failure = mgr._recovery_failures["dead.gguf"]
    assert failure.attempts == 1
    assert mgr.recovery_retry_after("dead.gguf", old.port, old.generation) > 0

    # During the cooldown no new load attempt is made.
    assert await mgr.recover_dead_model(
        "dead.gguf", old.port, old.generation) is None
    assert loads == ["dead.gguf"]

    # A forced/manual recovery bypasses the cooldown and grows the backoff.
    assert await mgr.recover_dead_model(
        "dead.gguf", old.port, old.generation, force=True) is None
    assert loads == ["dead.gguf", "dead.gguf"]
    assert mgr._recovery_failures["dead.gguf"].attempts == 2


@pytest.mark.asyncio
async def test_successful_recovery_clears_cooldown(isolated_manager, monkeypatch):
    mgr = isolated_manager
    old = _add_resident(mgr, generation=31)

    async def _fail(model_id):
        raise RuntimeError("bad engine config")

    monkeypatch.setattr(mgr, "_load_model_serialized", _fail)
    assert await mgr.recover_dead_model(
        "dead.gguf", old.port, old.generation) is None
    assert "dead.gguf" in mgr._recovery_failures

    replacement = _new_resident(mgr, "dead.gguf", old.port, generation=32)

    async def _succeed(model_id):
        mgr.loaded[model_id] = replacement
        return replacement

    monkeypatch.setattr(mgr, "_load_model_serialized", _succeed)
    result = await mgr.recover_dead_model(
        "dead.gguf", old.port, old.generation, force=True)

    assert result is replacement
    assert "dead.gguf" not in mgr._recovery_failures


@pytest.mark.asyncio
async def test_jit_load_respects_cooldown_but_manual_load_bypasses(isolated_manager,
                                                                  monkeypatch):
    mgr = isolated_manager
    _add_resident(mgr, generation=33)
    now = time.monotonic()
    mgr._recovery_failures["dead.gguf"] = RecoveryFailure(
        model_id="dead.gguf", dead_port=19555, generation=33, attempts=1,
        reason="reload failed", failed_at=time.time(), retry_at=now + 30)

    with pytest.raises(RecoveryCooldownError) as excinfo:
        await mgr.load_model("dead.gguf")
    assert excinfo.value.retry_after > 0

    replacement = _new_resident(mgr, "dead.gguf", 19555, generation=34)

    async def _succeed(model_id, force=False):  # noqa: ARG001
        mgr.loaded[model_id] = replacement
        return replacement

    monkeypatch.setattr(mgr, "_load_model_serialized", _succeed)
    result = await mgr.load_model(
        "dead.gguf", respect_recovery_cooldown=False)

    assert result is replacement
    assert "dead.gguf" not in mgr._recovery_failures


@pytest.mark.asyncio
async def test_failed_jit_reload_extends_existing_cooldown(isolated_manager,
                                                          monkeypatch):
    mgr = isolated_manager
    _add_resident(mgr, ready=False, generation=35)
    now = time.monotonic()
    mgr._recovery_failures["dead.gguf"] = RecoveryFailure(
        model_id="dead.gguf", dead_port=19555, generation=35, attempts=1,
        reason="reload failed", failed_at=time.time(), retry_at=now - 1)

    async def _fail(model_id, force=False):  # noqa: ARG001
        raise RuntimeError("still broken")

    monkeypatch.setattr(mgr, "_load_model_serialized", _fail)

    with pytest.raises(RuntimeError, match="still broken"):
        await mgr.load_model("dead.gguf")

    failure = mgr._recovery_failures["dead.gguf"]
    assert failure.attempts == 2, "expired recovery failure did not extend backoff"
    assert failure.retry_at > time.monotonic()


@pytest.mark.asyncio
async def test_post_cooldown_burst_makes_exactly_one_load_attempt(
        isolated_manager, monkeypatch):
    mgr = isolated_manager
    _add_resident(mgr, ready=False, generation=37)
    now = time.monotonic()
    mgr._recovery_failures["dead.gguf"] = RecoveryFailure(
        model_id="dead.gguf", dead_port=19555, generation=37, attempts=1,
        reason="reload failed", failed_at=time.time(), retry_at=now - 1)

    attempts = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fail(model_id, force=False):  # noqa: ARG001
        attempts.append(model_id)
        started.set()
        await release.wait()
        raise RuntimeError("still broken")

    monkeypatch.setattr(mgr, "_load_model_serialized", _fail)

    tasks = [
        asyncio.create_task(mgr.load_model("dead.gguf"))
        for _ in range(5)
    ]
    await asyncio.wait_for(started.wait(), timeout=2)
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert attempts == ["dead.gguf"], "burst raced into multiple load attempts"
    assert mgr._recovery_failures["dead.gguf"].attempts == 2, (
        "cooldown-suppressed waiters were counted as failed load attempts")
    assert all(isinstance(r, (RuntimeError, RecoveryCooldownError))
               for r in results)


@pytest.mark.asyncio
async def test_recovery_failure_is_observable_in_status(isolated_manager,
                                                       monkeypatch):
    mgr = isolated_manager
    old = _add_resident(mgr, generation=36)

    async def _fail(model_id):
        raise RuntimeError("missing binary")

    monkeypatch.setattr(mgr, "_load_model_serialized", _fail)
    await mgr.recover_dead_model("dead.gguf", old.port, old.generation)

    snapshot = mgr.recovery_failures_snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["model"] == "dead.gguf"
    assert snapshot[0]["attempts"] == 1
    assert "missing binary" in snapshot[0]["reason"]
    assert snapshot[0]["retry_after_s"] > 0


class _CooldownManager(_FakeManager):
    def recovery_retry_after(self, model_id, dead_port=None,
                             expected_generation=0):  # noqa: ARG002
        return 7.5


def test_forwarder_recovery_503_carries_retry_after(monkeypatch):
    f = _fwd(_CooldownManager())
    sent = []

    def _pipe(client_sock, raw_headers, body_prefix, target_port, label,
              engine_generation=0):  # noqa: ARG001
        raise _BackendUnreachable(
            target_port, ConnectionRefusedError(10061, "refused"))

    monkeypatch.setattr(f, "_pipe_to_backend", _pipe)
    monkeypatch.setattr(f, "_recover_dead_backend",
                        lambda model_id, port, expected_generation=0: None)
    monkeypatch.setattr(
        f, "_send_error_response",
        lambda sock, status, message, headers=None:
            sent.append((status, message, headers)))

    f._pipe_with_ticket(object(), b"", b"", 9001, "dead.gguf", _FakeQueue(),
                        expected_generation=40)

    assert sent and sent[0][0] == 503
    assert sent[0][2] == {"Retry-After": "8"}
    assert "retry after 8s" in sent[0][1]
