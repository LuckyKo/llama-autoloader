# BUG_0002 — TCP forwarder has no liveness timeout for an established but hung backend

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `1216df4`
**Found in:** Post-fix audit of BUG_0001 (`c31e08e`)
**Severity:** high
**Component:** `tcp_forwarder.py` (`_bidirectional_pipe`, `_pipe_to_backend_inner`, `_check_engine_liveness`, `_liveness_watchdog`)

## Symptom

BUG_0001 recovery only fires when the backend **refuses a connection**. If the engine process is
still alive and accepting connections but is wedged (GPU TDR, a stuck CUDA context, a deadlocked
generation loop, etc.), the request can hang indefinitely:

- `_bidirectional_pipe` loops on `select(..., 5)`; the 5-second timeout is only used to notice
  forwarder shutdown, not backend inactivity.
- There is no established-connection deadline, no `proc.poll()` check while piping, and no
  `/health` probe for an already-loaded model.
- The client may eventually time out, but until it does the forwarder keeps a worker thread,
  connection slot, and `active_requests` count occupied.

This is the same operational failure class as BUG_0001 but it bypasses the new detection path.

## Evidence

`tcp_forwarder.py:1011`:

```python
while self._running and open_socks:
    readable, _writable, exceptional = select.select(open_socks, [], open_socks, 5)
    ...
```

The loop only exits on EOF, reset, error, exceptional condition, or shutdown. A backend that
accepts the connection and then sends nothing never triggers any of those.

The connect-refusal recovery added for BUG_0001 starts only in the `_connect_backend` exception
handler (`tcp_forwarder.py:899-907`); it never observes a pipe that connected successfully.

## Root cause

The forwarder treats “TCP connect succeeded” as proof of engine health forever. There is no
established-connection liveness policy and no independent watchdog for a resident engine.

## Fix direction

Add a bounded liveness path that does not falsely kill legitimate long prompt evaluation:

1. A per-engine watchdog should probe `/health` (with a short connect/read timeout) while the
   engine is marked ready. A separate probe is preferable to inferring death from data silence,
   because prompt processing can legitimately emit no response bytes for minutes.
2. Require consecutive probe failures (and/or a minimum silence period) before declaring the
   engine dead, to avoid reacting to a transient scheduling stall.
3. On a confirmed hang: close the pipe, mark the instance unready, evict/reload through the
   BUG_0001 recovery path, and return 504/503 to the current request if no response has started.
4. Keep at most one probe per engine (no health-check stampede).

Do not implement this as a simple “no bytes for N seconds” timeout without accounting for long
prompt eval; that would kill healthy cold requests.

## Fix (implemented)

- `RawTCPForwarder` now runs a daemon liveness watchdog while accepting connections. It probes
  every resident `ready` engine's `/health` with a short connect/read timeout.
- Each pipe also checks liveness after backend silence (`AUTOLOADER_LIVENESS_SILENCE`, default
  10 s), so an in-flight hung request is bounded even if no watchdog tick has fired yet. Backend
  data resets the silence timer; client upload traffic does not, because it says nothing about
  generation health.
- A process that has exited is dead immediately. Otherwise **two consecutive** failed `/health`
  probes (`AUTOLOADER_LIVENESS_FAILURES=2`) are required, so a transient stall is not fatal.
- Probes are shared per `(port, generation)` and guarded by a per-engine lock plus a probe
  interval (`AUTOLOADER_LIVENESS_PROBE_INTERVAL=10`), preventing a health-check stampede.
- On confirmation the engine is marked `ready=False`, the existing BUG_0001/BUG_0003 recovery is
  scheduled (deduplicated, outside the worker), and active pipes are aborted. Connections that
  have not forwarded response bytes receive `504 Gateway Timeout`; a streaming response is closed
  rather than corrupted with a second HTTP response.
- Healthy cold prompt eval is preserved: health answers while no tokens are emitted, so silence
  alone never evicts an engine.

Tests: `tests/test_backend_liveness_backoff.py`
- silent connected backend returns a bounded 504;
- healthy silent backend remains alive;
- consecutive failure threshold and unready marking;
- a successful probe resets failures;
- exited process detected without a health probe;
- one concurrent probe per engine interval;
- watchdog sends a 504 only before a response has started.

## Verification

- Unit test with a fake backend that accepts and then never responds: the request finishes with a
  bounded 504 instead of hanging.
- Unit test that a healthy backend which is silent during a long prompt is not evicted.
- Full suite: 389 passed (the only intermittent failure observed was the pre-existing Windows
  `WinError 10055` socketpair flake, which passed when rerun).

## Related

- `BUG_0001_tcp_forwarder_no_dead_backend_recovery.md` — connect-refused recovery (fixed)
- `[[llama-server-silent-exit-10061]]` — silent-exit incident
