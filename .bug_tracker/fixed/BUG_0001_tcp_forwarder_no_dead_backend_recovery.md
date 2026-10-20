# BUG_0001 — TCP forwarder has no dead-backend detection or recovery in the LLM fast path

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `c31e08e`
**Found in:** Production log `N:\work\WD\AgentCascade\todo.md` lines 184-194 (2026-10-20 09:09:57–09:10:12)
**Severity:** high
**Component:** `tcp_forwarder.py:761-769` (`_pipe_to_backend_inner`), `tcp_forwarder.py:639-667` (`_resolve_target_port`)

## Symptom

When a llama-server backend dies silently (GPU TDR, OOM, silent process exit), the TCP forwarder
keeps returning **502 "llama-server unreachable"** to every subsequent request for that model.
It never evicts the dead model from the loaded registry or triggers a JIT reload. The model stays
"loaded" (in `model_manager.loaded` with `ready=True`) indefinitely until an external mechanism
(state restore, idle timeout, manual kill) evicts it.

Log evidence (2026-10-20):
```
09:09:57,750 [ERROR] autoloader: Failed to connect to llama-server on port 9021: [WinError 10061] No connection could be made because the target machine actively refused it
09:10:00,285 [ERROR] autoloader: Failed to connect to llama-server on port 9021: [WinError 10061] No connection could be made because the target machine actively refused it
09:10:03,274 [ERROR] autoloader: Failed to connect to llama-server on port 9021: [WinError 10061] No connection could be made because the target machine actively refused it
09:10:06,521 [ERROR] autoloader: Failed to connect to llama-server on port 9021: [WinError 10061] No connection could be made because the target machine actively refused it
09:10:09,062 [ERROR] autoloader: Failed to connect to llama-server on port 9021: [WinError 10061] No connection could be made because the target machine actively refused it
09:10:12,118 [ERROR] autoloader: Failed to connect to llama-server on port 9021: [WinError 10061] No connection could be made because the target machine actively refused it
```
~15 seconds of repeated 502s before recovery (via a subsequent state-restore-triggered reload).

## Evidence

1. **Code path** (`tcp_forwarder.py:297-306`): When `_resolve_target_port(model_id)` returns a port
   (model is in `loaded` dict with `ready=True`), the forwarder goes directly to `_pipe_with_ticket`
   → `_pipe_to_backend` → `_pipe_to_backend_inner`. The JIT load path (line 308-319) is ONLY reached
   when `_resolve_target_port` returns `None`.

2. **Failure handling** (`tcp_forwarder.py:764-769`):
   ```python
   try:
       target_sock = self._connect_backend(target_port)
   except Exception as e:
       log.error(f"Failed to connect to {label} on port {target_port}: {e}")
       self._send_error_response(client_sock, 502, f"{label} unreachable")
       return
   ```
   No eviction, no `ready=False`, no JIT reload trigger. Just log + 502 + return.

3. **No periodic health check**: `server.py` has `_wait_until_ready` (line 1868) which checks
   `proc.poll()` during initial load, but there is NO background health-check loop for already-loaded
   models. Once `ready=True`, nothing re-verifies the process is alive.

4. **Historical context**: This behavior has existed since the TCP forwarder was first introduced
   (commit `e4cbe26` — "fix(streaming): raw TCP forwarder to remove ASGI from SSE data path").
   Before that commit, all requests went through the FastAPI/uvicorn proxy which had proper
   recovery. The fast path bypassed those mechanisms.

5. **Related incident**: 2026-09-26 (see `[[llama-server-silent-exit-10061]]`) — same symptom
   (WinError 10061, ~2 min of failures) recovered via LRU eviction, not via JIT reload.

## Root cause

The TCP forwarder's "already loaded → pipe directly" path has no backend liveness verification.
It trusts `model_manager.loaded[mid].ready == True` as a guarantee that the llama-server process
is alive and accepting connections on that port. When the process dies silently (no stderr, no
crash dump — common with GPU TDR resets on Windows), the forwarder has no mechanism to:
- Detect the death (no `proc.poll()`, no `/health` probe)
- Evict the model from the loaded registry
- Trigger a JIT reload for the current or next request

The JIT load path exists (`_jit_load_and_wait`) but is only reachable when `_resolve_target_port`
returns `None` — which requires the model to NOT be in the `loaded` dict or NOT `ready`.

## Fix (implemented)

1. **Detection** — `tcp_forwarder._pipe_to_backend_inner` now classifies refusal-class connect
   errors (`ConnectionRefusedError`, `WinError 10061`, `ECONNREFUSED`) on the LLM fast path and
   raises `_BackendUnreachable`. Management/FastAPI traffic still gets the existing 502. The
   request has not been sent at this point, so retrying it is safe.

2. **Eviction + reload** — `_pipe_with_ticket` releases the FIFO ticket, schedules
   `ModelManager.recover_dead_model(model_id, dead_port)` on the main loop via
   `asyncio.run_coroutine_threadsafe`, and retries the SAME request once on the recovered port.
   On failure it returns 503 (not a permanent 502).

3. **Thundering-herd guard** — recoveries are deduplicated by dead port in
   `ModelManager._recovery_tasks`; the first caller creates the task and every other caller
   awaits the same shielded task. Requests woken by the dead port's queue being dropped join
   the active recovery instead of getting a spurious 503.

4. **Residency truth** — recovery marks the old instance `ready=False`, skips auto-save against
   the dead engine, and runs the normal verified unload. If the kill cannot be verified, the
   model stays tracked and recovery returns `None`; `_do_load_model` also refuses to spawn a
   replacement over a surviving not-ready process, so no duplicate engine is leaked.

5. **Stale-entry cleanup** — `_do_load_model` now unloads an existing `ready=False` entry before
   a JIT load (not just on `force=True`), closing the orphan-process path that could otherwise
   follow a failed recovery.

No periodic health check was added: the request-triggered path fixes the reported permanent-502
loop without introducing background model churn. A silent death with no traffic remains governed
by the existing idle reaper.

## Verification

- `tests/test_dead_backend_recovery.py` (14 tests): manager evict/reload, concurrent-caller
  dedup, failed-kill residency, already-recovered reuse, unknown-alias fallback, stale-entry
  cleanup, forwarder refusal classification, one-shot retry, 503 on recovery failure, queue-drop
  rejoin, and main-loop scheduling.
- Full suite: 356 passed; one test hit the known Windows `WinError 10055` event-loop setup flake
  and passed when rerun. No live model/binary required.

## Related

- `tests/test_dead_backend_recovery.py` — regression coverage
- `BUG_0002_forwarder_hung_backend_no_liveness_timeout.md` — post-fix audit: established hung backend
- `fixed/BUG_0003_recovery_evicts_healthy_replacement_on_reused_port.md` — FIXED in `6ba95c4`: same-port replacement identity
- `fixed/BUG_0004_active_request_tracking_port_not_engine_generation.md` — FIXED in `6ba95c4`: request accounting identity
- `fixed/BUG_0005_recovery_holds_manager_lock_across_teardown.md` — FIXED in `6ba95c4`: lock hold / worker exhaustion
- `BUG_0006_dead_backend_detection_ignores_send_failures_and_timeouts.md` — post-fix audit: non-refusal failures
- `BUG_0007_failed_backend_reload_has_no_backoff.md` — post-fix audit: retry/backoff policy
- `[[llama-server-silent-exit-10061]]` — 2026-09-26 incident with same symptom, recovered via LRU eviction
- `[[strata-bridge-lifecycle-facts]]` — bridge engine lifecycle context
- Commit `e4cbe26` — original TCP forwarder introduction (latent gap origin)
