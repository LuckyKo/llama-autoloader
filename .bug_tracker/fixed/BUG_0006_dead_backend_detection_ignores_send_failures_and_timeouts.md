# BUG_0006 — Dead-backend detection ignores post-connect send failures and connect timeouts

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `1216df4`
**Found in:** Post-fix audit of BUG_0001 (`c31e08e`)
**Severity:** low-medium
**Component:** `tcp_forwarder.py` (`_pipe_to_backend_inner`, `_backend_confirmed_dead`, `_is_connect_timeout`, `_is_connection_reset_or_abort`)

## Symptom

The BUG_0001 recovery path only recognizes refusal-class errors from `_connect_backend`. Other
failure modes still leave the model marked `ready=True`:

1. **Send-stage failure.** `target_sock.sendall(raw_headers + body_prefix)` raises
   `BrokenPipeError`/`ConnectionResetError`/`OSError` if the engine dies immediately after the TCP
   handshake. The handler logs, closes the backend socket, returns a bare 502, and does not mark
   the model unready.
2. **Connect timeout.** `socket.create_connection(..., timeout=10)` raises `TimeoutError` for a
   listener that is wedged or has a full accept queue. That is not `ConnectionRefusedError`, so it
   gets the old 502-only path even though the engine may be unusable.
3. **Other resets.** `ConnectionAbortedError` or a reset during connect are also not classified.

In the send-stage case the next request usually hits a refusal and recovers, so this is usually a
one-request delay. In the timeout case the backend may remain “loaded” and keep timing out on
every request with no recovery.

## Root cause

`_is_connection_refused` deliberately matches only refusal-class errors. There is no separate
“post-connect failure” policy and no timeout-specific liveness verification.

## Fix direction

Split the policy by stage:

- **Connect refusal:** current behavior — safe to retry the request after reload.
- **Send-stage failure:** the request may have been partially delivered, so **do not retry** it,
  but do mark the engine unready and schedule the same deduplicated recovery for later requests.
  Return 502 for the current request.
- **Connect timeout / ambiguous connection failure:** verify the process (`proc.poll()`) and/or
  probe `/health` with a short timeout before declaring death. If confirmed dead/unresponsive,
  evict/reload; otherwise retain the normal 502 without a spurious reload.

Keep the “no retry after partial send” distinction in the code and tests; do not simply widen
`_is_connection_refused` to catch every `OSError`.

## Fix (implemented)

- Connect failures are now classified by stage:
  - `_is_connection_refused` keeps the existing immediate `_BackendUnreachable` behavior.
  - `_is_connect_timeout` covers `TimeoutError`/`socket.timeout`, `ETIMEDOUT`, and WinError 10060.
  - `_is_connection_reset_or_abort` covers reset/abort (WinError 10053/10054, `ECONNRESET`,
    `ECONNABORTED`).
- An ambiguous connect failure is only promoted to `_BackendUnreachable` after
  `_backend_confirmed_dead()` verifies the process exited (`proc.poll()`) or a short `/health`
  probe fails. It always probes when no process result is conclusive, including when no registry
  entry owns the port, so a newer healthy instance is never evicted on a timeout alone.
- A send-stage `OSError` does **not** raise `_BackendUnreachable` and therefore never retries.
  The current request gets 502; the engine is marked unready and the deduplicated BUG_0001
  recovery is scheduled for subsequent requests.
- Internal-API forwarding keeps its old behavior: no engine liveness/marking is applied.

Tests: `tests/test_backend_liveness_backoff.py`
- send failure gets 502, marks the model unready, schedules recovery, and never retries;
- send failure on internal API leaves models untouched;
- connect timeout with failed health verification follows recovery;
- connect timeout against a live engine keeps a normal 502 and stays ready;
- refusal is conclusive and does not require a health probe.

## Verification

- Fake backend that accepts then resets/closes during `sendall`: current request gets 502, the
  model is marked unready and recovery is scheduled; no retry.
- Connect timeout against a non-accepting listener is verified via `/health` before recovery.
- A confirmed-dead timeout follows recovery; a live-but-slow backend is not evicted.
- Full suite: 389 passed (one pre-existing Windows `WinError 10055` flake passed when rerun).

## Related

- `BUG_0001_tcp_forwarder_no_dead_backend_recovery.md` — refusal-only recovery
- `BUG_0002_forwarder_hung_backend_no_liveness_timeout.md` — the established-connection variant
- `BUG_0007_failed_backend_reload_has_no_backoff.md` — failure policy after eviction
