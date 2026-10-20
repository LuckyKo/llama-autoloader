---
tags: [httpx, exceptions, transport-errors, slot-actions]
aliases: [dead-socket-signals, unreachable-engine-errors]
related: [[strata-restore-fifo-ordering-gap]]
confidence: verified
author: build-review-autoloader-safe-mode_20261019_004920.jsonl
---

# Dead-socket exception signals in httpx 0.28.1 / httpcore 1.0.9

Verified empirically (host `C:\Python312` and container both run httpx 0.28.1 / httpcore 1.0.9; requirements pin is only `httpx>=0.27.0` — re-check versions before trusting this list). A scratch probe (bind socket, accept-then-close thread) was run against `127.0.0.1` to enumerate what `httpx.AsyncClient.post` actually raises for a dead local engine:

| case | exception raised | message |
|---|---|---|
| nothing listening (connection refused) | `httpx.ConnectError` | `'All connection attempts failed'` |
| peer closes **before sending a response** | `httpx.RemoteProtocolError` | `'Server disconnected without sending a response.'` — raise site: `httpcore._async/http11.py:231` (guarded by `their_state == SEND_RESPONSE`) |
| peer closes **mid-response** (EOF while reading body) | `httpx.ReadError` | `''` — **empty `str()`**, so any log/summary of it needs the `repr(e)` fallback |

Negative findings (re-checkable by grep over the installed packages):
- `httpx.IncompleteReadError` appears **zero** times in httpx 0.28.1 / httpcore 1.0.9 — it is a dead name for this pinned stack, not a missing guard.
- `httpx.CloseError` is defined in httpx but has **no raise site** in httpcore 1.0.9 — never surfaces from the HTTP/1.1 path.
- `httpx.ConnectError is not httpcore.ConnectError` — parallel hierarchies; `isinstance` checks against the **httpx** classes are the correct ones for `self.client` calls.
- `httpx.NetworkError` is the category covering exactly Connect/Read/Write/Close (subclasses), deliberately excluding `TimeoutException` — a useful tuple element when "refused/closed" is the intent.

## Consequence for this repo

`_UNREACHABLE_ERRORS` in `server.py` (module constant, used by `_perform_slot_action` to break the 3-format retry loop on a dead engine) did not exist at all at committed HEAD f3d7f586b27d7100c18ccf1c0bd7528c3763e584 — the whole safe-mode/F5 block is uncommitted working-tree work. The 3-tuple below is an **intermediate working-tree state**, not HEAD: `(ConnectError, ConnectTimeout, RemoteProtocolError)` — missing `ReadError`. Effect is bounded (the next format's attempt gets a real `ConnectError` and breaks), but the EOF-mid-response case slips through the break. If the tuple is ever widened, prefer `(httpx.NetworkError, httpx.RemoteProtocolError)` and keep `ReadTimeout` excluded (a reachable-but-slow engine may answer a differently-shaped request).

Probe method: `socket.bind+listen` → `asyncio.to_thread(accept_and_close)` racing an `httpx.AsyncClient().post` to that port; deterministic EOF without sleeps.

## Resolution (applied 2026-10-19)

Shipped tuple is `(httpx.NetworkError, httpx.ConnectTimeout, httpx.RemoteProtocolError)` — NOT the bare two-element tuple from the recommendation above. Reason: `httpx.ConnectTimeout` subclasses `TimeoutException`, so `NetworkError` does NOT cover it and a blind swap silently drops connect-timeout aborts (behaviour regression, no test would have caught it). Keeping `ConnectTimeout` explicit preserves the old semantics while `NetworkError` adds the `ReadError('')` mid-response case. Guard test: `tests/test_safe_mode_slot_ordering.py::TestSlotActionRetry::test_dead_mid_response_stops_retrying` (fails 3≠1 on the old tuple).

Revert-proof gotcha: `git stash push server.py` reverts to committed HEAD, which predates the whole uncommitted safe-mode feature (`RECENT_UNLOAD_REFUSE_WINDOW_S` absent → collection error, not a clean fail). For uncommitted features, revert the single changed line in the working tree instead.
