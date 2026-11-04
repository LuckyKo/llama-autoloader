# BUG_0015 — `state/load` not serialized with in-flight generation requests (restore fails when KV is occupied)

**Status:** FIXED (2026-11-03)
**Filed:** 2026-10-26
**Found in:** Production log 2026-10-26 07:26:01–07:26:03 (`N:\work\stuff\Beta\llama-autoloader\logs\console.log`). Autoloader fires `POST /state/load` to restore `Maine.bin` immediately after model load, but the bench (or AgentCascade) sends a generation request that occupies KV pages before the restore completes. The restore then fails with HTTP 400.
**Severity:** high — state restore is a critical operation for session continuity; if it silently fails (500 to the caller), the agent's entire conversation context is lost and the next turn is a full cold re-prefill.

## Symptom

```
07:26:01,972 [INFO]    Restoring slot 0 state from file: ...Maine.bin
07:26:03,355 [INFO]    throughput | 5.0s | prefill 907.6 tok/s (4,529 tok) | decode 0.8 tok/s (4 tok)
07:26:03,930 [WARNING] Slot restore attempt failed [URL param: ?action=restore]:
                     HTTP 400 — {"error":{"code":"slot_restore_failed",
                     "message":"session snapshot does not fit the free KV capacity;
                     evict other sessions first",...}}
07:26:03,931 [WARNING] Slot restore attempt failed [URL param: ?action=restore]: HTTP 404 —
07:26:03,933 [WARNING] Slot restore attempt failed [JSON body params]: HTTP 404 —
07:26:03,933 [INFO]    POST /v1/models/.../state/load HTTP/1.1 500 Internal Server Error
```

The generation request (req#9, 4,529 tokens) is running and holding KV pages when the state restore is attempted. The restore needs 54K tokens of KV but can't fit because the in-flight request's session is occupying pages.

## Root Cause

The autoloader fires the state restore (HTTP `POST /state/load`) to NInfer without serializing it against concurrent generation requests. The NInfer server accepts the restore call while a generation request is already in flight, and NInfer's `state/load` handler returns 400 when the snapshot doesn't fit the remaining free KV capacity — it does not evict or wait for the in-flight session.

The autoloader's load sequence is:
1. `POST /v1/models/{id}/load` → 200 (model is now accepting requests)
2. Client immediately sends generation requests (bench, AgentCascade, etc.)
3. `POST /v1/models/{id}/state/load` → 400 (KV occupied by step 2's requests)

Steps 2 and 3 race. The autoloader should ensure no generation requests are in-flight (or are queued) during the state restore window.

## Expected Behavior

When the autoloader initiates a state restore after model load:
- In-flight generation requests should complete (or be evicted) before the restore is attempted, OR
- New generation requests should be queued/paused until the restore completes, OR
- The restore should retry (with backoff) until the KV capacity is available

The state restore should not fail with 500 simply because another client's request happened to be in-flight at that instant.

## Fix Options

1. **Autoloader-side queue**: The autoloader should track in-flight requests per model. Before issuing `state/load`, wait for all in-flight requests to complete (with a timeout). New requests arriving during the restore window should be queued.
2. **NInfer-side auto-evict**: `POST /state/load` should evict other sessions to make room (the error message already says "evict other sessions first" — the engine could do this itself instead of refusing).
3. **Combined**: Autoloader waits for in-flight requests to drain (option 1), and NInfer's `state/load` auto-evicts any remaining cached (non-active) sessions to guarantee capacity (option 2).

## Resolution (2026-11-03)

Closed by the **pre-load FIFO ticket reservation** in the *auto*-restore path — the same
mechanism that closed [[BUG_0018]] for the *manual* `load_state` path.

**Change** (`server.py`, `_do_load_model`, the span that was 3000–3155):
- The FIFO queue creation + `acquire_ticket` were hoisted from *after* `_wait_until_ready`
  (the model-ready → ticket-acquired window) to **before** it — right after `lm` is inserted
  into `self.loaded` and the port is known. The ticket is now reserved pre-load, so any chat
  request that lands in the JIT-load window queues *behind* the auto-restore instead of
  holding KV first.
- The ticket is held across the **whole load→auto-restore span** and released in a `finally`
  that covers success, `CancelledError`, and the not-ready 500 path, so a failed load never
  leaks the ticket (which would wedge the queue behind a holder that never returns).
- The auto-restore block now *uses* the pre-load ticket (no inner re-acquire) and is gated on
  `ticket is not None`.
- **Nested-acquisition guard:** the FIFO ticket is non-re-entrant (single `_granted_id`
  holder). The manual `load_state` path already holds the ticket and sets
  `_suppress_auto_restore`, so `_do_load_model` **skips** the pre-load re-acquire when that
  flag is set — otherwise a second `acquire_ticket` on the same queue would block behind its
  own holder and burn the 60 s timeout. This reuses the existing manager-level
  `_suppress_auto_restore` flag as the load-owner guard (it survives the `_load_model_serialized`
  task-join path, unlike a threaded parameter would).

**Verification:**
- `tests/test_safe_mode_slot_ordering.py::TestLoadWindowTicketOrdering::test_auto_restore_reserves_ticket_preload`
  — drives a (mocked) JIT load with `_wait_until_ready` gated in the window; a chat-side
  `acquire_ticket` fired in the window is granted only *after* the auto-restore runs
  (order `["restore", "restore-end", "llm"]`). Without the fix the order would be
  `["llm", "restore", ...]`.
- `tests/test_safe_mode_slot_ordering.py::TestLoadWindowTicketOrdering::test_auto_restore_ticket_released_on_failed_load`
  — a not-ready load releases the pre-load ticket in the `finally` (no leaked `_granted_id`).
- Full suite: **506 passed**; the only 3 errors are pre-existing `test_real_model.py`
  environment gaps (a hard-coded CUDA12 backend binary
  `backends/llama.cpp-win-x86_64-nvidia-cuda12-avx2-local/llama-server.exe` is absent from this
  checkout — fixture setup errors, unrelated to this change).

## Cross-references

- NInfer engine: the `state/load` 400 response is correct engine behavior (it can't silently evict an active generation). The bug is in the autoloader's sequencing.
- AgentCascade: the agent's next turn after a failed restore will be a full cold re-prefill (~45s for 38K tokens) because the state is lost.
- [[BUG_0018]] — same load-window root cause (ticket acquired after `load_model`). That one is the restore *succeeding* while the next request *misses* the restored KV (catalog race); this one is the restore *failing* (400) when an in-flight request holds KV. **Both are now closed by the same pre-load FIFO ticket reservation mechanism** — BUG_0018 in the manual `load_state` path (`_reserve_model_ticket`), BUG_0015 in the auto path (`_do_load_model`), so they are closed together.
