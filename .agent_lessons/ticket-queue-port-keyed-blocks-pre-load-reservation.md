---
tags: [autoloader, fifo, ticket-queue, model-queue, race-condition, state-restore, bug-0018]
aliases: [port-keyed-queue-blocks-pre-load-reservation, load-window-ticket-gap]
related: "[[strata-restore-fifo-ordering-gap]] [[autoloader-unload-drain-inflight-slot-action]] [[forwarder-generation-keyed-state-retirement]]"
confidence: verified
author: researcher_digger0018_20261103_110015.jsonl
---

# The FIFO ticket queue is keyed by PORT, so the JIT-load window cannot be reserved

**Fact:** `ModelQueueRegistry` is keyed by engine **port** (`Dict[int, ModelCommandQueue]`,
`ModelQueueRegistry.get_or_create(port, model_id)`), and a port only exists *after* the JIT load
completes. Consequence: no caller can reserve a ticket **before** a model load, so the whole load
window is unsynchronized. This is the structural reason behind
[[strata-restore-fifo-ordering-gap]]'s "ticket acquired only after the model is ready" note, and it is
the root cause of BUG_0018.

**Symptom (BUG_0018):** `POST /v1/models/{id}/state/load` returns 200 and logs the restored token
count, yet the next chat request re-prefills cold (`cache 0 (0.0%)`). The restore of a 109,455-token
session succeeded at 10:28:08.173 while a 305-token request ran *inside* the load window at 10:28:05;
that request cataloged its own continuation into the engine's context-cache first, so the restored
session lost the prefix match. The sequential control run hits 99.1% — the miss only occurs when a
concurrent request lands in the load window.
Ticket: `.bug_tracker/BUG_0018_state_restore_succeeds_but_next_request_reprefills.md`.

**Evidence (llama-autoloader @ `f510e2b`, 2026-11-03 — re-verify before trusting these numbers):**
- `server.py` `load_state`: acquires the ticket **after** the load — `queue_registry.get_or_create(lm.port, model_id)`
  then `acquire_ticket(timeout=120.0)` (`:4084-4085`), `_perform_slot_restore` (`:4087`).
- `server.py` `_do_load_model`: allocates the port (`_allocate_port`, `:3017`) and only then
  `get_or_create(port, model_id)` (`:3108`); its auto-restore is ticketed at `:3131`.
- `tcp_forwarder.py`: the post-JIT path takes its ticket after readiness (`:893-895`) and wins the
  race against `load_state` because the forwarder worker calls `acquire_ticket` directly while
  `server.py` goes through `await asyncio.to_thread(...)` (`:4085`).
- `_allocate_port` (`:2331`) is *stable* per model (`base_port + sorted-index`) but explicitly only
  "tends" to hold and falls back to a scan when the port is taken — so "predict the port and reserve
  on it" is a **silent no-op**, not a workaround.
- `ModelCommandQueue.port` is assigned and **never read** anywhere (whole-tree grep) — the port field
  exists only to satisfy the dict key.

**Constraints for the fix (each verified in code, do not re-derive them):**
1. The queue is **not re-entrant** (single `_granted_id`). A pre-load reservation plus the auto-restore's
   `acquire_ticket` on the same queue is not a hang — the waiter times out (`model_queue.py` removes
   itself from `_pending`), and `except Exception` swallows it as `"Auto-restore failed"`, burning the
   full 60 s acquire timeout on every safe-mode return. It must be *suppressed*, not ordered.
2. A `skip_auto_restore` **parameter** cannot work: `_load_model_serialized` joins an already-created
   load task (`task = self._loading_tasks[model_id]`) built by the *first* caller, so the flag is
   dropped exactly in the concurrent case the fix targets. Use a manager-level flag
   (`set[str]` checked at the auto-restore gate), armed around the `load_model` call.
3. Re-keying to `model_id` removes the registry's implicit alias merge (the debug-log branch that
   notes "aliases resolve to the same port"). Alias correctness then lives entirely in the callers —
   `tcp_forwarder.py:895` passes the **unresolved** request name (while `:873` passes the resolved
   one), and `_pipe_with_ticket`'s `model_id` argument is reused by the recovery-retry queues
   (`:1624`, `:1653`). Those three sites are where the ordering guarantee can be silently lost.
4. The unload drain (`_drain_inflight_slot_action`) counts `_inflight_slot_actions` keyed by **port**;
   a pre-load reservation is invisible to it. Safe (queue `stop()` does not clear `_granted_id`), but
   it must not be "fixed" by arming the counter on a predicted port.
5. Test churn for re-keying is bounded: 4 files (`test_model_queue.py`, `test_tcp_forwarder.py`,
   `test_dead_backend_recovery.py`, `test_safe_mode_slot_ordering.py`), ~26 lines.

**Plan:** `N:\work\WD\AgentWorkspace\plans\BUG_0018_load_window_ticket_plan.md` (recommended option:
re-key by model_id + reserve pre-load; options B and C rejected with reasons).
