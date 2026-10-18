---
tags: [strata, autoloader, FIFO, queue, ordering, state-restore, race-condition, tcp-forwarder]
aliases: [strata-restore-fifo-ordering, autoloader-per-model-queue]
related: "[[restored-slot-seqrm-reprocess]] [[option-a-prefix-reuse-infeasible-hybrid]] [[autoloader-slot-action-error-observability-gap]]"
confidence: verified
author: researcher_strata-restore-fifo-investigation_20261015_042252.jsonl
contributors:
  - coder_autoloader-fifo-queue-build_20261015_045500.jsonl
---

# Autoloader has no per-model FIFO queue → Strata restore races with LLM requests

**Fact:** The autoloader routes LLM requests via the TCP forwarder (port 1234, raw pipe) and state ops via FastAPI (port 1235, HTTP client). Both hit the Strata serve's HTTP port. The Strata serve's `self.fifo` (`threading.Lock`, serve/server.py:3042) **serializes** engine operations but does **not enforce ordering**. If the LLM request arrives at the serve before the RESTORE, the LLM request wins the lock and processes the full prompt without the restored state.

**The Strata engine's prefix matching works correctly.** After RESTORE, `live`/`checks`/`live_ok=true` are set (generate.cpp:10714-10734). The GEN handler checks `starts_with(live, ...)` and skips matching tokens via `RESUME` (generate.cpp:11034-11043, 11329). The problem is purely about **which operation reaches the engine first**.

**Evidence:**
- `tcp_forwarder.py:209-254`: `_handle_connection` pipes LLM requests directly to the engine port. No queue, no coordination with state ops.
- `server.py:1484-1542`: `save_state`/`load_state` use `self._lock` (asyncio lock) which only coordinates state ops with each other, NOT with the TCP forwarder.
- `serve/server.py:3042`: `self.fifo = threading.Lock()` — used by `_turn()` (line 3803) and `slot_action()` (line 3111). A `threading.Lock` provides mutual exclusion, not ordering.
- User log: restore logged at 04:02:39, full 15,480-token prompt reading at 04:02:47.

**Fix design:** Per-model ticket-based FIFO queue in the autoloader (`model_queue.py`). All operations (LLM, save, restore) acquire a ticket from the queue before executing. Tickets are granted in FIFO order. The TCP forwarder acquires a ticket before `_pipe_to_backend`; FastAPI acquires a ticket before `_perform_slot_action`. Full design in `reports/strata-restore-fifo-design.md`.

**Distinct from:** [[restored-slot-seqrm-reprocess]] (llama.cpp `seq_rm` failure — different root cause, different engine). [[option-a-prefix-reuse-infeasible-hybrid]] (llama.cpp hybrid model KV limitation — architectural, not a queueing issue).

**Open question:** Even with correct ordering, if the AC harness modifies the conversation between save and restore (e.g., adding the security agent's result), the re-tokenized prompt won't match the restored `live` tokens. This would cause a full reprocess even with the FIFO queue. Needs runtime verification.

---

## Implementation addendum (2026-10-15, coder)

Implemented in `model_queue.py` (`ModelCommandQueue` + `ModelQueueRegistry`), integrated into `tcp_forwarder.py` and `server.py`. Tests: `tests/test_model_queue.py` (9 tests). Existing `test_tcp_forwarder.py` + `test_regression.py` still pass (120).

**⚠ The design's §3.5 reference code had a latent deadlock — do NOT copy it verbatim.** Its `acquire_ticket` appended the ticket to `_pending` and *then* granted it immediately when no holder existed. The granted ticket stayed in `_pending`, so `release_ticket` popped and **re-granted the same ticket** instead of the next waiter → the queue deadlocked after the first holder (proved: only ticket 1 granted, 2..N time out). This is a textbook [[verify-reference-implementation-before-mirroring]] case — the "reviewer-passed" exemplar carried the defect.

**Fix applied:** `_pending` holds **only not-yet-granted waiters**. A ticket granted immediately is tracked solely by `_granted_id`/`_granted_event` and is never added to `_pending`. `release_ticket` then always grants a *different* waiter. Also added a guard in `acquire_ticket`: if a grant races with the wait timeout, honor the grant (`_granted_id == ticket_id`) rather than raising — a granted_id no one releases would otherwise wedge the queue. `release_ticket`/`stop()`/`pending_count()` are consistent with this invariant.

**Integration points actually wired:**
- `tcp_forwarder.py` `_handle_connection`: BOTH the loaded-model pipe path AND the post-JIT-load pipe path wrapped in `with queue.ticket(...)` (ticket acquired only *after* the model is ready, so the load itself is never gated).
- `server.py` `save_state`/`load_state`: ticket via `await asyncio.to_thread(queue.acquire_ticket)` + `try/finally release` (design §5.3, timeout 120s). These run while the asyncio `self._lock` is NOT held — verified no deadlock.
- `server.py` `_do_load_model`: creates the queue after the model is ready; the **auto-restore** of `.auto.bin` is ticketed (same race as the core fix; uncontested at load because forwarders are blocked on the load future).
- `server.py` `_unload_unlocked`: **auto-save is intentionally NOT ticketed** — teardown path, no later LLM request depends on it, and the engine's own FIFO already serializes it; ticketing it would hold the asyncio lock across a long ticket wait for no ordering gain (reviewer MAJOR). The queue IS stopped/dropped (`queue_registry.remove(port)`) before the process is killed. `ModelManager.stop()` calls `queue_registry.stop_all()`.

**Timeout note:** forwarder ticket timeout 300s; state-op ticket timeout 120s (per design). A generation longer than 120s can make a queued save/restore time out — deliberate, matches the design; revisit only if contested long generations are observed.
