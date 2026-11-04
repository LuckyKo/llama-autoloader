---
tags: [autoloader, fifo-ticket, state-restore, bug-0015, bug-0018, pre-load-reservation]
aliases: [preload-ticket-hoist, auto-restore-ticket-owner]
related: "[[ticket-queue-port-keyed-blocks-pre-load-reservation]] [[load-state-catch-all-swallows-inner-httpexception-codes]] [[bug-0014-release-marker-invariant]]"
confidence: verified
author: coder_coder0015_20261103_125313.jsonl
---

**Fact:** the autoloader's state restore is serialized against in-flight chat requests by
reserving the per-model FIFO ticket *before* the engine is ready. Two paths do this:
- **manual** `load_state` (`server.py`): `_reserve_model_ticket` (was `:4047`) grabs the ticket,
  sets `_suppress_auto_restore`, then calls `load_model`.
- **auto** `_do_load_model` (`server.py`, was span `3000-3155`): the ticket acquire was hoisted from
  *after* `_wait_until_ready` to *before* it (right after `lm` is inserted into `self.loaded` and the
  port is known). The ticket is held across load→auto-restore and released in a `finally` that covers
  success / CancelledError / not-ready / auto-restore-failure.

**The non-re-entrant guard (the subtle part):** the FIFO ticket is single-holder (`_granted_id`). The
manual path already holds it, so `_do_load_model` **skips** its pre-load acquire when
`model_id in self._suppress_auto_restore` — otherwise a second `acquire_ticket` on the same queue blocks
behind its own holder and burns the 60 s timeout. The flag is manager-level state (a `Set`), so it
survives the `_load_model_serialized` task-join; a threaded parameter would be dropped by the join.

**Why pre-load (the race it closes):** between "model ready" and "ticket acquired" a chat request can
reach the engine, hold KV / catalog a continuation, and then the auto-restore's `state/load` 400s
("snapshot does not fit the free KV capacity"). Reserving the ticket pre-load makes any chat request
landing in the JIT-load window queue *behind* the auto-restore (FIFO).

**Release is safe on a stopped queue:** `_unload_unlocked` calls `queue_registry.remove(model_id)`
(`stop()`), which clears `_pending` but NOT `_granted_id`; so a `release_ticket` on the dropped queue
still matches and no-ops cleanly.

**Test note:** `tests/test_safe_mode_slot_ordering.py::test_auto_restore_reserves_ticket_preload`
drives the *real* `_do_load_model` (mocking only `_wait_until_ready`/`_perform_slot_restore`/spawn), so
unlike the tautological `test_auto_restore_suppressed_while_reservation_held` (which patches
`load_model`), it actually exercises the pre-load hoist. It asserts a chat-side ticket fired in the
window is granted only *after* the auto-restore (order `["restore","restore-end","llm"]`).
