---
tags: [autoloader, fastapi, error-handling, state-restore, bug-0018, review]
aliases: [load-state-500-wrapping, catch-all-httpexception-collapse]
related: "[[slot-action-probe-loop-hides-engine-error]] [[ticket-queue-port-keyed-blocks-pre-load-reservation]]"
confidence: verified
author: reviewer_reviewer0018_20261103_115046.jsonl
---

**Fact:** in `ModelManager.load_state`, the whole body (load + restore) sits inside one `try` whose
last clause is `except Exception as e: raise HTTPException(500, f"restore failed: {e}")`. `HTTPException`
is an `Exception` subclass, so **every status code chosen inside that span — the new 503 "busy"
reservation timeout, the 504 load timeout, the 503 load failure, the 503 "no longer loaded" — is
re-wrapped as a 500** before it reaches the caller.

**Pin:** autoloader working tree (uncommitted BUG_0018 fix on top of `f510e2b`), 2026-11-03.
Symbols: `load_state` (`server.py`, reservation at `:4087`, load `:4104`, catch-all `:4133-4134`).
At `HEAD:f510e2b` the load block (`:4061-4071`) was **outside** the try, so 504/503 propagated — this
is a regression introduced by the BUG_0018 restructure, not a pre-existing quirk.

**Proof method (re-checkable):** `git show HEAD:server.py` and diff the try span; then grep the suite
for the message fragments (`"Timeout loading model"`, `"no longer loaded"`, `"busy (queue wait"`) —
**zero hits**, so no test guards these codes and a green suite proves nothing here.

**Fix pattern:** put `except HTTPException: raise` immediately before the catch-all (same fix shape as
`except EngineSlotError: raise` in `_perform_slot_action`, see
[[slot-action-probe-loop-hides-engine-error]] — the same swallowing class, one level higher).

**Also worth knowing from the same review (2026-11-03):**
- The queue registry is now keyed by **model_id**, not port. Every `get_or_create` call site must pass
  the *canonical* id; a site passing an alias silently creates a second queue and voids the FIFO
  ordering guarantee without failing loudly.
- `tests/test_safe_mode_slot_ordering.py::test_auto_restore_suppressed_while_reservation_held` is a
  **tautology**: it patches `load_model`, so `_do_load_model` (and the `_suppress_auto_restore` gate)
  never runs; the collected `ticket_calls` list is never asserted. A suppression test must reach the
  real `_do_load_model` or assert the flag inside the fake load.
