# BUG_0010 — Dead-backend recovery orphans a possibly-live process from `self.loaded` when teardown raises

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `28b95ae`
**Found in:** Same defect class as BUG_0009 (`_unload_unlocked`); this is the `_recover_dead_model_impl` teardown site, whose lesson `.agent_lessons/unload-teardown-exception-orphans-registry.md` had recorded as unfixed.
**Severity:** medium
**Component:** `server.py` (`_recover_dead_model_impl`)

## Symptom

If `_unload_finish` raised a `BaseException` after `_begin_unload_locked` had already popped the model, the model was removed from `self.loaded` while its process may still be running. The recovery path runs teardown **outside** `self._lock`, so the orphan is visible to other coroutines: no request can be routed to it, LRU eviction cannot see it, and nothing can retry the kill. No error line exists for the orphan itself — only the propagated exception.

## Root cause

The `except BaseException` handler cleared the `_unload_in_progress` flag but never re-inserted the model into `self.loaded`. The "a live process is always tracked in `self.loaded`" invariant was enforced on the normal path but bypassed on the exception path, which skips `_finalize_unload_locked` entirely.

## Fix (implemented)

The `except BaseException` handler in `_recover_dead_model_impl` now re-inserts the model under `self._lock` via `setdefault`, invalidates the status cache, logs the abort, and re-raises. The model is restored only when it may still be running, so a completed, verified kill (e.g. a `CancelledError` landing after the kill) is not resurrected as a routable entry.

## Verification

- `tests/test_dead_backend_recovery.py::test_recover_dead_model_restores_registry_when_teardown_raises` — mutation-proven: removing the `setdefault` re-insertion fails the "entry comes back" assertion.

## Related

- `BUG_0009_unload_unlocked_exception_orphans_registry.md` — sibling handler in `_unload_unlocked`, same defect class, fixed in `3926ace`
- `BUG_0005_recovery_holds_manager_lock_across_teardown.md` — why the recovery path tears down outside `self._lock`
- [[unload-teardown-exception-orphans-registry]] — the defect class, the site table, accepted residuals
