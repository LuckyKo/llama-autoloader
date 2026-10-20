# BUG_0005 — Dead-backend recovery holds the manager lock across slow teardown and can cause spurious 503s

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `6ba95c4`
**Found in:** Post-fix audit of BUG_0001 (`c31e08e`)
**Severity:** medium
**Component:** `server.py` (`_begin_unload_locked`, `_finalize_unload_locked`, `_recover_dead_model_impl`, `_model_lock`), `tcp_forwarder.py` (`_wait_for_recovery`, `_resolve_request_model`)

## Symptom

`_recover_dead_model_impl` holds the global `manager._lock` while awaiting the entire unload:

- `_unload_finish` may drain in-flight slot actions for up to
  `SLOT_ACTION_DRAIN_TIMEOUT_S = 15.0` seconds (`server.py:690`).
- `_kill_proc` waits up to 3 seconds after `terminate()` and another 3 seconds after `kill()`
  (`server.py:2212-2218`), i.e. up to ~6 seconds.
- Worst case is therefore roughly 21 seconds with the manager lock held.

During that window:

- forwarder worker threads calling `resolve_request_model()` block on the lock; the cross-thread
  resolver has a hard **5-second** timeout (`tcp_forwarder.py:634`), so they can return `None`
  and surface a spurious 503/JIT failure.
- `/v1/models`, `/v1/status`, and other loads/unloads that need the lock are blocked.
- every request for the dead model blocks a forwarder worker thread for the whole reload. With
  `_MAX_CONNECTIONS = 256` (`tcp_forwarder.py:86`), a single recovering model can occupy the
  entire connection budget for minutes, denying service to unrelated models.

The same lock behavior exists in ordinary unloads, but dead-backend recovery makes it happen
automatically on the request path during an incident.

## Root cause

Recovery reuses `_unload_unlocked`, whose contract assumes the caller holds `_lock` for the whole
teardown. The slow, I/O-bound drain/kill work is therefore serialized behind a lock that is also
needed for fast routing/status lookups.

## Fix (implemented)

1. **Two-phase unload** — `_unload_unlocked` now uses `_begin_unload_locked()` (pop from
   `loaded`, mark unready/in-progress) and `_finalize_unload_locked()` (restore on failed
   kill, clear the flag). Dead-backend recovery calls the fast phase under `_lock`,
   releases the lock, runs `_unload_finish()` (auto-save/drain/queue drop/kill), then
   reacquires `_lock` only to finalize.
2. **Per-model serialization** — `_model_lock(model_id)` serializes loads, explicit
   unloads, and recovery for one model. Recovery holds it across teardown and reload,
   so a load cannot spawn a replacement on the same stable port while the old process
   is dying. Different models use different locks and proceed concurrently.
3. **Reload under the same lock** — recovery calls `_load_model_serialized()` while
   still holding the per-model lock, closing the detached gap where an explicit unload
   could otherwise return success just before the replacement appeared.
4. **Bounded recovery waiters** — the TCP forwarder caps threads waiting on one
   `(port, generation)` recovery (`AUTOLOADER_MAX_RECOVERY_WAITERS`, default 32). Beyond
   the cap it returns an immediate retryable 503 instead of letting one dead model
   consume the whole `_MAX_CONNECTIONS` budget.

Routing/status still take `self._lock`, but now only for the short validate/detach and
finalize windows. They are never blocked for the 15 s drain or 6 s kill.

## Verification

- `tests/test_dead_backend_recovery.py::test_recovery_teardown_does_not_hold_manager_lock`
  — `resolve_model_id_strict()` completes while a fake `_unload_finish` is paused;
  `_unload_in_progress` is armed and cleared correctly.
- `tests/test_dead_backend_recovery.py::test_recoveries_for_different_models_run_concurrently`
  — two recoveries reach the fake teardown before either is released.
- `tests/test_dead_backend_recovery.py::test_explicit_unload_waits_for_inflight_recovery`
  — unload does not return success during teardown, then unloads the replacement.
- `tests/test_dead_backend_recovery.py::test_pipe_with_ticket_rejects_when_recovery_waiters_saturated`
  — the per-engine waiter cap returns a retryable 503 and releases its slot.
- Full suite: 370 passed after the BUG_0003/0004/0005 change; one Windows
  `WinError 10055` event-loop setup flake passed when rerun.

## Related

- `BUG_0001_tcp_forwarder_no_dead_backend_recovery.md` — recovery caller
- `BUG_0003_recovery_evicts_healthy_replacement_on_reused_port.md` — the long lock window widens
  the same-port replacement race
