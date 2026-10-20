# BUG_0003 — Dead-backend recovery can evict a healthy replacement that reused the same port

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `6ba95c4`
**Found in:** Post-fix audit of BUG_0001 (`c31e08e`)
**Severity:** medium-high
**Component:** `server.py` (`LoadedModel.generation`, `_recover_dead_model_impl`), `tcp_forwarder.py` (`_resolve_target`, `_recover_dead_backend`)

## Symptom

Recovery identifies the dead engine only by `(model_id_hint, dead_port)`. If another operation
reloads the model (or a different model) onto the **same stable port** before the recovery task
runs, recovery cannot tell the replacement from the instance that refused the connection. It
marks the fresh instance `ready=False` and unloads it again.

This is a TOCTOU race, not a theoretical port collision: `_allocate_port` deliberately assigns a
stable port per model, so a reload after an unload normally reuses the same port.

## Reproduction sequence

1. Request R resolves model X on port P and starts `_connect_backend(P)`.
2. The engine on P has died, so the connect is refused. R calls `recover_dead_model(X, P)`.
3. Before R's recovery task acquires `_lock`, a state restore / JIT load / manual reload replaces
   X with a fresh ready instance on the same stable port P.
4. R's recovery sees `loaded[X].port == P` and `ready=True`, assumes it is still the dead engine,
   sets `ready=False`, and runs a verified unload.
5. The healthy replacement is killed. The request may then reload it a second time, or fail.

The current “replacement on a different port” check only handles the case where the new instance
landed on a different port. Same-port replacement is indistinguishable.

## Root cause

Recovery is keyed by port with no engine-instance identity. `LoadedModel` has no generation token
that the forwarder can capture when it resolves the target port.

## Fix direction

Capture and carry the engine instance/generation from the routing decision:

1. Add a monotonically increasing generation (or use the `LoadedModel` object identity plus
   `starting_at`) when a model becomes ready.
2. Have the forwarder resolve `(port, generation)` (for example a new `_resolve_target()` helper)
   and pass `expected_generation` into `recover_dead_model`.
3. Under `_lock`, compare the current `loaded[hint]` generation with the expected value. If it is
   newer and ready, return it as the recovered model without unloading.
4. Key `_recovery_tasks` by the generation-bearing identity as well, so a delayed recovery cannot
   attach to a different engine that reused the port.

This fix also gives BUG_0004 the identity primitive it needs for request accounting.

## Fix (implemented)

1. `LoadedModel.generation` is a monotonically increasing manager-assigned identity,
   stamped in `_do_load_model` before the instance becomes visible.
2. The forwarder resolves `(port, generation)` via `_resolve_target()` and carries
   `expected_generation` through `_pipe_with_ticket` → `_recover_dead_backend` →
   `ModelManager.recover_dead_model`.
3. Recovery compares the current generation under `_lock`. If the port now holds a
   newer ready generation, it returns that replacement without marking it unready or
   unloading it. An in-flight newer generation is joined instead of killed.
4. `_recovery_tasks` is keyed by `(dead_port, expected_generation)`, so a delayed
   recovery for the old instance cannot attach to the replacement's task.

`expected_generation == 0` retains the legacy port-only behavior for fakes/older
callers; all real engines carry a positive generation.

## Verification

- `tests/test_dead_backend_recovery.py::test_recover_dead_model_preserves_newer_ready_generation`
  — replacement on the same port is returned and `_unload_unlocked` / `load_model`
  are never called.
- `tests/test_dead_backend_recovery.py::test_recover_dead_model_still_evicts_exact_dead_generation`
  — the exact dead generation is still evicted and reloaded.
- `tests/test_dead_backend_recovery.py::test_recover_dead_model_joins_unready_newer_generation`
  — a still-loading replacement is joined, not unloaded.
- `tests/test_dead_backend_recovery.py::test_recovery_tasks_are_scoped_by_generation`
  — recovery-in-progress lookup is generation-specific.
- `tests/test_dead_backend_recovery.py::test_pipe_with_ticket_carries_generation_through_recovery`
  — the forwarder propagates the identity through recovery and retry.
- Full suite: 370 passed after the BUG_0003/0004/0005 change; one Windows
  `WinError 10055` event-loop setup flake passed when rerun.

## Related

- `BUG_0001_tcp_forwarder_no_dead_backend_recovery.md` — original recovery implementation
- `BUG_0004_active_request_tracking_port_not_engine_generation.md` — shares the generation-token fix
- `BUG_0005_recovery_holds_manager_lock_across_teardown.md` — widens the race window
