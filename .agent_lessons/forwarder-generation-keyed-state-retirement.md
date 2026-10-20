---
tags: [tcp-forwarder, memory-leak, engine-generation, registry-pruning]
aliases: [generation-keyed-state-retirement, liveness-dict-leak]
related: [[raw-tcp-forwarder-architecture]] [[unload-teardown-exception-orphans-registry]] [[lru-eviction-result-check-and-mid-load-exclusion]]
confidence: verified
author: coder_polish-coder-c_20261020_170413.jsonl
contributors:
  - reviewer_polish-review-c_20261020_170958.jsonl
---

# Every `(port, generation)`-keyed dict in `RawTCPForwarder` needs an explicit retirement hook

`generation` is assigned by `ModelManager` and increments **monotonically forever** (a reload never reuses a generation number). So any per-engine dict keyed by `(port, generation)` grows by one entry per reload cycle unless something pops it. `port` reuse is handled by the generation component (see [[raw-tcp-forwarder-architecture]]); the flip side is that no key is ever overwritten, so stale keys are never evicted implicitly.

## Where the state is created

`RawTCPForwarder.__init__` declares these `(port, generation)`-keyed containers:
`_active_conns`, `_active_response_started`, `_recovery_waiters`, `_liveness_failures`,
`_liveness_last_probe`, `_liveness_probe_locks`, and the `_scheduled_recoveries` **set**.

The first three already self-prune (`_unregister_active`, `_release_recovery_waiter`). The liveness
dicts and the recovery set did not — each reload cycle leaked a `threading.Lock`, two float/int entries,
and one set member.

## The retirement hook (fixed at `d977c82` + POLISH-phase edit, 2026-10-20)

- `_handle_confirmed_hang` (`tcp_forwarder.py`, the only place that marks an engine `ready = False`
  and schedules recovery) now calls `_retire_liveness_state((port, generation))` before
  `_schedule_recovery`. `_retire_liveness_state` is a new helper next to `_liveness_probe_lock`
  that pops `_liveness_last_probe`, `_liveness_probe_locks` **and `_liveness_failures`** under
  `self._liveness_lock`.
- Why unconditional popping is correct here: unlike `_unregister_active` (house style, which pops the
  outer key only when the *inner* container empties), these are flat `key -> scalar` maps, and after
  the hang is confirmed the key can never be read again — `_check_engine_liveness` returns early on
  `not lm.ready`, and `_find_engine` stops matching the old generation once the manager bumps it.
  `_handle_confirmed_hang` has three call sites — `_check_engine_liveness` (process-exited and probe-threshold
  paths, both **outside** the probe lock) and `_pipe_to_backend_inner` (the BUG_0006 send-failure path) — and none
  of them holds `_liveness_lock`, so taking that lock inside `_retire_liveness_state` cannot deadlock.
  (Corrected 2026-10-20 by the review: the earlier "only from `_check_engine_liveness`" wording was a partial-grep
  miss; the no-deadlock conclusion was always right. See [[liveness-state-leak-persists-on-manager-side-retirement]].)
- `_schedule_recovery`'s dedup marker is now dropped in a `finally` inside its `_done` callback, i.e.
  **after** `fut.result()` confirms the recovery actually ran (pre/post-destructive-action discipline),
  covering both success and failure of the recovery coroutine.

## Second-order bug the same fix removes

Before this change the marker was discarded only on the two *error* paths of `_schedule_recovery`
(no running event loop / submit failed). A completed recovery — success **or** failure — left the key
in `_scheduled_recoveries` forever, so the early-return guard (`if key in self._scheduled_recoveries:
return`) permanently suppressed any later re-schedule for that generation. Dropping the marker at
completion restores retryability; the manager's own dedup + per-model cooldown (BUG_0007) is what
actually bounds repeat failures, not this forwarder-side marker.

## Checklist for future work

Any new `(port, generation)`-keyed container added to `RawTCPForwarder` must be popped in
`_retire_liveness_state` (liveness-class state) or in the path that already prunes the active/waiter
registries. Grep for `Dict[Tuple[int, int]` / `set()` in `__init__` and ask "what pops this key?"

Test coverage status: `tests/test_backend_liveness_backoff.py` exercises `_handle_confirmed_hang`
indirectly (via `_check_engine_liveness`, with `_schedule_recovery` monkeypatched) and asserts on
`_liveness_failures` only for the **healthy-probe reset** path (`test_successful_liveness_probe_resets_failures`).
Nothing asserts that the retirement path clears the liveness dicts or the `_scheduled_recoveries` marker, so
this leak class stays unpinned. If this area is touched again, add `assert not f._liveness_probe_locks`
(and the equivalent for `_liveness_last_probe` / `_scheduled_recoveries`) right after the confirmed-hang
assertions in `test_liveness_requires_consecutive_failures_and_marks_engine_unready`.

## Addendum (2026-10-20, after independent review)

The single-hook version above was **insufficient** — see
[[liveness-state-leak-persists-on-manager-side-retirement]]. Resolved by factoring the sweep into `_retire_stale_liveness_state(items)`, called from
`_liveness_watchdog` with the same `manager.loaded` snapshot the probe loop uses (retire any key in
`_liveness_probe_locks` not in the current `(port, generation)` set; reusing the existing `items`
snapshot means a manager without `loaded` skips the tick rather than mass-retiring). The key snapshot
inside it is taken **under** `_liveness_lock` — request threads insert probe locks concurrently, and
`list()` over a dict whose size changes raises `RuntimeError`, which would silently kill the watchdog
thread; the retires themselves run outside that block since `threading.Lock` is not reentrant. Review
caught this only on the second pass: re-check new code you wrote, not just the original patch.
plus a guard in `_check_engine_liveness` that skips the post-probe write when
`key not in self._liveness_probe_locks` (the popped probe lock is the retirement marker;
without it a probe racing a concurrent retirement re-created `_liveness_last_probe` and
`_liveness_failures`).

Pinned in tests: `tests/test_backend_liveness_backoff.py::test_liveness_requires_consecutive_failures_and_marks_engine_unready`
now asserts `not f._liveness_probe_locks`, `not f._liveness_last_probe`, `not f._liveness_failures` after the confirmed hang.
`_scheduled_recoveries` is still unasserted (that test monkeypatches `_schedule_recovery`, so an assertion there would
pass for the wrong reason); a real check needs a live event loop plus a fake `recover_dead_model`.

Unrelated bug logged, not fixed: a **cancelled** recovery logs `Background recovery ... failed:` with an empty message.
`asyncio.CancelledError` is not an `Exception`, but `concurrent.futures` re-wraps it as
`CancelledError`, which *is*, so the `except Exception` in `_done` catches it and `str(e)` is blank.

Verification: `python -m pytest tests/ -q -x -n 0` → 325 passed, twice (before and after the
addendum changes, 100.6 s).
