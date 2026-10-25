# BUG_0013 — an unload that races an in-flight load is booked as a failed dead-backend recovery

**Status:** FIXED (2026-10-25)
**Filed:** 2026-10-25
**Found in:** Production log 2026-10-25 08:21:16–08:21:17 (`N:\work\stuff\Beta\llama-autoloader\logs\console.log`), during a client's 363 s GPU test that called `POST /v1/unload_all` ~20 s in. Reported from the AgentCascade side, see `AgentCascade/.bug_tracker/fixed/BUG_0152_failed_unload_reported_as_unloaded_and_reload_killed_mid_load.md`.
**Severity:** medium — no data loss and the caller gets what it asked for (cards freed, 200), but the loader records a recovery failure that never happened, arms a 5 s load-suppression cooldown, and emits three ERROR lines that read like a crash.
**Component:** `server.py` — `ModelManager._unload_unlocked` / `_unload_finish` (~:3041-3350) vs. the load's readiness wait outside `_lock` (`_load_model_serialized` :2392-2415, `_do_load_model` :2854-2866), and the recovery caller `_recover_dead_model_impl` (:2745-2758)

## Symptom

A client's `unload_all` lands while the loader is reading a model's weights. The kill succeeds and `unload_all` answers **200**, but the loader blames itself:

```
08:21:04,851 [INFO]  Launching strata for qwen3.8-flash-next-coder-iq3_xxs on port 9024
08:21:08,089 [INFO]  [strata] starting the engine: reading the model's weights ...
08:21:16,412 [INFO]  Unloading qwen3.8-flash-next-coder-iq3_xxs (port 9024)        <- client unload_all
08:21:16,816 [INFO]  POST /v1/unload_all HTTP/1.1 200 OK
08:21:17,402 [ERROR] llama-server process exited prematurely with code 15
08:21:17,402 [ERROR] Model qwen3.8-flash-next-coder-iq3_xxs failed to become ready
08:21:17,403 [ERROR] Dead-backend recovery reload of 'qwen3.8-flash-next-coder-iq3_xxs' failed: Model ... failed to start
08:21:17,403 [ERROR] Dead-backend recovery for '...' failed (attempt 1); suppressing retries for 5.0s: reload failed: Model ... failed to start
```

Nothing here is a failure from the loader's point of view: it was asked to abort the load, it aborted it. There is no crashed engine and no failed relaunch.

## Evidence

- The load registers the model and then waits for readiness **outside** `self._lock` (`server.py:2854-2858`: `# Wait until ready OUTSIDE the lock!` → `ready = await self._wait_until_ready(...)`), while `unload_all` (`server.py:3353-3363`) takes `self._lock` and unloads everything it sees. So the two are concurrent *by design* and neither side signals the other.
- The unload path never touches `_loading_tasks`; the only cancel of load tasks in the file is shutdown (`server.py:3900-3904`: `# Cancel any in-flight loading and dead-backend recovery tasks ...`). `_is_loading()` (`server.py:2334`) exists and is used for LRU eligibility (BUG_0011), but not by unload.
- The load therefore learns about the kill only through `proc.poll()` in `_wait_until_ready` (`server.py:2211-2217`), which reports it as `llama-server process exited prematurely with code 15` and returns `False`.
- `_do_load_model` turns that into a failed start: `Model ... failed to become ready`, a **second** `_unload_unlocked(model_id)` (`server.py:2860-2862`, a no-op — the entry was already popped by the client's unload), then `raise RuntimeError("... failed to start")`.
- The recovery caller catches any non-cancellation exception from `_load_model_serialized(reload_mid)` (`server.py:2747-2756`) and books `_record_recovery_failure(...)`, which logs `suppressing retries for 5.0s` (`server.py:2452-2453`). A deliberate unload is indistinguishable from a genuine relaunch failure at that catch site.
- Timing: kill at `08:21:16,412`, detection at `08:21:17,402` — 990 ms, i.e. up to two 0.5 s polls. During that window the load task is running against a process that no longer exists and the model is already detached from `self.loaded`.
- Caller-side view (AgentCascade): `POST /v1/unload_all` → 200, `unloaded_all` true, cards actually free — verified by the client's own residency probe. So the response is not wrong; only the loader's bookkeeping and log are.

## Root cause

"Cancel this load because a client wants the cards" and "this load failed" are the same code path. `_unload_unlocked` kills the process without cancelling the load task, so the task observes the death as an unexpected exit and reports a failed start; the recovery wrapper then records a reload failure and arms the BUG_0007 cooldown. The pieces needed to do it right already exist:

- `_do_load_model` already handles cancellation correctly (`server.py:2899-2903`): `except asyncio.CancelledError:` → `_unload_unlocked(model_id)` → `raise`. That is exactly the semantics of "an unload aborted my load".
- `_recover_dead_model_impl` already re-raises `CancelledError` before its generic handler (`server.py:2748-2749`), so a cancelled load would **not** be booked as a recovery failure or arm a cooldown.

## Fix (proposed direction)

1. In the unload path, cancel the in-flight load task before/with the kill — e.g. in `_unload_unlocked` (and `_unload_finish` for the detached path), after `_begin_unload_locked` marks the release: `task = self._loading_tasks.get(mid); if task and not task.done(): task.cancel()`. The existing `CancelledError` branch then does the teardown and no `failed to become ready` / `failed to start` is produced.
2. Suppress `llama-server process exited prematurely with code 15` when the exit was caused by a deliberate unload (pass a "release requested" flag into `_wait_until_ready`, or check the manager's release marker before logging at ERROR). Keep the line for genuine crashes.
3. Skip the redundant second `_unload_unlocked` when the entry is already gone (currently harmless but it is what makes the log read like a double failure).
4. Do not change the HTTP contract: `unload_all` returning 200 for "killed a load" is correct — the release happened.

## Verification (to be provided with the fix)

- New unit test: slow fake engine (readiness never reached), `load_model` in a task, call `unload_model(mid)` mid-load, then assert
  (a) the load task ends `CancelledError`, not `RuntimeError`;
  (b) `recovery_retry_after(mid)` is `None` (no cooldown armed);
  (c) no `Dead-backend recovery ... failed` / `failed to become ready` log records;
  (d) `unload_model` returns `unloaded=True` and `models_loaded == 0`.
- Re-run the existing mid-load tests (`TestEvictLRU` mid-load exclusion, BUG_0012 teardown tests) to confirm no behaviour change to eviction/shutdown cancellation.

## Related

- `BUG_0011_lru_eviction_result_dropped_and_mid_load_cascade.md` — first place mid-load state was used to gate a destructive path (eviction); this ticket asks for the same predicate in unload.
- `BUG_0007_failed_backend_reload_has_no_backoff.md` — the cooldown that gets armed spuriously here.
- `BUG_0014_failed_deliberate_unload_is_resurrected_by_dead_backend_recovery.md` — same incident, the other half.
- AgentCascade `BUG_0152` — the client-side report (a failed unload reported as "unloaded"), fixed there.

## Fix (applied)
Implemented in `server.py`. The approach deviates from the proposed direction: instead of cancelling the load task, the load's readiness wait is told the release was requested and raises a dedicated exception, so the existing correct cleanup paths are reused.
- `LoadAbortedByUnload(RuntimeError)` (server.py:~703): "a requested release", never a failed load.
- `_load_aborts` set + `_mark_load_aborted_locked` (server.py:~2382): marks an in-flight load aborted ONLY for an outside release — returns False for no-task / finished-task / the load task's own `current_task()`, so force-reload, failed-load and cancelled-load cleanup never self-mark.
- `_wait_until_ready(..., abort_probe=...)` (server.py:~2239): when the probe is true a dead process is logged at INFO ("exited after a requested unload") instead of the ERROR "exited prematurely".
- `_do_load_model` abort branch (server.py:~2957): on `not ready` + marker set, discards the marker, logs INFO "aborted by an unload", raises `LoadAbortedByUnload`, and skips the redundant second `_unload_unlocked` (entry already detached).
- `_recover_dead_model_impl` catches `LoadAbortedByUnload` (server.py:~2838) and does NOT book a recovery failure or arm the BUG_0007 cooldown.

## Verification (actual)
New tests in `tests/test_unload_load_race.py` (6, all pass; the fixture now calls `mgr.scan()` per the house pattern to register the seeded GGUFs — the earlier red was a test-fixture omission, not a product defect):
- `test_unload_during_load_ends_the_load_aborted` — load ends as `LoadAbortedByUnload`, model detached, no cooldown, no ERROR, INFO present.
- `test_abort_marker_is_only_set_for_an_outside_release` — predicate sound for all four cases.
- `test_requested_unload_exit_is_logged_as_info_not_error` / `test_genuine_crash_is_still_an_error` — requested exit at INFO, real crashes still ERROR.
- `test_recovery_does_not_book_an_aborted_load_as_a_failure` / `test_recovery_still_books_a_real_reload_failure` — control: a genuine reload failure still arms the cooldown.

Result: `pytest tests/test_unload_load_race.py` → 6 passed. Recovery regression files (`test_dead_backend_recovery.py`, `test_backend_liveness_backoff.py`) unchanged and passing. Independently reviewed to PASS.
