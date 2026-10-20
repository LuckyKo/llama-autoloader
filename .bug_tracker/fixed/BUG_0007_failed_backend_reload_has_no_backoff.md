# BUG_0007 — Failed dead-backend reloads have no backoff and retry on every request

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `1216df4`
**Found in:** Post-fix audit of BUG_0001 (`c31e08e`)
**Severity:** low-medium
**Component:** `server.py` (`RecoveryFailure`, `recover_dead_model`, `load_model`, `recovery_retry_after`), `tcp_forwarder.py` (`_recovery_retry_after`, `_send_recovery_error`)

## Symptom

When a dead engine cannot be reloaded (missing binary, bad engine config, failed readiness, GPU
allocation failure), `recover_dead_model` returns `None` and the recovery task is removed. The
next request starts a brand-new recovery attempt immediately.

Under a client retry loop this can produce:

- a full spawn/readiness attempt on every request,
- repeated multi-minute waits for a load that is known to fail,
- repeated error-log spam and process churn,
- many forwarder worker threads held for the same doomed reload,
- repeated 503s with no indication of when retrying is useful.

There is no per-model failure state, cooldown, or circuit breaker. The problem is most visible
when the failure is deterministic (bad command line / missing backend), where every request
repeats the same expensive failure.

## Root cause

`_recovery_tasks` deduplicates **in-flight** recoveries only. Once a task fails it is removed, so
there is no memory of the failure or backoff window.

## Fix direction

Add per-model recovery failure state:

1. Record `(model_id, generation, failure_time, reason, attempt_count)` after a failed reload.
2. For a cooldown window, fail fast on new recovery requests with 503 and a `Retry-After` header
   (or an equivalent structured error) instead of spawning again.
3. Reset the failure state on a successful load, a manual load/unload, or a configuration change
   that plausibly makes the reload viable.
4. Use exponential backoff with a cap for repeated failures; keep the counters observable
   (`/v1/status` or logs) so an operator can see that recovery is being suppressed.
5. Keep the cooldown decision on the main loop and deduplicated per model/generation so a burst
   of requests does not race into multiple attempts.

Do not suppress forever: a manual “load now”/force action should bypass the cooldown.

## Fix implemented

- Added `RecoveryFailure` state keyed by resolved model id:
  `(model_id, dead_port, generation, attempts, reason, failed_at, retry_at)`.
- `_record_recovery_failure()` arms exponential backoff
  (`AUTOLOADER_RECOVERY_BACKOFF_BASE`, default 5 s; doubled per consecutive failure with
  `AUTOLOADER_RECOVERY_BACKOFF_MAX`, default 300 s).
- `recover_dead_model()` checks the cooldown on the main loop **before** creating a task, and
  closes the check/creation race under `manager._lock`. Non-forced calls return `None` while a
  cooldown is active; `force=True` bypasses it.
- `load_model()` enforces the same cooldown for ordinary JIT loads. It re-checks after acquiring
  the per-model lock, so a post-cooldown burst cannot race into multiple doomed loads.
- A failed JIT load while a prior recovery failure exists extends the backoff instead of starting
  a fresh chain.
- `recovery_retry_after()` is a lock-free read for forwarder worker threads. The forwarder adds a
  standard `Retry-After` header (rounded up) to recovery/JIT 503 responses. A model still inside
  its cooldown gets an immediate 503 without calling `load_model`.
- Reset points: successful load (manual/JIT/recovery), successful manual unload (including
  unload-all), explicit force load, and model configuration update.
- Observability: `/v1/status` now includes `recovery_failures` with attempts, reason, port,
  generation, and remaining seconds; each failure also logs at ERROR with the delay.

Tests: `tests/test_backend_liveness_backoff.py`
- failed recovery arms cooldown and suppresses the next call;
- forced recovery bypasses and grows the backoff;
- successful recovery clears state;
- JIT load respects the cooldown while manual load bypasses it;
- an expired failure is extended by another failed JIT reload;
- a concurrent post-cooldown burst makes exactly one load attempt;
- status snapshot exposes the failure;
- forwarder recovery 503 carries `Retry-After`.

## Verification

- Fake `load_model` failure; sequential and concurrent requests during the cooldown make exactly
  one load attempt and receive 503 with `Retry-After`.
- After the cooldown expires (or with a forced/manual load) one new attempt is made.
- A successful reload clears the failure state.
- Full suite: 389 passed (one pre-existing Windows `WinError 10055` flake passed when rerun).

## Related

- `BUG_0001_tcp_forwarder_no_dead_backend_recovery.md` — recovery implementation
- `BUG_0005_recovery_holds_manager_lock_across_teardown.md` — blocked worker threads amplify the
  cost of repeated failed reloads
- `BUG_0006_dead_backend_detection_ignores_send_failures_and_timeouts.md` — additional failure
  paths that feed recovery attempts
