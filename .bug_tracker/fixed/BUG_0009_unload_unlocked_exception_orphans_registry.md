# BUG_0009 — `_unload_unlocked` orphans a possibly-live process from `self.loaded` when teardown raises

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20 (uncommitted, on top of `03614f1`)
**Found in:** Follow-up of the `_recover_dead_model_impl` fix (same defect class, second site); the first fix's lesson `.agent_lessons/unload-teardown-exception-orphans-registry.md` recorded this site as unfixed.
**Severity:** high
**Component:** `server.py:_unload_unlocked` (docstring + `except BaseException` handler, ~2640-2690)

## Symptom
Silent. After an aborted teardown the engine process may still be running while the
registry no longer contains the model: no request can be routed to it, LRU eviction
cannot see it, and nothing can retry the kill. No error line exists for the orphan
itself — only the propagated exception.

## Evidence
`_begin_unload_locked` pops the entry (`server.py:2590`), and the pre-fix handler
restored nothing:

```python
except BaseException:
    if deliberate:
        self._unload_in_progress.discard(model_id)
    raise
```

Mutation proof against the added regression test
(`tests/test_dead_backend_recovery.py::test_unload_unlocked_restores_registry_when_teardown_raises`):
- removing `self.loaded.setdefault(model_id, lm)` → `KeyError: 'dead.gguf'` at the `mgr.loaded[...] is old` assertion.
- removing `raise` → `UnboundLocalError: cannot access local variable 'killed'`; the test's `pytest.raises(RuntimeError, match="teardown exploded")` fails on the unmatched message.

## Root cause
The "a live process is always tracked in `self.loaded`" invariant is enforced **per call
site** rather than at the choke point. `_finalize_unload_locked` restores on a *normal*
failed kill (`killed == False`), which makes a surrounding `except BaseException: raise`
look safe by analogy — but the exception path bypasses `_finalize_unload_locked` entirely.
None of the 10 call sites restores on exception either.

## Fix
`server.py:_unload_unlocked`, in the `except BaseException as exc:` handler, after the
flag discard: probe the process, then restore only if it may still be running.

```python
if lm.process is not None:
    still_running = lm.process.poll() is None
else:
    still_running = bool(lm.pid and lm.pid > 0 and psutil.pid_exists(lm.pid))
if still_running:
    self.loaded.setdefault(model_id, lm)
    self._invalidate_status()
    log.error(f"Unload of {model_id} aborted by {type(exc).__name__}; "
              f"process {lm.pid} may still be running, model restored to registry")
else:
    log.error(f"Unload of {model_id} aborted by {type(exc).__name__}; "
              f"process {lm.pid} is gone, not re-registered")
raise
```

Why the probe and not an unconditional restore: `_unload_finish` itself re-raises
`asyncio.CancelledError` **after** the queue drop and a completed, verified kill
(`server.py:2867-2870`: "the queue is dropped, the process killed and the cleanup
done, so the unload itself succeeded"). An unconditional restore would re-register that
dead engine as a routable `ready=True` entry — the exact inversion of the invariant
this fix protects. The exception type cannot distinguish the two cases, because a
cancellation can land before the kill just as easily as after it; only the process can.

Differences from the `_recover_dead_model_impl` site, both deliberate:
- **No `async with self._lock`** — the contract is "caller must hold `self._lock`";
  re-acquiring a held asyncio lock would deadlock. Pinned by `asyncio.wait_for(..., timeout=2.0)`.
- **`idle_reaper` can retry here** — unlike the recovery path, which forces
  `lm.ready = False` before detaching, this site leaves `ready` untouched, so a restored
  entry stays eligible for a reaper-driven unload retry. (Not true of the load-path call
  site, which sets `ready = False` at `server.py:2514` before unloading.)
- `setdefault` (not assignment) for symmetry with `_finalize_unload_locked:2617` and
  `2384`; no force-kill here — that belongs to the retry/next-unload path.

## Verification
- `python -m pytest tests/ -q -x -n 0` → **325 passed** (324 before the second test was added).
- New tests pass in isolation and pin both directions:
  `test_unload_unlocked_restores_registry_when_teardown_raises` — real live child process
  (`subprocess.Popen` + `lm.process`), asserts the entry comes back, `ready` stays `True`,
  the status generation changes, the flag was armed during teardown and is cleared after,
  and no finalize ran.
  `test_unload_unlocked_does_not_resurrect_verified_dead_process` — a reaped child plus a
  `CancelledError` from `_unload_finish`, asserting the entry is **not** re-registered.
- `syntax_check server.py` → valid.

## Related
- [[unload-teardown-exception-orphans-registry]] — the defect class, the site table, accepted residuals.
- `fixed/BUG_0005_recovery_holds_manager_lock_across_teardown.md` — why the *recovery* path tears down outside the lock; unrelated defect, and the source of the tag confusion (this is BUG_0009, not BUG_0005).
- [[status-cache-stale-commit-race]] — why `_invalidate_status()` must accompany every `self.loaded` mutation.

## Residuals (deliberately out of scope here)
- **Structural fix not taken:** the cleaner shape is for `_unload_finish` to *return*
  the verdict (`killed, reason, cancelled`) instead of swallowing it, so both callers
  route the cancelled-after-kill case through `_finalize_unload_locked:2616-2627` — which
  also restores the FIFO queue. Rejected here because it changes `_unload_finish`'s
  contract and therefore requires editing `_recover_dead_model_impl`, which was out of
  scope. Worth doing as a follow-up.
- `unload_all` (`server.py:2907-2909`) has no per-model isolation, and `stop()` awaits
  `unload_all()` unguarded (`:3450`), so one aborted teardown skips
  `queue_registry.stop_all()` (`:3453`) and `client.aclose()` (`:3454`). Pre-existing,
  but the cancelled path this fix handles is exactly what fires during shutdown.
- The exception path does not re-create the FIFO queue entry `_unload_finish` may have
  dropped. Self-healing: `tcp_forwarder.py:749/:771` call `get_or_create` per request.
- `psutil.pid_exists` is a cheap OS probe performed under the lock on an exceptional
  path only; the `lm.process.poll()` branch is non-blocking.
