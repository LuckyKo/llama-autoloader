---
tags: [unload, registry-invariant, exception-path, recovery]
aliases: [loaded-registry-orphan-on-teardown-exception]
related: [[status-cache-stale-commit-race]], [[raw-tcp-forwarder-architecture]]
confidence: verified
author: coder_coder-llama-fix_20261020_135051.jsonl
contributors:
  - reviewer_review-unload-orphan_20261020_144234.jsonl
---

# An exception inside `_unload_finish` orphans a live process out of `self.loaded`

The unload protocol is split in two: `_begin_unload_locked` **pops** the model from
`self.loaded`, then the slow drain/kill (`_unload_finish`) runs **outside** `self._lock`
(see [[status-cache-stale-commit-race]] for why status must be invalidated on every
`self.loaded` mutation, and `BUG_0005` in `.bug_tracker/fixed/` for why the slow part
must not hold the global lock).

The invariant this protects: **a running engine process must always be tracked in
`self.loaded`** — routing, the idle-reaper and LRU eviction all key off that dict.

The trap: `_finalize_unload_locked` restores the model via
`self.loaded.setdefault(model_id, lm)` on a *normal* failed kill (`killed == False`),
so a `try/except BaseException: ... raise` around `_unload_finish` looks safe by
analogy. It is not — the exception path bypasses `_finalize_unload_locked` entirely,
and nothing re-inserts the popped model.

## Sites

At `llama-autoloader` @ `03614f1` (pre-fix tree; fix applied uncommitted on top):

| call site | shape | restored on exception? |
|---|---|---|
| `_recover_dead_model_impl` (`server.py` ~2364-2375) | pop under lock → `_unload_finish` outside lock | **was NO → fixed** |
| `_unload_unlocked` (`server.py` ~2652-2672) | pop + `_unload_finish` both under caller's lock | **was NO → fixed (BUG_0009)** |

Fix for the first: in the `except BaseException as exc:` handler, after the
`_unload_in_progress.discard(...)`, do `async with self._lock: self.loaded.setdefault(reload_mid, detached_lm); self._invalidate_status()`, log an ERROR, re-raise. `setdefault` (not assignment) so a concurrent re-insertion is not clobbered. No force-kill here — that belongs to the normal unload/retry paths.

## `_unload_unlocked` — fixed as BUG_0009

Its docstring used to assert "On a failed kill the model is put back into `self.loaded`",
but that only covered the *normal* failed-kill path through `_finalize_unload_locked`. Its
`except BaseException: ... discard ... raise` had no restore, and none of its **10**
call sites restored on exception either — they all just propagated: LRU auto-evict,
`_do_load_model`'s pre-reload eviction, the failed-start rollback, the
`asyncio.CancelledError` rollback, the `_unload_model` wrapper, the two recovery paths
(`_recover_*` via `_model_lock`), `unload_all`, `idle_reaper`, and the interrupt/hard-stop
path — note that last one calls `manager._unload_unlocked(...)`, not `self.`, so a
`self\._unload_unlocked` grep **misses it**. Verify with a bare `_unload_unlocked` grep.

The fix is the same handler shape **minus the lock re-acquire**: every call site holds
`self._lock` by contract (`async with self._lock`), so the restore runs directly — wrapping
it in `async with self._lock` would deadlock on a non-reentrant asyncio lock, and the
regression test pins that with `asyncio.wait_for(..., timeout=2.0)`. A side benefit of
holding the caller's lock: the `_unload_in_progress.discard(...)` above the restore is inside
the lock too, so the sub-millisecond "neither flag nor registry marks the model" window
listed under residuals does **not** apply at this site.

**The restore at this site must be conditional on the process being alive.** `_unload_finish`
itself re-raises `asyncio.CancelledError` at `server.py:2867-2870` *after* the queue drop
and a completed, verified kill ("the queue is dropped, the process killed and the cleanup
done, so the unload itself succeeded"). An unconditional restore would re-register that
dead engine as a routable `ready=True` entry — the inverse invariant. **The exception type
cannot discriminate**: a cancellation lands before the kill just as easily as after it. Probe
`lm.process.poll() is None` (non-blocking, authoritative for our own child) and fall back to
`psutil.pid_exists(lm.pid)` for fake/legacy entries with `process=None`. The structural
alternative — have `_unload_finish` return `(killed, reason, cancelled)` so both callers
route the cancelled-after-kill case through `_finalize_unload_locked`, which also restores
the FIFO queue — is the better long-term shape, but it changes a contract shared with
`_recover_dead_model_impl`.



Root cause framing: the invariant is enforced **per call site** rather than at the
choke point (`_begin_unload_locked` is the single place that pops). The fix at
`_recover_dead_model_impl` is a second patch around that, not a structural guard.
A structural guard would mean the pop itself recording enough state to roll back.

## Gotcha worth remembering

`sys` is **not** imported in `server.py` (whole-file grep for `\bsys\.` → zero hits).
Do not write `sys.exc_info()` in log lines here; bind the exception
(`except BaseException as exc:`) and use `type(exc).__name__`.

## Verification

Full suite `python -m pytest tests/ -q -x -n 0` → 323 passed post-fix (322 passed + 1 setup `ERROR` on one run: the documented `WinError 10055` / event-loop self-pipe flake in [[windows-proactor-sock-test-error-transient]] — that test passes in isolation; ignore it, do not "fix" it).

Coverage added: `tests/test_dead_backend_recovery.py::test_recover_dead_model_restores_registry_when_teardown_raises` — stubs `_unload_finish` to `raise RuntimeError`, then asserts `recover_dead_model` returns `None`, `mgr.loaded[model_id] is old`, `old.ready is False`, the flag was armed during teardown (`flag_armed == [True]`, so the later absence assertion is not vacuous), no reload and no finalize were reached, and `mgr._recovery_failures[mid].reason` contains the original message. Why the bug stayed latent: **no** test in that file made `_unload_finish` raise — every stub returned normally, so only the ordinary failed-kill path (which does restore via `_finalize_unload_locked`) was ever exercised.

Both pins are **mutation-proven**, not inferred:
- commenting out the `setdefault` → `KeyError: 'dead.gguf'`.
- commenting out the `raise` → the reason assertion fails with `"eviction failed: cannot access local variable 'killed' ..."`. Note the trap: `finalizes == []` does **not** catch a missing re-raise, because Python evaluates the arguments to `_finalize_unload_locked` before entering the stub, so the `UnboundLocalError` fires while building them and the stub is never called. The `_recovery_failures[...].reason` assertion is what actually pins propagation.

Coverage added for the second site — two tests, one per direction:
- `test_unload_unlocked_restores_registry_when_teardown_raises` — holds `mgr._lock` (the contract), attaches a **real live child** to `lm.process` (`subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])`) so the liveness probe is exercised rather than stubbed, stubs `_unload_finish` to `raise RuntimeError`, and asserts the entry comes back, `old.ready is True`, `mgr._status_gen` changed, the flag was armed during teardown and is cleared after, and no finalize ran. Mutation-proven: removing `setdefault` → `KeyError: 'dead.gguf'`; removing `raise` → `UnboundLocalError: cannot access local variable 'killed'`, which `pytest.raises(RuntimeError, match="teardown exploded")` rejects on the unmatched message.
- `test_unload_unlocked_does_not_resurrect_verified_dead_process` — a **reaped** child plus `raise asyncio.CancelledError` from `_unload_finish`, asserting the id is absent from `self.loaded`. This is what pins the liveness conditional; a test with only the first case would pass with the buggy unconditional restore.

Note the trap when writing these: a synthetic `pid=424242, process=None` fixture makes the liveness probe return False and the restore silently vanish, so the first test looks broken when the code is right. Give the fixture a real process. Full suite: 325 passed.

## Correction to the usual reasoning

The entry restored by the **recovery** path is **always `ready=False`** (`lm.ready = False` is set before the detach). `idle_reaper` skips not-ready entries (`if lm.config.pinned or not lm.ready or lm.active_requests > 0: continue`), so the reaper is **not** a reclaim path for *that* restored entry. Only routing, LRU eviction (no `ready` filter) and the next explicit recovery/unload pass see it. Do not write "visible to the idle reaper" as a justification for the recovery-path fix.

This does **not** generalise to `_unload_unlocked`: nothing there touches `lm.ready`, so a model restored from an aborted deliberate unload is still `ready=True` and the reaper *can* retry the unload. Check the mutation history of `ready` per site, not the shape of the handler.

## Known accepted residuals (not fixed, minimal-change constraint)

- `setdefault` silently keeps a pre-existing live entry if a concurrent load re-registered the id first; the detached process is then still orphaned. Same behaviour as `_finalize_unload_locked`.
- `_unload_in_progress.discard(...)` runs **outside** the lock (every other mutation of that set is under it), leaving a sub-millisecond window where neither the flag nor the registry marks the model. Clearing it inside the lock risks a leaked flag on a cancelled acquire, which would refuse every later restore — so the order is deliberate.
- The restore does not re-create the FIFO queue entry that `_unload_finish` may already have removed (`_finalize_unload_locked` does). Self-healing via `get_or_create` — **verified true**: the raw-TCP hot path calls `queue_registry.get_or_create` on every LLM request (`tcp_forwarder.py`, in `_resolve_target_port`'s already-loaded branch and the post-load branch), so the next request recreates the queue.

## Addendum 2026-10-20 (review pass, uncommitted tree on top of `03614f1`)

**The BUG_0009 restore has a wrong-direction case that the table above does not cover.** `_unload_finish` deliberately re-raises **after** the destructive action: it catches the drain's `CancelledError` (`except asyncio.CancelledError: cancelled = True`), drops the queue, runs `_kill_proc`, and only then does `if cancelled: raise asyncio.CancelledError` with the comment "the queue is dropped, the process killed and the cleanup done, so the unload itself succeeded". A surrounding `except BaseException: self.loaded.setdefault(...)` cannot distinguish that from an abort *before* the kill, so it re-registers a **verifiably dead** process. Measured (real subprocess, cancelled mid-drain):

```
PROPAGATED: CancelledError
IN self.loaded AFTER cancelled unload: True
restored entry .ready = True      # and no "Process termination failed" line => killed was True
restored entry .pid    = 27988
is the engine process actually dead? True
```

This is the **dominant** trigger, not a theoretical edge: `_drain_inflight_slot_action` can only raise `CancelledError`; `ModelQueueRegistry.remove` is effectively unraiseable; `_kill_proc` guards every failure internally and returns `(False, reason)` instead of raising. So the realistic way the handler is reached is exactly the post-kill path. Pre-fix, that path was *correct* (a dead process was untracked).

**Fix direction:** do not branch inside the exception handler. Have `_unload_finish` return the verdict (e.g. `(killed, reason, cancelled)`) and let `_unload_unlocked` re-raise *after* `_finalize_unload_locked`, which already decides both outcomes **and** restores the FIFO queue.

**Blast radius:** `unload_all` builds its results in a list comprehension with no per-model isolation and `/v1/unload_all` has no handler, so one aborted teardown aborts the remaining unloads. `ModelManager.stop()` calls `await self.unload_all()` unguarded, so the following `queue_registry.stop_all()` and `await self.client.aclose()` are skipped — engine processes leak at shutdown.

**Corrections to this entry:**
- The "idle_reaper can also retry" property is **per call site, not per handler**. At `_do_load_model`'s failed-start and `asyncio.CancelledError` rollback sites the load path has already assigned `lm.ready = False` (`lm.ready = ready` in the same function), so the restored entry is *not* reaper-eligible. Whole-file grep: `.ready` is assigned in exactly two places — the recovery path (`lm.ready = False` before the detach) and the load path.
- `setdefault`'s "a concurrent re-registration wins" rationale is unreachable at the `_unload_unlocked` site: every writer of `self.loaded` runs under `self._lock`, and `note_request_start` only *reads* it from worker threads. The real reason is symmetry with the sibling restores.
- Test-strength claim is understated above: the BUG_0009 test uses `pytest.raises(RuntimeError, match="teardown exploded")`, so removing the `raise` makes it **fail correctly** (the `UnboundLocalError` message does not match). It is a stronger propagation pin than "caught by pytest.raises" suggests. Neither test pins the `_invalidate_status()` companion (`_status_gen` is a plain int — a before/after assert is trivial and non-flaky).
- Suite counts here are stale: current tree collects and passes **324** (`python -m pytest tests/ -q -n 0` → `324 passed in 103.12s`), not 323. `fixed/BUG_0008_*.md` separately claims 391 — tracker counts are not mutually consistent.


