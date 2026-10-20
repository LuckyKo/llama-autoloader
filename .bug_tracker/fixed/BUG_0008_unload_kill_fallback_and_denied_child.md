# BUG_0008 — Unload kill aborts on one denied child and the Windows fallback uses a nonexistent `os.CTRL_C_EVENT`

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `03614f1`
**Found in:** Production log 2026-10-20 13:03:08 (`/v1/unload_all` failed during the save → unload → heavy-memory → restore sequence)
**Severity:** high
**Component:** `server.py` (`ModelManager._unload_finish` → nested `_kill_proc`)

## Symptom

An explicit `/v1/unload_all` failed immediately:

```
13:03:08,918 [WARNING] Process termination failed for strata-coder-iq3_xxs (pid 22796): (pid=27344)
13:03:08,922 [ERROR]   ... process 22796 is still running
                        (fallback error: module 'os' has no attribute 'CTRL_C_EVENT')
13:03:08,922 [ERROR] Unload of strata-coder-iq3_xxs failed: ...
13:03:08,923 [INFO]  POST /v1/unload_all HTTP/1.1 500 Internal Server Error
```

The caller's intended sequence was **save state → unload the model → run heavy memory
operations → restore state**. Because the unload failed, the loader kept the model tracked while
its engine was already going away. The subsequent `state/load` then tried to restore into the dead
port 9022 and returned 500 (`All connection attempts failed`); the next chat request found the port
refusing connections and triggered the normal dead-backend recovery.

## Root cause

Two defects in the same termination path:

1. `psutil.Process(lm.pid).children(recursive=True)` or one child's `terminate()` raised
   `psutil.AccessDenied` for child pid 27344. The whole `_kill_proc()` body was wrapped in one
   `try/except`, so that single protected descendant aborted termination before the engine parent
   (22796) was ever terminated.
2. The fallback used `os.kill(lm.pid, os.CTRL_C_EVENT)`. `CTRL_C_EVENT` lives in `signal`, not
   `os`, so the fallback raised `AttributeError` before reaching `lm.process.kill()`. It could
   never work on any platform.

The result was a guaranteed unload failure whenever one descendant was inaccessible, even though
the engine parent itself was killable.

## Fix (implemented)

- Process enumeration is now best-effort: if recursive child enumeration fails, the code retries
  with direct children and continues instead of aborting the whole teardown.
- Each child `terminate()` / `kill()` is wrapped individually, so one `AccessDenied` /
  `ZombieProcess` / transient psutil error does not skip the engine parent.
- `psutil.wait_procs` failures are treated as "still alive" and proceed to the fallback.
- The fallback now uses `lm.process.kill()` (`TerminateProcess`) on Windows. It no longer references
  `os.CTRL_C_EVENT`. On Unix the process-group `SIGKILL` remains the first attempt, with
  `lm.process.kill()` as the backstop.
- Verification checks every known target (parent and accessible children), not just the parent.
  "Still running" or "cannot verify" remains a failed unload, preserving the BUG_0102 contract.
- If psutil cannot even open the parent, the fallback still kills by Popen handle and then verifies
  with a fresh psutil handle.

## Verification

`tests/test_unload_reporting.py`:
- `test_child_access_denied_does_not_skip_parent_termination` — one child raises `AccessDenied`;
  the parent must still be terminated and the unload must succeed at the process-kill stage.
- `test_windows_fallback_uses_popen_kill_not_missing_os_ctrl_c` — primary psutil attempts fail;
  the Windows fallback must call `Popen.kill()` and not `os.CTRL_C_EVENT`.
- Existing BUG_0102 tests still verify that a genuinely immortal process returns 500 and stays
  tracked.

- Full suite: **391 passed**.
- The production sequence now has a working unload step; a caller should still treat a
  non-200 `/v1/unload_all` as "do not start heavy-memory operations", but the loader no longer
  fails merely because one descendant is inaccessible.

## Related

- `BUG_0102` (external tracker) — unload must not report success when the kill failed
- `BUG_0005_recovery_holds_manager_lock_across_teardown.md` — teardown/locking context
- `BUG_0001_tcp_forwarder_no_dead_backend_recovery.md` — recovery that ran after the failed unload
