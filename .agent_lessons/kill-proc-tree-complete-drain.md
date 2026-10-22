---
tags: [process-tree, psutil, kill-verification, windows, bug-0012]
aliases: [kill-tree-drain, tree-complete-unload]
related: [[unload-teardown-exception-orphans-registry]], [[asyncio-lock-blocking-io-starvation]], [[best-effort-except-masks-crash-triage-fallback-warning]]
confidence: verified
author: coder_coder_bug0012_tree_drain_20261022_123621.jsonl
---

# BUG_0012: `_kill_proc` is tree-complete (was parent-only since a12bab7)

At `N:\work\stuff\Beta\llama-autoloader` @ `a12bab7` (fix applied on top): `_kill_proc` (nested in
`ModelManager._unload_finish`) now verifies the **entire engine process tree** is gone before
returning `unloaded=True`. Defect lineage: BUG_0008 (`03614f1`, verify-every-target) → narrowed to
parent-only by `a12bab7` ("verify parent only, not children") → BUG_0012. The parent-only check was
correct for the tracked pid but reported success while orphaned engine grandchildren (Strata engine
binary + vision encoder) still held ~40 GB of committed memory; the immediate state-restore auto-load
then died on the Windows commit limit (`VirtualAlloc error 1314`).

## What the fix does

After the existing parent-verify block confirms the parent is gone, `_kill_proc`:
1. `tree_pids = {p.pid for p in targets}` (the pre-kill snapshot);
2. calls module-level `_tree_survivors(tree_pids)`;
3. if survivors: logs WARNING with **pid + name + ppid** (closes the forensics gap — the old
   note logged pids only), terminates each, bounded-waits (`KILL_TREE_DRAIN_TIMEOUT_S = 15.0`,
   poll 0.25 s), **escalating to `kill()`** on the first re-scan that still finds survivors;
4. if any remain: logs ERROR and returns `False, "tree drain timed out: pids ..."`.

No changes to `_require_engine_gone`, `_unload_finish`, `load_state`, or the `_unload_unlocked`
call sites; no new guards/flags/locks; the `primary_error`/fallback structure is preserved.

## Non-obvious points worth remembering

- **`_tree_survivors` routes every candidate through a local `_resolve_live()` gate** that calls
  `psutil.Process(pid)` and checks `is_running() and status() != ZOMBIE`. This is what keeps the
  five existing `tests/test_unload_reporting.py` mock tests green: they patch `server.psutil.Process`
  (to a gone-parent mock) but NOT `psutil.process_iter`. Because the scan re-resolves through the
  patched `psutil.Process`, `_resolve_live` returns `None` for the fake pids (424242/27344) and the
  new scan is inert for them. If you ever bypass `_resolve_live` and use raw `process_iter` objects,
  those mock tests become non-deterministic (a real process with `ppid == 424242` would be killed).
- **Two passes are both required.** BFS via `children(recursive=False)` reaches descendants of any
  *live* tree pid; the `process_iter(["pid","ppid"])` pass matching `ppid in tree_pids` recovers
  direct orphans, which are unreachable via `Process(parent).children()` once the parent is dead
  (psutil resolves descendants through a ppid map rooted at the parent). On **Windows the recorded
  ppid stays sticky** after the parent exits — that is what makes the process-table pass work; on
  POSIX orphans reparent to init but were already in the pre-kill snapshot.
- **Known limitation (acceptable):** an orphan whose entire ppid chain passes only through DEAD
  intermediates that are NOT themselves in `tree_pids` is not reachable. In the real incident the
  survivors' ppid = the tracked wrapper (a snapshot pid), so the single process-table pass sufficed.
  Also out of scope: a wrapper that self-exits *before* `_kill_proc` starts returns success with no
  snapshot to drain (documented in-code).
- **The tree-drain must escalate to `kill()`, not just `terminate()`** — a descendant that ignores
  SIGTERM but accepts SIGKILL would otherwise stall the full 15 s window and still hold memory.
  (For the incident's `AccessDenied` descendant, `kill()` is also denied → correctly times out to a
  failed unload.)
- **Pid-recycling guard via `create_time`.** `_resolve_live` constructs a fresh `psutil.Process(pid)`,
  and `is_running()` on a fresh object is True for ANY live pid — including one the OS reused for an
  unrelated process after ours exited. `_kill_proc` therefore captures `expected_ctime = {p.pid: p.create_time()}`
  from the snapshot objects and `_resolve_live` rejects a pid whose current `create_time()` differs
  (treat as gone). Without it, a recycled tree pid makes the BFS descend into — and terminate — an
  unrelated subtree. The orphan pass carries the same guard: it accepts a candidate only when nothing
  live occupies its parent pid (a live process at a tree pid is either the original, already covered by
  BFS, or a recycled unrelated one). Note the comparison must be tolerant (`abs(diff) > 0.1`) and
  guarded so mock-based tests (MagicMock `create_time`) are unaffected — the isinstance(float) check
  keeps mocks passing through unguarded.

## Tests

`tests/test_kill_tree_drain.py`: `test_tree_drain_waits_for_grandchild` (real 3-tier tree, child's
terminate/kill forced to raise AccessDenied, then self-exits; asserts WARNING fires with `ppid=`,
success only after child+grandchild gone — note the descendants are reached via the pre-kill snapshot
+ BFS, NOT the process-table pass) and `test_tree_drain_times_out_reports_failure` (stub
`_tree_survivors` + shrink `KILL_TREE_DRAIN_TIMEOUT_S` to 0.5 s → `unloaded is False`). The
process-table (orphan) pass and the pid-recycling guard are covered by `test_tree_survivors_orphan_pass_and_recycling_guard`
(a deterministic unit test stubbing `psutil.process_iter`/`psutil.Process` directly). Full suite
`python -m pytest tests/ test_server.py -q -n 0` = **401 passed** (baseline 398 + 3).
