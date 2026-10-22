# BUG_0012 — Unload reports success while engine grandchildren still run; the immediate state-restore reload dies on the commit limit

**Status:** FIXED
**Filed:** 2026-10-22
**Found in:** Autoloader log excerpt pasted in `N:\work\WD\AgentWorkspace\todo.md` lines 109-203 (incident 2026-10-22 10:45:20-10:46:11); engine log `N:\work\WD\Strata\strata-coder-iq3_xxs.log`
**Severity:** high
**Component:** `server.py` (`ModelManager._unload_finish` → nested `_kill_proc`, teardown + lingering-children check) — interaction with the `load_state` auto-load-before-restore path

## Symptom

Safe-mode save→unload→restore sequence: `unload_all` returned **200**, but three child processes
survived the teardown holding the engine's committed memory. The state-restore path auto-loaded
the model ~0.5 s later; the new engine could not reserve its expert arena and died (exit 1);
`state/load` returned **503** and the KV restore was lost. The next chat request JIT-loaded the
model successfully — i.e. the failure was transient, caused by the overlap.

Verbatim (todo.md lines 118-123, 174-182):
```
10:45:20,526 Unloading strata-coder-iq3_xxs (port 9022)
10:45:20,550 [WARNING] Process termination needed a fallback for strata-coder-iq3_xxs (pid 1128): (pid=7908)
10:45:20,563 [WARNING] Child process(es) 11056, 7908, 26632 still running after unload of
          strata-coder-iq3_xxs (parent 1128 gone); they will be reaped by the OS
10:45:20,564 POST /v1/unload_all HTTP/1.1 200 OK
10:45:21,023 Auto-loading model 'strata-coder-iq3_xxs' before state restore
...
10:45:36,078 ArenaExpertSource: the arena could not be reserved (42915302400 B): large pages
          refused for 42916118528 B (GetLargePageMinimum=2097152, VirtualAlloc error 1314); using 4 KB pages
10:45:36,074 EngineDied: the engine exited before it was ready (exit code 1)
10:45:38,618 Failed to load model 'strata-coder-iq3_xxs' before state restore
10:45:38,618 POST /v1/models/qwen3.8-flash-next-coder-iq3_xxs/state/load HTTP/1.1 503 Service Unavailable
10:45:38,699 Model 'qwen3.8-flash-next-coder-iq3_xxs' not loaded; triggering JIT load...  (succeeds)
```

## Evidence

- The 200→auto-load gap is **459 ms** — the by-design safe-mode restore timing (see memory
  `unload-in-progress-flag-replaces-recent-unload-window`: restore "lands ~0.5-1 s after
  unload_all returned 200").
- `_kill_proc` (at `a12bab7`, `server.py` ~:2840-2950): terminates enumerated children, then the
  parent; two `wait_procs(timeout=3)` rounds; parent-only force-kill fallback; then a
  **log-only** lingering-children note (`still running after unload ... reaped by the OS`)
  followed by `return True, ""`. The warning line itself proves three processes were alive at
  the moment success was reported.
- `VirtualAlloc error 1314` = `ERROR_COMMITMENT_MINIMUM` — Windows **commit-limit** failure, not
  physical RAM. Corroborating chronic tightness from the same engine log (healthy run, tail
  ~:45520): `Windows commit: 4.5 GiB left of a 99.1 GiB limit (RAM + page file)` while the run
  plans 6.0 GiB of host allocations.
- Second load attempt ~18 s later succeeded (experts loaded 39.97 GiB at 2.40 GiB/s) → the
  orphans had been reaped → transient race, not a permanent condition.
- Process tree: the tracked pid is the Strata **wrapper** (`serve/server.py` @ `94f55cd6` spawns
  the engine binary via `popen("the Strata engine", ...)` ~:997 and a separate vision-encoder
  subprocess ~:2555) — engine processes are **grandchildren** of the tracked pid. (Individual
  orphan pids not yet traced — under investigation.)

## Root cause

Diagnosed at mechanism level: the engine-gone contract (`_require_engine_gone`, the
`UnloadResult.unloaded` contract — "a running engine process must always be tracked",
"never report success when the process is still running") is verified only against the
**tracked pid + its recursively-enumerated children snapshot taken before the kill**. When the
wrapper exits first, the engine grandchildren are orphaned: they are not in the kill targets,
the lingering check only logs them, and `_kill_proc` returns success. The immediate
auto-load-before-restore then spawns a replacement engine while the old engine's commit is
still live → arena reservation fails.

Open sub-questions (investigation delegated, 2026-10-22): why the graceful path failed
(`primary_error (pid=7908)` — which target, which exception); what the three pids actually
were; recurrence count of the lingering warning in `logs/console.log*`; whether BUG_0008's
"still running => failed unload" contract was ever intended to cover post-parent orphans.

## Fix

Implemented at HEAD `a12bab7` (server.py working tree; the change is on top of it). `_kill_proc`
is now **tree-complete**.

**`server.py` — `ModelManager._kill_proc` (the post-parent-kill phase):** the old log-only
lingering-children note ("... reaped by the OS") followed by `return True, ""` was replaced.
After the existing parent-verification block confirms the parent is gone, `_kill_proc` now:
1. builds `tree_pids = {p.pid for p in targets}` (the pre-kill snapshot: tracked parent + its
   children) and `expected_ctime = {p.pid: p.create_time() for p in targets}` (best-effort);
2. calls the new module-level helper `_tree_survivors(tree_pids, expected_ctime)`, which re-enumerates
   the whole tree — BFS down via `psutil.Process(pid).children(recursive=False)` from any live
   tree pid, PLUS one `psutil.process_iter(["pid","ppid"])` pass matching `ppid in tree_pids`
   to recover the direct orphans (unreachable via `Process(parent).children()` once the parent
   is dead; on Windows the recorded ppid stays sticky, which is what makes this pass work).
   Every candidate is re-resolved through a local `_resolve_live()` gate (`is_running()` and
   `status() != ZOMBIE`), so a test that patches `psutil.Process` controls the scan and the
   existing mock-based kill tests stay deterministic. The gate also enforces a **pid-recycling
   guard**: a candidate whose pid is in `expected_ctime` but whose current `create_time()` differs
   is an unrelated process that reused the pid, so it is treated as gone — otherwise the BFS would
   descend into (and terminate) an unrelated subtree. The orphan pass carries the same guard: a
   candidate is accepted only when nothing live occupies its parent pid.
3. if survivors are found: logs a WARNING listing **pid + name + ppid** (fixes the forensics
   gap — the old note logged pids only), terminates each per-process, then bounded-waits (poll
   every 0.25 s up to `KILL_TREE_DRAIN_TIMEOUT_S`), escalating to a force-`kill()` on the first
   re-scan that still finds survivors — mirroring the primary path's terminate → wait → kill →
   wait so a SIGTERM-ignoring descendant is still stopped;
4. if any remain after the window: logs an ERROR with the surviving pids and returns
   `False, "tree drain timed out: pids ... still running"`.

`return True, ""` now means the **entire engine process tree is verifiably gone**, making the
`_require_engine_gone` / `UnloadResult.unloaded` contract truthful.

**New module-level helpers** `_describe_proc` and `_tree_survivors` (above `class ModelManager`),
and one new constant `KILL_TREE_DRAIN_TIMEOUT_S = 15.0` (next to `SLOT_ACTION_DRAIN_TIMEOUT_S`).

No changes elsewhere: `_require_engine_gone`, `_unload_finish`, `load_state`, and all `_unload_unlocked`
call sites are untouched and now receive a truthful verdict. No new guards, flags, or locks. The
`primary_error`/fallback structure is preserved — this is an extension of the post-parent-kill
phase, not a rewrite. Cross-platform: the process table works via psutil on both Windows and POSIX.

Out-of-scope residual (documented in-code, not fixed): a wrapper that self-exits *before* `_kill_proc`
starts returns success without a drain (no snapshot to reach); this fix targets a parent that dies
*during* the call. The independent page-file hardening (commit limit vs ~64 GB RAM) remains a
separate concern.

## Verification

New regression tests in `tests/test_kill_tree_drain.py` (BUG_0012), mirroring the real-subprocess
style of `tests/test_unload_reporting.py` / `tests/test_dead_backend_recovery.py`:

- `test_tree_drain_waits_for_grandchild` — drives a REAL 3-tier process tree (tracked parent →
  child → grandchild) through the real `_kill_proc` / `_tree_survivors` path. The child's
  `terminate`/`kill` are forced to raise `psutil.AccessDenied` (the incident's signature: a
  descendant whose termination is denied and survives the primary path), then the child exits on
  its own after a short delay. Asserts: (a) the tree-drain WARNING fires while the descendant is
  still alive (success is NOT reported early), (b) the WARNING includes `ppid=` (the forensics
  gap), (c) `unloaded is True` only after BOTH the child and the real grandchild are gone.
  (The primary path's enumeration puts both descendants in the pre-kill snapshot, so the drain
  reaches them via the BFS from the live child; this test proves the drain WAITS for a real
  descendant and only then reports success.)
- `test_tree_drain_times_out_reports_failure` — the tree scan is stubbed to a persistent survivor
  that never dies and `KILL_TREE_DRAIN_TIMEOUT_S` is shortened to 0.5 s; asserts `unloaded is False`,
  `reason` contains `tree drain timed out` + the surviving pid, the model stays tracked, and an
  ERROR is logged with the pid.
- `test_tree_survivors_orphan_pass_and_recycling_guard` — a deterministic unit test that stubs
  `psutil.process_iter`/`psutil.Process` directly (no real subprocesses) to exercise the two paths
  test 1 does NOT reach: (A) the process-table pass recovers an orphan whose parent is a dead tree
  pid (BFS cannot descend from a dead pid), and (B) the pid-recycling guard rejects a tree pid that
  the OS reused for an unrelated live process (`create_time` mismatch) — neither the BFS nor the
  orphan pass descends into it or its child.

All fail against the pre-fix code (no `_tree_survivors`; success reported with live descendants).

Full suite `python -m pytest tests/ test_server.py -q -n 0`: **401 passed** (baseline at `a12bab7`
was 398; +3 new, zero regressions). The five existing `test_unload_reporting.py` BUG_0008 pins
(child AccessDenied does not skip the parent; Windows fallback uses `Popen.kill`; immortal process
returns a failed unload) remain green — the `_resolve_live` gate through the patched `psutil.Process`
keeps the new scan inert for them.

## Related

- Memory: `.agent_lessons/strata-reload-race-lingering-children-commit-limit.md` (full analysis + defect lineage)
- BUG_0008 (fixed) — same `_kill_proc`, best-effort enumeration + Windows fallback; its
  "still running => failed unload" contract is what this bug exposes as incomplete for orphans
- BUG_0009 / BUG_0010 (fixed) — registry orphan on teardown exception
- BUG_0011 (fixed) — LRU eviction result dropped → `_require_engine_gone`
- Memory: `unload-in-progress-flag-replaces-recent-unload-window`, `autoloader-lru-eviction-result-ignored`

## Investigation findings (2026-10-22)

Investigation only — no code changed. Sources: `logs/console.log` … `logs/console.log.5`
(retention starts 2026-10-15, so every count below is bounded by retention), `server.py` at
HEAD `a12bab7` (working tree clean for this file — see Q4), `git log`/pickaxe, psutil 7.2.2
source in the analysis sandbox, `N:\work\WD\Strata\serve\*`, `N:\work\WD\AgentCascade\agent_cascade\*`.
Log-coverage caveat: `console.log` reports mtime 09:56 but its last line is 10:46 — mtime is not
a coverage signal (see memory `log-mtime-does-not-imply-coverage`).

### Q1 — what `(pid=7908)` was, and why the graceful path failed

**The incident is in the autoloader's own log**, verbatim identical to the quoted excerpt:
`logs/console.log:42823-42826` (10:45:20,526 Unloading → 42824 fallback warning → 42825 lingering
warning → 42826 `POST /v1/unload_all 200`).

**Exception type: `psutil.AccessDenied` — confirmed by the message format, not guessed.**
psutil renders all its errors through `Error.__str__`, which builds `(k=repr(v), ...)` from the
`pid`/`ppid`/`name` infodict and prepends `msg` (`psutil/_common.py:191-200`). `AccessDenied`
defaults `msg` to `""` (`_common.py:245-254`), so `str(AccessDenied(pid=7908))` == `(pid=7908)`
(verified by execution). The alternatives cannot produce that bare string:
`NoSuchProcess` → `process no longer exists (pid=7908)` (`_common.py:209-220`),
`ZombieProcess` → `PID still exists but it's a zombie (pid=7908)` (`_common.py:226-239`),
`TimeoutExpired` → `timeout after N seconds (pid=7908)` (`_common.py:260-272`).
*Assumption:* the autoloader pins `psutil>=5.9.0` (`requirements.txt:4`); the format was verified
on 7.2.2 only.

**It refers to a descendant, never the tracked parent.** `primary_error` keeps only the FIRST
exception (`server.py:2811`, assigned at `:2819/:2830/:2836/:2851/:2858/:2865/:2873/:2877`).
Corpus check over all **105** `Process termination needed a fallback` lines in `logs/console.log*`
(snapshot 2026-10-22 12:19 local; `console.log` is live, so totals grow as the server runs — the
shape of the distribution is what matters, not the absolute number): the error pid equals the
tracked parent pid in **0/105** cases, and is one of the pids listed as still running on the
following line in **104/105** cases (the single exception is `console.log.1:33453`, where the
fallback is followed directly by a `Launching` line — no survivor list, i.e. the parent died and
nothing was left). The denied operation was therefore always performed on an enumerated descendant
that then survived the teardown.

**Call site — narrowed, but not uniquely determinable.**
Ruled out: `parent.terminate()` (`server.py:2854`) — that would carry `pid=1128`; and a completed
`wait_procs` round — the whole teardown ran in **24 ms** (42823 10:45:20,526 → 42824 10:45:20,550),
while both rounds are `timeout=3` (`server.py:2863`, `:2875`), so a psutil call on 7908 raised
immediately rather than a timeout elapsing. Remaining candidates, all on pid 7908:
`proc.terminate()` (`:2847`), `proc.kill()` (`:2869`), or `psutil.wait_procs` polling that child
(`:2863`/`:2875`) — psutil's `wait_procs.check_gone` catches only `TimeoutExpired`
(`psutil/__init__.py:1570`, `:1607-1610`), so an `AccessDenied` from `proc.wait()` propagates out
of `wait_procs` and is exactly the kind of fast raise the timing implies. The three are
indistinguishable in the log: they produce the identical `AccessDenied` string, only the first is
retained, and the absence of a `name=` field carries no information because psutil's Windows
wrapper always passes `self._name`, which is `None` in the platform Process object
(`psutil/_pswindows.py:660-668` `wrap_exceptions`, `:704-710` `__slots__`/`_name = None`).
**Exact raise site: cannot be determined from available evidence.** Confidence: exception type and
"descendant, not parent" — Confirmed; specific call — Low.

### Q2 — the surviving pids 11056 / 7908 / 26632

**Identity: cannot be determined from available evidence.**
- They appear nowhere else in any autoloader log: word-boundary grep for `11056|7908|26632` over
  `logs/console.log*` matches only the two incident lines (`console.log:42824-42825`); the single
  other `7908` hit is an unrelated timing value (`console.log.4:46314`).
- Strata logs do not record pids either: the same grep over `N:\work\WD\Strata\*.log` returns only
  coincidental numeric substrings (`11056 ms`, `79085 hits`, …); the wrapper never logs its engine's
  pid (the only `self.proc.pid` use is a metrics read, `serve/server.py:2251`).
- What *is* known structurally: pid 1128 was the Strata wrapper — `argv` at `console.log:42829` is
  `Strata/.venv/Scripts/python.exe N:/work/WD/Strata/serve/server.py --engine strata … --port 9022`.
  That wrapper spawns the engine binary via `popen("the Strata engine", …)` (`serve/server.py:997`),
  the image encoder via `popen("the image encoder", …)` (`:2555`), and short-lived probes via
  `subprocess.run` (`:3001`, `:3771`). Three survivors is consistent with engine + encoder + one
  engine child, but that is inference from the spawn inventory, not evidence about these pids.
- **Direct children vs grandchildren: not determinable.** The warning is built from `children`,
  which is `parent.children(recursive=True)` (`server.py:2823`) — one flat descendant list with no
  ppid recorded (psutil builds it from a ppid map, `psutil/__init__.py:951-1041`). If the recursive
  call had raised and the fallback `recursive=False` branch ran (`server.py:2832`), all three would
  be direct children; nothing in the log records which branch executed.

### Q3 — recurrence: the defect is routine, the harm so far is rare

Counts over all retained logs (`console.log` … `console.log.5`), snapshot 2026-10-22 12:19 local
(`console.log` is live — totals drift upward between readings; independently re-derived by a
reviewer pass at ~12:08 gave 102/80, the difference is new events, not a counting error):

| marker | count | dates | models |
|---|---|---|---|
| `Process termination needed a fallback` | 105 | 2026-10-20: 23, 10-21: 31, 10-22: 51 | iq1_m 94, iq3_xxs 11 |
| `Child process(es) … still running after unload` | 83 | **first 2026-10-20 19:10:50** (`console.log.1:49595`, 12 min after `a12bab7`), then 10-21: 31, 10-22: 51 (this incident `console.log:42825`) | iq1_m 72, iq3_xxs 11 |
| `Process termination failed for` | 87 | **2026-10-20 only** (03:50 → last before 18:58) | iq1_m 58, iq3_xxs 29 |

Pairing of the 105 fallback events: 83 ended as "lingering + success", 21 as "failed unload",
1 with no following marker. The `Process termination failed for` line stops appearing exactly when
`a12bab7` (2026-10-20 18:58) took effect in the running server, and the first lingering warning
lands 12 minutes later — independent corroboration of Q4.

Followed by a failed reload? Of the 83 lingering warnings, only **3** have any load/termination
failure within the next 250 log lines: 2026-10-22 09:51:10 (unrelated vision models failing to
become ready), 2026-10-22 10:38:44 (`EngineDied` for a *different* launch 4.5 min later,
`console.log:42714`), and this incident (`console.log:42825` → `EngineDied` `42843-42853` →
503 `42879`). Only this one has the full chain lingering → `Auto-loading model … before state
restore` (+2 lines) → `EngineDied` → 503.

Context for the wider race: 543 `Auto-loading model … before state restore` lines exist in the
retained logs; 29 are followed by a load failure within 250 lines, mostly on 2026-10-20 under the
pre-`a12bab7` code where the unload itself was reported failed — e.g. `console.log.1:33426-33448`
(auto-load 13:46:11,057 → old engine's unload FAILED, parent still running, 13:46:11,097 → new
engine exits code 15 at 13:46:16,111 → retry at 13:46:16,279 succeeds). So "a new engine starts
while the old engine's memory is still live" is a **recurring failure class**; the specific
commit-limit (`VirtualAlloc error 1314`) death following a *reported-successful* unload is confirmed
once. The reservation itself is chronically marginal: `arena could not be reserved` /
`VirtualAlloc error 1314` appears 23 times across `console.log.4` (2026-10-16 13:03),
`console.log.3` (2026-10-17 13:07-14:00), `console.log.1` (2026-10-20 08:35-08:38) and
`console.log` (2026-10-22 10:45:36), independent of this bug.

**Verdict for Q3:** not a one-off defect — the enabling behaviour (success reported with live
descendants) occurs routinely (83 occurrences across ~26 h of retained logs, 51 of them on the
incident day); the user-visible harm (lost KV restore + 503) is confirmed once with two
near-misses. Hit rate is size-dependent and will rise with the model's commit footprint.

### Q4 — code history and the contract question

- **Working tree check first** (the safe-mode work was assumed uncommitted): `git status
  --porcelain` at HEAD `a12bab7` shows `server.py` **unmodified** (only `.agent_lessons/memory_stats.json`
  and `config.yaml` modified, `BUG_0012_*.md` untracked); `git diff HEAD -- server.py` is empty.
  **Inference, high confidence but not directly observed:** the committed `_kill_proc` at
  `server.py:2805-2950` is the code that ran at 10:45:20 — supported by the log-marker change
  coinciding with `a12bab7` and by the incident behaving exactly as a12bab7's semantics predict,
  not by a runtime probe of the live process.
- **Pickaxe dating** (`git log -S"<string>" -- server.py`):
  - `"still running after unload"` → exactly one commit: **`a12bab7` 2026-10-20 18:58 "fix(kill): verify
    parent only, not children"**.
  - `"verify_targets"` → introduced by **`03614f1` 2026-10-20 13:15 (the BUG_0008 fix)**, removed by `a12bab7`.
  - `"needed a fallback"` → introduced by `03614f1`.
  → **The log-only lingering check + `return True` came from `a12bab7`, not from BUG_0008.**
  BUG_0008 introduced the opposite behaviour — verify every enumerated target, any survivor is a
  failed unload — which is precisely what the 87 `Process termination failed for` lines of
  2026-10-20 record (`a12bab7 --stat`: 30 insertions / 19 deletions, replacing the
  `verify_targets` loop with the parent-only check).
- **Did BUG_0008's contract ever cover processes outside the enumerated target list? No.** Under
  `03614f1` verification iterated `verify_targets = list(targets)` where `targets = [parent] +
  children` and `children` is the **pre-kill snapshot** taken at `server.py:2823` (fallback
  `:2832`). Anything not in that snapshot was never checked.
- **Current behaviour confirmed:** verification is parent-only (`server.py:2906-2935`,
  `parent_alive`); children are re-checked only for diagnostics over the same stale objects
  (`:2937-2949`) and then `return True, ""` (`:2950`). After the parent exits the tree cannot be
  re-reached from the tracked pid: `psutil.Process(1128).children()` raises `NoSuchProcess` once
  1128 is gone (psutil resolves descendants through a ppid map rooted at the parent,
  `psutil/__init__.py:951-1041`), and `_kill_proc` never re-enumerates. Confirmed.
- **New finding that changes the fix design:** a kill-on-close job object **already exists on the
  Strata side** — `serve/lifeline.py:6`, `:177-183` (`mechanism()` returns `"job"` on nt),
  `:257-258` `winjob.contain(child)`; `serve/winjob.py:1-9` ("Processes the child starts later join
  the same job"), `:41-60` (`_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`). It did **not** prevent these
  three survivors. Whether this tree was actually inside the job, or whether job termination is
  simply asynchronous relative to the 13 ms psutil re-check (`10:45:20,550` → `10:45:20,563`), is
  **unconfirmed** and should be settled before adding a job object on the autoloader side — the
  "structural alternative" in the Fix section is already implemented by the child project. Related
  unconfirmed reaper: `end_leftover_engines(exe)` (`serve/server.py:995`), run before each engine
  spawn; which mechanism actually freed the commit before the successful 10:45:38 load is unknown.

### Q5 — caller side (AgentCascade): no bounded wait exists

- Safe-mode seam: `ShellCmd._execute_safe` (`agent_cascade/tools/custom/shell_cmd.py:1062-1177`).
  `_release_model` (`:1128-1136`) → `state_ops.unload_all_models`
  (`agent_cascade/state_ops.py:265-287`): a single `httpx.post(.../v1/unload_all, timeout=60)`;
  no post-unload poll, wait or verification. The restore is issued in the `finally`
  (`shell_cmd.py:1147-1159`) → `restore_held_slot_state` → `state_ops.restore_instance_state`
  (`state_ops.py:126-262`) → `httpx.post(.../state/load, timeout=180)` at `:256`.
- No wait/poll/sleep exists anywhere in `state_ops.py` (grep for `sleep|poll|wait` matches only
  docstrings). `probe_memory_status` / `ram_vram_pressure` (`state_ops.py:295-333`) do read
  `/v1/status` RAM, but they run **before** the command to decide whether to force safe mode —
  never after the unload to confirm memory was released.
- → The 200 → `state/load` gap is **unconditional**: it is just the wrapped command's duration plus
  HTTP round-trips (~459 ms here). No caller-side bounded wait, no memory-clear check. (Which
  command occupied those 459 ms was not identified — open.)

### Confidence and open questions

- Confirmed: incident location and text; `AccessDenied` identification; error pid is always a
  descendant, never the parent (104/105, 0/105); the log-only check was introduced by `a12bab7`,
  not by BUG_0008; verification is parent-only and post-parent orphans are unreachable; no
  caller-side wait.
- High confidence: recurrence counts and the 2026-10-20 behaviour change (log corpus + pickaxe +
  commit diff agree independently).
- Independent review pass (2026-10-22) re-derived the corpus independently and confirmed claims 1,
  2 and 4 outright; it initially reported 102/80 vs my 101/79 — the difference is that
  `console.log` is a live sink and each reading picked up new events, so all counts here are
  labelled with their snapshot time. It also flagged two labelling issues, both corrected above
  (the "code that ran" inference, and the psutil line numbers being unverifiable against the
  installed version). Note the pre-existing sections could not be diffed against a git baseline:
  this BUG file is untracked (`??`); the append-only constraint was honoured by exact-match edits
  against the original text only (pre-edit backup:
  `N:\work\WD\AgentWorkspace\logs\backups\researcher_bug0012_forensics\BUG_0012_unload_success_with_living_engine_grandchildren.md.1792660214.bak`).
- Low confidence / unknown: which psutil call raised on 7908; what 11056/7908/26632 actually were
  and whether they were children or grandchildren; whether Strata's job object contained them;
  which mechanism reaped them; the psutil version actually installed for the autoloader (only the
  `>=5.9.0` pin is known).
- Suggested next actions: (1) determine whether the Strata job actually contained the engine tree
  before choosing the autoloader-side fix — if job-close is asynchronous, a bounded re-enumeration
  window is required regardless; (2) log `ppid` alongside the lingering pids so future occurrences
  are attributable; (3) if a regression test is written, drive a real parent+grandchild tree and
  assert absence of success-with-survivors, per the Verification section.
