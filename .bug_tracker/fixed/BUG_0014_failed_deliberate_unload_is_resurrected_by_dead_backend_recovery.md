# BUG_0014 — a failed *deliberate* unload is undone by dead-backend recovery (the model is resurrected onto the client's cards)

**Status:** FIXED (2026-10-25)
**Filed:** 2026-10-25
**Found in:** Production log 2026-10-25 08:20:40–08:21:08 (`N:\work\stuff\Beta\llama-autoloader\logs\console.log`). The `unload_all` came from AgentCascade safe mode (`state_ops.unload_all_models`) right before a 363 s GPU test; the reload landed 8 s into that test. Cross-report: `AgentCascade/.bug_tracker/fixed/BUG_0152_failed_unload_reported_as_unloaded_and_reload_killed_mid_load.md`.
**Severity:** high — the exact opposite of the requested action: a client asks for the cards, the release fails, and the loader then loads a ~47 GB model onto those cards while the client's GPU job is running. Also the reload is what made the follow-up unload kill a load mid-start (BUG_0013).
**Component:** `server.py` — `_begin_unload_locked` / `_finalize_unload_locked` (:2926-3010) and the unconditional reload in `_recover_dead_model_impl` (:2676-2750)

## Symptom

`unload_all` cannot verify the tree drain (an engine descendant survives the kill), so it correctly answers 500 and keeps the model tracked. Eight seconds later the loader decides that same model is a dead backend and **reloads it**:

```
08:20:40,947 [INFO]    Unloading qwen3.8-flash-next-coder-iq3_xxs (port 9024)
08:20:41,100 [WARNING] Engine descendants 752 (strata.exe, ppid=6804), 18896 (?, ppid=?) still running
                       after unload of qwen3.8-flash-next-coder-iq3_xxs (parent 12164 gone);
                       terminating and waiting for the whole tree before reporting success
08:20:56,145 [ERROR]   Tree drain timed out for qwen3.8-flash-next-coder-iq3_xxs (pid 12164):
                       752 (strata.exe, ppid=6804) still running
08:20:56,147 [ERROR]   Unload of qwen3.8-flash-next-coder-iq3_xxs failed: process 12164 is still
                       running (...); keeping the model tracked
08:20:56,147 [INFO]    POST /v1/unload_all HTTP/1.1 500 Internal Server Error
08:21:04,845 [ERROR]   Engine for 'qwen3.8-flash-next-coder-iq3_xxs' on port 9024 (generation 4)
                       is unresponsive: process exited; evicting and reloading
08:21:04,845 [ERROR]   Engine for qwen3.8-flash-next-coder-iq3_xxs on port 9024 is already marked
                       unready; evicting and reloading
08:21:04,851 [INFO]    Launching strata for qwen3.8-flash-next-coder-iq3_xxs on port 9024
08:21:08,089 [INFO]    [strata] starting the engine: reading the model's weights ...
```

The caller's intent ("release this model") is known to the manager — `_begin_unload_locked` returns `deliberate` — but it is discarded when the teardown fails.

## Evidence

- `_begin_unload_locked` marks the release: `self._unload_in_progress.add(model_id)` (`server.py:2951`) and returns `(lm, deliberate)`; `_finalize_unload_locked` discards that marker on both the success and the failure path (`server.py:2980`, `server.py:3001`). After a failed unload there is **no state left** saying "a client asked for this to be gone".
- `_unload_finish` is what fails: it cannot verify the kill because one descendant (`strata.exe` pid 752, parent `ppid=6804` — a process that no longer exists, i.e. an orphan reparented from an earlier engine generation) survives the 15 s drain. That is the BUG_0012 contract doing its job (never report success with a living engine process), but its consequence is a model that stays `loaded` with `ready=False` and a **dead engine port**.
- The tcp forwarder then does exactly what it was built for (BUG_0001): it sees connection-refused on :9024, calls `recover_dead_model`, and `_recover_dead_model_impl` takes the `else` branch (`server.py:2681`: `is already marked unready`), sets `lm.ready = False`, unloads the detached entry and calls `_load_model_serialized(reload_mid)` (`server.py:2747`) with **no check for a client-requested release**.
- The reload is not a repair of a spontaneously crashed engine: the engine port is dead *because the client's unload killed the wrapper*. Recovery is restoring a service the owner had just asked to stop.
- Timing: 500 at `08:20:56,147`, first detection at `08:21:04,845` (~8.7 s, the forwarder's liveness interval), reload starts at `08:21:04,851` — well inside the client's command, which was still running until ~08:27:20.
- Same shape was seen on 2026-10-22 (BUG_0012): 200 there, but the immediate state-restore reload also overlapped the teardown and died on the commit limit. There the reload was requested by the client; here it was invented by recovery.

## Root cause

Dead-backend recovery is unconditional: it does not distinguish "the engine died" from "the client asked us to kill it and the kill only half-succeeded". The manager computes that distinction at `_begin_unload_locked` (`deliberate`) and then throws it away on failure, so a failed release turns into a resurrection.

Secondary (trigger, not fixable here): the surviving descendant is an orphan whose parent is already gone (`ppid=6804`), so the client's unload can never verify or drain it, and every retry will fail the same way until that process exits. This is the same class as BUG_0012's "grandchildren outlive the tracked pid"; it is worth knowing that after BUG_0012's fix such an orphan no longer yields a false success — it yields a model that can never be unloaded.

## Fix (proposed direction)

1. Preserve the intent: on a **deliberate** unload that fails, keep the model in `_unload_in_progress` (or a new `_release_requested` set) instead of discarding it. A ready model must still be loadable by an explicit `load_model`, so clear the marker on any explicit load (`load_model(...)`), as `_load_model_serialized` already clears `_unload_in_progress` for in-flight loads (`server.py:2401`).
2. In `_recover_dead_model_impl`, before the `evicting and reloading` branch: if `reload_mid` is marked release-requested and `lm.ready` is False, **do not reload**. Log at INFO/WARNING, e.g.
   `not reloading '<model>' (port <p>): the client requested release and the unload failed (<reason>); leaving it tracked and not ready`.
   Return `None` to the forwarder (it already treats `None` as "cannot recover").
3. Include the survivor pid(s) in the unload failure reason surfaced to the client — `/v1/status` already carries the failed list; make sure the HTTP 500 body's `failed[].reason` names the pid and the command line so the operator can kill it manually.
4. Do not retry the drain from recovery either; a retry is the client's decision (AgentCascade deliberately never retries — a second unload seconds later is what kills an in-flight load, BUG_0013).

## Verification (to be provided with the fix)

- Unit test: unload a fake model whose `_unload_finish` fails (survivor pid), assert `unloaded=False` + model still tracked + release marker set; then call `recover_dead_model(mid, port, generation)` and assert it returns `None` **without** launching (no `Launching` log, `models_loaded` unchanged, no new generation).
- Unit test: after such a failure, an explicit `load_model(mid)` still loads and clears the marker.
- Regression: BUG_0001/BUG_0002 recovery tests (a genuinely crashed engine must still be reloaded) — the new guard must fire only when the release marker is set.

## Related

- `BUG_0012_unload_success_with_living_engine_grandchildren.md` — the teardown verification that now (correctly) fails here, and the orphan-grandchild class.
- `BUG_0001_tcp_forwarder_no_dead_backend_recovery.md` / `BUG_0002_forwarder_hung_backend_no_liveness_timeout.md` — the recovery path being (correctly) triggered; the gap is that it is unconditional.
- `BUG_0013_unload_racing_inflight_load_booked_as_failed_recovery.md` — the same incident, the other half (the client's second unload then killed that reload).
- AgentCascade `BUG_0152` — the client-side report, fixed there (a refused unload is no longer reported as "unloaded").

## Fix (applied)
Implemented in `server.py`.
- `_release_requested` is set in `_finalize_unload_locked` (server.py:~3091) ONLY when `client_requested` AND the kill failed — NOT on `deliberate`. `client_requested` is threaded from `_unload_unlocked` and passed `True` ONLY by the client entry points (`unload_model`, `unload_model_exact`, `unload_all`, hard-interrupt endpoint). This is the key correctness point: gating on `deliberate` (which is also true for a genuine dead-backend recovery whose teardown kill fails, and for a failed LRU eviction) would have suppressed recovery forever, turning BUG_0007's bounded backoff into indefinite suppression.
- `_recover_dead_model_impl` guard (server.py:~2756): if the model is release-requested, log WARNING "Not reloading '<mid>' …" and return `None` (no reload, no cooldown).
- An explicit/JIT `load_model` clears the marker (server.py:~2419); a successful unload clears it (server.py:~3096).
- `_unload_in_progress` arming in `_begin_unload_locked` stays on `deliberate` (unchanged BUG_0012/safe-mode discriminator).

## Verification (actual)
New tests in `tests/test_unload_reporting.py` (3, all pass):
- `test_failed_deliberate_unload_is_not_resurrected_by_recovery` — a client release whose kill fails sets the marker; recovery returns `None`, no `Popen` launch, no "Launching", "Not reloading" present, model stays tracked, no cooldown.
- `test_explicit_load_after_failed_release_clears_marker_and_loads` — an explicit load supersedes the failed release, clears the marker, reaches ready.
- `test_recovery_teardown_kill_failure_does_not_set_release_marker` — REGRESSION GUARD for the over-application defect: a genuine recovery whose teardown kill fails must NOT set the marker (fails against the initial `deliberate`-gated version, passes against the `client_requested` gate).

Result: `pytest tests/test_unload_reporting.py tests/test_unload_load_race.py` → 14 passed. Independent review ran twice: the first pass caught the `deliberate` over-application defect; the second confirmed the `client_requested` fix and gave PASS. Full autoloader suite: only 4 pre-existing bridge/ninfer WIP failures (`_perform_slot_save model_id` signature) + 3 env errors (real backend binary absent) remain — all proven to reproduce at HEAD, unrelated to this fix.
