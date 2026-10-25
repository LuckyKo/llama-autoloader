---
tags: [unload, recovery, release-marker, bug-0014, testing]
aliases: [release-requested-marker, not-reloading-warning]
related: [[bug-0013-load-abort-vs-recovery]]
confidence: verified
author: coder_bug0014_tests_20261025_121928.jsonl
contributors:
  - coder_bug0014_fix_20261025_124244.jsonl
---

# BUG_0014: a deliberate unload whose kill failed must NOT be resurrected by dead-backend recovery

A *deliberate* unload (caller asked for the release) that could not verify the kill keeps
the model **tracked** (its process may still be running) but records the release intent in
`ModelManager._release_requested` (a set of model ids). Dead-backend recovery then leaves
that model tracked-and-not-ready instead of reloading it — reloading would silently undo
the caller's release and take the GPU cards back mid-job.

## The three product-code points (server.py @ b97491c74f16c12c14f571cb378bcc624d789d58)
- `_finalize_unload_locked`: when `not killed` AND `deliberate`, keeps the model tracked and
  does `self._release_requested.add(model_id)` + logs
  `"Release of {model_id} was requested but did not complete"`.
- `_recover_dead_model_impl`: inside the generation-match branch,
  `if reload_mid in self._release_requested:` → logs
  `"Not reloading '<mid>' on port <p>: a client requested its release and the unload did not
  complete; leaving it tracked and not ready"` and `return None` (no evict, no Popen, no cooldown).
- `load_model`: `self._release_requested.discard(model_id)` — an explicit/JIT load is the
  legitimate revocation of a failed release. A *successful* unload also discards the marker
  (`_finalize_unload_locked` killed=True path).

## How the regression tests exercise it (tests/test_unload_reporting.py)
Reuse the existing `_add_resident` / `_make_kill_fail` helpers.
- **Not-resurrected test:** after the failed deliberate unload, set `lm.ready = False` (dead
  engine), patch `server.subprocess.Popen` with a MagicMock, call
  `recover_dead_model(mid, lm.port, lm.generation)`. Pass `lm.generation` so `_gen_matches`
  succeeds and the marker branch is reached. Assert `recovered is None`, model still in
  `mgr.loaded`, `mgr.recovery_retry_after(mid) is None`, Popen NOT called, and the
  "Not reloading" warning present.
- **Explicit-load-clears-marker test:** to make `load_model` reach ready via the REAL launch
  path (not the short-circuit), the tracked entry must be NOT-ready. A dead entry with
  `lm.pid = None` is verified-gone by `_kill_proc` (`if not lm.pid or lm.pid <= 0: return True,""`),
  so the load's eviction step passes. `_do_load_model`'s llama launch reads
  `self.gguf_paths[model_id]`, so the test must set `mgr.gguf_paths[mid] = Path(mid)`.
  Patch `server.subprocess.Popen` → a fake proc and `ModelManager._wait_until_ready` → async
  `return True`. Then `load_model(mid)` clears the marker and reaches ready.

## Pre-fix proof (why the test is not vacuous)
Neutering the marker branch in `_recover_dead_model_impl` (simulate pre-fix) makes the
not-resurrected test FAIL: recovery falls through to "evicting and reloading", evicts the
model (`mgr.loaded` becomes empty) and books a 5s cooldown. That is exactly the resurrection
bug. Reverting the neuter restores 7/7 in the file.

## Test-file note
The real recovery test files are `tests/test_dead_backend_recovery.py` and
`tests/test_backend_liveness_backoff.py` (NOT `test_recovery.py`). They mock `_load_model_serialized`
to avoid the real launch path; the BUG_0014 explicit-load test deliberately does NOT, to prove
a real load clears the marker.

## Addendum 2026-10-25 — marker gating was over-broad; corrected to client-initiated only
At `llama-autoloader` working tree on top of `b97491c74f16c12c14f571cb378bcc624d789d58` (uncommitted):
the "three product-code points" above were gated on the **`deliberate`** flag, which is wrong. `deliberate`
(`_begin_unload_locked`, computed `not self._is_loading(model_id)`) is ALSO True for INTERNAL deliberate
unloads — a genuine dead-backend recovery whose teardown kill fails (`recover_dead_model` → `_begin_unload_locked`
→ `_unload_finish` → `_finalize_unload_locked`) and a failed LRU eviction (`_evict_lru_if_needed` →
`_unload_unlocked`). Effect: a genuinely-crashed engine got `_release_requested` set, so the recovery guard
(`if reload_mid in self._release_requested: return None`) suppressed it FOREVER — contradicting BUG_0007's
bounded backoff and the ticket. This is the WRITER-side mirror of [[intent-flag-predicate-scope-audit]].

Corrected invariant: `_release_requested` means **"a CLIENT asked for this release"**, not "a deliberate unload
was attempted". Fix threaded a distinct `client_requested: bool = False` param:
- `_finalize_unload_locked` marker block now gates on `client_requested` (and `not killed`), NOT `deliberate`.
  The "Release ... was requested" log fires only for `client_requested`.
- `_unload_unlocked` gained `client_requested: bool = False` and passes it through to `_finalize_unload_locked`.
- `client_requested=True` ONLY at client entry points: `unload_model`, `unload_model_exact`, `unload_all`, and
  the hard-interrupt endpoint. ALL internal `_unload_unlocked` callers keep the default False (`_evict_lru_if_needed`,
  load's own cleanup, the `_unload` helper, idle_reaper/shutdown); recovery calls `_finalize_unload_locked` directly
  with the default False.
- UNCHANGED (correct): the `_unload_in_progress` arming in `_begin_unload_locked` stays gated on `deliberate` —
  that is the BUG_0012/safe-mode discriminator, a different semantic. Do not conflate the two flags.

Regression test: `tests/test_unload_reporting.py::test_recovery_teardown_kill_failure_does_not_set_release_marker`
drives the genuine recovery path (no client unload, `_make_kill_fail` forces the teardown kill to fail) and asserts
`mid NOT in mgr._release_requested`, no "Release ... was requested" log, and that recovery arms the bounded
`recovery_retry_after` backoff (not permanent suppression). It FAILS at the old `deliberate` gate (marker set +
misleading log) and passes after. See [[bug-0013-load-abort-vs-recovery]].
