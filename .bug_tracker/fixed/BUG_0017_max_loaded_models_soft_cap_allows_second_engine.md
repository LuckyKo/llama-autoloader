# BUG_0017 — `max_loaded_models` was a soft cap: a mid-load blocker let a second engine start

**Status:** FIXED
**Filed:** 2026-10-31
**Fixed:** 2026-10-31
**Commit:** `uncommitted (working tree)`
**Found in:** Production log `logs/console.log` 2026-10-31 04:33:13 (JIT load of `qwen3.8-flash-next-coder-iq1_m` while `qwen3.8-flash-next-coder-iq3_xxs` was mid-recovery-reload); same warning recurs in `logs/console.log.2` (2026-10-28) and `logs/console.log.4` (2026-10-27).
**Severity:** high — with `models.max_loaded_models: 1` two large models loaded at once; both engines died at the Windows commit limit and every retry failed.
**Component:** `server.py` (`_evict_lru_if_needed`, `_do_load_model`)

## Symptom

With `models.max_loaded_models: 1` (config.yaml:75), the autoloader launched two engines
concurrently:

```
04:33:13,866 [INFO] Launching strata for qwen3.8-flash-next-coder-iq3_xxs on port 9024
04:33:13,900 [INFO] Model 'qwen3.8-flash-next-coder-iq1_m' not loaded; triggering JIT load...
04:33:13,901 [WARNING] Max loaded models limit reached, but every other model is pinned,
                     busy (active request), mid-load, or the target itself; loading without evicting
04:33:13,904 [INFO] Launching strata for qwen3.8-flash-next-coder-iq1_m on port 9023
```

`iq3_xxs` alone plans ~47 GB of experts, `iq1_m` ~30 GB. Both then failed:

```
04:33:33,992 [ERROR] Model qwen3.8-flash-next-coder-iq1_m failed to become ready
04:34:19,038 [ERROR] Model qwen3.8-flash-next-coder-iq3_xxs failed to become ready
```

The `strata serve` log named the cause: "3.6 GiB left of a 99.9 GiB [commit] limit …
it can end at the limit with no message".

## Root cause

`_evict_lru_if_needed` returns `None` in two different situations: (a) the manager is
under `max_loaded_models`, and (b) the manager is at the cap but every candidate is
excluded (pinned, `active_requests != 0`, or mid-load). `_do_load_model` treated both
as "nothing to do" and proceeded to spawn. The mid-load exclusion itself is correct
(it prevents the BUG_0011 load/evict ping-pong), but the *caller's* response to it was
not: a JIT request arriving while the only resident model was mid-recovery-reload
started a second engine over the cap. Nothing re-enforces the cap afterwards
(`idle_reaper` only acts on `idle_timeout`).

The repository docs already described the intended hard-cap behavior
(`README.md:922`, `DOCUMENTATION.md:1040`: "if no eligible model can be evicted … the
load is refused with an error"), so the code and docs disagreed.

## Fix (implemented)

- `_evict_lru_if_needed` still returns `None` (the mid-load/pinned/busy predicate is
  unchanged), but its warning now says the load is being refused.
- `_do_load_model` distinguishes the two `None` cases with
  `evict_result is None and len(self.loaded) >= self.max_loaded_models` and raises
  `RuntimeError("Cannot load <id>: max_loaded_models (<n>) reached and no evictable
  model is available (pinned, busy, or mid-load)")` before allocating a port or
  spawning. A refusal kills nothing, so the BUG_0011 ping-pong fix is preserved.
- Forwarder JIT surfaces the refusal as a 503; `_recover_dead_engine` catches it and
  arms the existing BUG_0007 recovery backoff.

## Verification

- New test `test_server.py::TestEvictLRU::test_load_refused_when_only_mid_load_candidate`
  (max_loaded_models=1, one mid-load resident, assert `RuntimeError` and that the
  target was not installed).
- Existing `test_no_eviction_when_only_mid_load_candidates` still pins the `None`
  return from `_evict_lru_if_needed`.
- `python -m pytest tests/ test_server.py -q -n 0`: 491 passed, 4 environment-only
  errors (WinError 10055 socketpair; missing custom backend binary in
  `backends/llama.cpp-win-x86_64-nvidia-cuda12-avx2-local`).

## Related

- `BUG_0011_lru_eviction_result_dropped_and_mid_load_cascade.md` — introduced the
  mid-load exclusion and the eviction-result check; this bug closes its remaining
  "proceed over the cap" escape.
- `BUG_0007_failed_backend_reload_has_no_backoff.md` — consumes the new refusal on the
  recovery path.
- [[lru-eviction-result-check-and-mid-load-exclusion]] — 2026-10-31 addendum.
