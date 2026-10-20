# BUG_0011 — LRU eviction drops its result and mid-load models cascade into a load/evict ping-pong

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `d977c82`
**Found in:** Production log 2026-10-20 15:16 (`todo.md` lines 181-259)
**Severity:** high
**Component:** `server.py` (`_evict_lru_if_needed`, `_do_load_model`)

## Symptom

Two defects in the eviction path:

1. `_evict_lru_if_needed` returned `None`, so the load path proceeded after a **failed** eviction. The old model's processes were still running, consuming GPU memory while the new model loaded on top of them.
2. Mid-load models were eligible for LRU eviction, producing a load/evict ping-pong cascade: recovery evicted a model that was mid-load, then the load evicted the recovery target, and so on.

## Root cause

The eviction result was dropped — `_evict_lru_if_needed` had no return value and the call site performed no check. The candidate filter excluded pinned and busy models but not mid-load ones, so a model still being loaded could be selected as the eviction victim.

## Fix (implemented)

- `_evict_lru_if_needed` now returns `Optional[UnloadResult]`; the call site checks it and raises on failure, so a load that cannot make room refuses instead of proceeding over surviving processes.
- Mid-load models (those with an incomplete task in `_loading_tasks`) are excluded from eviction candidates, breaking the load/evict cascade.

## Verification

- Three new tests in `TestEvictLRU`: mid-load exclusion, all-mid-load no-candidate, and the call-site guard that refuses the load on a failed eviction.

## Related

- `BUG_0005_recovery_holds_manager_lock_across_teardown.md` — per-model lock context
- `BUG_0008_unload_kill_fallback_and_denied_child.md` — the process-tree kill that made the eviction fail
- [[lru-eviction-result-check-and-mid-load-exclusion]] — the eviction result-check and mid-load exclusion fix
