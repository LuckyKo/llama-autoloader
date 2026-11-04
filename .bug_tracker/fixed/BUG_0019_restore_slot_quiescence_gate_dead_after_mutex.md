# BUG_0019 — `restore_slot` quiescence gate evaluated after `execution_mutex_`, so the 409 `slot_busy` fast-reject is dead

**Status:** FIXED (2026-11-03) — gate moved before `execution_mutex_`; verified by the `slot-restore-non-quiescent` scenario (409 now fires under a busy catalog).
**Filed:** 2026-11-03
**Found in:** Runtime retest of the 9-file `CapacityShortfall`/full-drain change in `ninfer-all` (Maine session E1371/E1374). The `slot-restore-non-quiescent` test scenario failed before the fix (busy restore not rejected with 409).
**Severity:** medium — a restore issued while a generate is in-flight is not rejected with 409 `slot_busy`; it blocks on `execution_mutex_` and runs *after* the request settles, defeating the fast-reject contract. Not a data-corruption bug, but the gate is dead (never fires).
**Component:** `src/runtime/engine/engine_core.h` `EngineCore::restore_slot` (~547–615). **Code lives in `ninfer-all`** (not llama-autoloader); filed here because `ninfer-all` has no `.bug_tracker`.

## Symptom

A `restore_slot` call while a generate is in-flight (active exposure set non-empty) is expected to **fast-reject with 409 `slot_busy`**. Instead it blocks on `execution_mutex_` until the in-flight request settles, then proceeds as if quiescent — the 409 never fires.

## Root cause

`restore_slot` checked `active_exposure_set()` (the quiescence gate) **after** taking `std::scoped_lock lock(execution_mutex_)`. But the worker holds `execution_mutex_` across the *entire* generate unit, so by the time the gate runs the in-flight request has always settled and `active_exposure_set()` is empty. The gate is therefore unreachable through the public path — a textbook "fast-reject gate evaluated after the lock the competitor holds" defect.

## Fix

Move the `active_exposure_set()` check to **before** `std::scoped_lock lock(execution_mutex_)` (`engine_core.h` ~550–563). The surrounding comment already said "do not block behind it" — a gate that blocks on the lock is not fast-rejecting. This is a genuine **correctness** fix (the 409 gate was dead), not a feature.

## Verification

The `slot-restore-non-quiescent` scenario in `tests/models/qwen3_5/test_engine_prefix_real.cpp` now passes: *"busy restore rejected 409, quiescent restore succeeded."* Before the fix it failed (the 409 did not fire). Re-run via `bench/run_slot_tests.bat` → `logs/slot_tests.log` (RTX 5060 Ti, `build-win-120a`).

## Test-harness-only fixes from the same retest (NOT product bugs)

Two other fixes from the same retest live in `tests/models/qwen3_5/test_engine_prefix_real.cpp` and are test-harness only — no product change:
- **Config:** `slot_full_drain_options` set `kv_capacity=512` but `max_context=1024`; the engine requires `kv_capacity >= max_context`. Set `max_context=512`.
- **Pool sizing:** 3×250-token continuations did not saturate the 512-token pool (LRU eviction freed room); bumped to 4×400 tokens.

## Related

- [[BUG_0015]] / [[BUG_0018]] — the autoloader-side load-window ticket gap; this is the engine-side quiescence gate.
- Skill `state-guard-reachability-under-mutex` (Step 4: a fast-reject gate must run *before* the lock the competitor holds across the operation).
- `ninfer-all/.agent_lessons/slot-drain-retest-config-and-gate-ordering.md`.
