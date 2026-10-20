---
tags: [lru-eviction, unload-result, mid-load, max-loaded-models, ping-pong]
aliases: [lru-eviction-result-contract, mid-load-eviction-exclusion]
related: [[unload-teardown-exception-orphans-registry]], [[llama-autoloader-pytest-canonical-invocation]]
confidence: verified
author: coder_coder-llama-lru_20261020_155257.jsonl
contributors:
  - reviewer_lru-eviction-review_20261020_155658.jsonl
  - coder_polish-coder-a_20261020_164840.jsonl
---

# LRU eviction now returns a verdict; mid-load models are never eviction candidates

Two invariants were added to the LRU eviction path in `server.py`. Both are load-bearing for the "cascade of model kills" failure mode:

1. **`_evict_lru_if_needed` returns `Optional[UnloadResult]`** and its single production
   call site in `_do_load_model` checks it. A non-`None` result with `unloaded == False`
   means the slot was NOT freed (engine process still running) and the load is refused
   with `RuntimeError: Cannot load … LRU eviction of … failed`. Before this, the
   result was dropped and the load proceeded anyway — the same contract every other
   unload caller already enforces (see [[unload-teardown-exception-orphans-registry]]:
   LRU auto-evict was one of the call sites that "just propagated").

2. **Mid-load models are excluded from eviction candidates.** A model resident in
   `self.loaded` with an incomplete task in `self._loading_tasks` is the *most* eligible
   victim under the old predicate (stale `last_used`, `active_requests == 0`), which
   produced the ping-pong: recovery evicts a model mid-load, then that load evicts the
   recovery target. Liveness check is `task = self._loading_tasks.get(mid); mid is
   mid-load iff task and not task.done()` — the same check `_begin_unload_locked`
   uses for its `deliberate` flag.

## Evidence (pinned)

At `llama-autoloader` @ `3926acecf917796ed2fbf2a0073d97685a863219` (working tree, post-fix):

- `_evict_lru_if_needed` (`server.py:1951`): candidate predicate is
  `mid != target and not pinned and active_requests == 0 and not mid-load`.
- Call site in `_do_load_model`: guard mirrors the force-reload guard immediately
  above it (`result is not None and not result.unloaded → raise`).
- The raise happens **inside `async with self._lock` and before any target mutation**
  (`self.loaded[model_id] = lm` comes later), and a failed unload already restored the
  LRU entry via `_finalize_unload_locked`'s `setdefault` — so the registry is unchanged
  when the load is refused. Verified by test `test_load_refused_when_eviction_fails`.
- Surfacing: `/v1/models/{id}/load` → 503 with the message; forwarder JIT → logged,
  returns None; recovery reload → caught and arms the BUG_0007 backoff.

## Deliberate trade-off: over-cap loads are now the common case

When every candidate is pinned/busy/mid-load, `_evict_lru_if_needed` warns and returns
`None`, and **the load proceeds over `max_loaded_models`**. With `max_loaded_models: 1`
in config.yaml, a JIT request arriving while one model is mid-start yields two resident
engines. Nothing re-enforces the cap afterwards (`idle_reaper` only unloads on
`idle_timeout`). This is intentional — killing a just-started engine *is* the ping-pong
bug — and is documented in the function docstring. Do not "fix" it by re-admitting
mid-load models to the candidate set.

## Known follow-up (not done, out of scope of the fix)

The mid-load predicate now has three spellings: `_load_model_serialized`
(`model_id in self._loading_tasks and not …done()`), `_begin_unload_locked`
(local binding), and the comprehension inside `_evict_lru_if_needed` (double dict
lookup). Semantics are identical and the double lookup is safe (no `await` inside the
comprehension; all writers hold `self._lock`), but an invariant fixed twice already
(BUG_0005/BUG_0009) drifting in three places is the classic setup for the next bug.
Extract a shared `_is_mid_load_locked(mid)` helper if that area is touched again.

## Addendum 2026-10-20: the follow-up was done (as `_is_loading` + `_require_engine_gone`)

At `llama-autoloader` @ `d977c82bd3202cdc8ed7523190be478c211e6165` (POLISH phase, working tree):

Both duplicated invariants above are now single helpers on `ModelManager`. Grep the helper names
rather than the old inline shapes:

- **`_is_loading(model_id)`** (`ModelManager._is_loading`, immediately after `_model_lock`) —
  `task = self._loading_tasks.get(model_id); return task is not None and not task.done()`.
  All three former spellings now call it: the `_evict_lru_if_needed` candidate filter
  (`and not self._is_loading(mid)`), the join-the-in-flight-task branch in
  `_load_model_serialized`, and `_begin_unload_locked`'s `deliberate` flag
  (`deliberate = not self._is_loading(model_id)`). Truth-table-equivalent to all three originals:
  `asyncio.Task` has no `__bool__`/`__len__`, so the old `task and …` and the new
  `task is not None and …` coincide. **Callers must hold `self._lock`** (stated in its docstring).
- **`_require_engine_gone(result, context)`** (`ModelManager._require_engine_gone`, immediately
  above `_do_load_model`) — the force-reload guard and the LRU-eviction guard in `_do_load_model`
  both call it; the trigger condition is byte-identical to both original inline guards.

Two things that must NOT be "simplified" back:

1. **The `is task` identity check in `_load_model_serialized`'s `finally` is not a loading
   predicate** — it is an ownership guard for pruning the dict entry, and deliberately still reads
   `self._loading_tasks.get(model_id) is task`. Likewise the `.done()` checks on
   `_recovery_tasks` elsewhere are a different dict; grep by variable name before merging.
2. **`test_server.py::test_load_refused_when_eviction_fails` pins the message substring**
   `match="LRU eviction of x.gguf failed"`. So the eviction guard's context string must keep the
   victim's model id: it is built as
   `f"Cannot load {model_id}: LRU eviction of {evict_result.model_id if evict_result else model_id} failed"`.
   The conditional is load-bearing, not cosmetic — call-site f-strings evaluate *before* the helper
   runs, so an unguarded `evict_result.model_id` raises `AttributeError` on exactly the `None`
   path the guard tolerates. Wording drift that IS allowed: the shared suffix is now
   `": process {pid} is still running ({reason})"` (the old em-dash form and the force-reload
   prefix "previous engine" are gone); no test asserts those.

Full suite green at this pin: **397 passed** (`pytest tests/ test_server.py -q -n 0`; see
[[llama-autoloader-pytest-canonical-invocation]] for the invocation and
[[port-probe-tests-fd-exhaustion-flake]] / [[windows-proactor-sock-test-error-transient]] for the
`WinError 10055` `socket.socketpair` fixture-setup error that appears under `-x` runs and passes in
isolation).

## Test pinning

`test_server.py::TestEvictLRU` — `test_skips_mid_load_models` uses the discriminating
shape: the mid-load model is the LRU pick and the assertion is that a *different*
model got evicted (asserting "nothing evicted" would also pass with a filter that
wrongly excludes everything). `test_no_eviction_when_only_mid_load_candidates` pins
the `None` return. Full suite green (397 passed) at the pin above.
