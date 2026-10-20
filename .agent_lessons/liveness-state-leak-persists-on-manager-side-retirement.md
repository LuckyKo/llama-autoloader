---
tags: [tcp-forwarder, memory-leak, engine-generation, lifecycle-coverage, liveness-probe]
aliases: [liveness-state-leak-persists-on-manager-side-retirement, retire-liveness-state-single-hook]
related: "[[forwarder-generation-keyed-state-retirement]] [[unload-teardown-exception-orphans-registry]] [[lru-eviction-result-check-and-mid-load-exclusion]]"
confidence: verified
author: reviewer_polish-review-c_20261020_170958.jsonl
---

# `_retire_liveness_state` covers only the hang-triggered retirement path; manager-side unload/eviction still leaks

At `llama-autoloader` @ `d977c82` **plus the uncommitted POLISH-phase working-tree edits** (2026-10-20): the new
`_retire_liveness_state` helper in `RawTCPForwarder` (`tcp_forwarder.py`, adjacent to `_liveness_probe_lock`) is
reachable from exactly one function, `_handle_confirmed_hang`. That function itself has **three** call sites
(`_check_engine_liveness` process-exited path, `_check_engine_liveness` probe-threshold path, and
`_pipe_to_backend_inner` BUG_0006 send-failure path) — all of them are engine *death* paths.

An engine generation also stops existing without ever reaching it: `ModelManager._unload_unlocked`, LRU eviction,
and force-reload detach the registry entry and force `ready = False` on the manager side (`server.py`). The only
forwarder entry point the manager calls during such a transition is `cancel_active(port, generation)` (server.py,
soft-interrupt path), which touches `_active_conns` / `_active_response_started` only.

## Proof method (re-checkable)

Whole-repo grep of the three container names (`_liveness_last_probe`, `_liveness_failures`, `_liveness_probe_locks`)
finds removal sites at only two places: the healthy-probe reset inside `_check_engine_liveness`, and the new
`_retire_liveness_state`. Nothing else pops them, and `_liveness_watchdog` skips not-ready engines, so it never
revisits a dead key. Therefore a normal load → traffic → unload/evict cycle retains, forever, one `threading.Lock`
(`_liveness_probe_locks` is created by `_liveness_probe_lock` on every liveness check, which every piped LLM request
triggers) plus one `_liveness_last_probe` entry for the dead `(port, generation)`.

Second residual in the same fix: `_check_engine_liveness` writes `_liveness_last_probe[key]` / `_liveness_failures[key]`
after its `/health` probe, so a thread that is mid-probe re-creates 2 of the 3 entries a concurrent
`_retire_liveness_state` just popped (the process-exited path takes no probe lock, so it can retire while another
thread holds one). Measured with a two-thread scratch probe: `probe_locks` empty, the other two still keyed.

## Implication for future work

Prefer a path-independent sweep — in `_liveness_watchdog`, pop any key for which `_find_engine(*key)` returns `None`
— over adding a second hook per retirement path. It closes both residuals above and cannot regress probe-stampede
protection, since state for an unreachable or not-ready key is already gated off in `_check_engine_liveness`.

## Correction to the sibling memory

[[forwarder-generation-keyed-state-retirement]] states (its "Why unconditional popping is correct" bullet) that
"`_handle_confirmed_hang` is reached only from `_check_engine_liveness` outside the probe lock". That premise is
wrong — see the third call site in `_pipe_to_backend_inner` above. Its conclusion (no deadlock: no caller holds
`_liveness_lock`) still holds, verified against all three sites.
