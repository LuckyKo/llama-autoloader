---
tags: [engine-generation, generation-sentinel, cross-module-contract, tcp-forwarder, model-manager]
aliases: [gen-matches, legacy-generation, expected-generation-naming]
related: "[[forwarder-generation-keyed-state-retirement]] [[liveness-state-leak-persists-on-manager-side-retirement]]"
confidence: verified
author: coder_polish-coder-b_20261020_180019.jsonl
---

# One generation-match predicate and one "no filter" sentinel, shared by both modules

The "is this the engine the caller routed to?" check used to be re-implemented ad-hoc at
six sites with two different no-filter sentinels (`0` in the manager, `None` in the
forwarder's `cancel_active`/`recovery_in_progress` path) and two parameter names
(`generation` vs `expected_generation`). It is now a single shared definition.

## Where the single definition lives

`tcp_forwarder.py` (module level, right after the logger), **not** `server.py`:

- `LEGACY_GENERATION = 0`
- `gen_matches(recorded, expected_generation)` — one-sided: True when the **expectation**
  is legacy (any value `<= 0`) or the tokens are equal.
- `failure_record_applies(recorded, expected_generation)` — `gen_matches` **plus**
  `recorded == LEGACY_GENERATION`.

`server.py` imports them (`from tcp_forwarder import (LEGACY_GENERATION, RawTCPForwarder, failure_record_applies, gen_matches)`)
and exposes `ModelManager._gen_matches(recorded, expected_generation)` as a thin wrapper so manager
call sites read uniformly. The record-side check calls `failure_record_applies` directly — it is
deliberately a different rule, so hiding it behind a near-identical wrapper invites accidental unification.

**Why the constant lives in tcp_forwarder.py:** `server.py` already imports `tcp_forwarder`
(`server.py:60` at `d977c82`). A module-level `from server import ...` inside `tcp_forwarder.py`
would be cyclic and order-dependent — importing `tcp_forwarder` first would hit
`RawTCPForwarder` before it is defined. The acyclic direction is the only safe one.

## The sentinel contract (do not reintroduce `None`)

- Engine generations are manager-assigned and **always >= 1**; `0` is the
  backwards-compatible "unknown instance" form meaning *match by port only*.
- Every parameter carrying the caller's expectation is named **`expected_generation`**
  with default `LEGACY_GENERATION`: `ModelManager.note_request_start`,
  `_record_recovery_failure`, `recovery_retry_after`, `recover_dead_model`,
  `_recover_dead_model_impl`, `recovery_in_progress`,
  `RawTCPForwarder._find_engine`, `cancel_active`, `_recovery_retry_after`,
  `_send_recovery_error`.
- `recovery_in_progress(dead_port, expected_generation=LEGACY_GENERATION)` replaced an
  `Optional[int] = None` sentinel; the forwarder's queue path no longer translates
  `0 -> None` before calling it.
- Internal bookkeeping helpers keyed by a concrete instance keep the name `generation`
  (`_register_active`, `_unregister_active`, `_check_engine_liveness`, `_handle_confirmed_hang`,
  `_schedule_recovery`, `_reserve_recovery_waiter`) and the pipe path keeps
  `engine_generation`: those hold an actual token used as a dict key, not a filter.

## Why there are two predicates, not one

All four sites looked like one predicate, but they are two different rules:

- **Engine identity** (`_find_engine`, `note_request_start`, `_recover_dead_model_impl`): the
  old guards were `expected_generation > 0 and recorded != expected_generation` — one-sided.
  A resident carrying no token must NOT be treated as the specific instance the caller routed
  to; making the helper symmetric **widened** `_recover_dead_model_impl`'s skip branch, so a
  tokenless resident would be reported as a "healthy replacement" and recovery would stop
  silently (request retries the same dead engine → `_recovery_attempts >= 1` → 503).
- **Failure-record applicability** (`recovery_retry_after`): the old guard was
  `rec.generation not in (0, expected_generation)` — already lenient about the *record*.
  A record with generation 0 is model-level state written when the token could not be
  resolved, and it must keep suppressing retries. A one-sided helper would have dropped that
  leniency and allowed a doomed reload cycle.

Sentinel domain: the old code treated **any value `<= 0`** as legacy (and `_record_recovery_failure`
resolved `generation <= 0` from the registry). Preserved via `<=` comparisons; generations are
never negative in practice (the counter only increments), but the domain is unchanged.

## Naming gotcha

`LoadedModel.generation` / `RecoveryFailure.generation` are real attributes — never renamed.
Inside `_record_recovery_failure`, the local `generation` is the *resolved record token*
(seed from `expected_generation`, resolved from `self.loaded` when the legacy form was passed).

Tests call these parameters **positionally** except `expected_generation=` and
`engine_generation=` keywords, so renaming `engine_generation` would break
`tests/test_backend_liveness_backoff.py` and `tests/test_dead_backend_recovery.py`.
