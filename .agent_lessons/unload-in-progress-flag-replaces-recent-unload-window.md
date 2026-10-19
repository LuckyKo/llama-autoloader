---
tags: [kv-state, safe-mode, concurrency, autoloader]
aliases: [unload-in-progress-flag]
related: [[asyncio-lock-blocking-io-starvation]], [[httpx-dead-socket-exception-signals]], [[system-agents-never-use-shell-safe-mode]], [[safe-mode-kv-regressions-report]]
confidence: verified
author: coder_safe-mode-409-fix-build-1_20261019_111556.jsonl
---

# The 409 "recent unload" window was replaced by an in-progress flag

**llama-autoloader @ 6d447faa** (the commit that introduced the window) → replaced 2026-10-19,
uncommitted. See `reports/SAFE_MODE_KV_REGRESSIONS_20261019.md` and
`reports/SAFE_MODE_409_FIX_SUMMARY.md` in the agent workspace.

## Fact

`RECENT_UNLOAD_REFUSE_WINDOW_S` (10 s wall clock, armed at the start of `_unload_unlocked`,
raised as HTTP 409 by `load_state`) is a *symptom patch*: it refuses the safe-mode contract's
own post-command restore, which by design lands ~0.5–1 s after `POST /v1/unload_all` returned
200. The fix (`_unload_in_progress: set[str]`, armed only for deliberate releases, cleared in a
`finally` after queue-drop + process kill) makes the refusal intent-aware.

## Corrected timeline (the report got this wrong)

The report says BUG_0089's stale restore landed "2 ms DURING the unload". `logs/console.log.1:64154-64157`
says otherwise:

```
11:07:09,443 Unloading strata-coder-iq1_m (port 9021)
11:07:12,461 POST /v1/unload_all → 200 OK      ← teardown took ~3 s and had finished
11:07:12,463 Auto-loading 'strata-coder-iq1_m' before state restore
```

The stale restore landed **2 ms after the unload completed**. So the in-progress flag never
refused it, and never needed to: the caller that produced it (the advisor's early restore) was
already removed on the AgentCascade side. The 409 was guarding nothing but legitimate traffic —
see skill `new-guard-collides-with-deferred-legal-caller`.

## Why the DURING case is unreachable today

Every caller of `_unload_unlocked` holds `manager._lock` across the whole teardown
(`_do_load_model`'s `async with`, `unload_all`, `_unload`), and the kill is awaited inside it
— see [[asyncio-lock-blocking-io-starvation]]. `load_state` must acquire that same lock to
look the model up at all, so it can only observe the flag *after* the teardown released the
lock, at which point the `finally` already cleared it. Net: with the current lock discipline
the 409 cannot fire for any real caller; it is a race backstop that becomes live only if the
lock is ever narrowed around the kill.

Consequence for tests: `tests/test_safe_mode_slot_ordering.py` cannot drive a live
`load_state` during an in-progress unload. It pins (a) the guard refuses while the flag is set,
and (b) the flag's lifetime is exactly the real teardown (`test_flag_lifetime_is_exactly_the_teardown`
drives a real `unload_model` with only `_perform_slot_save` mocked).

## Gotcha

Reverting `server.py` to HEAD for a revert-proof breaks collection: the whole safe-mode feature
is uncommitted (`RECENT_UNLOAD_REFUSE_WINDOW_S` absent at HEAD) — see
[[httpx-dead-socket-exception-signals]].
