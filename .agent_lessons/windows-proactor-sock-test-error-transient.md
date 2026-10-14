---
tags: [llama-autoloader, pytest, windows, asyncio, flaky, proactor, triage]
aliases: [windows-proactor-sock-flake]
related: [[plan-review-cycle-before-implementation]], [[refute-agent-claimed-root-cause]]
confidence: verified
author: Maine_20261006.jsonl
---

# Windows ProactorEventLoop `_ssock` ERROR is transient flakiness, not a failure

## Symptom
A full `pytest tests/ -q` run on this repo occasionally reports **one** entry as
`ERROR at setup of <test>` (not FAILED) with this traceback tail:

```
asyncio/proactor_events.py:640: in __init__
...
AttributeError: 'ProactorEventLoop' object has no attribute '_ssock'
PytestUnraisableExceptionWarning: Exception ignored in: <function BaseEventLoop.__del__ ...>
```

Count drifts by ±1 between runs (e.g. "174 passed, 1 error" then "175 passed").

## What it is
A Windows ProactorEventLoop teardown race during pytest-asyncio fixture setup — the loop's
self-pipe socket `_ssock` is already gone when `close()` runs in `__del__`. It hits a
*random* test at setup time; it is NOT a defect in that test.

## How to confirm (before touching any code)
1. Run the "erroring" test **in isolation**: `python -m pytest tests/<file>::<test> -q`.
   If it passes → flake, not a real failure.
2. Re-run the full suite once more; if the count returns to all-green (or the error lands
   on a *different* test), it's confirmed transient.

## Rule
Do NOT "fix" the named test — there is nothing wrong with it. Confirm via isolation + one
clean re-run, then proceed. Only escalate if the isolated run also fails or the same test
errors repeatedly (then it's a real setup bug, not the `_ssock` race).
