---
tags: [pytest, windows, test-discovery, llama-autoloader]
aliases: [pytest-invocation, bare-pytest-fails]
confidence: verified
author: coder_ui_builder_20261006_112342.jsonl
contributors:
  - coder_loader-slots-build_20261009_070016.jsonl
  - reviewer_lru-eviction-review_20261020_155658.jsonl
  - coder_polish-coder-a_20261020_164840.jsonl
---

# `python -m pytest -q` at the repo root FAILS collection — use the explicit paths

**Fact:** in `N:\work\stuff\Beta\llama-autoloader`, a bare `python -m pytest -q` aborts the whole
run with `ERROR collecting temp/test_kv_fix.py` → `urllib3.exceptions.NewConnectionError:
[WinError 10061] Connection refused`, and prints **0 tests run** (exit code 2).

**Canonical invocation (the green suite):**
```
python -m pytest tests/ test_server.py -q
```

## Why

`pytest.ini` sets `norecursedirs = *.egg backends states static __pycache__ .git` and the root
`conftest.py` `collect_ignore`s the four ad-hoc live-HTTP scripts **at the repo root**. Neither
mechanism covers the `temp/` **directory** — `norecursedirs` has no `temp` entry and
`collect_ignore` is path-specific. So `temp/test_kv_fix.py` is collected, makes a live HTTP call to
127.0.0.1 at *import* time, and dies before any test runs.

## Symptom-vs-cause trap

The traceback is ~90 lines of urllib3/http.client frames pointing at a "connection" problem, which
reads like an environment issue. It is not — nothing about the server was running. Reading only the
last line ("1 error during collection") hides that **zero** tests executed, so a broken collection
can be mistaken for a broken feature.

## Check

Baseline for 2026-10-06 is **216 passed** (~70 s). After the Strata bridge-card build: **223 passed**
(+7). After the generic multi-engine bridge: **238 passed** (+15). After the save_state_max_keep
setting (2026-10-09): **245 passed** (+7). Use the counts, not the exit code, to detect regressions.
⚠️ Quoted baselines go stale fast in this repo — re-verify the pre-change count with a
`--collect-only` run before trusting a "N passed" figure from a plan.

## Addendum 2026-10-20 @ 3926acecf917796ed2fbf2a0073d97685a863219

`pytest.ini` `norecursedirs` now includes **`temp`** (and `_harness`):

```
norecursedirs = *.egg backends states static temp _harness __pycache__ .git
```

So the "neither mechanism covers `temp/`" diagnosis above no longer holds — a bare
`python -m pytest -q` at the repo root no longer dies on `temp/test_kv_fix.py`. The
explicit-path invocation is still the safe habit (and still required for anything
placed in a *new* directory), but the specific failure mode described above is fixed.
The "0 tests run can look like a broken feature" trap remains valid for any
collection error.

## Addendum 2026-10-20 @ d977c82bd3202cdc8ed7523190be478c211e6165

Current full-suite count for the canonical invocation is **397 passed in ~125 s**
(`python -m pytest tests/ test_server.py -q -n 0`). The older counts above are historical only.

**Do not add `-x` to this invocation.** Measured: an `-x` run aborted at ~64% with
`ERROR at setup of tests/test_safe_mode_slot_ordering.py::TestSlotActionRetry::test_read_timeout_still_tries_other_formats`
→ `OSError: [WinError 10055]` inside `asyncio`'s `socket.socketpair()` while *constructing a new
event loop*. That is Windows socket-queue/ephemeral-port exhaustion accumulated across thousands of
sequential event loops — **`-n 0` does not prevent it**, so serial mode is not evidence the cause is
code. It passed in isolation (`1 passed in 0.39s`), and the same event also emits a harmless
`AttributeError: 'ProactorEventLoop' object has no attribute '_ssock'`
`PytestUnraisableExceptionWarning` from the aborted loop's `__del__` — noise, not a second failure.

The trap is the reported count: `255 passed, 1 error` under `-x` is a **truncated** figure — the
remaining ~142 tests never ran, so a real regression later in collection order would be invisible.
Correct move after hitting it: re-run the same selection WITHOUT `-x` for complete coverage, then
report that number.

## Related

- [[llama-autoloader-tests-implementation]] (AgentCascade lessons) — original discovery-scoping fix.
- [[plan-review-backend-api-mismatch]] — the four-seam field plumbing rule.
- [[sequential-long-pytest-runs-windows]] — don't run pytest suites concurrently on this box.
