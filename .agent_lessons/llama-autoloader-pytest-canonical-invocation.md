---
tags: [pytest, windows, test-discovery, llama-autoloader]
aliases: [pytest-invocation, bare-pytest-fails]
confidence: verified
author: coder_ui_builder_20261006_112342.jsonl
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
(+7). Use the counts, not the exit code, to detect regressions.

## Related

- [[llama-autoloader-tests-implementation]] (AgentCascade lessons) — original discovery-scoping fix.
- [[plan-review-backend-api-mismatch]] — the four-seam field plumbing rule.
- [[sequential-long-pytest-runs-windows]] — don't run pytest suites concurrently on this box.
