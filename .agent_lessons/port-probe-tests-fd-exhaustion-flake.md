---
tags: [llama-autoloader, pytest, windows, flaky, port-probe, triage]
aliases: [port-probe-fd-exhaustion-flake]
related: [[windows-proactor-sock-test-error-transient]], [[multi-instance-three-ports]]
confidence: verified
author: reviewer_build-reviewer-llama_20261020_133137.jsonl
---

# `_port_in_use` port-probe tests FAIL under full-suite load (fd exhaustion), not a regression

## Symptom
A full `python -m pytest tests/ -q -n 0` run at HEAD of the 8-commit bug-fix set (base `c102df5`) reports **2 FAILED** (not ERROR):

- `tests/test_regression.py::TestMultiInstancePorts::test_port_in_use_true_when_listening`
- `tests/test_regression.py::TestMultiInstancePorts::test_free_port_is_returned_unchanged`

These are DIFFERENT from the `_ssock` "ERROR at setup" flake in [[windows-proactor-sock-test-error-transient]] — that one is an ERROR at fixture setup on a random test; this one is a hard FAILED on two *specific, named* port tests.

## What it is
`_port_in_use()` (server.py) uses a `socket.connect_ex` probe (see [[multi-instance-three-ports]]: a connect probe, not a bind probe). Under a full run the process nears the Windows fd/socket ceiling (`WinError 10055`), so `connect_ex` returns the wrong code and the "is this port listening / is this port free" assertions flip. The functions themselves are correct.

## How to confirm (before touching any code)
1. Run the two tests **in isolation**: `python -m pytest tests/test_regression.py::TestMultiInstancePorts -q` → 8/8 pass.
2. Run them at the base commit (worktree at `c102df5`) → pass.
3. `git diff c102df5..HEAD -- server.py` shows **zero** changes to `_port_in_use` / `_resolve_internal_port`.

Isolation-green + base-green + full-suite-red = fd-contention flake, not a regression from the change set.

## Rule
Do NOT "fix" `_port_in_use` or these tests because of a full-suite failure. Confirm via isolation + base-commit run + diff check, then proceed. Only escalate if the isolated run also fails.

## Verification-count gotcha (same review)
Bug files claim "356/389/391 passed" but this environment collects only **322** tests (some test files, e.g. real-model tests, are deselected here). Do not trust self-reported pass counts as evidence — re-run and record the actual number.
