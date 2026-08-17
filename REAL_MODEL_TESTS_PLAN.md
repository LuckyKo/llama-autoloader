# Plan: Real-Model Load Tests (Qwen2.5 0.5B)

Add `tests/test_real_model.py` — E2E tests that drive the **real, unmocked** load
pipeline against a tiny local GGUF model (`Qwen2.5-0.5B-Instruct-Q8_0.gguf`).

## Goal
Prove the true spawn → readiness → proxy round-trip works with an actual
`llama-server.exe`, closing the biggest coverage gap (real subprocess load + real
inference traffic). Must **gracefully skip/no-op** when the model or backend is
absent so CI on a machine without the model stays green.

## Target model
- Path: `N:\work\stuff\Beta\lmstudio-community\Qwen2.5-0.5B-Instruct-GGUF\Qwen2.5-0.5B-Instruct-Q8_0.gguf`
- Model id (filename, used by `scan()`): `Qwen2.5-0.5B-Instruct-Q8_0.gguf`
- Lives under `root_dir` (`n:/work/stuff/Beta`) → discovered by `ModelManager.scan()`.

## Graceful-skip gate (REQUIRED)
Module-level probe `_can_run_real_model()` returning a bool, used as a class-level
`@pytest.mark.skipif(not _can_run_real_model(), reason=...)`:
1. Resolve the model: confirm `Qwen2.5-0.5B-Instruct-Q8_0.gguf` is present under the
   manager's `root_dir` (via `manager.scan()` result / `manager.gguf_paths`). If not
   found → False.
2. Resolve a real backend binary via `manager.resolve_binary(...)` and confirm the
   file exists on disk. If no binary → False.
Return True only if BOTH hold. The probe must be cheap (no process spawn) and never
raise — any exception → False (skip).

## Isolation
Reuse existing conftest fixtures (`client` / `seeded_manager` / isolated manager swap
+ config redirect to tmp). Because the real load writes state under `save_state_dir`,
ensure that dir is redirected to a tmp path (conftest already redirects config; verify
`save_state_dir` lands in tmp, else point it there in this test's fixture) so the repo
`states/` is untouched.

## Tests (in `TestRealModelLoad`)
1. **test_real_load_ready**: POST `/v1/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf/load`.
   Assert 200, `ready is True`, `port > 0`. Then GET `/v1/status` → `models_loaded >= 1`
   and the model's entry shows `ready: true`.
2. **test_real_completion_roundtrip**: after load (or reuse loaded state), POST a tiny
   chat/completion request through the proxy endpoint and assert a 200 with non-empty
   generated content. Use a minimal prompt + `max_tokens` small (e.g. 16) to keep it
   fast. Discover the correct completion route from server.py before implementing.
3. **test_real_unload_cleans_up**: POST unload for the model → 200; GET `/v1/status` →
   that model no longer loaded (`models_loaded` back to baseline / entry gone); confirm
   the backend process was terminated (no lingering child).

## Hardening / correctness rules
- **No mocking** of `subprocess.Popen`, `_wait_until_ready`, or readiness — this is the
  whole point. The real binary must actually serve `/health`.
- **Bounded time**: rely on the manager's built-in readiness timeout, but keep each test
  well under ~60s for a 0.5B CPU load. If load fails to become ready, fail the test with
  a clear message (do NOT skip — if we got here the model+binary exist, so a failure is
  a real signal). 
- **Robust teardown**: in `finally`, unload any loaded model and `_kill_proc` its
  process (reuse the existing `_kill_proc` helper pattern), add a short `time.sleep(0.2)`
  for Windows TIME_WAIT port release, and log (not swallow) cleanup errors.
- **Idempotent/safe**: tests must not leave orphaned `llama-server.exe` processes.

## Acceptance gate (STRICT — I verify myself)
- `python -m pytest -q` → green. When the model+binary are present, the new real-model
  tests RUN and pass (report which ran vs skipped). When absent, they SKIP cleanly.
- **3 consecutive clean full-suite runs** all green (no flakiness, no leftover procs,
  no WinError 10055). Report each run's line.
- `python -m pytest test_server.py -q` → still 59 passed.
- Production untouched: `git diff --stat -- server.py config.yaml static/app.js test_server.py` empty.
- Repo `states/` and `config.yaml` not modified by the tests (verify via git status after a run).
- Syntax-check touched files.

Do NOT commit. Report: files changed, which real-model tests ran vs skipped on this
machine, the 3 consecutive run lines, and confirmation production + repo state are clean.
