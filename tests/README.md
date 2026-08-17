# llama-autoloader test suite

Automated tests for the FastAPI + llama.cpp model loader in this repo. Covers
regression (unit-level behavior of the core functions) and end-to-end startup
(the real HTTP surface driven through FastAPI's `TestClient`).

## Running

```powershell
cd N:\work\stuff\Beta\llama-autoloader
python -m pytest            # full suite (existing test_server.py + tests/)
python -m pytest tests/test_regression.py   # regression only
python -m pytest tests/test_e2e_startup.py  # E2E startup only
```

No real llama.cpp binary and no real GGUF model files are required — the default
suite passes without either. See "Mock / fake backend" below.

## Layout

| File | Purpose |
|------|---------|
| `tests/__init__.py` | Makes `tests/` a package (so `from tests import gguf_bytes` works). |
| `tests/gguf_bytes.py` | GGUF byte builders: `_make_dummy_gguf(name, max_ctx, ctx_vtype=...)` plus helpers to emit arbitrary KV fields of any GGUF value type. Reuses the exact byte layout proven in `test_server.py::_make_dummy_gguf`. |
| `tests/conftest.py` | Fixtures: `build_cfg`, `isolated_manager`, `seeded_manager`, `client`. |
| `tests/test_regression.py` | Regression tests (plan §3). |
| `tests/test_e2e_startup.py` | E2E startup tests (plan §4). |

A root-level `conftest.py` excludes the pre-existing ad-hoc scripts
(`test_direct.py`, `test_kv_cache_reuse.py`, `test_kv_fix.py`, `test_slot_restore.py`)
from collection — they make live HTTP calls at import time and are not unit tests.

## Regression vs E2E split

- **Regression** (`test_regression.py`): fast, no HTTP server. Exercises the pure
  logic directly: GGUF parser matrix, `_ensure_gguf_name` caching/idempotency/
  concurrency, `validate_config`, `resolve_binary` priority, port allocation +
  exhaustion, `to_launch_args`, and state save/list/load/delete path logic.
- **E2E startup** (`test_e2e_startup.py`): drives the real `server.app` over HTTP
  via `TestClient`. Covers startup happy path (`/health`, `/`, `/static/*`,
  `/v1/models`, `/v1/status`, `/v1/backends`, `/v1/settings`), startup failure
  mode, and the load/unload lifecycle.

## Isolation strategy (no production-code change)

Each test gets a fully isolated `ModelManager` built from a config rooted under
pytest's `tmp_path`. The fixture **swaps the module-level `server.manager` global**
to this instance, yields it, then restores the original and `await manager.stop()`
in teardown (guarded against double-stop). This means:

- No route or signature changes to `server.py` were needed.
- Background tasks (`idle_reaper`, `_status_cache_updater`) are cancelled by
  `manager.stop()`; each E2E test's manager is stopped exactly once.
- The `client` fixture wraps `TestClient(server.app)` as a context manager so the
  app's startup/shutdown events fire, and guarantees teardown exits the client and
  confirms `manager.stop()` ran.

## Mock / fake backend (load path)

The load path is exercised without a real binary in two ways:

1. **Deterministic (decision 3a)** — `subprocess.Popen` is patched with a fake
   process whose `.poll()` returns `None`, and `_wait_until_ready` is patched to
   return `True` (or `False` for the failure case). This drives the full
   `load_model → _do_load_model → ready/unload` pipeline over real HTTP.
2. **Best-effort real subprocess (decision 3b)** — a tiny in-process fake
   "llama-server" (a Python child that parses `--port <n>` from argv, binds
   `127.0.0.1:<n>`, serves `/health` → 200, and exits cleanly on termination) is
   launched through the *real* spawn path with the *real* `_wait_until_ready` HTTP
   poll. This test is marked `skipif` when a Python subprocess can't bind/serve on
   the platform, and always cleans up its child process + frees the port in teardown.

## What's covered (summary)

- GGUF metadata parsing: name/context extraction, vtype matrix, skip paths.
- `_ensure_gguf_name`: caching, idempotency, concurrent-call consistency.
- `validate_config`: valid config, out-of-range port, bad host, missing keys.
- `resolve_binary`: backend-dir priority, selected-backend override, fallback.
- Port allocation: sequential allocation, exhaustion → error.
- `to_launch_args`: full argv construction (model/host/port/ctx/n-gpu-layers/alias/slot-save-path).
- State save/list/load/delete path logic.
- E2E startup: health, index HTML, static assets, models populated from GGUF bytes,
  status coherence, backends, settings get/patch/validation.
- E2E load/unload lifecycle (mocked readiness + best-effort real subprocess).

## Notes / limitations

- The existing `test_server.py` (59 tests) is untouched and remains green.
- Windows-specific: the suite reuses a single shared asyncio event loop for sync
  tests to avoid Winsock buffer exhaustion, and uses a listening socket (no
  `SO_REUSEADDR`) to make port-availability checks fail deterministically.
