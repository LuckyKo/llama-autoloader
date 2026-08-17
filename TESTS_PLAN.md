# Plan (v2): `tests/` directory — regression + E2E startup tests

## Goal
Create a `tests/` package with (a) regression tests locking in current behavior of the
recently-refactored code, and (b) end-to-end tests that confirm the loader actually starts up
and "everything is in order" over the real HTTP surface.

## Verified facts (from reading server.py) — these drive the design
- `import server` runs module-level code: reads `$AUTOLOADER_CONFIG` or `config.yaml`, calls
  `ModelManager.validate_config`, builds global `manager = ModelManager(CFG)` (~L1308-1315), and
  mounts `/static` (~L1328). All route handlers read the module-level `manager` **at request time**
  (e.g. `health()` uses `manager._lock`; `_resolve_proxy_model` uses `manager.proxy`).
- `test_server.py` imports ONLY classes/functions (`ModelManager, ModelConfig, LoadedModel,
  _read_gguf_metadata, _read_gguf_name`) and builds its OWN `ModelManager(_make_cfg())` in a fixture.
  It does NOT use the global `manager` or `app`. → The two test files can safely share the imported
  `server` module without cross-contamination.
- Startup: `@app.on_event("startup") async def _startup(): await manager.start()` (~L1708).
  `manager.start()` runs `scan()` + spawns 2 bg tasks (`idle_reaper`, `_status_cache_updater`).
  `manager.stop()` cancels both, cancels in-flight loads, unloads all, and `await client.aclose()`.
- Real load: `_do_load_model` (~L720) builds argv via `cfg.to_launch_args(path, port, ...)` which
  passes `--port <port>` (and `--model`, `--host 127.0.0.1`, etc.), spawns a subprocess, then
  `_wait_until_ready(port, proc)` (~L658) polls `http://127.0.0.1:{port}/health` and treats **HTTP 200**
  as ready (no body check); it also returns False if `proc.poll()` shows premature exit.
- No pytest config exists; existing async tests use `@pytest.mark.asyncio` (strict mode).
  httpx 0.28 + FastAPI TestClient available.

## Design decisions (adjudicated against review)
1. **Isolation strategy — swap-and-restore the module global.** E2E conftest builds a fresh
   `ModelManager(isolated_cfg)` and temporarily sets `server.manager = <that instance>` for the test,
   restoring the original in teardown (try/finally). This exercises the REAL `server.app` HTTP surface
   with a fully isolated manager, requires NO route changes, and carries zero regression risk to
   production code. (Rejected: large `create_app()` factory refactor touching every route — overkill;
   rejected: relying on import-order env vars — fragile.)
2. **TestClient + async startup.** Use `fastapi.testclient.TestClient(server.app)` as a context manager
   so `@app.on_event("startup")` fires and bg tasks spawn. Wrap in a fixture with guaranteed teardown:
   exit the TestClient (triggers shutdown → `manager.stop()`) AND explicitly `await manager.stop()` if
   not already stopped, to cancel bg tasks and close the httpx client. Add an assertion that no stray
   asyncio tasks remain after teardown.
3. **Load-path coverage — two layers:**
   - (a) Deterministic unit/integration test: mock at the `subprocess.Popen` + `_wait_until_ready`
     boundary (or point `resolve_binary` at a stub) to exercise `load_model → ready=True → status` and
     `unload_model`. Always runs, no real binary needed.
   - (b) Best-effort REAL subprocess E2E: a tiny Python fake "llama-server" that parses `--port <n>`
     from argv, binds it, and serves `/health` → 200. Marked `pytest.mark.skipif` when the interpreter/
     platform can't run it. Exercises the true spawn→wait→ready pipeline. Optional; must not fail CI.
4. **pytest.ini.** Add a minimal `pytest.ini`: `asyncio_mode = auto` (so new async tests need no marker;
   existing marked tests still pass), and do NOT restrict testpaths (let pytest discover both
   `test_server.py` at root and `tests/`). Verify the existing 59 still pass after adding it. Do NOT move
   `test_server.py` (unnecessary churn).
5. **Shared helper.** One `tests/gguf_bytes.py` with `_make_dummy_gguf(...)`; imported by both regression
   and E2E tests (single source of truth).

## Deliverables
### 1. `pytest.ini` (repo root)
- `[pytest]`, `asyncio_mode = auto`. Confirm existing 59 tests still green.

### 2. `tests/` package
- `tests/__init__.py`
- `tests/gguf_bytes.py` — `_make_dummy_gguf(name, max_ctx, ctx_vtype=UINT32)` (+ helpers to emit extra
  KV fields of arbitrary types for skip-path testing). Reuse the byte layout already proven in
  test_server.py.
- `tests/conftest.py`:
  - Fixture `isolated_manager(tmp_path)`: builds a config dict (root_dir/save_state_dir/backends_dir all
    under tmp), creates dummy GGUF files in root_dir, constructs `ModelManager(cfg)`, swaps
    `server.manager` to it, yields it, restores original + `await manager.stop()` in finally.
  - Fixture `client(isolated_manager)`: `TestClient(server.app)` context manager with guaranteed teardown;
    asserts clean shutdown (no leaked tasks).
  - Dummy-GGUF creation helpers wired to `gguf_bytes`.

### 3. Regression tests — `tests/test_regression.py`
Lock in refactor behavior + important adjacent logic:
- GGUF binary parser: name+ctx; missing name; missing ctx; UINT64/INT64 ctx; skip over FLOAT32/BOOL/ARRAY;
  unknown vtype → (None,None); corrupt header → (None,None); kvs sanity bound.
- `_ensure_gguf_name`: sets gguf_name; defaults display name to stem when unedited; does NOT overwrite an
  explicit user name; sets max_ctx_size only when None; **idempotent** (second call performs no re-read —
  assert via a spy/counter on `_read_gguf_metadata`).
- **Concurrency**: N concurrent `_ensure_gguf_name` calls for the same model → metadata read happens
  correctly and config ends up consistent (no lost updates / no crash).
- `validate_config`: valid passes; each missing section / bad value raises ValueError.
- `resolve_binary` priority: per-model > global selected > first scanned > fallback binary.
- `_allocate_port` / `_is_port_available`; **port exhaustion** behavior when range is exhausted.
- `ModelConfig.to_launch_args`: expected argv incl. port, path, default args, mmproj when set, alias.
- State save/load: `save_state` → `list_states` shows it; `load_state` path; `delete_state`.

### 4. E2E startup tests — `tests/test_e2e_startup.py`
Confirm "loader starts up and everything is in order" over real HTTP (using `client` + `isolated_manager`):
- **Startup happy path** (root_dir has N dummy GGUFs with known name/ctx bytes):
    - `/health` and `/v1/health` → `{"status":"ok", ...}`.
    - `GET /` → 200, contains an index.html marker.
    - `/static/app.js`, `/static/styles.css` → 200.
    - `/v1/models` lists all N models; each has gguf_name/max_ctx_size populated from the GGUF bytes we
      wrote (proves scan + lazy metadata read end-to-end).
    - `/v1/status` coherent: `models_total == N`, `models_loaded == 0`, launcher host/port match config,
      backends list present.
    - `/v1/backends` resolved binary consistent with config selected_backend/fallback.
    - `/v1/settings` GET returns expected defaults; a settings update round-trips.
- **Startup failure mode**: invalid config (fails validate_config) surfaces an error rather than silently
  producing a healthy server.
- **Load/unload lifecycle E2E** (deterministic, mocked readiness per decision 3a): POST load → model in
  `/v1/status` with `ready: true` and correct port; then unload → removed from status. Proves the full
  pipeline over HTTP without real inference.
- **Real-subprocess load E2E** (best-effort, decision 3b, skip if unavailable): fake llama-server serves
  `/health`; POST load → ready:true; unload cleans up process and port.

### 5. `tests/README.md`
What's covered, how to run (`pytest`), regression vs E2E split, note that load-path uses mocks/fake
backend (no real GGUF required), and which tests are best-effort/skippable.

## Out of scope / decisions
- Do NOT rewrite or move existing `test_server.py` (keep passing as-is).
- Do NOT require real llama.cpp binaries or real model files for the default suite to pass.
- Minimal production-code change: NONE required (isolation via global swap). If a tiny testability seam
  proves necessary during implementation, it must not alter any existing behavior and must be reviewed.

## Acceptance criteria
- `pytest` (existing 59 + all new tests) passes green on this machine.
- Regression tests genuinely fail if the refactor's behavior regresses (spot-check by mentally reverting).
- E2E startup test exercises app startup via the real HTTP surface (TestClient), not just unit calls.
- No background-task / port leaks between tests (clean teardown verified).
- Existing 59 tests remain green (no regression from pytest.ini or any change).
