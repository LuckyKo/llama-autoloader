"""Shared fixtures for llama-autoloader regression + E2E startup tests.

Isolation strategy (plan decision #1): swap-and-restore the module-level
``server.manager`` global. The real ``server.app`` HTTP surface is exercised
with a fully isolated ``ModelManager`` built from tmp_path dirs, and the
original global is restored in teardown. No production-code changes required.

Import mechanics: pytest inserts the rootdir (repo root, where this conftest's
sibling test_server.py lives) into sys.path via its default "prepend" import
mode, so ``import server`` resolves to the production module regardless of the
current working directory.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable (defensive; pytest's prepend mode usually
# handles this already). This file lives in <root>/tests/, so parent is <root>.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import server  # noqa: E402  (production module under test)
from server import ModelManager  # noqa: E402


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_cfg(tmp_path: Path, *, root_dir=None, save_state_dir=None, backends_dir=None,
              selected_backend: str = "", binary: str = "llama-server",
              base_port: int = 19001, port: int = 19123, host: str = "127.0.0.1",
              max_loaded_models: int = 4, idle_timeout_seconds: int = 3600,
              poll_interval_seconds: float = 2, default_args: str = "",
              auto_save_state: bool = False) -> dict:
    """Build a minimal, fully-isolated config dict pointing at tmp_path dirs."""
    root_dir = Path(root_dir) if root_dir else tmp_path / "models"
    save_state_dir = Path(save_state_dir) if save_state_dir else tmp_path / "states"
    backends_dir = Path(backends_dir) if backends_dir else tmp_path / "backends"
    for d in (root_dir, save_state_dir, backends_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "models": {
            "root_dir": str(root_dir),
            "save_state_dir": str(save_state_dir),
            "idle_timeout_seconds": idle_timeout_seconds,
            "max_loaded_models": max_loaded_models,
            "auto_save_state": auto_save_state,
        },
        "llama_server": {
            "binary": binary,
            "backends_dir": str(backends_dir),
            "selected_backend": selected_backend,
            "default_args": default_args,
            "default_n_gpu_layers": 999,
        },
        "launcher": {
            "host": host,
            "port": port,
            "base_port": base_port,
        },
        "gpu": {"poll_interval_seconds": poll_interval_seconds},
    }


# ---------------------------------------------------------------------------
# Manager isolation fixture (swap-and-restore)
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_manager(tmp_path):
    """Build an isolated ModelManager, swap it into server.manager, yield it.

    Teardown: restore the original global and stop the manager exactly once
    (guarded against double-stop). Dummy GGUF files are NOT created here so the
    caller controls model contents; see the ``seeded_manager`` helper below or
    create files in ``manager.root_dir`` before use.
    """
    cfg = build_cfg(tmp_path)
    mgr = ModelManager(cfg)

    # Redirect config persistence to a tmp file so endpoints that write back
    # (PUT /v1/settings, backend selection) never touch the real config.yaml.
    import yaml as _yaml
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w") as f:
        _yaml.dump(cfg, f, default_flow_style=False)
    original_cfg_path = server.CONFIG_PATH
    server.CONFIG_PATH = str(cfg_file)

    original = server.manager
    server.manager = mgr
    try:
        yield mgr
    finally:
        # Restore the original global + config path so any late request sees a
        # valid manager and the real config file is untouched.
        server.manager = original
        server.CONFIG_PATH = original_cfg_path
        _stop_manager_once(mgr)


@pytest.fixture()
def seeded_manager(isolated_manager):
    """isolated_manager with three dummy GGUFs (known name/ctx bytes) already scanned."""
    from tests import gguf_bytes as gb

    root = isolated_manager.root_dir
    # model-a: has general.name + context_length (UINT32)
    gb.write_gguf(root / "model-a.gguf", name="Alpha Model", max_ctx=4096, ctx_vtype=gb.UINT32)
    # model-b: name + UINT64 context_length
    gb.write_gguf(root / "model-b.gguf", name="Beta Model", max_ctx=8192, ctx_vtype=gb.UINT64)
    # model-c: only a name (no context_length)
    gb.write_gguf(root / "model-c.gguf", name="Gamma Model", max_ctx=None)

    return isolated_manager


# ---------------------------------------------------------------------------
# Session-scoped event loop (root-cause fix for Windows ProactorEventLoop churn)
# ---------------------------------------------------------------------------

_SESSION_LOOP = None


def _get_session_loop() -> asyncio.AbstractEventLoop:
    """Return the single session-scoped event loop, creating it lazily.

    pytest-asyncio 1.x is configured (pytest.ini: asyncio_default_fixture_loop_scope =
    session) to run all async tests on ONE loop, so there is no per-test ProactorEventLoop
    churn. This helper gives synchronous teardown code a valid loop to drive coroutines
    on WITHOUT creating a throwaway loop (which would itself risk WinError 10055 and can
    abandon the coroutine). Created lazily; closed by the session-scoped fixture below.
    """
    global _SESSION_LOOP
    if _SESSION_LOOP is None or _SESSION_LOOP.is_closed():
        _SESSION_LOOP = asyncio.new_event_loop()
    return _SESSION_LOOP


@pytest.fixture(scope="session")
def _session_loop_cleanup():
    """Close the session loop exactly once at the very end of the test session."""
    yield
    if _SESSION_LOOP is not None and not _SESSION_LOOP.is_closed():
        try:
            _SESSION_LOOP.close()
        except Exception as e:  # pragma: no cover - defensive
            server.log.warning(f"closing session loop failed: {e}")


def _stop_manager_once(mgr) -> None:
    """Stop the manager exactly once, driving its coroutine on the session loop.

    Guarded so a second call (e.g. TestClient shutdown already stopped it) is a no-op.

    Root-cause fix: teardown runs in synchronous test code where NO event loop is running
    (the async test's loop has been torn down, and the TestClient portal loop is closed).
    The old ``asyncio.run(mgr.stop())`` created a throwaway ProactorEventLoop each time —
    on Windows that both risks OSError [WinError 10055] (socketpair buffer exhaustion) and
    could abandon the coroutine ("coroutine 'ModelManager.stop' was never awaited").

    Instead we drive ``mgr.stop()`` to completion on the shared session loop via a task, so
    the coroutine is ALWAYS awaited exactly once, no throwaway loop is created, and it never
    raises on Windows. If the session loop is somehow unavailable we fall back to a fresh
    loop but still await the coroutine (no abandonment).
    """
    if getattr(mgr, "_test_stopped", False):
        return
    mgr._test_stopped = True
    coro = mgr.stop()
    try:
        loop = _get_session_loop()
        task = loop.create_task(coro)
        loop.run_until_complete(task)  # always awaited to completion
    except Exception as e:  # pragma: no cover - defensive
        # If the session loop path failed (e.g. loop closed), ensure the coroutine is still
        # driven to completion on a fresh loop rather than abandoned.
        try:
            asyncio.run(coro)
        except Exception as e2:
            server.log.warning(f"manager.stop() during teardown failed: {e} / fallback: {e2}")
        else:
            server.log.warning(f"manager.stop() via session loop failed ({e}); recovered on fresh loop")


# ---------------------------------------------------------------------------
# TestClient fixture (drives real app startup/shutdown over HTTP)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(seeded_manager):
    """FastAPI TestClient as a context manager so @app.on_event('startup') fires.

    Depends on ``seeded_manager`` (which itself depends on ``isolated_manager``)
    so the real server.app surface runs against an isolated, pre-scanned manager.
    Teardown exits the client (triggering shutdown -> manager.stop()) and
    guarantees the manager is stopped exactly once; then asserts cleanliness.
    """
    from fastapi.testclient import TestClient

    with TestClient(server.app) as c:
        yield c

    # Client context exit already ran shutdown -> manager.stop(). Ensure it is
    # definitely stopped (guarded against double-stop), then verify cleanliness.
    _stop_manager_once(seeded_manager)
    assert not seeded_manager.loaded, "models should be unloaded after teardown"
    _assert_no_leaked_tasks(seeded_manager)


def _assert_no_leaked_tasks(manager) -> None:
    """Assert the manager's background tasks are gone/cancelled after stop().

    TestClient runs its own loop in a portal thread; by the time we're back in
    sync test code that loop is closed. The only thing we can robustly assert here
    is that the manager's background tasks are no longer live (stop() cancels them
    and clears the attributes).
    """
    for attr in ("_bg_task", "_status_task"):
        task = getattr(manager, attr, None)
        if task is not None:
            assert task.done(), f"{attr} should be done after stop()"
