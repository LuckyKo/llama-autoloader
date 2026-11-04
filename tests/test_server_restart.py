"""Tests for the llama-autoloader server-restart helper (server_restart.py).

Mirrors AgentCascade's tests/test_server_restart.py for the in-place re-exec
contract. The re-exec helper tests below are dependency-free (they import only
``server_restart`` and monkeypatch its ``os`` reference so the real re-exec /
``os._exit`` never runs). The final route-registration test imports ``server``
(pulling in FastAPI) to catch a decorator accidentally nested inside another
endpoint — a bug the helper-only tests cannot see.

Run with: python -m pytest tests/test_server_restart.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root is importable regardless of cwd (matches tests/conftest.py approach).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import server_restart  # noqa: E402
from server_restart import restart_server_process  # noqa: E402


class _Sentinel(Exception):
    """Raised by the patched os.execl so restart_server_process() never reaches os._exit."""


def test_restart_uses_inplace_execl_with_expected_args():
    """os.execl is called with exactly [sys.executable, sys.executable, *sys.argv].

    The patched execl raises a sentinel so the function aborts BEFORE os._exit;
    we then assert the exact argument tuple and that os._exit was NOT reached.
    """
    seen: dict = {}

    def _capture(*args, **kwargs):
        seen['args'] = args
        raise _Sentinel()

    with patch.object(server_restart.os, 'execl', side_effect=_capture), \
         patch.object(server_restart.os, '_exit') as mock_exit:
        with pytest.raises(_Sentinel):
            restart_server_process()

    expected = (sys.executable, sys.executable, *sys.argv)
    assert seen['args'] == expected
    mock_exit.assert_not_called()  # sentinel raised before the belt-and-braces os._exit(1)


def test_restart_execl_returns_reaches_exit1():
    """If execl returns (mocked), the belt-and-braces os._exit(1) is reached.

    In production os.execl only returns on failure (after it has already raised),
    so os._exit(1) is unreachable on the success path.
    """
    with patch.object(server_restart.os, 'execl') as mock_execl, \
         patch.object(server_restart.os, '_exit') as mock_exit:
        restart_server_process()

    mock_execl.assert_called_once_with(sys.executable, sys.executable, *sys.argv)
    mock_exit.assert_called_once_with(1)


def test_restart_does_not_use_detached_popen():
    """Guard against a regression to the detached-child approach (source-scan).

    The in-place contract means server_restart must NOT use subprocess/Popen —
    a detached Popen would outlive the console and break restart-in-place. A
    source scan (not just a module-attr check) also catches a future
    ``from subprocess import Popen`` inside the function body.
    """
    import inspect
    src = inspect.getsource(server_restart)
    # The docstring may mention the words (it explains the rationale), but the
    # code must not use a detached Popen: no import, no Popen() call, and no
    # module-level subprocess attribute (a from-import would add one).
    assert "import subprocess" not in src, "regression: server_restart imports subprocess"
    assert "Popen(" not in src, "regression: server_restart calls Popen (detached-child approach)"
    assert not hasattr(server_restart, "subprocess"), "regression: server_restart exposes subprocess"


def test_route_is_registered_on_app():
    """The /v1/restart route must be registered on the FastAPI app.

    Catches the case where the decorator is accidentally nested inside another
    endpoint (making it unreachable dead code). This is the one test that
    imports the full ``server`` module (FastAPI + uvicorn); the re-exec helper
    tests above stay dependency-free.
    """
    import server
    paths = {getattr(r, "path", "") for r in server.app.routes}
    assert "/v1/restart" in paths, (
        f"/v1/restart not registered; restart-ish routes: "
        f"{sorted(p for p in paths if 'restart' in p.lower())}"
    )
