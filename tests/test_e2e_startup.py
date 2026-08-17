"""End-to-end startup tests for llama-autoloader.

These drive the REAL ``server.app`` HTTP surface via FastAPI's TestClient, with a
fully isolated ``ModelManager`` (swap-and-restore of ``server.manager``, see
conftest). No real llama.cpp binary or real GGUF model files are required: the
load path is exercised with a mocked readiness boundary (decision 3a) and, as a
best-effort extra, a tiny in-process fake "llama-server" subprocess (decision 3b,
skipped if it can't run on this platform).

Covers (plan §4):
  - Startup happy path over real HTTP (/health, /, /static/*, /v1/models,
    /v1/status, /v1/backends, /v1/settings) with gguf_name + max_ctx_size
    populated from the GGUF bytes we wrote.
  - Startup failure mode (invalid config surfaces an error, not a healthy server).
  - Load/unload lifecycle E2E with mocked readiness.
  - Best-effort real-subprocess fake-backend load E2E (skipif when unavailable).
"""
from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import server
from server import ModelManager

log = logging.getLogger(__name__)


class _FakeProc:
    """Minimal stand-in for a running llama-server subprocess (mocked readiness)."""
    pid = 987654
    stdout = None
    stderr = None

    def poll(self):
        return None  # still running

    def kill(self):
        pass

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


# ===========================================================================
# Startup happy path (real HTTP via TestClient)
# ===========================================================================

class TestStartupHappyPath:
    def test_health_root(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["models_loaded"] == 0

    def test_health_v1_alias(self, client):
        r = client.get("/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_index_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        # index.html is served as HTMLResponse; it contains the app title.
        assert "<html" in r.text.lower()
        # The static dir ships an index.html with a recognizable marker.
        assert "llama" in r.text.lower() or "autoloader" in r.text.lower()

    def test_static_app_js(self, client):
        r = client.get("/static/app.js")
        assert r.status_code == 200
        assert len(r.content) > 0

    def test_static_styles_css(self, client):
        r = client.get("/static/styles.css")
        assert r.status_code == 200
        assert len(r.content) > 0

    def test_models_populated_from_gguf_bytes(self, client):
        """Proves scan + GGUF metadata read end-to-end: gguf_name and max_ctx_size
        come from the exact bytes written in conftest's seeded_manager."""
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = {m["id"]: m for m in r.json()["data"]}

        assert set(data) == {"model-a.gguf", "model-b.gguf", "model-c.gguf"}

        a = data["model-a.gguf"]
        assert a["gguf_name"] == "Alpha Model"
        assert a["max_ctx_size"] == 4096
        # Display name was promoted from metadata (was unedited stem at scan time).
        assert a["name"] == "Alpha Model"

        b = data["model-b.gguf"]
        assert b["gguf_name"] == "Beta Model"
        assert b["max_ctx_size"] == 8192

        c = data["model-c.gguf"]
        assert c["gguf_name"] == "Gamma Model"
        assert c["max_ctx_size"] is None  # no context_length in its GGUF bytes

    def test_status_coherent(self, client):
        r = client.get("/v1/status")
        assert r.status_code == 200
        s = r.json()
        assert s["models_total"] == 3
        assert s["models_loaded"] == 0
        launcher = s["launcher"]
        # Launcher host/port match the isolated config.
        assert launcher["host"] == "127.0.0.1"
        assert launcher["port"] == 19123
        # Backends list present (empty here, no backend dirs seeded).
        assert isinstance(launcher["backends"], list)

    def test_backends_endpoint(self, client):
        r = client.get("/v1/backends")
        assert r.status_code == 200
        body = r.json()
        # No backends dir contents -> empty list; resolved binary is the fallback.
        assert body["backends"] == []
        assert body["selected_backend"] == ""
        assert body["default_binary"] == "llama-server"
        assert body["resolved_global_binary"] == "llama-server"

    def test_settings_defaults(self, client):
        r = client.get("/v1/settings")
        assert r.status_code == 200
        s = r.json()
        assert s["idle_timeout_seconds"] == 3600
        assert s["max_loaded_models"] == 4
        assert s["base_port"] == 19001
        assert s["host"] == "127.0.0.1"
        assert s["port"] == 19123

    def test_settings_update_roundtrip(self, client):
        # PATCH a couple of settings; the endpoint persists to manager.cfg (the
        # isolated tmp config) and returns the updated view.
        r = client.put("/v1/settings", json={"idle_timeout_seconds": 120, "max_loaded_models": 2})
        assert r.status_code == 200
        s = r.json()
        assert s["idle_timeout_seconds"] == 120
        assert s["max_loaded_models"] == 2

    def test_settings_validation_rejects_bad(self, client):
        r = client.put("/v1/settings", json={"idle_timeout_seconds": -5})
        assert r.status_code == 400


# ===========================================================================
# Startup failure mode
# ===========================================================================

class TestStartupFailure:
    def test_invalid_config_raises_not_healthy(self, tmp_path):
        """A config that fails validate_config must surface an error at construction,
        not silently produce a healthy server."""
        from tests.conftest import build_cfg
        cfg = build_cfg(tmp_path)
        cfg["launcher"]["port"] = 99999  # out of range -> ValueError
        with pytest.raises(ValueError, match="port"):
            ModelManager.validate_config(cfg)

    def test_module_import_fails_on_bad_config(self, tmp_path):
        """import server with an invalid AUTOLOADER_CONFIG raises RuntimeError."""
        import os
        # Build a genuinely invalid yaml config (port out of range).
        cfg_file = tmp_path / "_test_bad_config.yaml"
        try:
            cfg_file.write_text(
                "launcher:\n  host: 127.0.0.1\n  port: 99999\n"
                "models:\n  root_dir: ./x\nllama_server:\n  binary: b\n"
            )
            env = {**os.environ, "AUTOLOADER_CONFIG": str(cfg_file)}
            # Fresh import in a subprocess so the module-level guard runs.
            code = (
                "import os; os.environ['AUTOLOADER_CONFIG']=r'%s';\n"
                "try:\n    import server\n    print('NO_ERROR')\n"
                "except RuntimeError as e:\n    print('RAISED', str(e)[:40])\n"
            ) % cfg_file.as_posix()
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            assert "NO_ERROR" not in out.stdout
            assert "RAISED" in out.stdout
        finally:
            if cfg_file.exists():
                cfg_file.unlink()


# ===========================================================================
# Load/unload lifecycle E2E (deterministic, mocked readiness — decision 3a)
# ===========================================================================

class TestLoadUnloadLifecycle:
    def test_load_ready_then_unload(self, client):
        """POST load -> model ready:true with a port; status reflects it; unload removes it."""
        # Mock the subprocess spawn + readiness boundary so no real binary is needed.
        fake_proc = _FakeProc()

        def fake_popen(argv, *a, **k):
            return fake_proc

        with patch("server.subprocess.Popen", side_effect=fake_popen), \
             patch.object(server.ModelManager, "_wait_until_ready", new=AsyncMock(return_value=True)):
            r = client.post("/v1/models/model-a.gguf/load")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["ready"] is True
            port = body["port"]
            assert isinstance(port, int) and port > 0

        # /v1/status now shows the model loaded & ready.
        s = client.get("/v1/status").json()
        assert s["models_loaded"] == 1
        up = {u["id"]: u for u in s["uptime_models"]}
        assert "model-a.gguf" in up
        assert up["model-a.gguf"]["ready"] is True
        assert up["model-a.gguf"]["port"] == port

        # /v1/models reflects loaded + ready + port.
        m = {x["id"]: x for x in client.get("/v1/models").json()["data"]}
        assert m["model-a.gguf"]["loaded"] is True
        assert m["model-a.gguf"]["ready"] is True
        assert m["model-a.gguf"]["port"] == port

        # Unload -> removed from status.
        r = client.post("/v1/models/model-a.gguf/unload")
        assert r.status_code == 200
        s2 = client.get("/v1/status").json()
        assert s2["models_loaded"] == 0
        m2 = {x["id"]: x for x in client.get("/v1/models").json()["data"]}
        assert m2["model-a.gguf"]["loaded"] is False
        assert m2["model-a.gguf"]["port"] is None

    def test_load_failure_unloads_and_503(self, client):
        """If readiness never comes, the load unloads and the endpoint returns 503."""
        fake_proc = _FakeProc()
        with patch("server.subprocess.Popen", side_effect=lambda *a, **k: fake_proc), \
             patch.object(server.ModelManager, "_wait_until_ready", new=AsyncMock(return_value=False)):
            r = client.post("/v1/models/model-a.gguf/load")
            assert r.status_code == 503
        # Model must not remain loaded after a failed load.
        s = client.get("/v1/status").json()
        assert s["models_loaded"] == 0

    def test_load_unknown_model_404(self, client):
        r = client.post("/v1/models/does-not-exist.gguf/load")
        assert r.status_code == 404


# ===========================================================================
# Best-effort REAL subprocess fake-backend load E2E (decision 3b)
# ===========================================================================

_FAKE_SERVER_SCRIPT = r"""
import sys, socket, threading, http.server

def _port(argv):
    if "--port" in argv:
        return int(argv[argv.index("--port") + 1])
    raise SystemExit("no --port in argv")

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *a):
        pass

def main():
    port = _port(sys.argv[1:])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # Keep the process alive until terminated; block on a socket accept loop.
    import time
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
"""


def _can_run_fake_backend() -> bool:
    """Return True if we can spawn a Python subprocess that binds a port here.

    Retries the probe up to 3 times: under full-suite load (many subprocesses
    spawned in quick succession) the first attempt's one-shot 5s deadline can be
    exceeded by slow Python startup / port bind, producing a false-negative skip.
    """
    import time
    import urllib.request

    for _attempt in range(3):
        try:
            # Quick capability probe: spawn the fake on an ephemeral port, hit /health.
            port = _free_port()
            proc = subprocess.Popen(
                [sys.executable, "-c", _FAKE_SERVER_SCRIPT, "--port", str(port)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            try:
                deadline = 5.0
                start = time.time()
                while time.time() - start < deadline:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as resp:
                            if resp.status == 200:
                                return True
                    except Exception:
                        pass
                    time.sleep(0.1)
            finally:
                _kill_proc(proc)
                # Let the OS release the bound port (Windows TIME_WAIT) before retrying.
                time.sleep(0.2)
        except Exception:
            pass
    return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_proc(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception as e:
        log.warning(f"Error terminating fake backend proc {getattr(proc, 'pid', '?')}: {e}")
    try:
        proc.kill()
    except Exception as e:
        log.warning(f"Error killing fake backend proc {getattr(proc, 'pid', '?')}: {e}")


@pytest.mark.skipif(not _can_run_fake_backend(), reason="fake backend subprocess can't run on this platform")
class TestRealSubprocessLoad:
    def test_real_spawn_wait_ready_unload(self, client):
        """Drive the TRUE spawn -> _wait_until_ready (real HTTP /health) -> ready
        pipeline using a genuine Python child process as the fake llama-server.

        We patch subprocess.Popen to launch our fake (which parses --port and binds
        it), but we do NOT patch _wait_until_ready — the real readiness poll runs
        against the child's live /health endpoint. This is best-effort and skipped
        if a Python subprocess can't bind/serve on this platform.
        """
        mgr = server.manager  # the isolated manager swapped in by conftest

        # Capture the REAL Popen before patching (server.subprocess IS the global
        # subprocess module, so patching it would otherwise recurse into itself).
        real_popen = subprocess.Popen

        def fake_popen(argv, *a, **k):
            # argv[0] is the "binary"; replace it with our real python child that
            # serves /health on the --port passed by to_launch_args.
            return real_popen([sys.executable, "-c", _FAKE_SERVER_SCRIPT] + list(argv[1:]),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        try:
            with patch("server.subprocess.Popen", side_effect=fake_popen):
                r = client.post("/v1/models/model-a.gguf/load")
                assert r.status_code == 200, r.text
                body = r.json()
                assert body["ready"] is True
                port = body["port"]

                # The child must actually be serving /health on that port.
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
                    assert resp.status == 200

                # Unload -> process killed, port freed.
                r = client.post("/v1/models/model-a.gguf/unload")
                assert r.status_code == 200
                s = client.get("/v1/status").json()
                assert s["models_loaded"] == 0
        finally:
            # Belt-and-suspenders cleanup: kill any lingering child and free the port.
            for lm in list(mgr.loaded.values()):
                _kill_proc(lm.process)
            # Let the OS release the bound port (Windows TIME_WAIT) before the next test.
            time.sleep(0.2)
