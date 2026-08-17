"""Real-model E2E load tests (plan: REAL_MODEL_TESTS_PLAN.md).

Drives the TRUE, unmocked load pipeline — real ``subprocess.Popen`` of a real
``llama-server.exe``, real readiness poll against the child's live ``/health``,
and a real chat/completion round-trip through the proxy — against a tiny local
GGUF model (Qwen2.5-0.5B-Instruct-Q8_0.gguf, ~531 MB).

Nothing about subprocess spawn or readiness is mocked: that is the entire point.
The tests are marked ``skipif`` via the module-level :func:`_can_run_real_model`
gate so a machine without the model or a usable backend stays green; once the
gate passes (model + binary present), a failure to become ready is a REAL
signal and FAILS the test rather than skipping.

Isolation reuses the conftest fixtures (swap-and-restore of ``server.manager``,
config redirect to tmp). The isolated manager's ``root_dir`` / ``backends_dir``
are pointed at the REAL model tree and backend binaries; ``save_state_dir``
stays under pytest's tmp_path so the repo ``states/`` and ``config.yaml`` are
never touched.
"""
from __future__ import annotations

import asyncio
import logging
import time
import urllib.request
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

import server
from tests.test_e2e_startup import _kill_proc  # reuse the existing kill helper

log = logging.getLogger(__name__)

# Target model: filename is the scan() key. Lives under root_dir (n:/work/stuff/Beta).
MODEL_ID = "Qwen2.5-0.5B-Instruct-Q8_0.gguf"
_ROOT_DIR = Path("N:/work/stuff/Beta")
_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "backends"


# ---------------------------------------------------------------------------
# Graceful-skip gate (plan §Graceful-skip gate)
# ---------------------------------------------------------------------------

def _can_run_real_model() -> bool:
    """Cheap, never-raising probe: True only if the target GGUF is present under
    root_dir AND a real llama-server backend binary resolves and exists on disk.

    Mirrors production resolution (server.ModelManager.resolve_binary):
    per-model override > selected backend > first scanned backend > fallback.
    Any exception -> False (skip). No process is spawned here.
    """
    try:
        model_path = _ROOT_DIR / "lmstudio-community" / "Qwen2.5-0.5B-Instruct-GGUF" / MODEL_ID
        if not model_path.is_file():
            log.info("real-model gate: model file missing at %s -> skip", model_path)
            return False

        # Read the per-model sidecar (ModelConfig.load semantics, without building a manager).
        import json as _json
        backend = ""
        sidecar = model_path.with_suffix(model_path.suffix + ".json")
        if sidecar.is_file():
            try:
                backend = _json.loads(sidecar.read_text()).get("backend", "") or ""
            except Exception:
                backend = ""

        # Priority 1: per-model backend override
        if backend:
            exe = _BACKENDS_DIR / backend / "llama-server.exe"
            if not exe.exists():
                exe = _BACKENDS_DIR / backend / "llama-server"
            if exe.is_file():
                log.info("real-model gate: using per-model backend %s", exe)
                return True

        # Priority 2/3: first scanned backend dir containing a binary
        if _BACKENDS_DIR.is_dir():
            for p in sorted(_BACKENDS_DIR.iterdir()):
                if not p.is_dir():
                    continue
                exe = p / "llama-server.exe"
                if not exe.exists():
                    exe = p / "llama-server"
                if exe.is_file():
                    log.info("real-model gate: using first scanned backend %s", exe)
                    return True

        log.info("real-model gate: no usable llama-server backend binary under %s -> skip", _BACKENDS_DIR)
        return False
    except Exception as e:  # never raise from the gate
        log.info("real-model gate probe failed (%s) -> skip", e)
        return False


# ---------------------------------------------------------------------------
# Fixtures (reuse conftest isolation; repoint dirs at the real model/backends)
# ---------------------------------------------------------------------------

@pytest.fixture()
def real_model_manager(isolated_manager):
    """isolated_manager with root_dir/backends_dir pointed at the REAL model tree
    and backend binaries, and a small ctx_size so the 0.5B load stays fast.

    ``save_state_dir`` already lands under pytest's tmp_path (conftest build_cfg),
    so the repo ``states/`` is never written. Config persistence is already
    redirected to tmp by conftest.
    """
    mgr = isolated_manager
    assert _ROOT_DIR.is_dir(), f"model root dir missing: {_ROOT_DIR}"
    assert _BACKENDS_DIR.is_dir(), f"backends dir missing: {_BACKENDS_DIR}"

    # Repoint at the real model tree + backend binaries (read-only use).
    mgr.root_dir = _ROOT_DIR.resolve()
    mgr.backends_dir = _BACKENDS_DIR.resolve()
    # Small context keeps the 0.5B CPU/CUDA load well under the readiness timeout.
    mgr.default_args = "--no-webui --parallel 1 --jinja"

    ids = mgr.scan()
    assert MODEL_ID in ids, (
        f"gate passed but scan() did not find {MODEL_ID!r} under {mgr.root_dir}; "
        f"scanned: {ids}"
    )
    # Note: TestClient.__enter__ triggers app startup → manager.start() → scan() again,
    # which re-reads sidecars. So we apply per-model overrides in _apply_test_overrides()
    # called at the start of each test (after client is up).

    # Belt-and-suspenders: prove save_state_dir is inside a tmp dir (never the repo).
    # On Windows, pytest's tmp_path resolves to e.g. C:\Users\...\AppData\Local\Temp\pytest-of-...
    ssd = str(mgr.save_state_dir)
    assert "tmp" in ssd.lower() or "temp" in ssd.lower() or "_pytest" in ssd, (
        f"save_state_dir not isolated: {ssd}"
    )
    # Also ensure it's NOT under the repo root.
    repo_root = _BACKENDS_DIR.parent.resolve()
    assert not Path(ssd).resolve().is_relative_to(repo_root), f"save_state_dir points into repo: {ssd}"

    return mgr


@pytest.fixture()
def real_client(real_model_manager):
    """TestClient over the real app surface with the real-model manager swapped in.

    Teardown mirrors conftest's ``client`` fixture: exit the client (shutdown ->
    manager.stop()), force-stop exactly once, then verify no model remains loaded
    and no background tasks leaked.
    """
    from fastapi.testclient import TestClient
    from tests.conftest import _stop_manager_once, _assert_no_leaked_tasks

    with TestClient(server.app) as c:
        yield c

    _stop_manager_once(real_model_manager)
    assert not real_model_manager.loaded, "models should be unloaded after teardown"
    _assert_no_leaked_tasks(real_model_manager)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_test_overrides(mgr) -> None:
    """Apply test-specific overrides to the model config AFTER startup scan.

    The sidecar for this model contains LM-Studio-specific args ("-dev cuda1")
    and n_gpu_layers=999 that are incompatible with the CPU-patched backend.
    We strip those so the real llama-server.exe can start cleanly on CPU.
    """
    cfg = mgr.models.get(MODEL_ID)
    if cfg is None:
        return
    # Small context keeps the 0.5B load fast and memory-light.
    if cfg.ctx_size > 4096:
        cfg.ctx_size = 4096
    # Strip GPU-specific sidecar args that vanilla llama.cpp doesn't understand.
    cfg.args = ""
    # CPU-only: no GPU offload (the CPU-patched backend has no CUDA).
    cfg.n_gpu_layers = 0


def _load_via_api(client, model_id: str = MODEL_ID):
    """POST the load endpoint; assert 200 + ready + port. Returns (port, pid)."""
    r = client.post(f"/v1/models/{model_id}/load")
    assert r.status_code == 200, (
        f"real load failed (gate passed, so model+binary exist — this is a real failure): {r.text}"
    )
    body = r.json()
    assert body["ready"] is True, f"model did not report ready: {body}"
    port = body["port"]
    assert isinstance(port, int) and port > 0, f"bad port in load response: {body}"
    pid = body["pid"]
    assert isinstance(pid, int), f"bad pid in load response: {body}"
    return port, pid


def _teardown_cleanup(mgr) -> None:
    """Robust cleanup: unload any loaded model + kill its process; log (don't
    swallow) errors; short sleep for Windows TIME_WAIT port release."""
    try:
        from tests.conftest import _get_session_loop
        loop = _get_session_loop()
        for mid in list(mgr.loaded.keys()):
            lm = mgr.loaded.get(mid)
            if lm is None:
                continue
            try:
                # Unload via the manager (kills the process group properly).
                task = loop.create_task(mgr.unload_model(mid))
                loop.run_until_complete(asyncio.wait_for(task, timeout=10))
            except Exception as e:
                log.warning("cleanup unload of %s failed: %s", mid, e)
            lm = mgr.loaded.get(mid)
            if lm is not None:
                _kill_proc(lm.process)
    except Exception as e:  # never let cleanup mask the test outcome silently
        log.warning("real-model teardown cleanup error: %s", e)
    finally:
        time.sleep(0.2)  # Windows TIME_WAIT port release


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _can_run_real_model(),
                    reason="target GGUF model or a usable llama-server backend binary is not present on this machine")
class TestRealModelLoad:
    """True spawn -> readiness (real HTTP /health) -> proxy round-trip against a
    real llama-server.exe and a real 0.5B GGUF. No mocking of Popen/readiness."""

    def test_real_load_ready(self, real_client):
        mgr = server.manager
        _apply_test_overrides(mgr)
        try:
            port, _pid = _load_via_api(real_client)

            # The child must actually be serving /health on that port (real process).
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                assert resp.status == 200

            # /v1/status shows it loaded & ready. The status cache may lag by one
            # poll_interval (2s) after load, so retry briefly until it reflects ready.
            s = None
            deadline = time.time() + 5.0
            while time.time() < deadline:
                s = real_client.get("/v1/status").json()
                up = {u["id"]: u for u in s["uptime_models"]}
                if MODEL_ID in up and up[MODEL_ID]["ready"]:
                    break
                time.sleep(0.3)
            assert s is not None
            assert s["models_loaded"] >= 1, f"expected >=1 loaded, got {s['models_loaded']}"
            up = {u["id"]: u for u in s["uptime_models"]}
            assert MODEL_ID in up, f"{MODEL_ID} not in uptime_models: {up}"
            assert up[MODEL_ID]["ready"] is True
            assert up[MODEL_ID]["port"] == port

            # /v1/models reflects loaded + ready + port (no cache — reads mgr directly).
            m = {x["id"]: x for x in real_client.get("/v1/models").json()["data"]}
            assert m[MODEL_ID]["loaded"] is True
            assert m[MODEL_ID]["ready"] is True
            assert m[MODEL_ID]["port"] == port
        finally:
            _teardown_cleanup(mgr)

    def test_real_completion_roundtrip(self, real_client):
        mgr = server.manager
        _apply_test_overrides(mgr)
        try:
            _load_via_api(real_client)

            # Tiny chat/completion through the proxy (real inference on CPU/CUDA).
            r = real_client.post("/v1/chat/completions", json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 16,
                "stream": False,
            })
            assert r.status_code == 200, f"completion proxy failed: {r.text[:500]}"
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            assert isinstance(content, str) and content.strip(), (
                f"expected non-empty generated content, got {content!r} from {str(data)[:500]}"
            )
        finally:
            _teardown_cleanup(mgr)

    def test_real_unload_cleans_up(self, real_client):
        mgr = server.manager
        _apply_test_overrides(mgr)
        pid = None
        try:
            _port, pid = _load_via_api(real_client)

            # Backend process must be alive while loaded.
            assert psutil.pid_exists(pid), f"backend process {pid} not running after load"

            r = real_client.post(f"/v1/models/{MODEL_ID}/unload")
            assert r.status_code == 200, r.text

            # Status no longer shows it loaded.
            s = real_client.get("/v1/status").json()
            assert s["models_loaded"] == 0, f"expected 0 loaded after unload, got {s['models_loaded']}"
            up = {u["id"]: u for u in s["uptime_models"]}
            assert MODEL_ID not in up

            # Backend process terminated (no lingering child).
            if pid is not None:
                deadline = time.time() + 5.0
                while psutil.pid_exists(pid) and time.time() < deadline:
                    time.sleep(0.1)
                assert not psutil.pid_exists(pid), f"backend process {pid} still alive after unload"
        finally:
            _teardown_cleanup(mgr)
