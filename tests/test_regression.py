"""Regression tests locking in current behavior of server.py core logic.

These are unit/integration-level tests that do NOT drive the HTTP surface (that's
test_e2e_startup.py). They build their own isolated ModelManager via the shared
``isolated_manager`` fixture and assert on behavior directly.

Covers (plan §3):
  - GGUF binary parser matrix (_read_gguf_metadata / _read_gguf_name)
  - _ensure_gguf_name: caching, idempotency, concurrency
  - validate_config
  - resolve_binary priority
  - _allocate_port / _is_port_available + port exhaustion
  - ModelConfig.to_launch_args
  - state save/list/load/delete (path logic; network parts mocked)
"""
from __future__ import annotations

import asyncio
import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import server
from server import (
    ModelConfig,
    ModelManager,
    _read_gguf_metadata,
    _read_gguf_name,
)
from tests import gguf_bytes as gb


# NOTE: Tests that drive coroutines are written as ``async def`` (pytest-asyncio
# auto mode, see pytest.ini). Each such test gets its own event loop with correct
# creation/cleanup lifecycle — this avoids the Windows ProactorEventLoop
# socketpair buffer exhaustion (OSError 10055) that a module-scope shared loop
# would cause at collection time. Synchronous tests stay plain ``def``.

# ===========================================================================
# GGUF binary parser matrix
# ===========================================================================

class TestGGUFParser:
    def test_missing_file(self):
        assert _read_gguf_metadata(Path("/nonexistent/x.gguf")) == (None, None)

    def test_invalid_magic(self, tmp_path):
        p = tmp_path / "bad.gguf"
        p.write_bytes(b"NOT_GGUF_DATA" * 50)
        assert _read_gguf_metadata(p) == (None, None)

    def test_corrupt_header_counts(self, tmp_path):
        # Magic OK but kvs is absurd -> early return (None, None)
        p = tmp_path / "corrupt.gguf"
        p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1_000_001))
        assert _read_gguf_metadata(p) == (None, None)

    def test_name_and_ctx_uint32(self, tmp_path):
        p = tmp_path / "m.gguf"
        gb.write_gguf(p, name="N", max_ctx=4096, ctx_vtype=gb.UINT32)
        assert _read_gguf_metadata(p) == ("N", 4096)

    def test_name_and_ctx_uint64(self, tmp_path):
        p = tmp_path / "m.gguf"
        gb.write_gguf(p, name="U64", max_ctx=32768, ctx_vtype=gb.UINT64)
        assert _read_gguf_metadata(p) == ("U64", 32768)

    def test_name_and_ctx_int64(self, tmp_path):
        p = tmp_path / "m.gguf"
        gb.write_gguf(p, name="I64", max_ctx=65536, ctx_vtype=gb.INT64)
        assert _read_gguf_metadata(p) == ("I64", 65536)

    def test_name_and_ctx_int32(self, tmp_path):
        p = tmp_path / "m.gguf"
        gb.write_gguf(p, name="I32", max_ctx=2048, ctx_vtype=gb.INT32)
        assert _read_gguf_metadata(p) == ("I32", 2048)

    def test_missing_name(self, tmp_path):
        p = tmp_path / "m.gguf"
        gb.write_gguf(p, name=None, max_ctx=4096)
        assert _read_gguf_metadata(p) == (None, 4096)

    def test_missing_ctx(self, tmp_path):
        p = tmp_path / "m.gguf"
        gb.write_gguf(p, name="OnlyName", max_ctx=None)
        assert _read_gguf_metadata(p) == ("OnlyName", None)

    def test_skips_float32_bool_array(self, tmp_path):
        skip_fields = [
            ("some.float", gb.FLOAT32, struct.pack("<f", 1.5)),
            ("some.bool", gb.BOOL, b"\x01"),
            ("some.arr", gb.ARRAY, struct.pack("<IQ", 4, 3) + struct.pack("<III", 1, 2, 3)),
        ]
        p = tmp_path / "m.gguf"
        gb.write_gguf(p, name="SkipModel", max_ctx=4096, skip_fields=skip_fields)
        assert _read_gguf_metadata(p) == ("SkipModel", 4096)

    def test_unknown_vtype_returns_none(self, tmp_path):
        p = tmp_path / "m.gguf"
        p.write_bytes(gb.make_gguf_unknown_vtype(unknown_vtype=13))
        assert _read_gguf_metadata(p) == (None, None)

    def test_read_gguf_name_delegates(self, tmp_path):
        p = tmp_path / "m.gguf"
        gb.write_gguf(p, name="DelegateName", max_ctx=1024)
        assert _read_gguf_name(p) == "DelegateName"

    def test_read_gguf_name_failure_returns_none(self):
        assert _read_gguf_name(Path("/nonexistent/x.gguf")) is None


# ===========================================================================
# _ensure_gguf_name
# ===========================================================================

def _make_ensure_manager(tmp_path, name="Ensure Name", max_ctx=4096):
    """Build a manager with one model whose GGUF has known metadata."""
    cfg = build_cfg_local(tmp_path)
    mgr = ModelManager(cfg)
    p = mgr.root_dir / "m.gguf"
    gb.write_gguf(p, name=name, max_ctx=max_ctx)
    mgr.scan()
    return mgr


def build_cfg_local(tmp_path):
    from tests.conftest import build_cfg
    return build_cfg(tmp_path)


class TestEnsureGGUFName:
    async def test_sets_gguf_name_and_max_ctx(self, tmp_path):
        mgr = _make_ensure_manager(tmp_path, name="Alpha", max_ctx=4096)
        # scan() already populated these; reset to prove lazy fill works.
        cfg = mgr.models["m.gguf"]
        cfg.gguf_name = ""
        cfg.max_ctx_size = None
        assert cfg.name == "Alpha"  # scan set display name from metadata

        with patch.object(server, "_read_gguf_metadata", wraps=server._read_gguf_metadata) as spy:
            await mgr._ensure_gguf_name("m.gguf")

        assert cfg.gguf_name == "Alpha"
        assert cfg.max_ctx_size == 4096
        assert spy.call_count >= 1

    async def test_defaults_display_name_to_stem_when_unedited(self, tmp_path):
        mgr = _make_ensure_manager(tmp_path, name="FreshName", max_ctx=2048)
        cfg = mgr.models["m.gguf"]
        # Simulate a config where display name is still the stem (unedited).
        cfg.name = "m"  # stem of m.gguf
        cfg.gguf_name = ""
        cfg.max_ctx_size = None
        await mgr._ensure_gguf_name("m.gguf")
        assert cfg.gguf_name == "FreshName"
        assert cfg.name == "FreshName"  # unedited -> promoted to metadata name

    async def test_does_not_overwrite_explicit_user_name(self, tmp_path):
        mgr = _make_ensure_manager(tmp_path, name="MetadataName", max_ctx=2048)
        cfg = mgr.models["m.gguf"]
        cfg.name = "My Custom Display Name"  # user-edited
        cfg.gguf_name = ""
        cfg.max_ctx_size = None
        await mgr._ensure_gguf_name("m.gguf")
        assert cfg.gguf_name == "MetadataName"
        assert cfg.name == "My Custom Display Name"  # preserved

    async def test_sets_max_ctx_only_when_none(self, tmp_path):
        mgr = _make_ensure_manager(tmp_path, name="X", max_ctx=4096)
        cfg = mgr.models["m.gguf"]
        cfg.max_ctx_size = 7777  # pre-existing value
        cfg.gguf_name = ""
        await mgr._ensure_gguf_name("m.gguf")
        assert cfg.max_ctx_size == 7777  # not overwritten

    async def test_idempotent_no_reread(self, tmp_path):
        mgr = _make_ensure_manager(tmp_path, name="CacheMe", max_ctx=4096)
        cfg = mgr.models["m.gguf"]
        assert cfg.gguf_name == "CacheMe" and cfg.max_ctx_size == 4096

        with patch.object(server, "_read_gguf_metadata", wraps=server._read_gguf_metadata) as spy:
            await mgr._ensure_gguf_name("m.gguf")
        assert spy.call_count == 0, "cached config must not trigger a re-read"

    async def test_concurrent_ensure_consistent(self, tmp_path):
        mgr = _make_ensure_manager(tmp_path, name="Conc", max_ctx=4096)
        cfg = mgr.models["m.gguf"]
        # Force a cold cache so the concurrent calls all race to read.
        cfg.gguf_name = ""
        cfg.max_ctx_size = None

        with patch.object(server, "_read_gguf_metadata", wraps=server._read_gguf_metadata) as spy:
            await asyncio.gather(*[mgr._ensure_gguf_name("m.gguf") for _ in range(20)])

        assert cfg.gguf_name == "Conc"
        assert cfg.max_ctx_size == 4096
        # The design permits benign duplicate reads: the read happens OUTSIDE the lock
        # (asyncio.to_thread), so concurrent cold-cache callers can all race to read.
        # The count is bounded by N (one read per task, no unbounded re-reads) — we fire
        # 20 tasks, so the reliable ceiling is exactly N=20. (Tighter values like 10 are
        # flaky: with a cold cache all 20 observe it before any result is written back.)
        assert 1 <= spy.call_count <= 20


# ===========================================================================
# validate_config
# ===========================================================================

class TestValidateConfig:
    def test_valid(self):
        from tests.conftest import build_cfg
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = build_cfg(Path(d))
            ModelManager.validate_config(cfg)  # should not raise

    def _cfg(self, tmp_path):
        from tests.conftest import build_cfg
        return build_cfg(tmp_path)

    def test_missing_launcher(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg.pop("launcher")
        with pytest.raises(ValueError, match="launcher"):
            ModelManager.validate_config(cfg)

    def test_bad_port_range(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg["launcher"]["port"] = 70000
        with pytest.raises(ValueError, match="port"):
            ModelManager.validate_config(cfg)

    def test_missing_models_section(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg.pop("models")
        with pytest.raises(ValueError, match="models"):
            ModelManager.validate_config(cfg)

    def test_missing_root_dir(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg["models"]["root_dir"] = None
        with pytest.raises(ValueError, match="root_dir"):
            ModelManager.validate_config(cfg)

    def test_bad_idle_timeout(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg["models"]["idle_timeout_seconds"] = 0
        with pytest.raises(ValueError, match="idle_timeout_seconds"):
            ModelManager.validate_config(cfg)

    def test_bad_max_loaded(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg["models"]["max_loaded_models"] = 0
        with pytest.raises(ValueError, match="max_loaded_models"):
            ModelManager.validate_config(cfg)

    def test_missing_llama_server(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg.pop("llama_server")
        with pytest.raises(ValueError, match="llama_server"):
            ModelManager.validate_config(cfg)

    def test_missing_binary(self, tmp_path):
        cfg = self._cfg(tmp_path)
        cfg["llama_server"]["binary"] = ""
        with pytest.raises(ValueError, match="binary"):
            ModelManager.validate_config(cfg)


# ===========================================================================
# resolve_binary priority
# ===========================================================================

class TestResolveBinary:
    def _mk_backend(self, mgr, name, exe_name="llama-server.exe"):
        bdir = mgr.backends_dir / name
        bdir.mkdir(parents=True, exist_ok=True)
        exe = bdir / exe_name
        exe.write_text("fake")
        return str(exe)

    def test_priority_1_per_model(self, tmp_path):
        from tests.conftest import build_cfg
        mgr = ModelManager(build_cfg(tmp_path))
        per_model_exe = self._mk_backend(mgr, "permodel")
        global_exe = self._mk_backend(mgr, "global")
        mgr.selected_backend = "global"
        cfg = ModelConfig.default_for(Path("m.gguf"))
        cfg.backend = "permodel"
        assert mgr.resolve_binary(cfg) == per_model_exe

    def test_priority_2_global_selected(self, tmp_path):
        from tests.conftest import build_cfg
        mgr = ModelManager(build_cfg(tmp_path))
        global_exe = self._mk_backend(mgr, "global")
        mgr.selected_backend = "global"
        cfg = ModelConfig.default_for(Path("m.gguf"))
        assert mgr.resolve_binary(cfg) == global_exe

    def test_priority_3_first_scanned(self, tmp_path):
        from tests.conftest import build_cfg
        mgr = ModelManager(build_cfg(tmp_path))
        # No selected backend; create a scanned backend.
        exe = self._mk_backend(mgr, "scanned")
        cfg = ModelConfig.default_for(Path("m.gguf"))
        assert mgr.resolve_binary(cfg) == exe

    def test_priority_4_fallback(self, tmp_path):
        from tests.conftest import build_cfg
        mgr = ModelManager(build_cfg(tmp_path))
        # No backends present -> fall back to self.binary
        cfg = ModelConfig.default_for(Path("m.gguf"))
        assert mgr.resolve_binary(cfg) == mgr.binary

    def test_none_model_uses_global_or_scanned(self, tmp_path):
        from tests.conftest import build_cfg
        mgr = ModelManager(build_cfg(tmp_path))
        global_exe = self._mk_backend(mgr, "global")
        mgr.selected_backend = "global"
        assert mgr.resolve_binary(None) == global_exe


# ===========================================================================
# Port allocation
# ===========================================================================

class TestPortAllocation:
    def test_allocate_returns_available(self, isolated_manager):
        port = isolated_manager._allocate_port()
        assert isinstance(port, int)
        assert port >= isolated_manager.base_port
        assert isolated_manager._is_port_available(port) is True

    def test_allocate_increments_next(self, isolated_manager):
        before = isolated_manager.next_port
        p1 = isolated_manager._allocate_port()
        assert isolated_manager.next_port > before
        assert p1 >= before

    def test_is_port_available_false_when_bound(self, isolated_manager):
        # Bind a LISTENING socket (no SO_REUSEADDR) so a second bind genuinely
        # fails on both Windows and POSIX. (SO_REUSEADDR alone does not prevent
        # double-bind on Windows.)
        import socket
        port = isolated_manager._allocate_port()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        try:
            assert isolated_manager._is_port_available(port) is False
        finally:
            s.close()

    def test_exhaustion_raises(self, isolated_manager):
        # Force every candidate in the 500-window to be "used" so allocation fails.
        with patch.object(isolated_manager, "_is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="No available ports"):
                isolated_manager._allocate_port()

    def test_skips_used_ports(self, isolated_manager):
        from server import LoadedModel
        # Occupy base_port via a fake loaded model.
        cfg = ModelConfig.default_for(Path("x.gguf"))
        lm = LoadedModel("x.gguf", Path("x.gguf"), cfg, isolated_manager.base_port,
                         MagicMock(), ready=True, pid=1)
        isolated_manager.loaded["x.gguf"] = lm
        port = isolated_manager._allocate_port()
        assert port != isolated_manager.base_port


# ===========================================================================
# to_launch_args
# ===========================================================================

class TestToLaunchArgs:
    def test_basic(self):
        # Use a Windows-safe path (str(Path("/m")) -> "\\m" on Windows).
        cfg = ModelConfig(ctx_size=4096, n_gpu_layers=50)
        model_path = Path("C:/models/m.gguf")
        args = cfg.to_launch_args(model_path, 8080, "")
        assert args[0:2] == ["--model", str(model_path)]
        assert "--host" in args and "127.0.0.1" in args
        assert args[args.index("--port") + 1] == "8080"
        assert args[args.index("--ctx-size") + 1] == "4096"
        assert args[args.index("--n-gpu-layers") + 1] == "50"

    def test_alias_when_model_id(self):
        cfg = ModelConfig()
        args = cfg.to_launch_args(Path("/m"), 8080, "", model_id="my-model.gguf")
        assert args[args.index("--alias") + 1] == "my-model.gguf"

    def test_no_alias_without_model_id(self):
        cfg = ModelConfig()
        args = cfg.to_launch_args(Path("/m"), 8080, "")
        assert "--alias" not in args

    def test_extra_default_args_split(self):
        cfg = ModelConfig()
        args = cfg.to_launch_args(Path("/m"), 8080, "--no-webui --parallel 1")
        assert "--no-webui" in args
        assert "1" in args

    def test_model_args_appended_last(self):
        cfg = ModelConfig(args="--temp 0.7")
        args = cfg.to_launch_args(Path("/m"), 8080, "")
        assert args[-2:] == ["--temp", "0.7"]

    def test_mmproj_included_when_exists(self, tmp_path):
        mmproj = tmp_path / "clip.gguf"
        mmproj.write_bytes(b"x")
        cfg = ModelConfig(use_mmproj=True)
        args = cfg.to_launch_args(Path("/m"), 8080, "", mmproj_full_path=mmproj)
        assert "--mmproj" in args
        assert str(mmproj) in args

    def test_mmproj_excluded_when_missing(self, tmp_path):
        missing = tmp_path / "nope.gguf"
        cfg = ModelConfig(use_mmproj=True)
        args = cfg.to_launch_args(Path("/m"), 8080, "", mmproj_full_path=missing)
        assert "--mmproj" not in args

    def test_slot_save_dir(self, tmp_path):
        cfg = ModelConfig()
        args = cfg.to_launch_args(Path("/m"), 8080, "", slot_save_dir=tmp_path)
        assert "--slot-save-path" in args
        assert str(tmp_path) in args


# ===========================================================================
# State save / list / load / delete (path logic; network mocked)
# ===========================================================================

class TestState:
    @pytest.fixture()
    def mgr(self, isolated_manager):
        # One loaded model so save_state/load_state have a target.
        cfg = ModelConfig.default_for(Path("m.gguf"))
        lm = MagicMock()
        lm.port = 19500
        lm.ready = True
        lm.config = cfg
        lm.state_path = None
        isolated_manager.models["m.gguf"] = cfg
        isolated_manager.loaded["m.gguf"] = lm
        return isolated_manager

    async def test_save_state_writes_file(self, mgr):
        # The real _perform_slot_save writes the state file over HTTP; emulate
        # that side effect so save_state's post-conditions hold.
        async def fake_save(port, path, timeout=None):
            Path(path).write_bytes(b"state-data")

        with patch.object(mgr, "_perform_slot_save", new=AsyncMock(side_effect=fake_save)) as mock_save:
            path = await mgr.save_state("m.gguf", "default")
        assert path.exists()
        assert path.name == "m.gguf.default.bin"
        # _perform_slot_save called with the model's port and the state path.
        args, _ = mock_save.call_args
        assert args[0] == 19500
        assert Path(args[1]) == path

    async def test_list_states(self, mgr):
        (mgr.save_state_dir / "m.gguf.default.bin").write_bytes(b"1")
        (mgr.save_state_dir / "m.gguf.convo1.bin").write_bytes(b"2")
        labels = await mgr.list_states("m.gguf")
        assert set(labels) == {"gguf.default", "gguf.convo1"}

    async def test_list_states_empty(self, mgr):
        assert await mgr.list_states("m.gguf") == []

    async def test_load_state_restores_when_loaded(self, mgr):
        (mgr.save_state_dir / "m.gguf.default.bin").write_bytes(b"state")
        with patch.object(mgr, "_perform_slot_restore", new=AsyncMock()) as mock_restore:
            path = await mgr.load_state("m.gguf", "default")
        assert path.exists()
        args, _ = mock_restore.call_args
        assert args[0] == 19500

    async def test_delete_state(self, mgr):
        p = mgr.save_state_dir / "m.gguf.default.bin"
        p.write_bytes(b"x")
        await mgr.delete_state("m.gguf", "default")
        assert not p.exists()

    async def test_delete_state_missing_raises(self, mgr):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await mgr.delete_state("m.gguf", "missing")
        assert exc.value.status_code == 404
