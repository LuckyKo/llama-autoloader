"""
Unit tests for llama-autoloader server.py – ModelManager core logic.

Run with:  pytest test_server.py -v
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from server import ModelManager, ModelConfig, LoadedModel, _read_gguf_metadata, _read_gguf_name

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(overrides: dict | None = None) -> dict:
    """Build a minimal config dict suitable for ModelManager construction."""
    cfg = {
        "models": {
            "root_dir": str(Path(__file__).parent / "test_models"),
            "save_state_dir": str(Path(__file__).parent / "test_states"),
            "idle_timeout_seconds": 300,
            "max_loaded_models": 1,
            "auto_save_state": False,
        },
        "llama_server": {
            "binary": "llama-server",
            "backends_dir": "./backends",
            "selected_backend": "",
            "default_args": "",
        },
        "launcher": {
            "host": "127.0.0.1",
            "port": 9123,
            "base_port": 9001,
        },
        "gpu": {"poll_interval_seconds": 2},
    }
    if overrides:
        for section, vals in overrides.items():
            if section in cfg:
                cfg[section].update(vals)
            else:
                cfg[section] = vals
    return cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_model_dir(tmp_path):
    """Create a temporary directory with fake .gguf model files."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Model A
    a_gguf = models_dir / "model-a.gguf"
    a_gguf.write_bytes(b"GGUF_HEADER" * 100)
    (a_gguf.with_suffix(".gguf.json")).write_text(
        json.dumps({"name": "Model A", "ctx_size": 4096})
    )

    # Model B (marked as default)
    b_gguf = models_dir / "model-b.gguf"
    b_gguf.write_bytes(b"GGUF_HEADER" * 200)
    (b_gguf.with_suffix(".gguf.json")).write_text(
        json.dumps({"name": "Model B", "default": True, "ctx_size": 8192})
    )

    # Model with no sidecar config
    c_gguf = models_dir / "model-c.gguf"
    c_gguf.write_bytes(b"GGUF_HEADER" * 50)

    # mmproj file (should be filtered out)
    mm_gguf = models_dir / "clip-mmproj.gguf"
    mm_gguf.write_bytes(b"GGUF_HEADER" * 30)

    # Nested model
    sub = models_dir / "subdir"
    sub.mkdir()
    d_gguf = sub / "model-d.gguf"
    d_gguf.write_bytes(b"GGUF_HEADER" * 80)

    return models_dir


@pytest.fixture()
def tmp_state_dir(tmp_path):
    """Create a temporary state directory with some fake state files."""
    s_dir = tmp_path / "states"
    s_dir.mkdir()
    # list_states globs {model_id}.*.bin  (model_id includes .gguf extension)
    (s_dir / "model-a.gguf.default.bin").write_bytes(b"state")
    (s_dir / "model-a.gguf.convo1.bin").write_bytes(b"state")
    (s_dir / "model-b.gguf.default.bin").write_bytes(b"state")
    return s_dir


@pytest.fixture()
def manager(tmp_model_dir, tmp_state_dir):
    """Create a ModelManager pointing at temp dirs and scan models."""
    cfg = _make_cfg({
        "models": {
            "root_dir": str(tmp_model_dir),
            "save_state_dir": str(tmp_state_dir),
            "max_loaded_models": 2,
        },
    })
    mgr = ModelManager(cfg)
    # scan() is synchronous; just call it directly
    mgr.scan()
    return mgr


# ---------------------------------------------------------------------------
# scan()
# ---------------------------------------------------------------------------

class TestScan:
    def test_discovers_models(self, manager):
        ids = list(manager.models.keys())
        assert "model-a.gguf" in ids
        assert "model-b.gguf" in ids
        assert "model-c.gguf" in ids
        assert "model-d.gguf" in ids

    def test_filters_mmproj(self, manager):
        ids = list(manager.models.keys())
        assert "clip-mmproj.gguf" not in ids

    def test_pairs_mmproj(self, manager):
        # There's one mmproj and 4 models; fallback pairs it with every model
        assert len(manager.mmproj_paths) > 0

    def test_reads_config_sidecar(self, manager):
        cfg_a = manager.models["model-a.gguf"]
        assert cfg_a.name == "Model A"
        assert cfg_a.ctx_size == 4096

    def test_defaults_when_no_sidecar(self, manager):
        cfg_c = manager.models["model-c.gguf"]
        assert cfg_c.name == "model-c"  # stem
        assert cfg_c.ctx_size == 8192   # default

    def test_model_sizes_cached(self, manager):
        for mid in manager.models:
            assert mid in manager._model_sizes
            assert manager._model_sizes[mid] >= 0.0


# ---------------------------------------------------------------------------
# resolve_model_id()
# ---------------------------------------------------------------------------

class TestResolveModelId:
    @pytest.mark.asyncio
    async def test_exact_match(self, manager):
        result = await manager.resolve_model_id("model-a.gguf")
        assert result == "model-a.gguf"

    @pytest.mark.asyncio
    async def test_stem_match(self, manager):
        result = await manager.resolve_model_id("model-a")
        assert result == "model-a.gguf"

    @pytest.mark.asyncio
    async def test_case_insensitive(self, manager):
        result = await manager.resolve_model_id("MODEL-A.gguf")
        assert result == "model-a.gguf"

    @pytest.mark.asyncio
    async def test_name_match(self, manager):
        result = await manager.resolve_model_id("Model A")
        assert result == "model-a.gguf"

    @pytest.mark.asyncio
    async def test_fallback_to_default(self, manager):
        manager.models.clear()
        manager.loaded.clear()
        cfg = ModelConfig.default_for(Path("only.gguf"))
        cfg.default = True
        manager.models["only.gguf"] = cfg
        result = await manager.resolve_model_id("nonexistent")
        assert result == "only.gguf"

    @pytest.mark.asyncio
    async def test_fallback_to_single_model(self, manager):
        manager.models.clear()
        manager.loaded.clear()
        manager.models["solo.gguf"] = ModelConfig.default_for(Path("solo.gguf"))
        result = await manager.resolve_model_id("nonexistent")
        assert result == "solo.gguf"

    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self, manager):
        manager.models.clear()
        result = await manager.resolve_model_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_to_loaded_when_no_default(self, manager):
        # Multiple models, none marked default, one is loaded
        manager.models.clear()
        manager.loaded.clear()
        manager.models["x.gguf"] = ModelConfig.default_for(Path("x.gguf"))
        manager.models["y.gguf"] = ModelConfig.default_for(Path("y.gguf"))
        lm = LoadedModel(
            model_id="x.gguf", gguf_path=Path("x.gguf"),
            config=manager.models["x.gguf"], port=9001,
            process=MagicMock(), ready=True, pid=123,
        )
        manager.loaded["x.gguf"] = lm
        result = await manager.resolve_model_id("nonexistent")
        assert result == "x.gguf"


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------

class TestListModels:
    @pytest.mark.asyncio
    async def test_returns_model_info(self, manager):
        models = await manager.list_models()
        assert len(models) > 0
        for m in models:
            assert "id" in m
            assert "name" in m
            assert "size_mb" in m
            assert "loaded" in m
            assert "ready" in m

    @pytest.mark.asyncio
    async def test_size_mb_present(self, manager):
        models = await manager.list_models()
        for m in models:
            assert isinstance(m["size_mb"], (int, float))
            assert m["size_mb"] >= 0.0


# ---------------------------------------------------------------------------
# get_config() / update_config()
# ---------------------------------------------------------------------------

class TestConfig:
    @pytest.mark.asyncio
    async def test_get_config(self, manager):
        cfg = await manager.get_config("model-a.gguf")
        assert cfg is not None
        assert cfg.name == "Model A"

    @pytest.mark.asyncio
    async def test_get_config_by_stem(self, manager):
        cfg = await manager.get_config("model-a")
        assert cfg is not None

    @pytest.mark.asyncio
    async def test_get_config_unknown(self, manager):
        # resolve_model_id falls back to default/single, so clear those too
        manager.models.clear()
        cfg = await manager.get_config("nonexistent")
        assert cfg is None

    @pytest.mark.asyncio
    async def test_update_config(self, manager):
        new_cfg = ModelConfig(name="Updated A", ctx_size=16384)
        result = await manager.update_config("model-a.gguf", new_cfg)
        assert result.name == "Updated A"
        assert result.ctx_size == 16384
        # Persisted in registry
        cur = await manager.get_config("model-a.gguf")
        assert cur.name == "Updated A"

    @pytest.mark.asyncio
    async def test_update_config_writes_sidecar(self, manager):
        new_cfg = ModelConfig(name="Written A", ctx_size=2048)
        await manager.update_config("model-a.gguf", new_cfg)
        sidecar_path = manager.gguf_paths["model-a.gguf"].with_suffix(".gguf.json")
        data = json.loads(sidecar_path.read_text())
        assert data["name"] == "Written A"
        assert data["ctx_size"] == 2048


# ---------------------------------------------------------------------------
# list_states()
# ---------------------------------------------------------------------------

class TestListStates:
    @pytest.mark.asyncio
    async def test_lists_existing_states(self, manager):
        labels = await manager.list_states("model-a.gguf")
        # list_states extracts label as p.stem.split(".", 1)[1]
        # For file "model-a.gguf.default.bin", stem="model-a.gguf.default"
        # split(".",1) → ["model-a", "gguf.default"] → label="gguf.default"
        assert len(labels) == 2
        assert "gguf.default" in labels
        assert "gguf.convo1" in labels

    @pytest.mark.asyncio
    async def test_empty_states(self, manager):
        labels = await manager.list_states("model-c.gguf")
        assert labels == []


# ---------------------------------------------------------------------------
# _evict_lru_if_needed()
# ---------------------------------------------------------------------------

class TestEvictLRU:
    @pytest.mark.asyncio
    async def test_evicts_lru_when_max_reached(self, manager):
        # max_loaded_models=2.  Load 2 models manually.
        manager.models.clear()
        manager.loaded.clear()
        cfg_x = ModelConfig.default_for(Path("x.gguf"))
        cfg_y = ModelConfig.default_for(Path("y.gguf"))
        cfg_z = ModelConfig.default_for(Path("z.gguf"))
        manager.models.update({
            "x.gguf": cfg_x, "y.gguf": cfg_y, "z.gguf": cfg_z,
        })

        lm_x = LoadedModel("x.gguf", Path("x.gguf"), cfg_x, 9001, MagicMock(),
                           last_used=time.time() - 100, ready=True, pid=1)
        lm_y = LoadedModel("y.gguf", Path("y.gguf"), cfg_y, 9002, MagicMock(),
                           last_used=time.time() - 50, ready=True, pid=2)
        manager.loaded.update({"x.gguf": lm_x, "y.gguf": lm_y})

        # Load z → should evict x (LRU)
        async with manager._lock:
            await manager._evict_lru_if_needed("z.gguf")

        assert "x.gguf" not in manager.loaded
        assert "y.gguf" in manager.loaded

    @pytest.mark.asyncio
    async def test_skips_pinned_models(self, manager):
        manager.models.clear()
        manager.loaded.clear()
        cfg_x = ModelConfig.default_for(Path("x.gguf"))
        cfg_x.pinned = True
        cfg_y = ModelConfig.default_for(Path("y.gguf"))
        cfg_z = ModelConfig.default_for(Path("z.gguf"))
        manager.models.update({"x.gguf": cfg_x, "y.gguf": cfg_y, "z.gguf": cfg_z})

        lm_x = LoadedModel("x.gguf", Path("x.gguf"), cfg_x, 9001, MagicMock(),
                           last_used=time.time() - 200, ready=True, pid=1)
        lm_y = LoadedModel("y.gguf", Path("y.gguf"), cfg_y, 9002, MagicMock(),
                           last_used=time.time() - 50, ready=True, pid=2)
        manager.loaded.update({"x.gguf": lm_x, "y.gguf": lm_y})

        async with manager._lock:
            await manager._evict_lru_if_needed("z.gguf")

        # x is pinned → y should be evicted instead
        assert "x.gguf" in manager.loaded
        assert "y.gguf" not in manager.loaded

    @pytest.mark.asyncio
    async def test_no_eviction_under_limit(self, manager):
        manager.models.clear()
        manager.loaded.clear()
        cfg_a = ModelConfig.default_for(Path("a.gguf"))
        manager.models["a.gguf"] = cfg_a
        lm_a = LoadedModel("a.gguf", Path("a.gguf"), cfg_a, 9001, MagicMock(),
                           last_used=time.time(), ready=True, pid=1)
        manager.loaded["a.gguf"] = lm_a

        async with manager._lock:
            await manager._evict_lru_if_needed("b.gguf")

        assert len(manager.loaded) == 1


# ---------------------------------------------------------------------------
# get_cached_status()
# ---------------------------------------------------------------------------

class TestCachedStatus:
    @pytest.mark.asyncio
    async def test_returns_dict(self, manager):
        status = await manager.get_cached_status()
        assert isinstance(status, dict)
        assert "launcher" in status
        assert "models_loaded" in status
        assert "models_total" in status

    @pytest.mark.asyncio
    async def test_none_cache_triggers_build(self, manager):
        manager._status_cache = None
        status = await manager.get_cached_status()
        assert status is not None
        assert len(status) > 0

    @pytest.mark.asyncio
    async def test_caching_returns_same_keys(self, manager):
        s1 = await manager.get_cached_status()
        s2 = await manager.get_cached_status()
        assert set(s1.keys()) == set(s2.keys())


# ---------------------------------------------------------------------------
# Lock discipline (concurrent access)
# ---------------------------------------------------------------------------

class TestLockDiscipline:
    @pytest.mark.asyncio
    async def test_concurrent_resolve(self, manager):
        """Fire off many concurrent resolve_model_id calls; none should crash."""
        tasks = [
            manager.resolve_model_id(f"model-{i}")
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 20

    @pytest.mark.asyncio
    async def test_concurrent_list_models(self, manager):
        tasks = [manager.list_models() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert all(len(r) > 0 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_config_access(self, manager):
        tasks = [manager.get_config("model-a.gguf") for _ in range(10)]
        results = await asyncio.gather(*tasks)
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_status(self, manager):
        tasks = [manager.get_cached_status() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        assert all(isinstance(r, dict) for r in results)


# ---------------------------------------------------------------------------
# _sanitize_model_id / _sanitize_label
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_sanitize_model_id(self):
        assert ModelManager._sanitize_model_id("model.gguf") == "model.gguf"
        # replaces \, /, .. then strips whitespace
        assert ModelManager._sanitize_model_id("a\\b..c") == "abc"
        assert ModelManager._sanitize_model_id("x/y..z") == "xyz"

    def test_sanitize_label(self):
        assert ModelManager._sanitize_label("default") == "default"
        assert ModelManager._sanitize_label("a\\b..c") == "abc"
        assert ModelManager._sanitize_label("  spaced  ") == "spaced"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_valid_config(self):
        assert ModelManager.validate_config(_make_cfg()) is None

    def test_missing_launcher(self):
        cfg = _make_cfg()
        cfg.pop("launcher")
        with pytest.raises(ValueError, match="launcher"):
            ModelManager.validate_config(cfg)

    def test_missing_root_dir(self):
        cfg = _make_cfg()
        cfg["models"]["root_dir"] = None
        with pytest.raises(ValueError, match="root_dir"):
            ModelManager.validate_config(cfg)

    def test_missing_binary(self):
        cfg = _make_cfg()
        cfg["llama_server"]["binary"] = ""
        with pytest.raises(ValueError, match="binary"):
            ModelManager.validate_config(cfg)

    def test_port_out_of_range(self):
        cfg = _make_cfg()
        cfg["launcher"]["port"] = 70000
        with pytest.raises(ValueError, match="port"):
            ModelManager.validate_config(cfg)


# ---------------------------------------------------------------------------
# ModelConfig helpers
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_default_for(self):
        cfg = ModelConfig.default_for(Path("mymodel.gguf"))
        assert cfg.name == "mymodel"
        assert cfg.ctx_size == 8192
        assert cfg.n_gpu_layers == 999
        assert cfg.default is False
        assert cfg.pinned is False

    def test_load_and_save_roundtrip(self, tmp_path):
        gguf = tmp_path / "test.gguf"
        gguf.touch()
        cfg = ModelConfig.load(gguf)
        cfg.name = "Changed"
        cfg.save(gguf)
        cfg2 = ModelConfig.load(gguf)
        assert cfg2.name == "Changed"

    def test_to_launch_args(self):
        cfg = ModelConfig(ctx_size=4096, n_gpu_layers=50)
        args = cfg.to_launch_args(Path("/m"), 8080, "")
        assert "--ctx-size" in args
        assert "4096" in args
        assert "--n-gpu-layers" in args
        assert "50" in args

    def test_to_launch_args_alias(self):
        cfg = ModelConfig()
        args = cfg.to_launch_args(Path("/m"), 8080, "", model_id="my-model.gguf")
        assert "--alias" in args
        assert "my-model.gguf" in args

    def test_to_launch_args_no_alias_without_model_id(self):
        cfg = ModelConfig()
        args = cfg.to_launch_args(Path("/m"), 8080, "")
        assert "--alias" not in args

    def test_max_ctx_size_field(self):
        cfg = ModelConfig(ctx_size=8192, max_ctx_size=32768)
        assert cfg.max_ctx_size == 32768


# ---------------------------------------------------------------------------
# _allocate_port
# ---------------------------------------------------------------------------

class TestPortAllocation:
    def test_allocate_port_basic(self, manager):
        # Just verify it can allocate a port without error
        port = manager._allocate_port()
        assert isinstance(port, int)
        assert port >= manager.base_port

    def test_allocate_port_increments_next(self, manager):
        base_next = manager.next_port
        port = manager._allocate_port()
        assert manager.next_port > base_next


# ---------------------------------------------------------------------------
# LoadedModel touch
# ---------------------------------------------------------------------------

class TestLoadedModel:
    def test_touch_updates_last_used(self):
        t1 = time.time()
        lm = LoadedModel(
            model_id="a.gguf", gguf_path=Path("a.gguf"),
            config=ModelConfig.default_for(Path("a.gguf")),
            port=9001, process=MagicMock(), ready=False, pid=1,
        )
        time.sleep(0.01)
        t2 = time.time()
        lm.touch()
        assert t1 <= lm.last_used <= t2


def _make_dummy_gguf(name: str | None = None, max_ctx: int | None = None,
                     ctx_vtype: int = 4) -> bytes:
    """Build a minimal GGUF header with optional general.name and llama.context_length.

    ctx_vtype selects the value type used for context_length (default 4=UINT32);
    pass 10=UINT64 or 11=INT64 to exercise the wider integer branches in skip_val.
    """
    import struct
    buf = bytearray(b"GGUF")
    kvs = 0
    if name is not None: kvs += 1
    if max_ctx is not None: kvs += 1
    buf.extend(struct.pack("<IQQ", 3, 0, kvs))

    if name is not None:
        key = "general.name".encode('utf-8')
        val = name.encode('utf-8')
        buf.extend(struct.pack("<Q", len(key)) + key)
        buf.extend(struct.pack("<I", 8)) # STRING
        buf.extend(struct.pack("<Q", len(val)) + val)

    if max_ctx is not None:
        key = "llama.context_length".encode('utf-8')
        buf.extend(struct.pack("<Q", len(key)) + key)
        if ctx_vtype == 4:      # UINT32
            buf.extend(struct.pack("<I", 4))
            buf.extend(struct.pack("<I", max_ctx))
        elif ctx_vtype == 10:   # UINT64
            buf.extend(struct.pack("<I", 10))
            buf.extend(struct.pack("<Q", max_ctx))
        elif ctx_vtype == 11:   # INT64
            buf.extend(struct.pack("<I", 11))
            buf.extend(struct.pack("<q", max_ctx))
        else:
            raise ValueError(f"unsupported ctx_vtype {ctx_vtype}")

    return bytes(buf)


def _make_gguf_with_skips(name: str | None = None, max_ctx: int | None = None,
                          skip_fields: list | None = None) -> bytes:
    """Build a GGUF header with optional extra KV fields (to be skipped by the parser)
    placed before general.name and llama.context_length.

    Each entry in skip_fields is (key, vtype, payload_bytes). This exercises the
    skip_val path for FLOAT32 (6), BOOL (7) and ARRAY (9) value types.
    """
    import struct
    skip_fields = skip_fields or []
    buf = bytearray(b"GGUF")
    kvs = len(skip_fields)
    if name is not None: kvs += 1
    if max_ctx is not None: kvs += 1
    buf.extend(struct.pack("<IQQ", 3, 0, kvs))

    for key, vtype, payload in skip_fields:
        kb = key.encode('utf-8')
        buf.extend(struct.pack("<Q", len(kb)) + kb)
        buf.extend(struct.pack("<I", vtype))
        buf.extend(payload)

    if name is not None:
        key = "general.name".encode('utf-8')
        val = name.encode('utf-8')
        buf.extend(struct.pack("<Q", len(key)) + key)
        buf.extend(struct.pack("<I", 8)) # STRING
        buf.extend(struct.pack("<Q", len(val)) + val)

    if max_ctx is not None:
        key = "llama.context_length".encode('utf-8')
        buf.extend(struct.pack("<Q", len(key)) + key)
        buf.extend(struct.pack("<I", 4)) # UINT32
        buf.extend(struct.pack("<I", max_ctx))

    return bytes(buf)


def _make_gguf_unknown_vtype(unknown_vtype: int = 13) -> bytes:
    """Build a GGUF header whose first KV value has an unrecognized type code.

    The parser's skip_val should raise on it and the outer handler returns (None, None).
    """
    import struct
    buf = bytearray(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, 0, 1))
    key = "some.unknown".encode('utf-8')
    buf.extend(struct.pack("<Q", len(key)) + key)
    buf.extend(struct.pack("<I", unknown_vtype))
    # A little trailing payload that would be misread if the parser failed to abort.
    buf.extend(b"\x00" * 16)
    return bytes(buf)


class TestReadGGUFMetadata:
    def test_returns_none_on_missing_file(self):
        """When file doesn't exist, function should return (None, None) without raising."""
        name, max_ctx = _read_gguf_metadata(Path("/nonexistent/path/model.gguf"))
        assert name is None
        assert max_ctx is None

    def test_returns_none_on_invalid_gguf_data(self, tmp_path):
        """When file exists but isn't valid GGUF, function should return (None, None)."""
        fake_gguf = tmp_path / "fake.gguf"
        fake_gguf.write_bytes(b"NOT_A_REAL_GGUF_FILE" * 100)
        name, max_ctx = _read_gguf_metadata(fake_gguf)
        assert name is None
        assert max_ctx is None

    def test_handles_missing_name_field(self, tmp_path):
        """When general.name field is missing, name should be None but context_length still read."""
        gguf_file = tmp_path / "no_name.gguf"
        gguf_file.write_bytes(_make_dummy_gguf(name=None, max_ctx=4096))
        name, max_ctx = _read_gguf_metadata(gguf_file)
        assert name is None
        assert max_ctx == 4096

    def test_handles_missing_context_length_field(self, tmp_path):
        """When context_length field is missing, max_ctx should be None but name still read."""
        gguf_file = tmp_path / "no_ctx.gguf"
        gguf_file.write_bytes(_make_dummy_gguf(name="Test Model", max_ctx=None))
        name, max_ctx = _read_gguf_metadata(gguf_file)
        assert name == "Test Model"
        assert max_ctx is None

    def test_handles_parse_error_in_context_length(self, tmp_path):
        """When file has a corrupt header (bad magic/counts), gracefully returns (None, None)."""
        gguf_file = tmp_path / "bad.gguf"
        gguf_file.write_bytes(b"GGUF" + b"\xff" * 20)
        name, max_ctx = _read_gguf_metadata(gguf_file)
        assert name is None
        assert max_ctx is None

    def test_context_length_uint64(self, tmp_path):
        """context_length stored as UINT64 (vtype 10) is read correctly."""
        gguf_file = tmp_path / "ctx_u64.gguf"
        gguf_file.write_bytes(_make_dummy_gguf(name="U64 Model", max_ctx=32768, ctx_vtype=10))
        name, max_ctx = _read_gguf_metadata(gguf_file)
        assert name == "U64 Model"
        assert max_ctx == 32768

    def test_context_length_int64(self, tmp_path):
        """context_length stored as INT64 (vtype 11) is read correctly."""
        gguf_file = tmp_path / "ctx_i64.gguf"
        gguf_file.write_bytes(_make_dummy_gguf(name="I64 Model", max_ctx=65536, ctx_vtype=11))
        name, max_ctx = _read_gguf_metadata(gguf_file)
        assert name == "I64 Model"
        assert max_ctx == 65536

    def test_skips_float32_bool_and_array_fields(self, tmp_path):
        """Extra KV fields (FLOAT32, BOOL, ARRAY of uint32) before the target fields
        are skipped correctly, and name + context_length are still read."""
        skip_fields = [
            ("some.float", 6, struct.pack("<f", 1.5)),                 # FLOAT32
            ("some.bool", 7, b"\x01"),                                 # BOOL
            ("some.arr", 9, struct.pack("<IQ", 4, 3) + struct.pack("<III", 1, 2, 3)),  # ARRAY of uint32
        ]
        gguf_file = tmp_path / "skip.gguf"
        gguf_file.write_bytes(_make_gguf_with_skips(name="Skip Model", max_ctx=4096, skip_fields=skip_fields))
        name, max_ctx = _read_gguf_metadata(gguf_file)
        assert name == "Skip Model"
        assert max_ctx == 4096

    def test_unknown_value_type_returns_none(self, tmp_path):
        """An unrecognized value type aborts parsing and returns (None, None) gracefully."""
        gguf_file = tmp_path / "unknown.gguf"
        gguf_file.write_bytes(_make_gguf_unknown_vtype(unknown_vtype=13))
        name, max_ctx = _read_gguf_metadata(gguf_file)
        assert name is None
        assert max_ctx is None

    def test_read_gguf_name_delegates_correctly(self):
        """_read_gguf_name should return just the name from _read_gguf_metadata."""
        with patch("server._read_gguf_metadata") as mock_fn:
            mock_fn.return_value = ("Mocked Name", 8192)
            assert _read_gguf_name(Path("test.gguf")) == "Mocked Name"