"""
Unit tests for llama-autoloader server.py – ModelManager core logic.

Run with:  pytest test_server.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytest_asyncio

import server
import yaml
from fastapi import HTTPException

from server import ModelManager, ModelConfig, LoadedModel, UnloadResult, _read_gguf_metadata, _read_gguf_name

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

    def test_lone_unrelated_mmproj_is_not_auto_paired(self, manager):
        """A lone projector with no name relationship must NOT be attached to the models.

        Previously the "exactly one mmproj in the tree" fallback paired that projector
        with EVERY model in the library regardless of name. That is the broadest form of
        the mispairing bug (it bypassed the directory-scoped rule entirely): an unrelated
        vision projector silently became every text model's --mmproj.
        """
        assert "clip-mmproj.gguf" not in manager.models   # filtered from the registry
        assert len(manager._mmproj_paths) == 1           # ... but still detected as one
        assert manager.mmproj_paths == {}, "no name match, so nothing may be paired"

    def test_lone_named_mmproj_still_pairs_via_fallback(self, tmp_path):
        """The single-projector fallback still works when the name DOES match a model."""
        from server import ModelManager
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "llava-v1.gguf").write_bytes(b"GGUF_HEADER" * 10)
        (models_dir / "mmproj-llava.gguf").write_bytes(b"GGUF_HEADER" * 5)
        cfg = _make_cfg({"models": {"root_dir": str(models_dir),
                                        "save_state_dir": str(tmp_path / "states"),
                                        "backends_dir": str(tmp_path / "backends")}})
        mgr = ModelManager(cfg)
        mgr.scan()
        assert mgr.mmproj_paths.get("llava-v1.gguf") == models_dir / "mmproj-llava.gguf"

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
        # label is derived by stripping the known "{mid_clean}." prefix from p.stem,
        # so the number of dots in the model id doesn't matter
        assert len(labels) == 2
        assert "default" in labels
        assert "convo1" in labels

    @pytest.mark.asyncio
    async def test_dotted_model_name(self, manager, tmp_model_dir):
        # Regression: model ids with embedded dots (e.g. "Qwen3.8-...") used to
        # mangle the label via split(".", 1)[1] → "8-27B-..." instead of "default"
        dotted = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
        # Register the model in root_dir and re-scan so resolve_model_id hits
        # the exact-match branch (without it, resolution falls back to the
        # default-marked model-b.gguf and this test would be vacuous)
        (tmp_model_dir / dotted).write_bytes(b"GGUF_HEADER" * 50)
        manager.scan()
        assert await manager.resolve_model_id(dotted) == dotted
        (manager.save_state_dir / f"{dotted}.default.bin").write_bytes(b"state")
        labels = await manager.list_states(dotted)
        assert labels == ["default"]

    @pytest.mark.asyncio
    async def test_dotted_label(self, manager, tmp_model_dir):
        # A label containing a dot must survive intact (not truncated at the dot)
        dotted = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
        (tmp_model_dir / dotted).write_bytes(b"GGUF_HEADER" * 50)
        manager.scan()
        (manager.save_state_dir / f"{dotted}.v1.2.bin").write_bytes(b"state")
        labels = await manager.list_states(dotted)
        assert labels == ["v1.2"]

    @pytest.mark.asyncio
    async def test_empty_states(self, manager):
        labels = await manager.list_states("model-c.gguf")
        assert labels == []


# ---------------------------------------------------------------------------
# _cleanup_old_states()
# ---------------------------------------------------------------------------

class TestCleanupOldStates:
    def _make_state_files(self, state_dir, model_id, count, base_mtime):
        """Create `count` state files with distinct mtimes, oldest→newest."""
        paths = []
        for i in range(count):
            p = state_dir / f"{model_id}.state{i}.bin"
            p.write_bytes(b"state")
            t = base_mtime + i * 100
            os.utime(p, (t, t))
            paths.append(p)
        return paths

    def _remaining(self, state_dir, model_id):
        return sorted(
            p.name for p in state_dir.iterdir()
            if p.name.startswith(f"{model_id}.") and p.name.endswith(".bin")
        )

    @pytest.mark.asyncio
    async def test_keeps_newest_by_mtime(self, manager):
        # 6 state files, max_keep=3 → exactly the 3 newest survive
        paths = self._make_state_files(manager.save_state_dir, "model-c.gguf", 6, 1_000_000)
        manager.save_state_max_keep = 3
        await manager._cleanup_old_states("model-c.gguf")
        assert self._remaining(manager.save_state_dir, "model-c.gguf") == [
            paths[3].name, paths[4].name, paths[5].name,
        ]

    @pytest.mark.asyncio
    async def test_max_keep_one(self, manager):
        paths = self._make_state_files(manager.save_state_dir, "model-c.gguf", 4, 2_000_000)
        manager.save_state_max_keep = 1
        await manager._cleanup_old_states("model-c.gguf")
        assert self._remaining(manager.save_state_dir, "model-c.gguf") == [paths[-1].name]

    @pytest.mark.asyncio
    async def test_live_read_after_construction(self, manager):
        # Constructed with the default (5); changing the attr AFTER construction
        # must take effect — proves cleanup reads the instance attr, not a frozen constant.
        assert manager.save_state_max_keep == 5
        paths = self._make_state_files(manager.save_state_dir, "model-c.gguf", 6, 3_000_000)
        manager.save_state_max_keep = 2
        await manager._cleanup_old_states("model-c.gguf")
        assert self._remaining(manager.save_state_dir, "model-c.gguf") == [
            paths[4].name, paths[5].name,
        ]

    @pytest.mark.asyncio
    async def test_noop_when_under_limit(self, manager):
        paths = self._make_state_files(manager.save_state_dir, "model-c.gguf", 3, 4_000_000)
        manager.save_state_max_keep = 5
        await manager._cleanup_old_states("model-c.gguf")
        assert self._remaining(manager.save_state_dir, "model-c.gguf") == [p.name for p in paths]


# ---------------------------------------------------------------------------
# _enforce_state_budget()
# ---------------------------------------------------------------------------

class TestStateBudget:
    def _isolate_states(self, manager, tmp_path):
        """Point the manager at a fresh EMPTY state dir.

        The `manager` fixture pre-seeds model-a/model-b state files; the budget
        groups by the KNOWN MODEL REGISTRY (self.models), not by what happens to
        already be on disk, so an empty dir keeps the byte math deterministic.
        """
        d = tmp_path / "budget_states"
        d.mkdir()
        manager.save_state_dir = d
        return d

    def _make_state_files(self, state_dir, model_id, count, base_mtime, size=1000):
        """Create `count` state files for a model with distinct mtimes (oldest→newest)."""
        paths = []
        for i in range(count):
            p = state_dir / f"{model_id}.state{i}.bin"
            p.write_bytes(b"x" * size)
            t = base_mtime + i * 100
            os.utime(p, (t, t))
            paths.append(p)
        return paths

    def _write_state(self, state_dir, name, size, mtime=None):
        p = state_dir / name
        p.write_bytes(b"x" * size)
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p

    def _remaining_bins(self, state_dir):
        return sorted(p.name for p in state_dir.iterdir() if p.name.endswith(".bin"))

    @pytest.mark.asyncio
    async def test_disabled_zero_evicts_nothing(self, manager, tmp_path):
        d = self._isolate_states(manager, tmp_path)
        manager.save_state_max_bytes = 0
        self._make_state_files(d, "model-c.gguf", 3, 1_000_000, size=5000)
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == [
            "model-c.gguf.state0.bin", "model-c.gguf.state1.bin", "model-c.gguf.state2.bin",
        ]

    @pytest.mark.asyncio
    async def test_under_budget_evicts_nothing(self, manager, tmp_path):
        d = self._isolate_states(manager, tmp_path)
        manager.save_state_max_bytes = 10_000
        self._make_state_files(d, "model-c.gguf", 3, 1_000_000, size=100)
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == [
            "model-c.gguf.state0.bin", "model-c.gguf.state1.bin", "model-c.gguf.state2.bin",
        ]

    @pytest.mark.asyncio
    async def test_evicts_oldest_first_until_under_budget(self, manager, tmp_path):
        # 3 × 1000 = 3000 > budget 2000 → evict only the single oldest, stop at 2000.
        d = self._isolate_states(manager, tmp_path)
        self._make_state_files(d, "model-c.gguf", 3, 1_000_000, size=1000)
        manager.save_state_max_bytes = 2000
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == [
            "model-c.gguf.state1.bin", "model-c.gguf.state2.bin",
        ]

    @pytest.mark.asyncio
    async def test_floor_one_newest_per_model_preserved(self, manager, tmp_path):
        # Budget smaller than any single file → evict all but the newest per model.
        d = self._isolate_states(manager, tmp_path)
        self._make_state_files(d, "model-c.gguf", 3, 1_000_000, size=1000)
        manager.save_state_max_bytes = 1
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == ["model-c.gguf.state2.bin"]

    @pytest.mark.asyncio
    async def test_keep_path_never_evicted(self, manager, tmp_path):
        # keep_path is the OLDEST file; one eviction is required → must skip the
        # protected keep_path and evict the next-oldest instead.
        d = self._isolate_states(manager, tmp_path)
        paths = self._make_state_files(d, "model-c.gguf", 3, 1_000_000, size=1000)
        manager.save_state_max_bytes = 2000
        await manager._enforce_state_budget(keep_path=paths[0])
        assert paths[0].exists()
        assert self._remaining_bins(d) == [
            "model-c.gguf.state0.bin", "model-c.gguf.state2.bin",
        ]

    @pytest.mark.asyncio
    async def test_scratch_files_never_counted_or_evicted(self, manager, tmp_path):
        # A large model-prefixed scratch file must NOT be counted against the budget
        # nor evicted. The single real state file (100 B) is under budget on its own.
        d = self._isolate_states(manager, tmp_path)
        self._write_state(d, "model-c.gguf.probe.bin", 100_000)
        self._write_state(d, "probe.bin", 100_000)
        self._write_state(d, "viaraw.bin", 100_000)
        self._write_state(d, "model-c.gguf.state0.bin", 100)
        manager.save_state_max_bytes = 1000
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == [
            "model-c.gguf.probe.bin", "model-c.gguf.state0.bin", "probe.bin", "viaraw.bin",
        ]

    @pytest.mark.asyncio
    async def test_cross_model_eviction_is_global_oldest_first(self, manager, tmp_path):
        # Two models, each with 2 files. Eviction order is GLOBAL by mtime (not
        # per-model), while the floor is per-model. Total 4000 > budget 2000.
        # model-c files are globally oldest → its non-floor file is evicted first,
        # then model-a's non-floor file; both floors survive.
        d = self._isolate_states(manager, tmp_path)
        self._make_state_files(d, "model-c.gguf", 2, 1_000_000, size=1000)
        self._make_state_files(d, "model-a.gguf", 2, 2_000_000, size=1000)
        manager.save_state_max_bytes = 2000
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == [
            "model-a.gguf.state1.bin", "model-c.gguf.state1.bin",
        ]

    @pytest.mark.asyncio
    async def test_nested_model_prefix_counted_once(self, manager, tmp_path):
        # A model id that is a dotted prefix of another (``foo.gguf`` and
        # ``foo.gguf.lora.gguf``): the lora state matches BOTH prefixes. With
        # correct single-ownership (longest prefix wins) total is 4000 and only
        # the globally-oldest non-floor (foo.gguf.a) is evicted; a naive double
        # count would inflate the total and evict the lora state.
        from server import ModelConfig
        d = self._isolate_states(manager, tmp_path)
        manager.models.clear()
        manager.models.update({
            "foo.gguf": ModelConfig.default_for(Path("foo.gguf")),
            "foo.gguf.lora.gguf": ModelConfig.default_for(Path("foo.gguf.lora.gguf")),
        })
        self._write_state(d, "foo.gguf.a.bin", 1000, 1_000_000)          # evictable (not foo floor)
        self._write_state(d, "foo.gguf.lora.gguf.x.bin", 1000, 1_000_100)  # owned by lora, must survive
        self._write_state(d, "foo.gguf.b.bin", 1000, 1_000_200)          # floor for foo.gguf
        self._write_state(d, "foo.gguf.lora.gguf.y.bin", 1000, 2_000_000)  # floor for lora
        manager.save_state_max_bytes = 3000
        await manager._enforce_state_budget()
        assert not (d / "foo.gguf.a.bin").exists()
        assert self._remaining_bins(d) == [
            "foo.gguf.b.bin", "foo.gguf.lora.gguf.x.bin", "foo.gguf.lora.gguf.y.bin",
        ]

    @pytest.mark.asyncio
    async def test_start_sweep_reclaims_over_budget_dir(self, manager, tmp_path):
        # A states/ dir already over budget from a previous run must be reclaimed
        # by the async startup sweep in ModelManager.start(). start() re-scans
        # models but never touches save_state_dir, so the isolated dir survives.
        d = self._isolate_states(manager, tmp_path)
        self._make_state_files(d, "model-c.gguf", 3, 1_000_000, size=1000)
        manager.save_state_max_bytes = 2000
        await manager.start()
        try:
            assert self._remaining_bins(d) == [
                "model-c.gguf.state1.bin", "model-c.gguf.state2.bin",
            ]
        finally:
            await manager.stop()

    # ---- Orphan handling: files with no registered owner (renamed/removed
    # model, or id CASE drift) are counted against the budget and reclaimed.
    # An id like "orphan-gone.gguf" / "strata-coder-iq1_m" is NOT in the
    # registry, so no model prefix matches it -> owner None.

    @pytest.mark.asyncio
    async def test_orphan_files_no_floor_all_reclaimed(self, manager, tmp_path):
        # ORPHANS have NO floor, so with a budget smaller than everything,
        # ALL orphan files are evicted.
        d = self._isolate_states(manager, tmp_path)
        self._write_state(d, "orphan-gone.gguf.state0.bin", 1000, 1_000_000)
        self._write_state(d, "orphan-gone.gguf.state1.bin", 1000, 1_000_100)
        manager.save_state_max_bytes = 1
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == []

    @pytest.mark.asyncio
    async def test_renamed_model_old_id_treated_as_orphan(self, manager, tmp_path):
        # A model was RENAMED: the NEW id is registered, the OLD id's state
        # files remain on disk. Old-id files match NO registered prefix -> orphans
        # -> reclaimed oldest-first. Reclaiming the two orphans alone does NOT get
        # under budget (remaining stays 2000 > 1500), so the loop reaches the new-id
        # file and the newest-per-KNOWN-model floor is what protects it. This test
        # therefore covers both orphan reclaim and the floor.
        from server import ModelConfig
        d = self._isolate_states(manager, tmp_path)
        manager.models.clear()
        manager.models["qwen3.8-flash-next-coder-iq1_m"] = ModelConfig.default_for(
            Path("qwen3.8-flash-next-coder-iq1_m")
        )
        self._write_state(d, "strata-coder-iq1_m.state0.bin", 1000, 1_000_000)
        self._write_state(d, "strata-coder-iq1_m.state1.bin", 1000, 1_000_100)
        self._write_state(d, "qwen3.8-flash-next-coder-iq1_m.default.bin", 1000, 2_000_000)
        manager.save_state_max_bytes = 1500
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == ["qwen3.8-flash-next-coder-iq1_m.default.bin"]

    @pytest.mark.asyncio
    async def test_known_model_floor_survives_when_only_orphans_insufficient(self, manager, tmp_path):
        # Eviction demand exceeds what the orphans alone can free: both orphans
        # are reclaimed (no floor), but the KNOWN model's newest file is protected
        # by the floor even though the budget can no longer be met.
        d = self._isolate_states(manager, tmp_path)
        self._write_state(d, "model-c.gguf.new.bin", 1000, 3_000_000)  # known model, floor
        self._write_state(d, "orphan-gone.gguf.a.bin", 1000, 500_000)
        self._write_state(d, "orphan-gone.gguf.b.bin", 1000, 600_000)
        manager.save_state_max_bytes = 500
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == ["model-c.gguf.new.bin"]

    @pytest.mark.asyncio
    async def test_orphan_only_over_budget_still_fires(self, manager, tmp_path):
        # Regression guard for the exact bug: the ONLY over-budget content is
        # orphan (renamed/removed) files. The old owner-only scan saw total=0
        # ("under budget") and evicted nothing; now orphans are counted and the
        # oldest is reclaimed.
        d = self._isolate_states(manager, tmp_path)
        self._write_state(d, "qwen3.8-27b-gsq-rco-iq3_s.old.bin", 1000, 1_000_000)
        self._write_state(d, "qwen3.8-27b-gsq-rco-iq3_s.new.bin", 1000, 1_000_100)
        manager.save_state_max_bytes = 1500
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == ["qwen3.8-27b-gsq-rco-iq3_s.new.bin"]

    @pytest.mark.asyncio
    async def test_case_drift_old_lowercase_is_orphan_new_cased_floored(self, manager, tmp_path):
        # The NInfer-card bug: an id's CASE drifted between save time and the
        # current registry. Prefix matching is case-sensitive, so the OLD lowercase
        # file matches NO registered prefix -> orphan (reclaimed); the current-cased
        # file IS owned and is floored. Confirms no crash and correct ownership.
        from server import ModelConfig
        d = self._isolate_states(manager, tmp_path)
        manager.models.clear()
        manager.models["Qwen3.8-27B-GSQ-RCO-IQ3_S"] = ModelConfig.default_for(
            Path("Qwen3.8-27B-GSQ-RCO-IQ3_S")
        )
        self._write_state(d, "qwen3.8-27b-gsq-rco-iq3_s.auto.bin", 1000, 1_000_000)  # old case -> orphan
        self._write_state(d, "Qwen3.8-27B-GSQ-RCO-IQ3_S.maine.bin", 1000, 1_000_100)  # current case -> floored
        manager.save_state_max_bytes = 1500
        await manager._enforce_state_budget()
        assert self._remaining_bins(d) == ["Qwen3.8-27B-GSQ-RCO-IQ3_S.maine.bin"]

    @pytest.mark.asyncio
    async def test_orphan_keep_path_never_evicted(self, manager, tmp_path):
        # keep_path is an ORPHAN and the oldest file: it must still be protected
        # (keep_path overrides the no-floor rule), so the next-oldest orphan is
        # evicted instead.
        d = self._isolate_states(manager, tmp_path)
        keep = self._write_state(d, "orphan-gone.gguf.a.bin", 1000, 1_000_000)
        self._write_state(d, "orphan-gone.gguf.b.bin", 1000, 1_000_100)
        manager.save_state_max_bytes = 1000
        await manager._enforce_state_budget(keep_path=keep)
        assert keep.exists()
        assert self._remaining_bins(d) == ["orphan-gone.gguf.a.bin"]


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

    @pytest.mark.asyncio
    async def test_skips_mid_load_models(self, manager):
        # The mid-load model is the LRU pick; the filter must make the OTHER
        # model the victim instead. Asserting "nothing evicted" would also pass
        # with a filter that wrongly excludes everything, so this shape is
        # the discriminating one.
        manager.models.clear()
        manager.loaded.clear()
        cfg_x = ModelConfig.default_for(Path("x.gguf"))
        cfg_y = ModelConfig.default_for(Path("y.gguf"))
        cfg_z = ModelConfig.default_for(Path("z.gguf"))
        manager.models.update({"x.gguf": cfg_x, "y.gguf": cfg_y, "z.gguf": cfg_z})

        lm_x = LoadedModel("x.gguf", Path("x.gguf"), cfg_x, 9001, MagicMock(),
                           last_used=time.time() - 100, ready=True, pid=1)
        lm_y = LoadedModel("y.gguf", Path("y.gguf"), cfg_y, 9002, MagicMock(),
                           last_used=time.time() - 50, ready=True, pid=2)
        manager.loaded.update({"x.gguf": lm_x, "y.gguf": lm_y})

        # x.gguf is mid-load: an incomplete task in _loading_tasks.
        task = asyncio.create_task(asyncio.sleep(30))
        manager._loading_tasks["x.gguf"] = task
        try:
            async with manager._lock:
                result = await manager._evict_lru_if_needed("z.gguf")
        finally:
            task.cancel()
            manager._loading_tasks.pop("x.gguf", None)

        assert result is not None
        assert result.model_id == "y.gguf"
        assert "x.gguf" in manager.loaded   # mid-load model survives
        assert "y.gguf" not in manager.loaded

    @pytest.mark.asyncio
    async def test_no_eviction_when_only_mid_load_candidates(self, manager):
        manager.models.clear()
        manager.loaded.clear()
        cfg_x = ModelConfig.default_for(Path("x.gguf"))
        cfg_y = ModelConfig.default_for(Path("y.gguf"))
        cfg_z = ModelConfig.default_for(Path("z.gguf"))
        manager.models.update({"x.gguf": cfg_x, "y.gguf": cfg_y, "z.gguf": cfg_z})

        lm_x = LoadedModel("x.gguf", Path("x.gguf"), cfg_x, 9001, MagicMock(),
                           last_used=time.time() - 100, ready=True, pid=1)
        lm_y = LoadedModel("y.gguf", Path("y.gguf"), cfg_y, 9002, MagicMock(),
                           last_used=time.time() - 50, ready=True, pid=2)
        manager.loaded.update({"x.gguf": lm_x, "y.gguf": lm_y})

        task = asyncio.create_task(asyncio.sleep(30))
        manager._loading_tasks["x.gguf"] = task
        task2 = asyncio.create_task(asyncio.sleep(30))
        manager._loading_tasks["y.gguf"] = task2
        try:
            async with manager._lock:
                result = await manager._evict_lru_if_needed("z.gguf")
        finally:
            task.cancel()
            task2.cancel()
            manager._loading_tasks.clear()

        assert result is None
        assert len(manager.loaded) == 2

    @pytest.mark.asyncio
    async def test_load_refused_when_eviction_fails(self, manager):
        # Call-site guard: a failed eviction verdict (process still running)
        # must refuse the load and leave the registry unchanged.
        manager.models.clear()
        manager.loaded.clear()
        cfg_x = ModelConfig.default_for(Path("x.gguf"))
        cfg_y = ModelConfig.default_for(Path("y.gguf"))
        cfg_z = ModelConfig.default_for(Path("z.gguf"))
        manager.models.update({"x.gguf": cfg_x, "y.gguf": cfg_y, "z.gguf": cfg_z})

        lm_x = LoadedModel("x.gguf", Path("x.gguf"), cfg_x, 9001, MagicMock(),
                           last_used=time.time() - 100, ready=True, pid=1)
        lm_y = LoadedModel("y.gguf", Path("y.gguf"), cfg_y, 9002, MagicMock(),
                           last_used=time.time() - 50, ready=True, pid=2)
        manager.loaded.update({"x.gguf": lm_x, "y.gguf": lm_y})

        failed = UnloadResult("x.gguf", False, 1, "kill could not be verified")
        with patch.object(manager, "_unload_unlocked",
                         new=AsyncMock(return_value=failed)):
            with pytest.raises(RuntimeError, match="LRU eviction of x.gguf failed"):
                await manager._do_load_model("z.gguf")

        assert "x.gguf" in manager.loaded
        assert "z.gguf" not in manager.loaded
        assert len(manager.loaded) == 2


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

    def test_save_state_max_keep_valid(self):
        cfg = _make_cfg()
        cfg["models"]["save_state_max_keep"] = 3
        assert ModelManager.validate_config(cfg) is None

    def test_save_state_max_keep_zero(self):
        cfg = _make_cfg()
        cfg["models"]["save_state_max_keep"] = 0
        with pytest.raises(ValueError, match="save_state_max_keep"):
            ModelManager.validate_config(cfg)

    def test_save_state_max_keep_not_int(self):
        cfg = _make_cfg()
        cfg["models"]["save_state_max_keep"] = "5"
        with pytest.raises(ValueError, match="save_state_max_keep"):
            ModelManager.validate_config(cfg)

    def test_save_state_max_bytes_valid(self):
        cfg = _make_cfg()
        cfg["models"]["save_state_max_bytes"] = 1000
        assert ModelManager.validate_config(cfg) is None

    def test_save_state_max_bytes_zero_allowed(self):
        # 0 is a valid "disabled" value, not an error.
        cfg = _make_cfg()
        cfg["models"]["save_state_max_bytes"] = 0
        assert ModelManager.validate_config(cfg) is None

    def test_save_state_max_bytes_negative_rejected(self):
        cfg = _make_cfg()
        cfg["models"]["save_state_max_bytes"] = -1
        with pytest.raises(ValueError, match="save_state_max_bytes"):
            ModelManager.validate_config(cfg)

    def test_save_state_max_bytes_bool_rejected(self):
        # The guarded validator must reject a bool (True == 1 would otherwise
        # read as a 1-byte budget). Mirrors the PATCH validator, not the
        # bool-guard-less save_state_max_keep validator above.
        cfg = _make_cfg()
        cfg["models"]["save_state_max_bytes"] = True
        with pytest.raises(ValueError, match="save_state_max_bytes"):
            ModelManager.validate_config(cfg)

    def test_save_state_max_bytes_not_int_rejected(self):
        cfg = _make_cfg()
        cfg["models"]["save_state_max_bytes"] = "1000"
        with pytest.raises(ValueError, match="save_state_max_bytes"):
            ModelManager.validate_config(cfg)


# ---------------------------------------------------------------------------
# PUT /v1/settings — save_state_max_bytes config seam (apply + persist)
# ---------------------------------------------------------------------------

class TestSettingsPersist:
    class _FakeRequest:
        """Minimal stand-in for fastapi.Request: update_settings only calls .json()."""
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    @pytest.mark.asyncio
    async def test_patch_applies_and_persists_max_bytes(self, manager, tmp_path, monkeypatch):
        # Drive the real handler: the new key must be applied to the live manager
        # AND written to config.yaml. Without the persist seam a PATCH is silently
        # lost on restart — this is the regression the test guards.
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(manager.cfg))
        monkeypatch.setattr(server, "manager", manager)
        monkeypatch.setattr(server, "CONFIG_PATH", str(cfg_path))
        await server.update_settings(self._FakeRequest({"save_state_max_bytes": 12345}))
        written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert written["models"]["save_state_max_bytes"] == 12345
        assert manager.save_state_max_bytes == 12345

    @pytest.mark.asyncio
    async def test_patch_rejects_negative_max_bytes(self, manager, monkeypatch):
        monkeypatch.setattr(server, "manager", manager)
        with pytest.raises(HTTPException):
            await server.update_settings(self._FakeRequest({"save_state_max_bytes": -1}))

    @pytest.mark.asyncio
    async def test_patch_rejects_bool_max_bytes(self, manager, monkeypatch):
        monkeypatch.setattr(server, "manager", manager)
        with pytest.raises(HTTPException):
            await server.update_settings(self._FakeRequest({"save_state_max_bytes": True}))

    @pytest.mark.asyncio
    async def test_get_settings_exposes_max_bytes(self, manager, monkeypatch):
        # The GET seam: the new key must be returned and reflect the live value.
        monkeypatch.setattr(server, "manager", manager)
        manager.save_state_max_bytes = 98765
        body = await server.get_settings()
        assert body["save_state_max_bytes"] == 98765


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