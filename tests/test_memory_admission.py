"""Tests for memory-aware admission (phase 2).

LRU is still the only eviction policy. The change is *when* we evict:
  * incoming model fits (per-GPU VRAM + RAM, minus headroom and reservations) -> co-load
  * it does not fit -> evict LRU, one at a time, until it fits or no candidate remains
  * no measurable profile (new/unknown, or a changed launch config) -> evict everything
    evictable so the model can be measured solo (``models.vram.unknown: evict_all``)
  * still no room -> refuse (never soft-cap)

These tests never touch nvidia-smi or real processes.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import server
from server import GPUInfo, LoadedModel, ModelConfig, UnloadResult

from tests.conftest import build_cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu(index=0, free=20000, total=24576, uuid="GPU-aaaa"):
    return GPUInfo(
        index=index,
        name="NVIDIA GeForce RTX 4090",
        total_mb=total,
        used_mb=total - free,
        free_mb=free,
        utilization_pct=0.0,
        uuid=uuid,
        driver_version="555.99",
    )


def _manager(tmp_path, *, max_loaded=3, headroom=0, ram_headroom=0,
             unknown="evict_all", occupancy=True):
    return server.ModelManager(build_cfg(
        tmp_path,
        max_loaded_models=max_loaded,
        occupancy_profiles=occupancy,
        vram={
            "enabled": True,
            "headroom_mb": headroom,
            "ram_headroom_mb": ram_headroom,
            "unknown": unknown,
            "reclaim_wait_s": 0,
        },
    ))


def _artifact(tmp_path):
    path = tmp_path / "models" / "target.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF_HEADER")
    return path


def _argv(artifact):
    return ["llama-server", "-m", str(artifact), "-c", "8192"]


def _resident(mgr, mid, *, last_used=None, pinned=False, active=0, port=9100):
    cfg = ModelConfig.default_for(Path(mid))
    cfg.pinned = pinned
    lm = LoadedModel(
        mid, Path(mid), cfg, port, MagicMock(),
        last_used=time.time() - (last_used if last_used is not None else 100),
        ready=True, pid=0,  # pid 0 => nothing to kill (fast, safe)
    )
    lm.active_requests = active
    mgr.models[mid] = cfg
    mgr.loaded[mid] = lm
    return cfg, lm


def _set_need(mgr, cfg, artifact, gpus, gpus_mb, ram_mb):
    """Store a measured profile for exactly this launch config and return its key."""
    key = mgr._occupancy_load_context(cfg, artifact, _argv(artifact), gpus)["key"]
    cfg.occupancy_profiles[key] = {"gpus_mb": gpus_mb, "ram_mb": ram_mb}
    return key


async def _admit(mgr, cfg, artifact):
    async with mgr._lock:
        await mgr._make_room_for("target.gguf", cfg, artifact, _argv(artifact))


# ---------------------------------------------------------------------------
# Measured profile: co-load when it fits
# ---------------------------------------------------------------------------

async def test_legacy_commit_based_profile_admits_by_system_delta(tmp_path):
    """Profiles written before the RSS fix must not reserve GPU-backed commit as RAM.

    The old schema stored ``ram_mb = max(system Δ, commit)``. A fully-VRAM model can
    commit ~45 GB while only ~8 GB is resident, so reading the legacy field verbatim
    would refuse a load that actually fits.
    """
    mgr = _manager(tmp_path, max_loaded=3)
    artifact = _artifact(tmp_path)
    gpus = [_gpu(free=60000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 20000}
    cfg = ModelConfig.default_for(artifact)
    key = mgr._occupancy_load_context(cfg, artifact, _argv(artifact), gpus)["key"]
    cfg.occupancy_profiles[key] = {
        "gpus_mb": {"GPU-aaaa": 1000},
        "ram_mb": 45954,          # legacy: polluted by the commit charge
        "ram_delta_mb": 8198,     # the resident-ish figure
        "ram_private_mb": 45954,  # diagnostics only
    }

    await _admit(mgr, cfg, artifact)

    assert mgr._occupancy_reservations["target.gguf"]["ram_mb"] == 8198


async def test_co_loads_when_need_fits(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3)
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(free=20000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 12000}, 10000)

    await _admit(mgr, cfg, artifact)

    assert list(mgr.loaded) == ["a.gguf"]              # resident kept
    assert mgr._occupancy_reservations["target.gguf"]["gpus_mb"] == {"GPU-aaaa": 12000}
    assert mgr._occupancy_reservations["target.gguf"]["ram_mb"] == 10000


async def test_per_gpu_split_must_fit_each_device(tmp_path):
    """A sum-fit is not enough: -ts can put most of the model on one device."""
    mgr = _manager(tmp_path, max_loaded=3)
    artifact = _artifact(tmp_path)
    gpus = [_gpu(0, free=8000, uuid="GPU-aaaa"),
            _gpu(1, free=20000, uuid="GPU-bbbb")]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)
    # Sum = 10000 <= 28000 free, but GPU0 needs 9000 > 8000.
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 9000, "GPU-bbbb": 1000}, 0)

    with pytest.raises(RuntimeError, match="exceeds available"):
        await _admit(mgr, cfg, artifact)

    assert mgr._occupancy_reservations == {}


async def test_headroom_is_subtracted(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3, headroom=2048)
    artifact = _artifact(tmp_path)
    gpus = [_gpu(free=10000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)
    # 10000 free - 2048 headroom => 9000 available, need 10000.
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 10000}, 0)

    with pytest.raises(RuntimeError, match="exceeds available"):
        await _admit(mgr, cfg, artifact)


async def test_ram_headroom_is_subtracted(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3, ram_headroom=8192)
    artifact = _artifact(tmp_path)
    gpus = [_gpu(free=20000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 20000}
    cfg = ModelConfig.default_for(artifact)
    # 20000 free - 8192 RAM headroom => 11808 available, need 15000.
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 1000}, 15000)

    with pytest.raises(RuntimeError, match="exceeds available"):
        await _admit(mgr, cfg, artifact)


async def test_one_short_card_forces_eviction_even_when_total_fits(tmp_path):
    """The recorded split is a per-card need: a total-fit does not admit."""
    mgr = _manager(tmp_path, max_loaded=3)
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(0, free=8000, uuid="GPU-aaaa"),
            _gpu(1, free=20000, uuid="GPU-bbbb")]
    free_after = [_gpu(0, free=12000, uuid="GPU-aaaa"),
                  _gpu(1, free=20000, uuid="GPU-bbbb")]
    calls = {"n": 0}

    def gpus_fn():
        calls["n"] += 1
        return gpus if calls["n"] == 1 else free_after

    mgr.gpus = gpus_fn
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)
    # Sum = 10000 fits the 28000 free, but GPU0 is 1000MB short until the eviction.
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 9000, "GPU-bbbb": 1000}, 0)

    await _admit(mgr, cfg, artifact)

    assert mgr.loaded == {}                            # evicted for the short card
    assert mgr._occupancy_reservations["target.gguf"]["gpus_mb"] == {
        "GPU-aaaa": 9000, "GPU-bbbb": 1000}


async def test_zero_need_on_full_card_still_co_loads(tmp_path):
    """A model recorded as using only GPU1 must not be blocked by a full GPU0."""
    mgr = _manager(tmp_path, max_loaded=3)
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(0, free=0, uuid="GPU-aaaa"),
            _gpu(1, free=20000, uuid="GPU-bbbb")]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 0, "GPU-bbbb": 9000}, 0)

    await _admit(mgr, cfg, artifact)

    assert list(mgr.loaded) == ["a.gguf"]              # co-load, nothing evicted


def test_memory_map_labels_use_live_gpu_indices():
    gpus = [_gpu(0, uuid="GPU-aaaa"), _gpu(1, uuid="GPU-bbbb")]
    text = server.ModelManager._format_memory_map(
        {"GPU-aaaa": 1000, "GPU-bbbb": 2000}, gpus)
    assert text == "{GPU0: 1000MB, GPU1: 2000MB}"


async def test_ram_need_can_force_eviction(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3)
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(free=20000)]
    mgr.gpus = lambda: gpus
    ram_free = iter([10000, 50000])
    mgr.system_ram = lambda: {"free_mb": next(ram_free, 50000)}
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 1000}, 30000)

    await _admit(mgr, cfg, artifact)

    assert mgr.loaded == {}                            # had to evict for RAM
    assert "target.gguf" in mgr._occupancy_reservations


# ---------------------------------------------------------------------------
# Measured profile: LRU until it fits, then refuse
# ---------------------------------------------------------------------------

async def test_evicts_lru_until_need_fits(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3)
    artifact = _artifact(tmp_path)
    _resident(mgr, "old.gguf", last_used=500)
    _resident(mgr, "new.gguf", last_used=10)
    gpus = [_gpu(free=5000)]
    free_after = [_gpu(free=25000)]
    calls = {"n": 0}

    def gpus_fn():
        calls["n"] += 1
        return gpus if calls["n"] == 1 else free_after

    mgr.gpus = gpus_fn
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 20000}, 0)

    await _admit(mgr, cfg, artifact)

    assert list(mgr.loaded) == ["new.gguf"]            # only LRU evicted
    assert "target.gguf" in mgr._occupancy_reservations


async def test_count_cap_forces_eviction_even_when_memory_fits(tmp_path):
    mgr = _manager(tmp_path, max_loaded=1)
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(free=20000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 1000}, 0)

    await _admit(mgr, cfg, artifact)

    assert mgr.loaded == {}                            # cap wins over free memory
    assert "target.gguf" in mgr._occupancy_reservations


async def test_refuses_when_no_evictable_candidate(tmp_path):
    mgr = _manager(tmp_path, max_loaded=2)
    artifact = _artifact(tmp_path)
    _resident(mgr, "pinned.gguf", pinned=True)
    gpus = [_gpu(free=4000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 9000}, 0)

    with pytest.raises(RuntimeError, match="no evictable model"):
        await _admit(mgr, cfg, artifact)

    assert list(mgr.loaded) == ["pinned.gguf"]
    assert mgr._occupancy_reservations == {}


async def test_refuses_when_eviction_cannot_be_verified(tmp_path):
    mgr = _manager(tmp_path, max_loaded=2)
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(free=1000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}

    async def failed_unload(mid, client_requested=False):
        return UnloadResult(mid, False, 123, "still running")

    mgr._unload_unlocked = failed_unload
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 9000}, 0)

    with pytest.raises(RuntimeError, match="eviction of a.gguf failed"):
        await _admit(mgr, cfg, artifact)


async def test_busy_and_mid_load_residents_are_not_victims(tmp_path):
    mgr = _manager(tmp_path, max_loaded=2)
    artifact = _artifact(tmp_path)
    _resident(mgr, "busy.gguf", active=1)
    _resident(mgr, "loading.gguf")
    task = asyncio.create_task(asyncio.sleep(30))
    mgr._loading_tasks["loading.gguf"] = task
    gpus = [_gpu(free=1000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 9000}, 0)
    try:
        with pytest.raises(RuntimeError, match="no evictable model"):
            await _admit(mgr, cfg, artifact)
    finally:
        task.cancel()
        mgr._loading_tasks.pop("loading.gguf", None)

    assert set(mgr.loaded) == {"busy.gguf", "loading.gguf"}


async def test_other_load_reservation_reduces_available_memory(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3)
    artifact = _artifact(tmp_path)
    gpus = [_gpu(free=15000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    mgr._occupancy_reservations["spawning.gguf"] = {
        "gpus_mb": {"GPU-aaaa": 10000}, "ram_mb": 0}
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 8000}, 0)

    with pytest.raises(RuntimeError, match="exceeds available"):
        await _admit(mgr, cfg, artifact)


# ---------------------------------------------------------------------------
# Unknown (no profile for this hardware+launch)
# ---------------------------------------------------------------------------

async def test_unknown_evicts_all_so_it_can_be_measured(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3, unknown="evict_all")
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf", last_used=500)
    _resident(mgr, "b.gguf", last_used=10)
    gpus = [_gpu(free=20000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)      # no occupancy_profiles entry

    await _admit(mgr, cfg, artifact)

    assert mgr.loaded == {}                      # every evictable resident was evacuated
    assert mgr._occupancy_reservations == {}     # unknown need => nothing to reserve


async def test_unknown_does_not_evict_pinned_but_still_admits_under_cap(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3, unknown="evict_all")
    artifact = _artifact(tmp_path)
    _resident(mgr, "pinned.gguf", pinned=True)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(free=20000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)

    await _admit(mgr, cfg, artifact)

    assert list(mgr.loaded) == ["pinned.gguf"]   # evictable one gone, pinned kept


async def test_unknown_refuses_when_cap_full_of_non_evictable(tmp_path):
    mgr = _manager(tmp_path, max_loaded=1, unknown="evict_all")
    artifact = _artifact(tmp_path)
    _resident(mgr, "pinned.gguf", pinned=True)
    gpus = [_gpu(free=20000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)

    with pytest.raises(RuntimeError, match="unmeasured model"):
        await _admit(mgr, cfg, artifact)

    assert list(mgr.loaded) == ["pinned.gguf"]


async def test_unknown_refuse_policy(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3, unknown="refuse")
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(free=20000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)

    with pytest.raises(RuntimeError, match="unknown is 'refuse'"):
        await _admit(mgr, cfg, artifact)

    assert list(mgr.loaded) == ["a.gguf"]


async def test_unknown_allow_keeps_legacy_count_only_behaviour(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3, unknown="allow")
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(free=1000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    cfg = ModelConfig.default_for(artifact)

    await _admit(mgr, cfg, artifact)

    assert list(mgr.loaded) == ["a.gguf"]        # co-load allowed, unmeasured
    assert mgr._occupancy_reservations == {}


# ---------------------------------------------------------------------------
# Disabled admission / config validation
# ---------------------------------------------------------------------------

async def test_disabled_admission_uses_count_only_path(tmp_path):
    mgr = _manager(tmp_path, max_loaded=1, occupancy=False)
    artifact = _artifact(tmp_path)
    _resident(mgr, "a.gguf")
    gpus = [_gpu(free=100000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 100000}
    cfg = ModelConfig.default_for(artifact)
    _set_need(mgr, cfg, artifact, gpus, {"GPU-aaaa": 1}, 1)

    await _admit(mgr, cfg, artifact)

    assert mgr.vram_admission_enabled is False
    assert mgr.loaded == {}                      # legacy cap eviction
    assert mgr._occupancy_reservations == {}


def test_config_validation_rejects_bad_vram_block():
    base = {
        "models": {"root_dir": "x", "vram": {"unknown": "banana"}},
        "launcher": {"host": "127.0.0.1", "port": 9123},
    }
    with pytest.raises(ValueError, match="models.vram.unknown"):
        server.ModelManager.validate_config(base)

    base["models"]["vram"] = {"headroom_mb": -1}
    with pytest.raises(ValueError, match="headroom_mb"):
        server.ModelManager.validate_config(base)


# ---------------------------------------------------------------------------
# End-to-end through _do_load_model (mocked spawn/readiness)
# ---------------------------------------------------------------------------

class _FakeProc:
    pid = 424242
    stdout = None
    stderr = None

    def poll(self):
        return None

    def kill(self):
        pass

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


async def test_do_load_model_evicts_all_for_new_model_and_clears_reservation(tmp_path):
    mgr = _manager(tmp_path, max_loaded=3, unknown="evict_all")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "target.gguf").write_bytes(b"GGUF_HEADER")
    _resident(mgr, "a.gguf")
    _resident(mgr, "b.gguf")
    gpus = [_gpu(free=20000)]
    mgr.gpus = lambda: gpus
    mgr.system_ram = lambda: {"free_mb": 80000}
    mgr.scan()
    mgr.system_ram = lambda: {"free_mb": 80000}
    mgr.gpus = lambda: gpus

    with patch("server.subprocess.Popen", side_effect=lambda *a, **k: _FakeProc()), \
         patch.object(server.ModelManager, "_wait_until_ready",
                      new=AsyncMock(return_value=True)):
        lm = await mgr._do_load_model("target.gguf")

    assert lm.ready is True
    assert set(mgr.loaded) == {"target.gguf"}          # both residents evacuated
    assert mgr._occupancy_reservations == {}           # reservation cleared after load
    assert lm.occupancy_baseline.get("solo") is True   # clean baseline for measurement
