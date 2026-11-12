"""Tests for the coarse VRAM/RAM occupancy profiler (Phase 1).

The profiler is deliberately simple: one before-spawn baseline, then ONE sample taken
on the first streaming request while the model is solo-loaded, per-GPU deltas plus
``max(system RAM delta, process RSS)``, stored in the model sidecar keyed by
``<hardware fp>:<launch-config fp>``. After that the model is never resampled until its
launch config (hence the key) changes.

These tests never shell out to nvidia-smi; ``gpus()`` and ``system_ram()`` are patched.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import server
from server import (
    GPUInfo,
    LoadedModel,
    ModelConfig,
    _gpu_identity,
    _gpu_key,
    _occupancy_argv_flavor,
    _occupancy_hash,
)

from tests.conftest import build_cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu(index=0, used=1000, total=24576, uuid="GPU-aaaa", driver="555.99"):
    return GPUInfo(
        index=index,
        name="NVIDIA GeForce RTX 4090",
        total_mb=total,
        used_mb=used,
        free_mb=total - used,
        utilization_pct=0.0,
        uuid=uuid,
        driver_version=driver,
    )


def _manager(tmp_path, **cfg_kw):
    cfg_kw.setdefault("occupancy_profiles", True)
    return server.ModelManager(build_cfg(tmp_path, **cfg_kw))


def _gguf(tmp_path, mid="model.gguf", content=b"GGUF_HEADER"):
    path = tmp_path / "models" / mid
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _loaded(mid, cfg, gguf_path, **overrides):
    lm = LoadedModel(
        model_id=mid,
        gguf_path=gguf_path,
        config=cfg,
        port=9005,
        process=MagicMock(),
        pid=0,  # psutil.Process(0) fails -> RSS/commit are 0 unless a test patches it
    )
    for k, v in overrides.items():
        setattr(lm, k, v)
    return lm


def _baseline(gpus_used=None, ram_used=1000):
    return {
        "key": "hw:cfg",
        "hw": "NVIDIA GeForce RTX 4090 / driver 555.99",
        "engine": "llama",
        "backend": "",
        "argv_flavor": "llama-server -m model.gguf",
        "model_file": {},
        "gpus_used_mb": gpus_used if gpus_used is not None else {"GPU-aaaa": 1000},
        "ram_used_mb": ram_used,
        "taken_at": 0.0,
    }


# ---------------------------------------------------------------------------
# Argv normalization / fingerprints
# ---------------------------------------------------------------------------

def test_argv_flavor_drops_volatile_flags_only():
    argv = [
        "llama-server", "-m", "model.gguf", "-c", "8192", "-ts", "100,80",
        "--port", "9005", "--host", "127.0.0.1", "--alias", "fast",
        "--slot-save-path", "C:/saves", "--model-id", "abc",
        "--speculative.n-max", "8",
    ]
    flavor = _occupancy_argv_flavor(argv)
    assert "model.gguf" in flavor
    assert "-c 8192" in flavor
    assert "-ts 100,80" in flavor
    assert "--speculative.n-max 8" in flavor
    for dropped in ("9005", "127.0.0.1", "fast", "C:/saves", "abc"):
        assert dropped not in flavor
    # --flag=value forms are dropped too.
    assert "--port=9005" not in _occupancy_argv_flavor(["--port=9005", "-m", "x"])


def test_hardware_fingerprint_is_about_the_box_not_the_load():
    a = [_gpu(0, used=1000), _gpu(1, used=2000, uuid="GPU-bbbb")]
    b = [_gpu(0, used=9999), _gpu(1, used=12345, uuid="GPU-bbbb")]
    assert _occupancy_hash(_gpu_identity(a)) == _occupancy_hash(_gpu_identity(b))

    # A different device, driver or total makes it a different profile.
    assert _occupancy_hash(_gpu_identity(a)) != _occupancy_hash(
        _gpu_identity([_gpu(0, uuid="GPU-cccc"), _gpu(1, uuid="GPU-bbbb")]))
    assert _occupancy_hash(_gpu_identity(a)) != _occupancy_hash(
        _gpu_identity([_gpu(0, driver="556.01"), _gpu(1, uuid="GPU-bbbb")]))


async def test_baseline_key_is_stable_and_invalidates_on_launch_changes(tmp_path):
    mgr = _manager(tmp_path)
    mgr.gpus = lambda: [_gpu(0)]
    mgr.system_ram = lambda: {"used_mb": 8000}
    artifact = _gguf(tmp_path)
    os.utime(artifact, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    cfg = ModelConfig.default_for(artifact)
    argv = ["llama-server", "-m", str(artifact), "-c", "8192", "-ts", "100,80"]

    first = await mgr._capture_occupancy_baseline("model.gguf", cfg, artifact, argv)
    second = await mgr._capture_occupancy_baseline("model.gguf", cfg, artifact, argv)
    assert first["key"] == second["key"]
    assert first["key"].count(":") == 1

    # Volatile per-instance flags must not change the key.
    other_port = argv + ["--port", "9999", "--alias", "whatever"]
    assert (await mgr._capture_occupancy_baseline(
        "model.gguf", cfg, artifact, other_port))["key"] == first["key"]

    # A real launch change (context size) must.
    assert (await mgr._capture_occupancy_baseline(
        "model.gguf", cfg, artifact, argv[:-1] + ["4096"]))["key"] != first["key"]

    # So must a changed model file (size/mtime).
    artifact.write_bytes(b"GGUF_HEADER_V2")
    os.utime(artifact, ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000))
    assert (await mgr._capture_occupancy_baseline(
        "model.gguf", cfg, artifact, argv))["key"] != first["key"]


async def test_baseline_disabled_or_probe_failure_yields_empty(tmp_path):
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    argv = ["llama-server", "-m", str(artifact)]

    disabled = _manager(tmp_path / "disabled", occupancy_profiles=False)
    assert await disabled._capture_occupancy_baseline("model.gguf", cfg, artifact, argv) == {}

    broken = _manager(tmp_path / "broken")
    broken.gpus = MagicMock(side_effect=RuntimeError("nvidia-smi not found"))
    assert await broken._capture_occupancy_baseline("model.gguf", cfg, artifact, argv) == {}


async def test_no_gpu_still_profiles_ram(tmp_path):
    mgr = _manager(tmp_path)
    mgr.gpus = lambda: []
    mgr.system_ram = lambda: {"used_mb": 8000}
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)

    base = await mgr._capture_occupancy_baseline(
        "model.gguf", cfg, artifact, ["llama-server", "-m", str(artifact)])
    assert base["gpus_used_mb"] == {}
    assert "no GPU" in base["hw"]

    mgr.gguf_paths["model.gguf"] = artifact
    mgr.system_ram = lambda: {"used_mb": 11000}
    lm = _loaded("model.gguf", cfg, artifact, occupancy_key=base["key"],
                 occupancy_baseline=base, ready=True, occupancy_wanted=True,
                 starting_at=1.0, ready_at=2.0)
    await mgr._capture_occupancy([("model.gguf", lm)], [])
    assert lm.occupancy_best["vram_mb"] == 0
    assert lm.occupancy_best["gpus_mb"] == {}
    assert lm.occupancy_best["ram_mb"] == 3000  # system delta; process RSS unavailable


# ---------------------------------------------------------------------------
# Delta math and RAM choice
# ---------------------------------------------------------------------------

async def test_per_gpu_delta_math(tmp_path):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    mgr.gguf_paths["model.gguf"] = artifact
    mgr.system_ram = lambda: {"used_mb": 1000}
    base = _baseline(gpus_used={"GPU-aaaa": 1000, "GPU-bbbb": 500}, ram_used=1000)
    lm = _loaded("model.gguf", cfg, artifact, occupancy_key="hw:cfg",
                 occupancy_baseline=base, ready=True, occupancy_wanted=True,
                 starting_at=1.0, ready_at=2.0)

    during = [_gpu(0, used=13000), _gpu(1, used=6500, uuid="GPU-bbbb")]
    await mgr._capture_occupancy([("model.gguf", lm)], during)

    best = lm.occupancy_best
    assert best["gpus_mb"] == {"GPU-aaaa": 12000, "GPU-bbbb": 6000}
    assert best["vram_mb"] == 18000
    # Per-GPU identity (not a single sum): the split is what admission will check.
    assert best["source"] == "load"


async def test_ram_is_max_of_system_delta_and_rss(tmp_path, monkeypatch):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    mgr.gguf_paths["model.gguf"] = artifact
    # System delta is small in this scenario; the model's resident set is the real cost.
    mgr.system_ram = lambda: {"used_mb": 1500}
    base = _baseline(gpus_used={}, ram_used=1000)

    class _Mem:
        rss = 3072 * 1024 * 1024      # 3072 MiB resident
        private = 4096 * 1024 * 1024  # 4096 MiB commit (informational)

    monkeypatch.setattr(server.psutil, "Process",
                        lambda pid: MagicMock(memory_info=lambda: _Mem()))

    lm = _loaded("model.gguf", cfg, artifact, pid=1234, occupancy_key="hw:cfg",
                 occupancy_baseline=base, ready=True, occupancy_wanted=True,
                 starting_at=1.0, ready_at=2.0)
    await mgr._capture_occupancy([("model.gguf", lm)], [])

    best = lm.occupancy_best
    assert best["ram_delta_mb"] == 500
    assert best["ram_rss_mb"] == 3072
    assert best["ram_private_mb"] == 4096
    assert best["ram_mb"] == 3072  # max(delta, RSS), not commit


async def test_gpu_backed_commit_does_not_inflate_ram(tmp_path, monkeypatch):
    """A GPU-resident engine can commit far more than it keeps resident in RAM.

    WDDM evictable-budget engines (ninfer) reserve/commit tens of GB for GPU
    allocations that never become resident pages. The RAM estimate must reflect the
    resident set, not the commit charge, or a fully-VRAM model looks like it needs
    40+ GB of RAM.
    """
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    mgr.gguf_paths["model.gguf"] = artifact
    mgr.system_ram = lambda: {"used_mb": 1000}  # no measurable system-wide growth
    base = _baseline(gpus_used={}, ram_used=1000)

    class _Mem:
        rss = 8228 * 1024 * 1024       # measured resident set
        private = 43178 * 1024 * 1024  # GPU-backed commit charge

    monkeypatch.setattr(server.psutil, "Process",
                        lambda pid: MagicMock(memory_info=lambda: _Mem()))

    lm = _loaded("model.gguf", cfg, artifact, pid=1234, occupancy_key="hw:cfg",
                 occupancy_baseline=base, ready=True, occupancy_wanted=True,
                 starting_at=1.0, ready_at=2.0)
    await mgr._capture_occupancy([("model.gguf", lm)], [])

    best = lm.occupancy_best
    assert best["ram_delta_mb"] == 0
    assert best["ram_rss_mb"] == 8228
    assert best["ram_private_mb"] == 43178  # diagnostics only
    assert best["ram_mb"] == 8228


async def test_missing_device_and_negative_deltas_clamp_to_zero(tmp_path):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    mgr.gguf_paths["model.gguf"] = artifact
    mgr.system_ram = lambda: {"used_mb": 500}  # below baseline -> negative delta
    base = _baseline(gpus_used={"GPU-aaaa": 1000, "GPU-gone": 4000}, ram_used=1000)
    lm = _loaded("model.gguf", cfg, artifact, occupancy_key="hw:cfg",
                 occupancy_baseline=base, ready=True, occupancy_wanted=True,
                 starting_at=1.0, ready_at=2.0)

    # GPU-gone is missing; GPU-aaaa shrank (driver reclaim) instead of growing.
    await mgr._capture_occupancy([("model.gguf", lm)], [_gpu(0, used=500)])

    assert lm.occupancy_best["gpus_mb"] == {"GPU-aaaa": 0, "GPU-gone": 0}
    assert lm.occupancy_best["ram_delta_mb"] == 0
    assert lm.occupancy_best["vram_mb"] == 0


async def test_co_resident_models_are_not_sampled(tmp_path):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    mgr.gguf_paths["model.gguf"] = artifact
    mgr.system_ram = lambda: {"used_mb": 9000}
    base = _baseline()
    a = _loaded("model.gguf", cfg, artifact, occupancy_key="hw:cfg",
                occupancy_baseline=base, ready=True, occupancy_wanted=True)
    b = _loaded("other.gguf", cfg, artifact, occupancy_key="hw:cfg2",
                occupancy_baseline=base, ready=True, occupancy_wanted=True)

    await mgr._capture_occupancy([("model.gguf", a), ("other.gguf", b)], [_gpu(0)])

    # No profile, and both still want a sample for when they are solo again.
    assert a.occupancy_best == {}
    assert b.occupancy_best == {}
    assert a.occupancy_wanted is True
    assert b.occupancy_wanted is True


# ---------------------------------------------------------------------------
# Sidecar storage
# ---------------------------------------------------------------------------

def test_sidecar_merge_preserves_user_fields_and_other_profiles(tmp_path):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    mgr.gguf_paths["model.gguf"] = artifact
    sidecar = artifact.with_suffix(artifact.suffix + ".json")
    sidecar.write_text(json.dumps({
        "args": "--mlock --no-mmap",       # user field that must survive
        "ctx_size": 32768,
        "occupancy_profiles": {"hw-old:cfg-old": {"vram_mb": 1}},
    }), encoding="utf-8")

    entry = {"vram_mb": 14000, "ram_mb": 21000, "source": "stream"}
    assert mgr._write_occupancy_profile("model.gguf", "hw-new:cfg-new", entry, cfg) is True

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["args"] == "--mlock --no-mmap"
    assert data["ctx_size"] == 32768
    assert data["occupancy_profiles"]["hw-old:cfg-old"] == {"vram_mb": 1}
    assert data["occupancy_profiles"]["hw-new:cfg-new"] == entry


def test_load_normalizes_legacy_commit_based_profile(tmp_path):
    """Old sidecars stored RAM as max(system Δ, commit); load reads the Δ instead."""
    artifact = _gguf(tmp_path)
    sidecar = artifact.with_suffix(artifact.suffix + ".json")
    sidecar.write_text(json.dumps({
        "name": "legacy",
        "occupancy_profiles": {
            "hw:legacy": {
                "gpus_mb": {"GPU-aaaa": 24000},
                "ram_mb": 45954,          # polluted by GPU-backed commit
                "ram_delta_mb": 8198,
                "ram_private_mb": 45954,
            },
            "hw:fresh": {
                "gpus_mb": {"GPU-aaaa": 24000},
                "ram_mb": 9000,
                "ram_delta_mb": 4000,
                "ram_rss_mb": 9000,
                "ram_private_mb": 40000,
            },
        },
    }), encoding="utf-8")

    cfg = ModelConfig.load(artifact)

    legacy = cfg.occupancy_profiles["hw:legacy"]
    assert legacy["ram_mb"] == 8198
    assert legacy["ram_private_mb"] == 45954  # kept for diagnostics
    # New-schema entries are untouched.
    assert cfg.occupancy_profiles["hw:fresh"]["ram_mb"] == 9000


def test_sidecar_is_created_when_absent(tmp_path):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    mgr.gguf_paths["model.gguf"] = artifact

    assert mgr._write_occupancy_profile(
        "model.gguf", "hw:cfg", {"vram_mb": 5}, cfg) is True
    sidecar = artifact.with_suffix(artifact.suffix + ".json")
    assert json.loads(sidecar.read_text(encoding="utf-8"))["occupancy_profiles"] == {
        "hw:cfg": {"vram_mb": 5}}


def test_bridge_card_is_only_updated_when_it_already_exists(tmp_path):
    mgr = _manager(tmp_path)
    cards = tmp_path / "cards"
    cards.mkdir()
    mgr.cards_dir = cards
    cfg = ModelConfig.default_for(Path("my-card.json"))
    cfg.engine = "strata"
    cfg.engine_config = str(tmp_path / "strata.json")
    cfg.artifact = str(tmp_path / "serve.py")

    # Not an existing overlay -> nothing is written (profiler must not create cards).
    assert mgr._write_occupancy_profile("card", "hw:cfg", {"vram_mb": 1}, cfg) is False
    assert list(cards.glob("*.json")) == []

    # An existing overlay is updated in place, preserving its editable fields.
    overlay = cards / "strata.json"
    overlay.write_text(json.dumps({"name": "Card", "pinned": True}), encoding="utf-8")
    assert mgr._write_occupancy_profile("card", "hw:cfg", {"vram_mb": 2048}, cfg) is True
    data = json.loads(overlay.read_text(encoding="utf-8"))
    assert data["name"] == "Card"
    assert data["pinned"] is True
    assert data["occupancy_profiles"]["hw:cfg"] == {"vram_mb": 2048}


async def test_writes_once_and_never_resamples(tmp_path):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    mgr.gguf_paths["model.gguf"] = artifact
    mgr.system_ram = lambda: {"used_mb": 4000}
    base = _baseline(gpus_used={"GPU-aaaa": 1000}, ram_used=1000)
    lm = _loaded("model.gguf", cfg, artifact, occupancy_key="hw:cfg",
                 occupancy_baseline=base, ready=True, occupancy_wanted=True,
                 starting_at=1.0, ready_at=2.0)

    writes = []
    real = mgr._write_occupancy_profile

    def counting(mid, key, entry, cfg_):
        writes.append(copy.deepcopy(entry))
        return real(mid, key, entry, cfg_)

    mgr._write_occupancy_profile = counting

    # The one shot: one sample, one write.
    await mgr._capture_occupancy([("model.gguf", lm)], [_gpu(0, used=5000)])
    assert len(writes) == 1
    assert lm.occupancy_saved is True
    assert cfg.occupancy_profiles["hw:cfg"]["vram_mb"] == 4000
    frozen = copy.deepcopy(lm.occupancy_best)

    # Re-arming a loaded engine must NOT resample: the number is frozen until the
    # hardware/launch key changes.
    lm.occupancy_wanted = True
    await mgr._capture_occupancy([("model.gguf", lm)], [_gpu(0, used=9000)])
    assert len(writes) == 1
    assert lm.occupancy_best == frozen
    assert cfg.occupancy_profiles["hw:cfg"] == frozen
    assert lm.occupancy_wanted is False


# ---------------------------------------------------------------------------
# Arming: one measurement per key, until the card changes
# ---------------------------------------------------------------------------

def _register(mgr, cfg, lm):
    mgr.models[lm.model_id] = cfg
    mgr.loaded[lm.model_id] = lm


def test_request_start_arms_sampling_once_per_key(tmp_path):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    lm = _loaded("model.gguf", cfg, artifact, occupancy_key="hw:cfg", ready=True)
    _register(mgr, cfg, lm)

    assert mgr.note_request_start(lm.port, 0) is lm
    assert lm.occupancy_wanted is True

    # Once the profile is stored, later requests must not re-arm the profiler.
    lm.occupancy_best = {"gpus_mb": {"GPU-aaaa": 1}, "ram_mb": 1, "ram_rss_mb": 1}
    lm.occupancy_wanted = False
    cfg.occupancy_profiles["hw:cfg"] = dict(lm.occupancy_best)

    mgr.note_request_end(lm)
    assert lm.occupancy_wanted is False  # request end no longer arms

    mgr.note_request_start(lm.port, 0)
    assert lm.occupancy_wanted is False  # existing profile -> no resample


def test_request_start_does_not_arm_when_profile_exists(tmp_path):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    cfg.occupancy_profiles["hw:cfg"] = {
        "gpus_mb": {"GPU-aaaa": 24000}, "ram_mb": 8000, "ram_rss_mb": 8000}
    lm = _loaded("model.gguf", cfg, artifact, occupancy_key="hw:cfg", ready=True)
    _register(mgr, cfg, lm)

    mgr.note_request_start(lm.port, 0)

    assert lm.occupancy_wanted is False


def test_legacy_profile_is_measured_once_to_upgrade(tmp_path):
    """Pre-RSS entries are upgraded by one fresh measurement (then frozen again)."""
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    cfg.occupancy_profiles["hw:cfg"] = {
        "gpus_mb": {"GPU-aaaa": 24000},
        "ram_mb": 45954,
        "ram_delta_mb": 8198,
        "ram_private_mb": 45954,   # no ram_rss_mb -> legacy schema
    }
    lm = _loaded("model.gguf", cfg, artifact, occupancy_key="hw:cfg", ready=True)
    _register(mgr, cfg, lm)

    mgr.note_request_start(lm.port, 0)

    assert lm.occupancy_wanted is True


def test_changed_launch_key_arms_a_fresh_measurement(tmp_path):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    # A profile for the OLD launch config must not suppress the new key's measurement.
    cfg.occupancy_profiles["hw:old"] = {"gpus_mb": {"GPU-aaaa": 1}, "ram_mb": 1}
    lm = _loaded("model.gguf", cfg, artifact, occupancy_key="hw:new", ready=True)
    _register(mgr, cfg, lm)

    mgr.note_request_start(lm.port, 0)

    assert lm.occupancy_wanted is True


# ---------------------------------------------------------------------------
# Status surfacing
# ---------------------------------------------------------------------------

async def test_status_reports_hardware_and_measured_occupancy(tmp_path, monkeypatch):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    mgr.gguf_paths["model.gguf"] = artifact
    key = "hw:cfg"
    cfg.occupancy_profiles[key] = {"vram_mb": 12000, "ram_mb": 20000, "gpus_mb": {}}
    lm = _loaded("model.gguf", cfg, artifact, occupancy_key=key, ready=True,
                 occupancy_best={"vram_mb": 12000, "ram_mb": 20000})
    mgr.loaded["model.gguf"] = lm

    monkeypatch.setattr(mgr, "gpus", lambda: [_gpu(0)])
    monkeypatch.setattr(mgr, "per_model_ram_vram", lambda loaded_items=None: {
        "model.gguf": {"ram_mb": 123, "vram_mb": 0}})
    monkeypatch.setattr(mgr, "_capture_occupancy",
                        lambda loaded_snap, gpus_list: _noop())

    status = await mgr._build_status()

    assert status["hardware"] == {
        "fingerprint": _occupancy_hash(_gpu_identity([_gpu(0)])),
        "label": "NVIDIA GeForce RTX 4090 / driver 555.99",
    }
    assert status["per_model"]["model.gguf"]["occupancy"]["vram_mb"] == 12000
    assert status["per_model"]["model.gguf"]["occupancy"]["ram_mb"] == 20000
    assert status["occupancy_profiles"] is True


async def test_status_without_profiles_still_works(tmp_path, monkeypatch):
    mgr = _manager(tmp_path)
    artifact = _gguf(tmp_path)
    cfg = ModelConfig.default_for(artifact)
    lm = _loaded("model.gguf", cfg, artifact)
    mgr.loaded["model.gguf"] = lm

    monkeypatch.setattr(mgr, "gpus", lambda: [])
    monkeypatch.setattr(mgr, "per_model_ram_vram", lambda loaded_items=None: {})
    monkeypatch.setattr(mgr, "_capture_occupancy",
                        lambda loaded_snap, gpus_list: _noop())

    status = await mgr._build_status()
    assert status["per_model"]["model.gguf"] == {"ram_mb": 0, "vram_mb": 0}


async def _noop():
    return None
