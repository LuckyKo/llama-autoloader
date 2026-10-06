"""
Unit tests for the Strata "bridge card" adapter (strata_adapter.py) and the
strata-aware paths in server.py. No GPU, no subprocess.

Run with:  pytest tests/test_strata_adapter.py -v

These tests prove the two hard constraints:
  1. The llama.cpp path is byte-identical (``_launch_llama`` reproduces
     ``cfg.to_launch_args(...)`` exactly and returns ``(argv, None, None)``).
  2. The user's Strata JSON is never mutated (``derive_config`` reads it
     read-only and writes only to ``<autoloader>/temp/strata/<model_id>.json``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from server import ModelManager, ModelConfig
from strata_adapter import derive_config, _launch_llama, _launch_strata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _make_cfg(overrides: dict | None = None) -> dict:
    """Build a minimal config dict suitable for ModelManager construction."""
    cfg = {
        "models": {
            "root_dir": "",
            "save_state_dir": "",
            "idle_timeout_seconds": 300,
            "max_loaded_models": 2,
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


def _strata_manager(tmp_path, models_dir, strata_dir, cards_dir) -> ModelManager:
    """Build a ModelManager with a strata: section pointing at temp dirs.

    venv_python / serve_script may not exist on disk; they are only used to build
    the argv (never executed in these unit tests).
    """
    cfg = _make_cfg({
        "models": {"root_dir": str(models_dir), "save_state_dir": str(models_dir / "states")},
        "strata": {
            "enabled": True,
            "root_dir": str(strata_dir),
            "venv_python": str(strata_dir / ".venv" / "python.exe"),
            "serve_script": str(strata_dir / "serve" / "server.py"),
            "configs_dir": str(strata_dir),
            "cards_dir": str(cards_dir),
            "ready_timeout_seconds": 600,
        },
    })
    return ModelManager(cfg)


def _write_engine_config(strata_dir: Path, name: str, payload: dict) -> Path:
    p = strata_dir / f"{name}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _launch_strata: golden argv
# ---------------------------------------------------------------------------

def test_launch_strata_golden_argv(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    strata_dir = tmp_path / "strata"; strata_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(strata_dir, "strata-coder-iq1_m",
                         {"exe": "strata.exe", "model_name": "Coder IQ1_M"})
    mgr = _strata_manager(tmp_path, models_dir, strata_dir, cards_dir)
    mgr.temp_dir = tmp_path / "temp"          # isolate derived-config writes
    mgr.scan()

    mid = "strata-coder-iq1_m"
    assert mid in mgr.models
    card = mgr.models[mid]
    card.min_free_vram_mib = 40960

    argv, cwd, env = _launch_strata(mgr, card, mid, 9007)
    expected = [
        str(strata_dir / ".venv" / "python.exe"),
        str(strata_dir / "serve" / "server.py"),
        "--engine", "strata",
        "--config", str(mgr.temp_dir / "strata" / f"{mid}.json"),
        "--port", "9007",
        "--host", "127.0.0.1",
        "--min-free-vram-mib", "40960",
    ]
    assert argv == expected
    assert cwd == Path(str(strata_dir))
    assert env is not None
    assert (mgr.temp_dir / "strata" / f"{mid}.json").exists()


def test_launch_strata_no_vram_flag_and_extra_args(tmp_path):
    """min_free_vram_mib == 0 → no --min-free-vram-mib; extra_args appended verbatim."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    strata_dir = tmp_path / "strata"; strata_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(strata_dir, "card", {"exe": "s.exe"})
    mgr = _strata_manager(tmp_path, models_dir, strata_dir, cards_dir)
    mgr.temp_dir = tmp_path / "temp"
    mgr.scan()
    card = mgr.models["card"]
    card.extra_args = ["--temperature", "0.2"]
    argv, cwd, env = _launch_strata(mgr, card, "card", 9010)
    assert "--min-free-vram-mib" not in argv
    assert argv[-2:] == ["--temperature", "0.2"]


# ---------------------------------------------------------------------------
# derive_config: no mutation + overlay
# ---------------------------------------------------------------------------

def test_derive_config_no_mutation_and_overlay(tmp_path):
    strata_dir = tmp_path / "strata"; strata_dir.mkdir()
    src = _write_engine_config(
        strata_dir, "my-model",
        {"exe": "strata.exe", "gpu": [0, 1], "args": ["--foo"],
         "port": 8080, "model_name": "M"})
    out = strata_dir / "out" / "derived.json"
    before = _sha256(src)
    derive_config(src, out, {
        "aliases": ["my-model"],
        "cors_origins": ["*"],
        "lazy_load": False,
        "idle_unload_s": 0,
    })
    after = _sha256(src)
    assert before == after, "derive_config must never mutate the source"
    data = json.loads(out.read_text(encoding="utf-8"))
    # overlaid keys
    assert data["aliases"] == ["my-model"]
    assert data["cors_origins"] == ["*"]
    assert data["lazy_load"] is False
    assert data["idle_unload_s"] == 0
    # untouched keys
    assert data["exe"] == "strata.exe"
    assert data["gpu"] == [0, 1]
    assert data["args"] == ["--foo"]
    assert data["port"] == 8080
    assert data["model_name"] == "M"


def test_derive_config_creates_parent(tmp_path):
    """out_path.parent.mkdir(parents=True, exist_ok=True) is called (nested path)."""
    strata_dir = tmp_path / "strata"; strata_dir.mkdir()
    src = _write_engine_config(strata_dir, "m", {"exe": "s.exe"})
    out = tmp_path / "a" / "b" / "c" / "derived.json"
    derive_config(src, out, {"aliases": ["m"]})
    assert out.exists()


# ---------------------------------------------------------------------------
# _launch_llama: byte-identical argv + (argv, None, None)
# ---------------------------------------------------------------------------

def test_launch_llama_byte_identical(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    (models_dir / "model-a.gguf").write_bytes(b"GGUF_HEADER")
    cfg = _make_cfg({"models": {"root_dir": str(models_dir),
                                "save_state_dir": str(models_dir / "states")}})
    mgr = ModelManager(cfg)
    mgr.scan()
    mid = "model-a.gguf"
    card = mgr.models[mid]
    mmproj_p = mgr.mmproj_paths.get(mid)
    expected_argv = [mgr.resolve_binary(card)] + card.to_launch_args(
        mgr.gguf_paths[mid], 9005, mgr.default_args,
        mmproj_full_path=mmproj_p, slot_save_dir=mgr.save_state_dir, model_id=mid)
    argv, cwd, env = _launch_llama(mgr, card, mid, mgr.gguf_paths[mid], 9005)
    assert argv == expected_argv, "llama argv must be byte-identical to today's"
    assert cwd is None
    assert env is None


# ---------------------------------------------------------------------------
# _launch_strata: env scrub
# ---------------------------------------------------------------------------

def test_launch_strata_env_scrub(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    strata_dir = tmp_path / "strata"; strata_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(strata_dir, "card", {"exe": "s.exe"})
    mgr = _strata_manager(tmp_path, models_dir, strata_dir, cards_dir)
    mgr.temp_dir = tmp_path / "temp"
    mgr.scan()
    monkeypatch.setenv("STRATA_API_KEY", "secret")
    monkeypatch.setenv("STRATA_ALLOWED_HOSTS", "evil.example")
    monkeypatch.setenv("PATH", "C:\\keep-me")
    argv, cwd, env = _launch_strata(mgr, mgr.models["card"], "card", 9007)
    assert "STRATA_API_KEY" not in env
    assert "STRATA_ALLOWED_HOSTS" not in env
    assert env.get("PATH") == "C:\\keep-me"


# ---------------------------------------------------------------------------
# _scan_strata_cards: idempotence / collision / skip rules
# ---------------------------------------------------------------------------

def test_scan_strata_cards_idempotent(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    strata_dir = tmp_path / "strata"; strata_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(strata_dir, "strata-coder-iq1_m",
                         {"exe": "s.exe", "model_name": "C"})
    mgr = _strata_manager(tmp_path, models_dir, strata_dir, cards_dir)
    mgr.scan()
    mid = "strata-coder-iq1_m"
    assert mid in mgr.models
    assert mgr.models[mid].engine == "strata"
    assert mgr.models[mid].strata_config == str(strata_dir / f"{mid}.json")
    assert mgr.models[mid].name == "C"
    # Idempotent rescan: strata id stable, config unchanged, no duplication.
    mgr.scan()
    assert mid in mgr.models
    assert mgr.models[mid].strata_config == str(strata_dir / f"{mid}.json")
    assert mgr.models[mid].engine == "strata"


def test_scan_strata_card_skip_rules(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    strata_dir = tmp_path / "strata"; strata_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(strata_dir, "good", {"exe": "s.exe"})
    # Without "exe" → not an engine config → skipped.
    _write_engine_config(strata_dir, "no-exe", {"model_name": "NoExe"})
    # *.shared-settings.json → skipped even if it has "exe".
    (strata_dir / "foo.shared-settings.json").write_text(json.dumps({"exe": "x.exe"}))
    # Bad JSON → warned, not fatal.
    (strata_dir / "broken.json").write_text("{ not valid json")
    mgr = _strata_manager(tmp_path, models_dir, strata_dir, cards_dir)
    mgr.scan()
    assert "good" in mgr.models
    assert "no-exe" not in mgr.models
    assert "foo" not in mgr.models
    assert "broken" not in mgr.models


def test_scan_strata_card_collision_gguf_wins(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    strata_dir = tmp_path / "strata"; strata_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    # gguf id "clash.gguf" == strata config stem "clash.gguf" (file clash.gguf.json).
    (models_dir / "clash.gguf").write_bytes(b"GGUF_HEADER")
    _write_engine_config(strata_dir, "clash.gguf", {"exe": "s.exe", "model_name": "Clash"})
    mgr = _strata_manager(tmp_path, models_dir, strata_dir, cards_dir)
    mgr.scan()
    assert "clash.gguf" in mgr.models
    assert mgr.models["clash.gguf"].engine == "llama", "a gguf card of the same id must win"


def test_scan_strata_inert_without_section(tmp_path):
    """No strata: section → _scan_strata_cards is a no-op (feature inert)."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    (models_dir / "model-a.gguf").write_bytes(b"GGUF_HEADER")
    cfg = _make_cfg({"models": {"root_dir": str(models_dir),
                                "save_state_dir": str(models_dir / "states")}})
    mgr = ModelManager(cfg)
    mgr.scan()
    assert list(mgr.models.keys()) == ["model-a.gguf"]
    assert mgr.strata_cfg == {}


# ---------------------------------------------------------------------------
# update_config: strata-aware save (no mutation of the user's JSON)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_config_strata_writes_card(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    strata_dir = tmp_path / "strata"; strata_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    src = _write_engine_config(strata_dir, "card", {"exe": "s.exe", "model_name": "C"})
    src_hash = _sha256(src)
    mgr = _strata_manager(tmp_path, models_dir, strata_dir, cards_dir)
    mgr.scan()
    mid = "card"
    cfg = mgr.models[mid]
    cfg.pinned = True
    cfg.name = "C updated"
    await mgr.update_config(mid, cfg)
    # card overlay written to cards_dir
    card_path = cards_dir / f"{mid}.json"
    assert card_path.exists()
    data = json.loads(card_path.read_text())
    assert data["pinned"] is True
    assert data["name"] == "C updated"
    assert "engine" not in data
    # user's strata JSON untouched
    assert _sha256(src) == src_hash
    # in-memory config updated
    assert mgr.models[mid].pinned is True


# ---------------------------------------------------------------------------
# save_strata_card / save
# ---------------------------------------------------------------------------

def test_save_strata_card_only_editable_fields(tmp_path):
    card_path = tmp_path / "cards" / "my-card.json"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig.default_for(Path("my-card.json"))
    cfg.name = "My Card"
    cfg.pinned = True
    cfg.default = True
    cfg.tags = ["strata", "coder"]
    cfg.min_free_vram_mib = 40960
    cfg.extra_args = ["--foo", "bar"]
    cfg.engine = "strata"
    cfg.strata_config = "C:\\strata\\my-card.json"
    cfg.save_strata_card(card_path)
    data = json.loads(card_path.read_text())
    assert data == {
        "name": "My Card",
        "description": "",
        "pinned": True,
        "default": True,
        "tags": ["strata", "coder"],
        "min_free_vram_mib": 40960,
        "extra_args": ["--foo", "bar"],
    }
    assert "engine" not in data
    assert "strata_config" not in data


def test_save_gguf_sidecar_unchanged(tmp_path):
    gguf = tmp_path / "model-x.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = ModelConfig.default_for(gguf)
    cfg.name = "X"
    cfg.save(gguf)
    sidecar = gguf.with_suffix(gguf.suffix + ".json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["name"] == "X"
    assert "engine" in data      # asdict carries every field (gguf sidecar is full config)
    assert "ctx_size" in data
