"""
Unit tests for the config-driven engine bridge adapter (bridge_adapter.py) and the
bridge-aware paths in server.py. No GPU, no subprocess.

Run with:  pytest tests/test_bridge_engines.py -v

These tests prove the three hard constraints:
  1. The llama.cpp path is byte-identical (``_launch_llama`` reproduces
     ``cfg.to_launch_args(...)`` exactly and returns ``(argv, None, None)``).
  2. The user's engine JSON is never mutated (``derive_config`` reads it
     read-only and writes only to ``<autoloader>/temp/strata/<model_id>.json``).
  3. A bridge launch is data-driven: ``_launch_bridge`` renders the engine's own argv
     template, so a second engine needs config only — no new code path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

import server
from server import ModelManager, ModelConfig, _DEFAULT_ARGV
from bridge_adapter import derive_config, _launch_llama, _launch_bridge


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


def _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir) -> ModelManager:
    """Build a ModelManager with an ``engines.strata`` entry pointing at temp dirs.

    The entry mirrors what the legacy ``strata:`` section migrates to, but is written
    in the new shape so the tests exercise the registry itself rather than the
    migration path. The interpreter and serve script are created as empty files: boot
    validation checks they EXIST (and only ever warns), and the add endpoint rejects an
    engine flagged ``_errors``, so the happy-path fixtures must be valid. They are never
    executed — no subprocess in these tests.
    """
    (engine_dir / ".venv").mkdir(parents=True, exist_ok=True)
    (engine_dir / ".venv" / "python.exe").write_bytes(b"")
    (engine_dir / "serve").mkdir(parents=True, exist_ok=True)
    (engine_dir / "serve" / "server.py").write_text("", encoding="utf-8")
    cfg = _make_cfg({
        "models": {"root_dir": str(models_dir), "save_state_dir": str(models_dir / "states")},
        "engines": {
            "strata": {
                "enabled": True,
                "label": "Strata",
                "interpreter": str(engine_dir / ".venv" / "python.exe"),
                "serve_script": str(engine_dir / "serve" / "server.py"),
                "cwd": str(engine_dir),
                "argv": list(_DEFAULT_ARGV),   # a real engine config must supply its own argv
                "derive_config": True,
                "config_overlay": {"cors_origins": ["*"], "lazy_load": False, "idle_unload_s": 0},
                "config_overlay_dynamic": {"aliases": ["{model_id}"]},
                "env_scrub": ["STRATA_API_KEY", "STRATA_ALLOWED_HOSTS"],
                "min_free_vram_flag": "--min-free-vram-mib",
                "config_key": "exe",
                "configs_dir": str(engine_dir),
                "cards_dir": str(cards_dir),
                "ready_timeout_seconds": 600,
            },
        },
    })
    return ModelManager(cfg)


def _write_engine_config(engine_dir: Path, name: str, payload: dict) -> Path:
    p = engine_dir / f"{name}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _launch_bridge: golden argv
# ---------------------------------------------------------------------------

def test_launch_bridge_golden_argv(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "strata-coder-iq1_m",
                         {"exe": "strata.exe", "model_name": "Coder IQ1_M"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.temp_dir = tmp_path / "temp"          # isolate derived-config writes
    mgr.scan()

    mid = "strata-coder-iq1_m"
    assert mid in mgr.models
    card = mgr.models[mid]
    card.min_free_vram_mib = 40960

    argv, cwd, env = _launch_bridge(mgr, card, "strata", mid, 9007)
    expected = [
        str(engine_dir / ".venv" / "python.exe"),
        str(engine_dir / "serve" / "server.py"),
        "--engine", "strata",
        "--config", str(mgr.temp_dir / "strata" / f"{mid}.json"),
        "--port", "9007",
        "--host", "127.0.0.1",
        "--min-free-vram-mib", "40960",
    ]
    assert argv == expected
    assert cwd == Path(str(engine_dir))
    assert env is not None
    assert (mgr.temp_dir / "strata" / f"{mid}.json").exists()


def test_launch_bridge_no_vram_flag_and_extra_args(tmp_path):
    """min_free_vram_mib == 0 → no --min-free-vram-mib; extra_args appended verbatim."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "card", {"exe": "s.exe"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.temp_dir = tmp_path / "temp"
    mgr.scan()
    card = mgr.models["card"]
    card.extra_args = ["--temperature", "0.2"]
    argv, cwd, env = _launch_bridge(mgr, card, "strata", "card", 9010)
    assert "--min-free-vram-mib" not in argv
    assert argv[-2:] == ["--temperature", "0.2"]


# ---------------------------------------------------------------------------
# derive_config: no mutation + overlay
# ---------------------------------------------------------------------------

def test_derive_config_no_mutation_and_overlay(tmp_path):
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    src = _write_engine_config(
        engine_dir, "my-model",
        {"exe": "strata.exe", "gpu": [0, 1], "args": ["--foo"],
         "port": 8080, "model_name": "M"})
    out = engine_dir / "out" / "derived.json"
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
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    src = _write_engine_config(engine_dir, "m", {"exe": "s.exe"})
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
# _launch_bridge: env scrub
# ---------------------------------------------------------------------------

def test_launch_bridge_env_scrub(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "card", {"exe": "s.exe"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.temp_dir = tmp_path / "temp"
    mgr.scan()
    monkeypatch.setenv("STRATA_API_KEY", "secret")
    monkeypatch.setenv("STRATA_ALLOWED_HOSTS", "evil.example")
    monkeypatch.setenv("PATH", "C:\\keep-me")
    argv, cwd, env = _launch_bridge(mgr, mgr.models["card"], "strata", "card", 9007)
    assert "STRATA_API_KEY" not in env
    assert "STRATA_ALLOWED_HOSTS" not in env
    assert env.get("PATH") == "C:\\keep-me"


def test_launch_bridge_no_env_scrub_inherits(tmp_path):
    """env_scrub absent ⇒ env is None (child inherits the full parent env)."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "card", {"exe": "s.exe"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.temp_dir = tmp_path / "temp"
    # Drop the scrub list so this engine inherits the environment untouched.
    mgr.engines["strata"].pop("env_scrub", None)
    mgr.scan()
    argv, cwd, env = _launch_bridge(mgr, mgr.models["card"], "strata", "card", 9008)
    assert env is None


def test_legacy_strata_card_dir_still_registers(tmp_path, monkeypatch):
    """A pre-registry card in ./strata_cards (old shape: strata_config, no engine) still
    shows up read-only after the refactor — and a scan never writes back to it."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "legacy-model", {"exe": "s.exe"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    # LEGACY_CARDS_DIRS is the relative "./strata_cards" — resolve it against tmp_path.
    legacy_dir = tmp_path / "strata_cards"; legacy_dir.mkdir()
    legacy_card = legacy_dir / "legacy-model.json"
    legacy_card.write_text(json.dumps({"strata_config": str(engine_dir / "legacy-model.json")}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)   # so the relative "./strata_cards" resolves to tmp_path
    mgr.scan()
    mid = "legacy-model"
    assert mid in mgr.models
    assert mgr.models[mid].engine == "strata"
    # The legacy card file is read-only: a rescan must not rewrite it into the new shape.
    before = legacy_card.read_text(encoding="utf-8")
    mgr.scan()
    assert legacy_card.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# _scan_bridge_cards: idempotence / collision / skip rules
# ---------------------------------------------------------------------------

def test_scan_engine_cards_idempotent(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "strata-coder-iq1_m",
                         {"exe": "s.exe", "model_name": "C"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    mid = "strata-coder-iq1_m"
    assert mid in mgr.models
    assert mgr.models[mid].engine == "strata"
    assert mgr.models[mid].engine_config == str(engine_dir / f"{mid}.json")
    assert mgr.models[mid].name == "C"
    # Idempotent rescan: strata id stable, config unchanged, no duplication.
    mgr.scan()
    assert mid in mgr.models
    assert mgr.models[mid].engine_config == str(engine_dir / f"{mid}.json")
    assert mgr.models[mid].engine == "strata"


def test_scan_strata_card_skip_rules(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "good", {"exe": "s.exe"})
    # Without "exe" → not an engine config → skipped.
    _write_engine_config(engine_dir, "no-exe", {"model_name": "NoExe"})
    # *.shared-settings.json → skipped even if it has "exe".
    (engine_dir / "foo.shared-settings.json").write_text(json.dumps({"exe": "x.exe"}))
    # Bad JSON → warned, not fatal.
    (engine_dir / "broken.json").write_text("{ not valid json")
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    assert "good" in mgr.models
    assert "no-exe" not in mgr.models
    assert "foo" not in mgr.models
    assert "broken" not in mgr.models


def test_scan_strata_card_collision_gguf_wins(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    # gguf id "clash.gguf" == strata config stem "clash.gguf" (file clash.gguf.json).
    (models_dir / "clash.gguf").write_bytes(b"GGUF_HEADER")
    _write_engine_config(engine_dir, "clash.gguf", {"exe": "s.exe", "model_name": "Clash"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    assert "clash.gguf" in mgr.models
    assert mgr.models["clash.gguf"].engine == "llama", "a gguf card of the same id must win"


def test_scan_engine_inert_without_section(tmp_path):
    """No strata: section → _scan_bridge_cards is a no-op (feature inert)."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    (models_dir / "model-a.gguf").write_bytes(b"GGUF_HEADER")
    cfg = _make_cfg({"models": {"root_dir": str(models_dir),
                                "save_state_dir": str(models_dir / "states")}})
    mgr = ModelManager(cfg)
    mgr.scan()
    assert list(mgr.models.keys()) == ["model-a.gguf"]
    assert mgr.engines == {}


# ---------------------------------------------------------------------------
# update_config: strata-aware save (no mutation of the user's JSON)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_config_strata_writes_card(tmp_path):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    src = _write_engine_config(engine_dir, "card", {"exe": "s.exe", "model_name": "C"})
    src_hash = _sha256(src)
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
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
    assert data["engine_config"] == str(src)   # self-describing card
    # DELIBERATE REVERSAL: cards now record the engine too. Previously "engine" was
    # derived at scan time and never written, so a card could not say which engine it
    # belonged to once multiple engines were configured.
    assert data["engine"] == "strata"
    # user's strata JSON untouched
    assert _sha256(src) == src_hash
    # in-memory config updated
    assert mgr.models[mid].pinned is True


# ---------------------------------------------------------------------------
# save_engine_card / save
# ---------------------------------------------------------------------------

def test_save_engine_card_only_editable_fields(tmp_path):
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
    cfg.engine_config = "C:\\strata\\my-card.json"
    cfg.save_engine_card(card_path)
    data = json.loads(card_path.read_text())
    assert data == {
        "name": "My Card",
        "description": "",
        "pinned": True,
        "default": True,
        "tags": ["strata", "coder"],
        "min_free_vram_mib": 40960,
        "extra_args": ["--foo", "bar"],
        # Self-describing card: the engine name and config path are both recorded so a
        # rescan can re-register a user-added card without configs_dir.
        "engine": "strata",
        "engine_config": "C:\\strata\\my-card.json",
    }


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


# ---------------------------------------------------------------------------
# POST /v1/models/strata (WebUI add flow) — endpoint driven directly so the
# status codes are asserted without the TestClient portal loop.
# ---------------------------------------------------------------------------

def _use_manager(monkeypatch, mgr: ModelManager) -> None:
    """Point the module-level global the endpoint closes over at ``mgr``."""
    monkeypatch.setattr(server, "manager", mgr)


async def test_add_strata_card_registers_and_persists_across_scan(tmp_path, monkeypatch):
    """Add → card file written → rescan keeps it (user-added cards persist)."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    # Engine config OUTSIDE configs_dir: only the card file can re-register it.
    extra = tmp_path / "extra"; extra.mkdir()
    src = _write_engine_config(extra, "added", {"exe": "s.exe", "model_name": "Added Model"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    assert "added" not in mgr.models
    _use_manager(monkeypatch, mgr)

    res = await server.add_strata_card_ep(server.StrataCardCreate(strata_config=str(src)))
    assert res == {"id": "added"}
    card = json.loads((cards_dir / "added.json").read_text(encoding="utf-8"))
    assert card["engine_config"] == str(src)
    assert card["name"] == "Added Model"        # defaults to the engine's model_name

    mgr.scan()
    cfg = mgr.models["added"]
    assert cfg.engine == "strata"
    assert cfg.engine_config == str(src)
    assert cfg.name == "Added Model"

    payload = {p["id"]: p for p in await mgr.list_models()}["added"]
    assert payload["engine"] == "strata"
    assert payload["engine_config"] == str(src)
    assert payload["min_free_vram_mib"] == 0
    assert payload["extra_args"] == []
    assert payload["resolved_binary"] == ""     # no llama binary for strata


async def test_add_strata_card_name_override(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    extra = tmp_path / "extra"; extra.mkdir()
    src = _write_engine_config(extra, "custom", {"exe": "s.exe", "model_name": "From Config"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    _use_manager(monkeypatch, mgr)

    res = await server.add_strata_card_ep(
        server.StrataCardCreate(strata_config=str(src), name="User Chosen"))
    assert res == {"id": "custom"}
    assert mgr.models["custom"].name == "User Chosen"
    card = json.loads((cards_dir / "custom.json").read_text(encoding="utf-8"))
    assert card["name"] == "User Chosen"


def test_discovered_card_leaves_no_orphan_file(tmp_path):
    """No auto-seed: deleting a configs_dir entry removes the card AND writes no file."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    src = _write_engine_config(engine_dir, "disc", {"exe": "s.exe"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    assert "disc" in mgr.models
    assert not (cards_dir / "disc.json").exists(), "scan must not auto-seed card files"

    src.unlink()
    mgr.scan()
    assert "disc" not in mgr.models


def test_card_overlay_cannot_flip_engine(tmp_path):
    """Overlay is applied FIRST; engine/strata_config/auto_save_state are forced after."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    src = _write_engine_config(engine_dir, "flip", {"exe": "s.exe", "model_name": "F"})
    (cards_dir / "flip.json").write_text(json.dumps({
        "engine": "llama",
        "engine_config": "D:\\bogus\\flip.json",
        "auto_save_state": True,
        "pinned": True,
    }), encoding="utf-8")
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    cfg = mgr.models["flip"]
    assert cfg.engine == "strata"
    assert cfg.engine_config == str(src)
    assert cfg.auto_save_state is False
    assert cfg.pinned is True           # editable overlay fields still apply


async def test_add_strata_card_invalid_config_400(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    extra = tmp_path / "extra"; extra.mkdir()
    no_exe = _write_engine_config(extra, "bad", {"model_name": "NoExe"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    _use_manager(monkeypatch, mgr)

    with pytest.raises(HTTPException) as ei_no_exe:
        await server.add_strata_card_ep(server.StrataCardCreate(strata_config=str(no_exe)))
    assert ei_no_exe.value.status_code == 400
    assert not (cards_dir / "bad.json").exists()

    with pytest.raises(HTTPException) as ei_missing:
        await server.add_strata_card_ep(
            server.StrataCardCreate(strata_config=str(extra / "nope.json")))
    assert ei_missing.value.status_code == 400
    assert "bad" not in mgr.models and "nope" not in mgr.models


async def test_add_strata_card_id_collision_409(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    src = _write_engine_config(engine_dir, "dup", {"exe": "s.exe"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    assert "dup" in mgr.models           # discovered first
    _use_manager(monkeypatch, mgr)

    with pytest.raises(HTTPException) as ei:
        await server.add_strata_card_ep(server.StrataCardCreate(strata_config=str(src)))
    assert ei.value.status_code == 409
    assert mgr.models["dup"].engine_config == str(src)   # untouched


async def test_list_models_llama_payload_gains_only_defaulted_fields(tmp_path):
    """Byte-identical llama: the payload gains engine/strata_config/min_free_vram_mib/extra_args
    at their defaults; every other key is unchanged."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    (models_dir / "model-a.gguf").write_bytes(b"GGUF_HEADER")
    cfg = _make_cfg({"models": {"root_dir": str(models_dir),
                                "save_state_dir": str(models_dir / "states")}})
    mgr = ModelManager(cfg)
    mgr.scan()
    payload = (await mgr.list_models())[0]

    assert set(payload) == {
        "id", "name", "description", "tags", "default", "pinned", "auto_save_state",
        "backend", "engine", "engine_config", "min_free_vram_mib", "extra_args",
        "resolved_binary", "has_mmproj", "use_mmproj", "mmproj_file", "ctx_size",
        "max_ctx_size", "n_gpu_layers", "estimated_vram_mb", "args", "gguf_name",
        "size_mb", "loaded", "ready", "port", "pid", "last_used", "state_path",
    }
    assert payload["engine"] == "llama"
    assert payload["engine_config"] == ""
    assert payload["min_free_vram_mib"] == 0
    assert payload["extra_args"] == []
    # untouched llama behaviour
    assert payload["ctx_size"] == 8192
    assert payload["n_gpu_layers"] == 999
    assert payload["has_mmproj"] is False


# ---------------------------------------------------------------------------
# Engine registry: normalisation, boot validation, generic dispatch
# ---------------------------------------------------------------------------

def _manager_with_engines(overrides: dict, models_dir) -> ModelManager:
    """ModelManager whose only non-default config is ``engines:``."""
    cfg = _make_cfg({"models": {"root_dir": str(models_dir),
                                "save_state_dir": str(models_dir / "states")},
                     "engines": overrides})
    mgr = ModelManager(cfg)
    mgr.temp_dir = models_dir.parent / "temp"
    mgr.temp_dir.mkdir(parents=True, exist_ok=True)
    return mgr


def test_normalize_engines_legacy_strata_section():
    """A pre-registry ``strata:`` section becomes engines.strata with the default argv.

    This is the migration every existing user hits; if it silently changes shape their
    bridge cards stop launching.
    """
    legacy = {
        "enabled": True,
        "root_dir": "N:/work/WD/Strata",
        "venv_python": "N:/work/WD/Strata/.venv/Scripts/python.exe",
        "serve_script": "N:/work/WD/Strata/serve/server.py",
        "configs_dir": "N:/work/WD/Strata",
    }
    engines = server._normalize_engines({"strata": legacy})

    assert set(engines) == {"strata"}
    spec = engines["strata"]
    # old key names are remapped, not dropped
    assert spec["interpreter"] == legacy["venv_python"]
    assert spec["serve_script"] == legacy["serve_script"]
    assert spec["cwd"] == legacy["root_dir"]
    # the behaviour that used to be hardcoded in the launcher is now data
    assert spec["derive_config"] is True
    assert spec["config_overlay_dynamic"] == {"aliases": ["{model_id}"]}
    assert spec["env_scrub"] == ["STRATA_API_KEY", "STRATA_ALLOWED_HOSTS"]
    assert spec["min_free_vram_flag"] == "--min-free-vram-mib"
    assert spec["config_key"] == "exe"
    assert spec["ready_timeout_seconds"] == 600
    assert spec["argv"][0] == "{interpreter}" and spec["argv"][1] == "{serve_script}"
    # the rest of the legacy section is carried over verbatim (per plan §3), so unknown
    # keys a user had are preserved rather than silently dropped
    assert spec["configs_dir"] == legacy["configs_dir"]


def test_normalize_engines_prefers_explicit_engines_section():
    """An explicit ``engines:`` wins; the legacy section is not consulted."""
    out = server._normalize_engines({
        "engines": {"other": {"interpreter": "x", "serve_script": "y"}},
        "strata": {"venv_python": "legacy"},
    })
    assert set(out) == {"other"}
    assert server._normalize_engines({}) == {}


def test_boot_validation_collects_errors_not_raises(tmp_path):
    """A broken engine warns and is flagged, but llama still boots and keeps 120s."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    (models_dir / "model-a.gguf").write_bytes(b"GGUF_HEADER")
    mgr = _manager_with_engines({"strata": {
        "enabled": True, "label": "Strata",
        "interpreter": str(tmp_path / "nope" / "python.exe"),   # does not exist
        "serve_script": "",                                       # not set either
        "config_key": "exe",
    }}, models_dir)
    mgr.scan()

    spec = mgr.engines["strata"]
    assert spec["_errors"], "boot validation must record why the engine is unusable"
    assert "interpreter" in spec["_errors"]
    assert "serve_script" in spec["_errors"]
    # warn-only: llama is fully operational despite the broken engine
    assert "model-a.gguf" in mgr.models
    assert mgr.models["model-a.gguf"].is_llama
    assert mgr.ready_timeout_for("model-a.gguf") == 120.0


def test_unknown_template_token_is_boot_error(tmp_path):
    """A typo'd {token} is caught at boot, not at launch."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    interp = tmp_path / "python.exe"; interp.write_bytes(b"")
    script = tmp_path / "server.py"; script.write_text("", encoding="utf-8")
    mgr = _manager_with_engines({"strata": {
        "enabled": True, "interpreter": str(interp), "serve_script": str(script),
        "argv": ["{serve_script}", "{nope}"], "config_key": "exe",
    }}, models_dir)

    errors = mgr.engines["strata"]["_errors"]
    assert "{nope}" in errors


async def test_bridge_load_missing_interpreter_returns_400(tmp_path):
    """REGRESSION GUARD for the KeyError('venv_python') crash.

    Before the registry, a strata card whose engine config was misconfigured reached
    the launcher, which subscripted the missing key and raised KeyError — a 500 with a
    traceback. It must now be a clean 400 naming the problem.
    """
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    src = _write_engine_config(engine_dir, "card", {"exe": "s.exe", "model_name": "C"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()

    # Simulate the engine entry losing its interpreter after the card was registered
    # (e.g. the venv was deleted / config edited). Validation flags it...
    mgr.engines["strata"]["interpreter"] = str(engine_dir / "gone" / "python.exe")
    mgr._validate_engines()
    assert mgr.engines["strata"]["_errors"]

    # ...and loading must be a 400, not a KeyError.
    with pytest.raises(HTTPException) as ei:
        await mgr._do_load_model("card")
    assert ei.value.status_code == 400
    assert "misconfigured" in str(ei.value.detail)
    assert "strata" not in mgr.loaded


async def test_bridge_load_unknown_engine_name_returns_400(tmp_path):
    """A card naming an engine that is not in the registry is a 400, not a traceback."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "card", {"exe": "s.exe", "model_name": "C"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()

    mgr.models["card"].engine = "not-an-engine"
    with pytest.raises(HTTPException) as ei:
        await mgr._do_load_model("card")
    assert ei.value.status_code == 400
    assert "not-an-engine" in str(ei.value.detail)
    assert "not-an-engine" not in mgr.loaded


def test_unknown_engine_in_card_file_is_skipped_with_warning(tmp_path, caplog):
    """A card naming an unregistered engine is ignored, never registered as a llama."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    (engine_dir / ".venv").mkdir(parents=True, exist_ok=True)
    (engine_dir / ".venv" / "python.exe").write_bytes(b"")
    (engine_dir / "serve").mkdir(parents=True, exist_ok=True)
    (engine_dir / "serve" / "server.py").write_text("", encoding="utf-8")
    _write_engine_config(engine_dir, "real", {"exe": "s.exe", "model_name": "R"})
    (cards_dir / "ghost.json").write_text(json.dumps({
        "engine": "ghost", "engine_config": str(engine_dir / "real.json")}),
        encoding="utf-8")
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)

    with caplog.at_level("WARNING"):
        mgr.scan()

    assert "ghost" not in mgr.models
    assert "real" in mgr.models


def test_launch_bridge_second_engine_config_only(tmp_path):
    """D1: a second engine needs config only — no engine-specific code exists.

    ``acme`` uses a totally different argv shape, no derived config and its own env
    scrub, and is driven by the same _launch_bridge.
    """
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "acme"; engine_dir.mkdir()
    src = _write_engine_config(engine_dir, "acme-model", {"bin": "acme.exe", "name": "A"})
    mgr = _manager_with_engines({
        "acme": {
            "enabled": True, "label": "Acme", "config_key": "bin",
            "argv": ["acme.exe", "run", "--id", "{model_id}", "--cfg", "{derived_config}"],
            "derive_config": False,
            "env_scrub": ["FOO_KEY"],
            "cwd": str(engine_dir),
            "configs_dir": str(engine_dir),   # discovery: any *.json holding "bin"
        },
    }, models_dir)
    mgr.scan()
    card = mgr.models["acme-model"]
    assert card.engine == "acme"
    card.extra_args = ["--verbose"]

    argv, cwd, env = _launch_bridge(mgr, card, "acme", "acme-model", 9200)

    assert argv == ["acme.exe", "run", "--id", "acme-model",
                    "--cfg", str(src), "--verbose"]
    # derive_config: false ⇒ the user's file is passed straight through, nothing written
    assert cwd == engine_dir
    assert env is not None and "FOO_KEY" not in env


def test_ready_timeout_for_bridge_card(tmp_path):
    """D4: one helper, both copies agree — engine value for bridge, 120s for llama."""
    from tcp_forwarder import _JIT_LOAD_TIMEOUT
    models_dir = tmp_path / "models"; models_dir.mkdir()
    (models_dir / "model-a.gguf").write_bytes(b"GGUF_HEADER")
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "card", {"exe": "s.exe", "model_name": "C"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()

    assert mgr.ready_timeout_for("card") == 600.0          # the engine's value
    assert mgr.ready_timeout_for("model-a.gguf") == 120.0    # llama/unknown default
    # the forwarder's own non-bridge constant is untouched by this refactor
    assert _JIT_LOAD_TIMEOUT == 300.0
    # an id that does not exist must not explode (the JIT path probes before loading)
    assert mgr.ready_timeout_for("no-such-model") == 120.0


async def test_update_config_can_change_engine_config_path(tmp_path, monkeypatch):
    """E27: the config path is editable through the API (the UI now sends it).

    Before this refactor ModelConfigUpdate had no engine_config field at all, so the
    WebUI could not repoint a card — the input existed but was never wired.
    """
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "card", {"exe": "s.exe", "model_name": "C"})
    moved = _write_engine_config(engine_dir, "card-moved", {"exe": "s2.exe", "model_name": "C2"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    _use_manager(monkeypatch, mgr)

    # valid new path applies, in memory and on disk
    await server.update_model_config("card", server.ModelConfigUpdate(engine_config=str(moved)))
    assert mgr.models["card"].engine_config == str(moved)
    saved = json.loads((cards_dir / "card.json").read_text(encoding="utf-8"))
    assert saved["engine_config"] == str(moved)

    # invalid new path is rejected with 400 and changes nothing
    with pytest.raises(HTTPException) as ei:
        await server.update_model_config("card", server.ModelConfigUpdate(
            engine_config=str(engine_dir / "does-not-exist.json")))
    assert ei.value.status_code == 400
    assert mgr.models["card"].engine_config == str(moved)

    # a card may not re-engine itself: no "engine" field exists to set
    assert "engine" not in server.ModelConfigUpdate.model_fields


async def test_legacy_card_file_with_strata_config_key_loads(tmp_path):
    """§5: a pre-registry card (strata_config, no engine) still registers as strata."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    _write_engine_config(engine_dir, "strata-coder-iq1_m",
                         {"exe": "strata.exe", "model_name": "Coder IQ1_M"})
    # exactly the shape the old code wrote
    (cards_dir / "strata-coder-iq1_m.json").write_text(json.dumps({
        "name": "Coder IQ1_M",
        "strata_config": str(engine_dir / "strata-coder-iq1_m.json"),
        "pinned": True,
    }), encoding="utf-8")
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()

    mid = "strata-coder-iq1_m"
    assert mid in mgr.models
    card = mgr.models[mid]
    assert card.engine == "strata"
    assert card.engine_config == str(engine_dir / "strata-coder-iq1_m.json")
    assert card.pinned is True
    assert card.is_bridge

    # re-saving rewrites it in the new shape (legacy key gone, engine recorded)
    card.pinned = False
    await mgr.update_config(mid, ModelConfig(**{k: getattr(card, k) for k in (
        "name", "description", "tags", "default", "pinned", "auto_save_state",
        "min_free_vram_mib", "extra_args", "engine", "engine_config")}))
    data = json.loads((cards_dir / f"{mid}.json").read_text(encoding="utf-8"))
    assert data["engine"] == "strata"
    assert data["engine_config"] == str(engine_dir / "strata-coder-iq1_m.json")
    assert "strata_config" not in data


async def test_add_engine_card_generic_endpoint_and_alias(tmp_path, monkeypatch):
    """D6: POST /v1/models/engine is generic; the strata path is a working alias."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir()
    src = _write_engine_config(elsewhere, "newcard", {"exe": "s.exe", "model_name": "N"})
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    _use_manager(monkeypatch, mgr)

    res = await server.add_engine_card_ep(
        server.EngineCardCreate(engine="strata", engine_config=str(src)))
    assert res == {"id": "newcard"}
    assert (cards_dir / "newcard.json").exists()

    # deprecated alias reaches the same generic implementation
    other = _write_engine_config(elsewhere, "aliascard", {"exe": "s.exe", "model_name": "A"})
    res2 = await server.add_strata_card_ep(server.StrataCardCreate(strata_config=str(other)))
    assert res2 == {"id": "aliascard"}

    # unknown engine → clean 400
    with pytest.raises(HTTPException) as ei_unknown:
        await server.add_engine_card_ep(
            server.EngineCardCreate(engine="nope", engine_config=str(src)))
    assert ei_unknown.value.status_code == 400
    assert "nope" in str(ei_unknown.value.detail)


async def test_list_engines_reports_errors(tmp_path, monkeypatch):
    """GET /v1/engines feeds the WebUI dropdown and explains unusable engines."""
    models_dir = tmp_path / "models"; models_dir.mkdir()
    engine_dir = tmp_path / "strata"; engine_dir.mkdir()
    cards_dir = tmp_path / "cards"; cards_dir.mkdir()
    mgr = _bridge_manager(tmp_path, models_dir, engine_dir, cards_dir)
    mgr.scan()
    _use_manager(monkeypatch, mgr)

    out = await server.list_engines()
    assert [e["name"] for e in out] == ["strata"]
    entry = out[0]
    assert entry["label"] == "Strata"
    assert entry["enabled"] is True
    assert entry["ready"] is True
    assert entry["error"] == ""
    assert entry["cards_dir"] == str(cards_dir)

    # a broken engine is reported, not hidden
    mgr.engines["strata"]["interpreter"] = str(engine_dir / "gone" / "python.exe")
    mgr._validate_engines()
    broken = (await server.list_engines())[0]
    assert broken["ready"] is False
    assert "interpreter" in broken["error"]

    # and a disabled one is flagged too
    mgr.engines["strata"].pop("_errors", None)
    mgr.engines["strata"]["interpreter"] = str(engine_dir / ".venv" / "python.exe")
    mgr.engines["strata"]["serve_script"] = str(engine_dir / "serve" / "server.py")
    mgr.engines["strata"]["enabled"] = False
    off = (await server.list_engines())[0]
    assert off["ready"] is False
    assert "disabled" in off["error"]
