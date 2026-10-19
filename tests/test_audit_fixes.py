"""Regression tests for the audit fixes.

Covers strict model resolution, pre-validation config copies, settings
persistence, mmproj provenance, duplicate GGUF detection and filename safety.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

import server
from server import ModelManager
from tests import gguf_bytes as gb


@pytest.mark.asyncio
async def test_strict_resolution_does_not_fall_back(seeded_manager):
    mgr = seeded_manager
    mgr.scan()

    assert await mgr.resolve_model_id_strict("does-not-exist.gguf") is None
    assert await mgr.resolve_model_id_strict("model-a") == "model-a.gguf"
    assert await mgr.resolve_model_id_strict("Alpha Model") == "model-a.gguf"


def test_update_model_config_duplicate_name_does_not_mutate_live(client, seeded_manager):
    """A rejected rename must leave the live registry untouched (no sidecar, no 500 damage)."""
    mgr = seeded_manager
    before_a = mgr.models["model-a.gguf"].name
    before_b = mgr.models["model-b.gguf"].name

    try:
        response = client.put("/v1/models/model-a.gguf/config", json={"name": before_b})
    except ValueError:
        # TestClient re-raises server exceptions by default; the behavior under
        # test is the absence of mutation, not the HTTP status.
        pass
    else:
        assert response.status_code == 500

    assert mgr.models["model-a.gguf"].name == before_a
    assert mgr.models["model-b.gguf"].name == before_b

    # And a valid edit still persists.
    response = client.put("/v1/models/model-a.gguf/config", json={"name": "Alpha Renamed"})
    assert response.status_code == 200, response.text
    assert mgr.models["model-a.gguf"].name == "Alpha Renamed"


def test_settings_host_port_persist_without_other_changes(client, seeded_manager):
    response = client.put("/v1/settings", json={"host": "0.0.0.0", "port": 9999})
    assert response.status_code == 200, response.text

    persisted = yaml.safe_load(Path(server.CONFIG_PATH).read_text(encoding="utf-8"))
    assert persisted["launcher"]["host"] == "0.0.0.0"
    assert persisted["launcher"]["port"] == 9999
    # Runtime binding is intentionally unchanged until restart.
    assert seeded_manager.port != 9999


def test_settings_reject_invalid_port_and_backend(client):
    assert client.put("/v1/settings", json={"port": 70000}).status_code == 400
    response = client.put("/v1/settings", json={"selected_backend": "not-installed"})
    assert response.status_code == 400
    assert "not an installed backend" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_config_preserves_auto_mmproj_provenance(isolated_manager):
    mgr = isolated_manager
    gb.write_gguf(mgr.root_dir / "solo.gguf", name="Solo Model")
    gb.write_gguf(mgr.root_dir / "mmproj-F16.gguf", name="Projector")
    mgr.scan()

    mid = "solo.gguf"
    assert mgr.mmproj_source.get(mid) == "auto"

    # An edit that does not touch mmproj_file must not turn "auto" into "explicit".
    await mgr.update_config(mid, copy.deepcopy(mgr.models[mid]))
    assert mgr.mmproj_source.get(mid) == "auto"

    # An explicit change is still recorded as explicit (absolute path differs
    # from the auto-stamped filename even though it points at the same file).
    cfg = copy.deepcopy(mgr.models[mid])
    cfg.mmproj_file = str(mgr.root_dir / "mmproj-F16.gguf")
    cfg.description = "changed"
    await mgr.update_config(mid, cfg)
    assert mgr.mmproj_source.get(mid) == "explicit"


def test_duplicate_gguf_filename_is_skipped_not_silently_overwritten(isolated_manager):
    mgr = isolated_manager
    first = mgr.root_dir / "one"
    second = mgr.root_dir / "two"
    first.mkdir()
    second.mkdir()
    gb.write_gguf(first / "dup.gguf", name="First")
    gb.write_gguf(second / "dup.gguf", name="Second")

    mgr.scan()

    assert list(mgr.models) == ["dup.gguf"]
    assert mgr.models["dup.gguf"].gguf_name == "First"  # sorted paths -> first wins deterministically


def test_sanitize_safe_string_produces_valid_windows_filenames():
    assert ModelManager.sanitize_safe_string("") == "_"
    assert ModelManager.sanitize_safe_string("..") == "_"
    assert ModelManager.sanitize_safe_string("CON") == "_CON"
    assert ModelManager.sanitize_safe_string("aux.txt") == "_aux.txt"
    assert ModelManager.sanitize_safe_string('bad<>:"|?*name') == "bad_______name"
    assert ModelManager.sanitize_safe_string("trailing. ") == "trailing"
