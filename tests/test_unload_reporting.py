"""Regression tests for BUG_0102: unload must not claim success when the kill failed.

Every caller of unload treats a successful response as "the engine is gone and its
memory is free" (AC safe-mode, media_gen's VRAM seam, LRU eviction). Swallowing a
termination failure broke that contract: the endpoint returned 200 while the process
was still running. These tests pin the failure reporting and the residency truth.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import psutil
import pytest

import server
from server import LoadedModel, ModelConfig


class _ImmortalProc:
    """psutil.Process stand-in that refuses every termination attempt."""

    def __init__(self, pid: int):
        self.pid = pid

    def children(self, recursive=True):  # noqa: ARG002 - psutil API
        return []

    def terminate(self):
        raise psutil.AccessDenied(self.pid)

    def kill(self):
        raise psutil.AccessDenied(self.pid)

    def is_running(self) -> bool:
        return True

    def status(self) -> str:
        return psutil.STATUS_RUNNING


def _add_resident(mgr, model_id="immortal.gguf", pid=424242):
    cfg = ModelConfig.default_for(Path(model_id))
    lm = LoadedModel(
        model_id=model_id,
        gguf_path=Path(model_id),
        config=cfg,
        port=19555,
        process=MagicMock(),
        pid=pid,
    )
    lm.ready = True
    mgr.models[model_id] = cfg
    mgr.loaded[model_id] = lm
    return lm


def _make_kill_fail(monkeypatch):
    """Force both the psutil path and the process-group fallback to fail."""
    proc = _ImmortalProc(424242)
    monkeypatch.setattr(server.psutil, "Process", lambda pid: proc)

    def _refuse(*args, **kwargs):  # noqa: ARG001 - monkeypatch target
        raise psutil.AccessDenied(424242)

    monkeypatch.setattr(server.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(server.os, "killpg", _refuse, raising=False)


@pytest.mark.asyncio
async def test_unload_all_reports_failure_and_keeps_model_tracked(isolated_manager, monkeypatch):
    mgr = isolated_manager
    _add_resident(mgr)

    with monkeypatch.context() as m:
        m.setattr(server.platform, "system", lambda: "Linux")
        _make_kill_fail(m)
        result = await mgr.unload_all()

    assert result["unloaded_all"] is False
    assert result["unloaded"] == []
    assert len(result["failed"]) == 1
    failure = result["failed"][0]
    assert failure["model"] == "immortal.gguf"
    assert failure["pid"] == 424242
    assert "still running" in failure["reason"]

    # The registry must stay truthful: the engine is still resident, so /v1/models
    # keeps showing it and a later unload attempt can retry.
    assert "immortal.gguf" in mgr.loaded

    # Real psutil is back after the context: a retry must now succeed and clear it.
    retry = await mgr.unload_all()
    assert retry["unloaded_all"] is True
    assert "immortal.gguf" not in mgr.loaded


def test_unload_all_endpoint_returns_500_on_failed_kill(client, seeded_manager, monkeypatch):
    _add_resident(seeded_manager)

    with monkeypatch.context() as m:
        m.setattr(server.platform, "system", lambda: "Linux")
        _make_kill_fail(m)
        response = client.post("/v1/unload_all")

    assert response.status_code == 500, response.text
    body = response.json()
    assert body["unloaded_all"] is False
    assert body["failed"][0]["model"] == "immortal.gguf"

    # The model must still be listed as loaded while its process survives.
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["immortal.gguf"]["loaded"] is True

    # With the patches gone the retry succeeds, leaving a clean fixture teardown.
    retry = client.post("/v1/unload_all")
    assert retry.status_code == 200, retry.text
    assert retry.json()["unloaded_all"] is True
    assert "immortal.gguf" not in seeded_manager.loaded


def test_single_unload_endpoint_returns_500_on_failed_kill(client, seeded_manager, monkeypatch):
    _add_resident(seeded_manager)

    with monkeypatch.context() as m:
        m.setattr(server.platform, "system", lambda: "Linux")
        _make_kill_fail(m)
        response = client.post("/v1/models/immortal.gguf/unload")

    assert response.status_code == 500, response.text
    body = response.json()
    assert body["id"] == "immortal.gguf"
    assert body["unloaded"] is False
    assert "still running" in body["reason"]
    assert "immortal.gguf" in seeded_manager.loaded

    retry = client.post("/v1/models/immortal.gguf/unload")
    assert retry.status_code == 200, retry.text
    assert retry.json()["unloaded"] is True
    assert "immortal.gguf" not in seeded_manager.loaded
