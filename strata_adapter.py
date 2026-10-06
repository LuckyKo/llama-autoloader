"""Strata "bridge card" adapter for llama-autoloader.

This module is the ONLY engine-specific seam for Strata bridge cards. It exposes two
launch functions that return ``(argv, cwd, env)`` tuples for ``subprocess.Popen``:

- ``_launch_llama(...)`` — today's llama-server launch, byte-identical (env/cwd are None).
- ``_launch_strata(...)`` — builds the Strata ``serve/server.py`` argv, writes a derived
  config (a read-only copy of the user's JSON with a few keys overlaid), and returns a
  scrubbed env.

With no ``strata:`` section in config.yaml, neither function is ever called and the
existing llama path is untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _deep_overlay(base: Dict[str, Any], overlays: Dict[str, Any]) -> Dict[str, Any]:
    """Return a NEW dict: ``base`` deep-merged with ``overlays`` on top (no mutation of ``base``)."""
    merged = dict(base)
    for key, value in overlays.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_overlay(merged[key], value)
        else:
            merged[key] = value
    return merged


def derive_config(user_cfg_path, out_path, overlays: Dict[str, Any]) -> Path:
    """Read the user's engine config (utf-8-sig, READ-ONLY), deep-overlay only the
    allow-listed keys, and write the result to ``out_path`` (indent=2).

    Creates ``out_path.parent`` if needed. NEVER writes to the source file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(user_cfg_path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Strata config {user_cfg_path} is not a JSON object")
    data = _deep_overlay(data, overlays)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path


def _launch_llama(mgr, cfg, model_id: str, path, port: int):
    """Return ``(argv, cwd, env)`` — byte-identical to today's llama-server launch.

    ``env`` and ``cwd`` are both ``None`` so ``subprocess.Popen`` inherits the parent
    environment and working directory, exactly as before this feature.
    """
    mmproj_p = mgr.mmproj_paths.get(model_id)
    argv = [mgr.resolve_binary(cfg)] + cfg.to_launch_args(
        path, port, mgr.default_args, mmproj_full_path=mmproj_p,
        slot_save_dir=mgr.save_state_dir, model_id=model_id)
    return argv, None, None


def _launch_strata(mgr, cfg, model_id: str, port: int):
    """Return ``(argv, cwd, env)`` for a Strata serve process on ``port``.

    The derived config is a read-only copy of the user's Strata JSON with the autoloader's
    ``model_id`` injected as an alias, CORS opened to ``*``, and lazy-load/idle-unload
    forced off (non-lazy spawn is mandatory; see plan §1). The user's JSON is never written.
    """
    sc = mgr.strata_cfg
    derived = mgr._strata_derived_path(model_id)
    derive_config(cfg.strata_config, derived, {
        "aliases": [model_id],
        "cors_origins": ["*"],
        "lazy_load": False,
        "idle_unload_s": 0,
    })
    argv = [sc["venv_python"], sc["serve_script"],
            "--engine", "strata", "--config", str(derived),
            "--port", str(port), "--host", "127.0.0.1"]
    if cfg.min_free_vram_mib:
        argv += ["--min-free-vram-mib", str(cfg.min_free_vram_mib)]
    argv += list(cfg.extra_args)
    # STRATA_API_KEY must be DELETED (not blanked): an empty key is a hard exit(2) in
    # serve/server.py. STRATA_ALLOWED_HOSTS is dropped so the card stays loopback-only.
    env = {k: v for k, v in os.environ.items()
           if k not in ("STRATA_API_KEY", "STRATA_ALLOWED_HOSTS")}
    return argv, Path(sc["root_dir"]), env
