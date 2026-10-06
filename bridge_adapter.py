"""Engine "bridge card" adapter for llama-autoloader.

This module is the ONLY engine-specific seam for bridge cards (any engine declared
under ``engines:`` in config.yaml). It exposes two launch functions that return
``(argv, cwd, env)`` tuples for ``subprocess.Popen``:

- ``_launch_llama(...)`` — today's llama-server launch, byte-identical (env/cwd are None).
- ``_launch_bridge(...)`` — renders a config-supplied ``argv`` template for engine
  ``engine_name``, optionally writes a derived config, and optionally scrubs env.

Adding an engine is a pure config change; no code here is engine-specific.

With no ``engines:`` section in config.yaml, ``_launch_bridge`` is never called and
the existing llama path is untouched.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

_TOKEN_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


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
        raise ValueError(f"Engine config {user_cfg_path} is not a JSON object")
    data = _deep_overlay(data, overlays)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path


def _expand_token(token: str, values: Dict[str, Any]) -> str:
    """Substitute ``{name}`` placeholders in one argv token.

    An unknown ``{name}`` is a config bug and raises ``ValueError`` rather than
    silently launching a broken command line.
    """

    def _sub(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name not in values:
            raise ValueError(
                f"Unknown template token {{{name}}} in argv template "
                f"(known: {', '.join(sorted(values))})"
            )
        return str(values[name])

    return _TOKEN_RE.sub(_sub, token)


def _expand(obj: Any, values: Dict[str, Any]) -> Any:
    """Recursively substitute ``{token}`` placeholders in every string inside ``obj``.

    Non-string leaves (ints, bools, None) pass through untouched.
    """
    if isinstance(obj, str):
        return _expand_token(obj, values)
    if isinstance(obj, list):
        return [_expand(v, values) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand(v, values) for k, v in obj.items()}
    return obj


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


def _launch_bridge(mgr, cfg, engine_name: str, model_id: str, port: int):
    """Return ``(argv, cwd, env)`` for an engine bridge process on ``port``.

    Everything engine-specific is data from ``mgr.engines[engine_name]``: the ``argv``
    template, the optional derived-config overlays, the env scrub list and the cwd.

    When ``derive_config`` is on (the default) the derived config is a read-only copy
    of the user's engine JSON with the allow-listed overlays merged in; the user's JSON
    is never written.
    """
    spec = mgr.engines[engine_name]        # caller validated; KeyError impossible
    ec = Path(cfg.engine_config)
    if spec.get("derive_config", True):
        derived = mgr._derived_config_path(engine_name, model_id)
        derive_config(ec, derived, {**spec.get("config_overlay", {}),
                                    **_expand(spec.get("config_overlay_dynamic", {}),
                                              {"model_id": model_id})})
        cfg_arg = str(derived)
    else:
        cfg_arg = str(ec)
    tokens = {"engine": engine_name, "model_id": model_id, "port": port,
              "host": mgr.host, "derived_config": cfg_arg,
              "interpreter": spec.get("interpreter", ""),
              "serve_script": spec.get("serve_script", "")}
    argv_template = spec.get("argv")
    if not argv_template:
        raise ValueError(f"engines.{engine_name}.argv is required (no default; each engine supplies its own launch template)")
    argv = _expand(argv_template, tokens)
    if cfg.min_free_vram_mib and spec.get("min_free_vram_flag"):
        argv += [spec["min_free_vram_flag"], str(cfg.min_free_vram_mib)]
    argv += list(cfg.extra_args)
    env = None
    if spec.get("env_scrub"):
        # Scrubbed vars are DELETED from the child env, not blanked (an empty value can be a
        # hard exit for some engines — see each engine's config.yaml entry for why).
        env = {k: v for k, v in os.environ.items() if k not in set(spec["env_scrub"])}
    return argv, Path(spec["cwd"]) if spec.get("cwd") else ec.parent, env
