---
tags: [model-resolution, gguf_name, alias-collision, resolve_model_id, autoloader]
aliases: [model-name-collision, wrong-model-loads, gguf-alias-ambiguity]
confidence: verified
---

# Model-ID Resolution Collides on `gguf_name` (internal GGUF name) — 2026-08-30

Two different GGUFs of the same base model (e.g. two Qwen3.8-27B quants from different
sources) both carry the **same internal `general.name`** in their GGUF metadata
(e.g. `"Qwen3.8-27B"`). The autoloader stores that as `cfg.gguf_name` and — critically —
treats it as a valid **request alias**. So requesting by one model's ID can silently
resolve to the *other* file, even though each has its own distinct user-facing `name`.

## How resolution works (`server.py: ModelManager.resolve_model_id`, ~L499)
Lookup order for an incoming `model` string (case-insensitive):
1. exact registry key = **filename** (`mid = p.name`, set at scan, ~L418) — e.g. `Qwen3.8-27B-UD-Q4_K_XL.gguf`
2. filename stem / full id match
3. `cfg.name` (the user-facing display name from the sidecar `.json`)
4. **`cfg.gguf_name`** ← the collision point

If none match, it falls back to: currently-loaded ready model → `default` model → single model → `None`.

## Why the bug is invisible / hard to spot
- The registry **key is the filename**, so two files never collide as *keys* — the dict holds both fine.
- `gguf_name` is auto-filled from GGUF metadata at scan time (~L485) and lazily via `_ensure_gguf_name` (~L586). A sidecar `.json` can pin it too (`"gguf_name": "..."`), but the code only fills it **when empty** — so a hand-written identical `gguf_name` in two sidecars is kept as-is.
- The collision only manifests when a client sends a model string that equals the shared internal name (or omits the param and hits the loaded/default fallback).

## Concrete case (2026-08-30)
- `n:/work/stuff/Beta/jrell/Qwen3.8-27B-i1-IQ4_XS/...GGUF-Smaller.gguf` → sidecar `name="Qwen3.8-27BIQ4-XS"`, `gguf_name="Qwen3.8-27B"`
- `n:/work/stuff/Beta/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf` → sidecar `name="Qwen3.8-27B"`, `gguf_name="Qwen3.8-27B"`
- Both alias to `qwen3.8-27b`. Whichever is registered first (dict order) wins the `gguf_name` match. User worked around it by renaming the jrell file to `.gguf.dis` so discovery skips it.

## Fixes / mitigations
- **Immediate (no code):** always request models by their **exact filename** (the registry key) — that path is unambiguous. Or make each `name`/`gguf_name` unique per file in the sidecar `.json`.
- **Code-level (if you want it robust):** don't let `gguf_name` act as a request alias when it's non-unique across the registry, or drop `gguf_name` from the alias loop entirely and rely on filename + user `name`. If keeping it, resolve ambiguity by preferring an exact-filename/stem match before falling to `gguf_name`, and log a warning when two models share a `gguf_name`.

## RESOLVED (2026-08-30) — the editable `name` is now THE retrieval key
Implemented in `server.py` (see [[autoloader-name-retrieval-key-fix]] for the change detail):
- `resolve_model_id()` no longer matches `gguf_name`. Order is now: exact filename → stem/full id → **editable `name`** (case-insensitive). The loaded/default/single fallbacks are unchanged.
- New `_unique_display_name(desired, owner_mid, stem)` helper keeps auto-promoted display names unique (appends `(stem)` / `(stem-N)` on collision, logs a warning). Wired into both the scan path and `_ensure_gguf_name()`.
- `update_config()` rejects (ValueError) a `name` already used by another model (case-insensitive), since `name` is now the retrieval key.
- `gguf_name` remains display-only metadata (still populated + returned by `/v1/models`).
- Regression coverage: `tests/test_regression.py::TestModelIdResolution` (two models sharing `gguf_name="Qwen3.8-27B"` with distinct names → each resolves to the correct file; shared `gguf_name` is NOT an alias; duplicate-name rejection incl. case-insensitive; scan disambiguation). Full suite: 141 passed.

## Patterns to keep
- Registry key = **filename** is the one guaranteed-unique identity; anything derived from GGUF metadata (`general.name`) is *not* unique across quants of the same base model.
- Any new alias field added to resolution must be checked for cross-model uniqueness, or it becomes a silent wrong-model hazard.
