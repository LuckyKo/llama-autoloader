---
tags: [llama-autoloader, bridge-card, model-id, config-stem, overlay, server.py]
aliases: [bridge-card-model-id-invariant]
related: [[model-id-gguf-name-collision]] [[bridge-generic-plan-dig]]
confidence: verified
author: coder_bridge-id-coder_20261024_144848.jsonl
contributors:
  - coder_bridge-id-coder_20261024_144848.jsonl
---

# Bridge-card model id vs on-disk overlay key (two independent keys)

A bridge card has THREE distinct identifiers that must NOT be conflated. They fail
independently, so audit each separately.

- **Registry id** (`ModelManager.models[mid]`, the `{model_id}` argv token, GET `/v1/models`
  `id`): derived from the card's `model_name` with **spaces → dashes**, falling back to the
  config-file **stem** when `model_name` is unset/empty. Helper `ModelManager._bridge_card_model_id(cfg, fallback_stem)`
  returns `self._sanitize_model_id((cfg.name or fallback_stem).replace(" ", "-"))`. Called by
  `_scan_bridge_cards` (discovered cards) and `_register_card_file` (user-added cards).
- **Display name** (`ModelConfig.name`, GET `/v1/models` `name`): keeps its spaces. Only the
  derived id gets dashes.
- **On-disk overlay card filename** (`cards_dir/<X>.json`): keyed by the **config-file STEM**,
  NOT the registry id. Read path: `_scan_bridge_cards` passes `cards_dir/f"{p.stem}.json"` into
  `_make_engine_config`. Write path (`update_config`) and delete path (`delete_model_ep`) must use the
  same config-stem key. This is the real-world case: Strata cards carry `model_name`, so id != stem.
- **Derived temp config**: keyed by the **registry id** (self-consistent per id — do NOT re-key it).

## Non-obvious: `update_config` keys by the REGISTERED config's engine_config, not new_cfg's

At `server.py` `update_config` (bridge branch): `cfg = self.models[mid]` is the **currently registered**
config. `config_stem = Path(cfg.engine_config).stem if cfg.engine_config else ""` then
`save_path = cards_dir/f"{config_stem or _sanitize_model_id(mid)}.json"`, and `new_cfg.save_engine_card(save_path)`
writes the NEW content there. So **repointing a card's `engine_config` records the new path INSIDE the
original-key overlay file — it does NOT move the file.** The overlay stays keyed by the config the card was
registered from. (Verified: `test_update_config_can_change_engine_config_path` reads `card.json`, not `card-moved.json`.)

Consequence: after repointing + a rescan, the repoint is lost (scan re-forces `engine_config` from the config file
at `_make_engine_config`). This is pre-existing behavior, not introduced by the id change.

## Invariant to preserve

registry id + `{model_id}` token + `/v1/models` id = `model_name`(dashes) · `/v1/models` name = spaced display name
· on-disk overlay card filename = config-file stem · derived temp = registry id.

## Test guards (tests/test_bridge_engines.py)

- `test_bridge_card_model_id_from_name_and_stem`: model_name "Foo Bar Baz" → id "Foo-Bar-Baz";
  no model_name → id = file stem; asserts `/v1/models` id/name.
- `test_discovered_card_edit_survives_rescan`: edits `extra_args` on a discovered card whose id
  ("Keep-Me") differs from its config stem ("keepme"), asserts the overlay is written at `keepme.json`
  (never `Keep-Me.json`) and survives a rescan — guards the read/write key alignment above.

## Addendum (2026-10-24, hardening follow-up)

- **Single source of truth for the overlay key:** `ModelManager._bridge_card_overlay_stem(cfg, mid)`
  (next to `_bridge_card_model_id`) returns `Path(cfg.engine_config).stem if cfg.engine_config else _sanitize_model_id(mid)`.
  `update_config` (save_path) and `delete_model_ep` (unlink) both route through it. The scan read path
  (`_scan_bridge_cards`, `cards_dir/f"{p.stem}.json"`) and the add path (`add_engine_card_ep`, `cfg_path.stem`)
  hold the path directly (not a cfg), so they use the stem inline — annotated as the source of the same key.
  Derived-temp (`_derived_config_path`) stays keyed by the registry id.
- **Name/ID is CHANGEABLE via `PUT /v1/models/{id}/config` (revised 2026-10-24 — supersedes the earlier reject-on-rename):** a rename derives `new_mid = _bridge_card_model_id(new_cfg, config_stem)` (config_stem from `_bridge_card_overlay_stem`), re-keys the registry, and PERSISTS the new name to the card's SOURCE config file so a rescan/restart keeps it. Guard ONLY the real conflicts: (a) `new_mid` already in `self.models` → 400; (b) name already used by another model → the existing name-uniqueness `ValueError` (the `update_model_config` endpoint wraps `except ValueError` → 400); (c) `mid in self.loaded` → 400 "unload first" (a running process's model id can't change in place). `new_mid == mid` (no rename) is byte-identical to the old flow.
  - Helper `_write_bridge_name_to_config(engine_config_path, name)`: reads utf-8-sig, sets `data["model_name"]`, writes via `_atomic_write_text(json.dumps(indent=2, ensure_ascii=False))` preserving ALL other keys; best-effort no-op on missing/unparseable/non-dict. Called ONLY when `cfg.engine_config and new_cfg.name and new_cfg.name != cfg.name` — so unrelated edits (extra_args/pinned) leave the user's strata JSON byte-identical.
  - Second lock block: `self.models[new_mid] = new_cfg`; if `new_mid != mid` pop the old id and `self._recovery_failures.pop(mid)`; `_clear_recovery_failure(new_mid)`. `self.loaded` is NOT touched (rename refused while loaded).
  - Exception taxonomy (revised 2026-10-24): the name-uniqueness guard raises a dedicated `ModelNameConflict(ValueError)` (module-level, next to `RecoveryCooldownError`); the `update_model_config` endpoint catches ONLY `ModelNameConflict` → 400, so genuine internal ValueErrors (e.g. `_read_gguf_info` "Unknown GGUF value type") propagate as 500s. It subclasses ValueError so any broad catcher still works.
  - NOTE (still true): a test that aliases the live object (`cfg = mgr.models[mid]` then mutates and passes it back) defeats any guard that detects an edit by comparing to the live value — drive it through the endpoint (which deep-copies).
- **Collisions made observable (FIX 2):** `_scan_bridge_cards` (discovered) and `_register_card_file` (user-added)
  now `log.warning(...)` when a derived id collides (include config path + id) instead of silently dropping, and
  `log.warning(...)` when a display name is case-insensitively ambiguous with an already-registered model (card still
  registered). Safe against rescan-noise because `scan()` rebuilds `self.models` fresh (`self.models = found`)
  before calling `_scan_bridge_cards()`, so a collision warning only fires on a genuine within-scan collision.
