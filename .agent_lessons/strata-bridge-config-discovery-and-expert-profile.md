---
tags: [strata, bridge-card, autoloader, discovery, expert-profile]
aliases: [strata-config-is-the-special-config, per-pack-expert-profile]
related: [[strata-bridge-lifecycle-facts]], [[autoloader-assigned-name-canonical-source]]
confidence: verified
author: researcher_strata-iq3-research_20261012_042241.jsonl
---

# Strata bridge: a config file IS the "special config"; expert-profile must match the model header

Two non-obvious facts verified 2026-10-12 while planning an IQ3_XXS bridge.

## 1. "Special config" = a `*.json` with an `exe` key in configs_dir — no card required
- `_scan_bridge_cards` globs `*.json` in `engines.strata.configs_dir` (`N:/work/WD/Strata`, config.yaml:23) and registers a bridge card keyed on the **config file's stem** (server.py:736-741). A file qualifies iff it parses to a dict and has `config_key`=`exe` (server.py:700-702). `*.shared-settings.json` skipped (server.py:737-738).
- The paired card at `cards_dir/{stem}.json` (server.py:739) is an **optional overlay** — `_make_engine_config` applies it only `if card_path.exists()` (server.py:704). No card → still a valid ModelConfig. So **auto-discovery from the Strata config alone is sufficient**; a card is only needed to set `min_free_vram_mib`/`extra_args` (ModelConfig fields set via card overlay server.py:704-708, consumed bridge_adapter.py:129-131).
- `cards_dir` for strata resolves to `./engine_cards` (server.py:413; no `cards_dir` in config.yaml). `strata_cards/` is a **read-only legacy fallback** (`LEGACY_CARDS_DIRS`, server.py:339/758-762). New cards go to `engine_cards/`.
- No quant-type branching exists in server.py or serve/ — "only IQ1 has one" just means it's the only such file. Adding `strata-coder-iq3_xxs.json` is the whole fix; no server.py/config.yaml edits.
- Card `name` is overridden by the config's `model_name` (server.py:715) — cosmetic.

## 2. Expert-profile is model-layout-specific; each pack ships its own
STRP headers (LE u32 @8/12/16 = layers/experts/total):
- `Strata/data/expert-profile-coder.bin` (96KB): **(48, 256, 12288)** — identical to `packs/coder-iq1_m/expert-profile.bin`.
- `packs/coder-iq3_xxs/expert-profile.bin` (192KB): **(48, 512, 24576)** — different expert count.
- serve/server.py:822 (docstring) describes the rule; executable match at serve/server.py:835-837 (`learned == base` on header layers/experts). So the 256-expert shared coder profile will NOT match the 512-expert IQ3 model.
- **Rule: an engine config's `--expert-profile` must point at the profile whose header matches that model.** Each pack ships its own `expert-profile.bin`. The IQ1 config happens to use the shared coder file because it coincidentally equals the IQ1 pack's profile. For IQ3, use the pack's own `expert-profile.bin`.

See [[strata-bridge-lifecycle-facts]] for the serve-layer spawn/CORS/port facts. Full plan: `reports/strata-iq3-config-bridge-plan.md`.
