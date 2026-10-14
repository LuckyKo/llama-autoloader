---
tags: [settings, config-plumbing, yaml-dump, llama-autoloader]
aliases: [save-state-retention, max-keep-wiring]
related: [[llama-autoloader-pytest-canonical-invocation]], [[plan-review-backend-api-mismatch]]
confidence: verified
author: coder_loader-slots-build_20261009_070016.jsonl
---

# `models.save_state_max_keep` — 7-seam wiring + config.yaml reformat trap

## Fact

`models.save_state_max_keep` (int, >= 1, default 5) makes the save-state retention count
UI-configurable and **live** (PUT takes effect without restart). Built 2026-10-09; suite at
**245 passed**.

The 7 seams (all must agree on the key name):
1. `server.py` `ModelManager.__init__` — `self.save_state_max_keep = int(cfg["models"].get("save_state_max_keep", 5))`
2. `server.py` `_validate_config()` — int >= 1 (mirrors `max_loaded_models` style; `isinstance(v, int)` accepts bool — intentional, matches existing)
3. `server.py` `_cleanup_old_states()` — `max_states = max(1, int(self.save_state_max_keep))`. **Live-read**: reads the instance attr at call time, NO module-level constant. This is the seam that makes the setting actually work at runtime.
4. `server.py` `/v1/settings` — FOUR sites: GET response, PUT validate, PUT **assign** (`manager.save_state_max_keep = body[...]` — the critical runtime-effect seam), PUT persist into `cfg["models"]`.
5. `config.yaml` `models:` — `save_state_max_keep: 5` (alphabetical: after `save_state_dir`).
6. `static/index.html` — `<input id="s-save-state-max-keep" type="number" min="1">`.
7. `static/app.js` — THREE sites: loadSettings populate, saveSettings parse+validate (int >= 1), PUT body.

Key-name mapping: `save_state_max_keep` (Python/YAML/PUT body) ↔ `s-save-state-max-keep` (DOM) ↔ `saveStateMaxKeep` (JS local).

Tests: `TestCleanupOldStates` (4 tests, uses `model-c.gguf` which has no fixture state files) + 3 `TestConfigValidation` cases. The live-read test sets `manager.save_state_max_keep = 2` post-construction and asserts behavior changes — guard against import-time freeze.

## config.yaml reformat trap (gotcha)

The running server's `yaml.dump` persistence (`PUT /v1/settings`, backend selection) **rewrites
config.yaml in canonical form**: comments stripped, keys sorted alphabetically, flow-style lists
→ block-style. So the working-tree config.yaml diverges from git HEAD by ~50 lines of pure
reformatting **before any of your edits**. `git diff config.yaml` therefore shows massive noise
that is NOT yours. When reporting diffs, scope to your actual lines (e.g. diff against a
pre-edit backup) and flag the pre-existing reformatting so the committer knows. The reformat is
semantically lossless except comments.

## Related

- [[llama-autoloader-pytest-canonical-invocation]] — canonical test invocation + count history (245).
- [[plan-review-backend-api-mismatch]] — the four-seam field plumbing rule this extends to seven.
