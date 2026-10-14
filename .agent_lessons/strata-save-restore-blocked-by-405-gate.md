---
tags: [llama-autoloader, strata, bridge, save-state, 405, is_llama]
aliases: [strata-save-restore-blocked-by-405-gate, require-llama-engine-blocks-strata-state]
related: [[strata-bridge-lifecycle-facts]], [[strata-dedup-deliberate-delta]], [[strata-session-save-restore-is-fully-implemented]], [[save-state-max-keep-setting]]
confidence: verified
author: researcher_strata-save-restore-scope_20261014_110135.jsonl
---

# The autoloader blocks Strata save/restore in 3 places — the first is a 405 gate, not a missing feature

Verified against `N:\work\stuff\Beta\llama-autoloader` at HEAD `2c42030`, 2026-10-14.

A request to `POST /v1/models/<strata-card>/state/save` returns **405 "Operation not supported for this engine"** and **never reaches Strata**. The block is at the autoloader's own HTTP layer:

1. `_require_llama_engine` (`server.py:1053-1060`) raises `HTTPException(405, …)` when `cfg is not None and not cfg.is_llama`. Called from `save_state_ep` at **`:2401`** and `load_state_ep` at **`:2416`** (also `toggle_vision_ep` at `:2246`).
2. `base.auto_save_state = False` forced after the card overlay (`server.py:714`).
3. Auto-save-on-unload gated on `lm.config.is_llama` (`server.py:1311`; `is_llama` defined `:208-209`).

Plus: `--slot-save-path` is added **only** to the llama argv (`server.py:292`), never for a bridge launch — so even past the gate, Strata answers **501** (`Strata serve/server.py:3117-3118`).

**The protocol is already compatible.** `save_state()`/`load_state()` themselves do *not* gate; they reach `_perform_slot_action` (`:1444-1465`) whose first attempt is `POST /slots/0?action=save {"filename": <basename>}` (`:1448`) — byte-for-byte what Strata expects. Enabling it needs **one code change** (relax the `:2401`/`:2416` gate; keep it for the vision toggle, which *is* llama-only) **plus two config changes** (`slot_save_path` in `engines.strata.config_overlay`, and `models.save_state_dir` pointed at the *same absolute directory*).

**Directory mismatch is the subtle one.** The autoloader computes `state_path = save_state_dir / f"{mid}.{label}.bin"` (`:1490`, dir resolved `:382`) but sends only the **basename**; Strata writes into its own `slot_save_path` (`serve/server.py:3130`). If the dirs differ, `lm.state_path` (`:1497`), `load_state`'s `state_path.exists()` (`:1475`), `_cleanup_old_states` (`:1565`) and the delete path (`:1559-1563`) all look in the wrong place.

**Two memories were stale — see [[strata-bridge-lifecycle-facts]]:**
- That file's lines 42-43 ("`StrataEngine` has no `/slots` save/restore equivalent") were true on 2026-10-06 and are **now false**: `session_file()` exists at `serve/server.py:2029`, handler at `:3111`.
- [[strata-dedup-deliberate-delta]] documents the deliberate `cfg is None → 405` delta. That delta has since been **reverted**: current code is `if cfg is not None and not cfg.is_llama` with a docstring saying `cfg is None` "falls through … matching the original inline behaviour".
