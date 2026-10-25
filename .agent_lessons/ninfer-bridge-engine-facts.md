---
tags: [llama-autoloader, bridge-engine, native-exe, ninfer, readiness, health-path, model-discovery]
aliases: [ninfer-bridge-engine-facts]
related: [[native-exe-engine-blockers]], [[strata-bridge-lifecycle-facts]], [[stt-integrates-as-bridge-engine]], [[autoloader-registry-is-gguf-shaped]]
confidence: verified
author: researcher_ninfer-bridge-researcher_20261024_065153.jsonl
---

# NInfer as a llama-autoloader bridge engine: /health is ready-gated (NOT a blocker); the positional `argv[1]` artifact IS the real divergence

NInfer (`N:\work\WD\ninfer-all`, native CUDA `.exe`) maps onto the autoloader bridge **more cleanly than koboldcpp/audio.cpp** in one respect and **less cleanly** in another. Verified against autoloader @ `b97491c` (HEAD; `server.py`/`bridge_adapter.py` unmodified) and NInfer `src/serve/*` @ working tree 2026-10-24.

## Blocker 2 from [[native-exe-engine-blockers]] does NOT apply to NInfer
- The autoloader polls `GET /health` for 200 (`_wait_until_ready`, `server.py:2095`). NInfer serves `/health` (`http_server.cpp:608-609`) and **gates it on a real `ready_` flag**: pre-routing handler returns **503 `model_loading`** for every route while loading (`http_server.cpp:471-489`, `Retry-After: 2`), and `handle_health` returns **200** only when the registry is serving (`Loaded`/`Sleeping`) else 503 (`http_server.cpp:994-1013`).
- ⇒ `GET /health` 200 is a **self-gating readiness signal**; `ready_timeout_seconds` works as-is, no `health_path` needed. This is strictly cleaner than Strata (whose `/health` is always 200 — see [[strata-bridge-lifecycle-facts]]).

## The real divergence: per-model selection is a positional `argv[1]`, not a config file
- NInfer usage is `ninfer-serve.exe <model.ninfer> [options]` (`serve_options.cpp:85-86`); `options.artifact_path = argv[1]` (`serve_options.cpp:593-596`) — **position-fixed**, not flexible. There is **no config-file model** (unlike Strata, whose per-model config is a JSON in `configs_dir` with an `exe` key).
- The autoloader's per-card data that reaches argv is only `{model_id}` (`bridge_adapter.py:122`) and `cfg.extra_args` appended at the **END** (`bridge_adapter.py:131`). **Neither places a path at `argv[1]`** → the per-card artifact path cannot be supplied by the current token set.
- **Clean fix (needs code):** add an `{artifact}` token bound to a new `ModelConfig.artifact` field (next to `engine_config`, `server.py:505-509`), populated by `_make_engine_config` from the stub card's `config_key` value; place `{artifact}` at `argv[1]`.
- **Config-only stopgap:** `_expand_token` (`bridge_adapter.py:55-71`) does **substring** regex sub, so a template entry `"…/models/{model_id}.ninfer"` expands per-card — forces artifact-filename == model_id == stub-stem. Plus point `interpreter`+`serve_script` both at the exe to satisfy validation (unused because argv never references those tokens). Two hacks.

## Other confirmed NInfer surface (for the argv template)
- Bind: `--host` (`serve_options.cpp:619`, default 127.0.0.1), `--port` (`:623`). Model name: `--model-id ID` (`:635`) — direct analogue of Strata `aliases` but NInfer has a real flag. Auth: `--api-key` (`:631`, empty ⇒ no auth) → **omit it, `env_scrub` empty** (no STRATA_API_KEY-style hard-exit-on-empty). Context: `--max-context` (`:642`).
- GPU: `--device N` single (`.h:68`); `--devices A,B,...` multi-stage pipeline (`.h:72`, parse `:519-527`) is **Linux-only**; on Windows use single `--device` + `--wddm-evictable-budget` (`.h:100`). **No `--min-free-vram-mib`** → `min_free_vram_flag` has no target; bound VRAM via `max_loaded_models:1` LRU + NInfer's `--vram-headroom-mib` (`:652`).
- Session state: `--slot-save-path DIR` (`:313`) = same `/slots` protocol the autoloader uses for Strata. Optional; if enabled, align it with `models.save_state_dir` (same absolute dir).

## Blocker 3 (config_key structural gate) STILL applies — unavoidable
- `_make_engine_config` requires the discovered config JSON to contain `config_key` (default `exe`, `server.py:1485,1493-1495`). NInfer has no such config → author **stub JSONs** in `configs_dir` containing e.g. `config_key: artifact` + the artifact path; set `derive_config: false` (NInfer ignores any derived JSON). Mirrors the koboldcpp "stub config" pattern in [[native-exe-engine-blockers]].
- `.ninfer` files are NOT auto-discovered (the GGUF `scan()` globs only `*.gguf`; `_scan_bridge_cards` never touches GGUF, `server.py:1517`). NInfer models must be authored cards. `list_models` (`:1817`) already guards bridge cards (`:1830`, `resolve_binary(cfg) if cfg.is_llama else ""` → empty `resolved_binary` for non-llama); `resolve_binary` (`:1195-1228`) is llama-only and unused by bridge engines. (NOTE: the `path = None if is_bridge` at `:1929-1930` is in `update_config`, NOT `list_models`.)

## Verdict
**Not achievable config-only with a clean design.** Needs: (1) relax `_validate_engines` (`:1078-1085`) to gate interpreter/serve_script on argv usage, and (2) an `{artifact}` token for the positional path. Everything else (spawn, port alloc, `/health` readiness, OpenAI/Anthropic forwarding, psutil teardown, single-slot LRU) is reused unchanged.

Full report: `N:\work\WD\AgentWorkspace\reports\ninfer-bridge-plan.md`.

---

## Addendum 2026-10-24 — runtime facts after the bridge shipped (autoloader @ `b97491c` working tree, NInfer `build-win-120a`)

These were verified live (not from the plan) and supersede/extend the "session state" and vision notes above.

### Vision is embedded in the artifact; enable with `--vision`, no mmproj
- The Qwen3.8 NInfer artifacts carry a `vision` component (`model_type: qwen3_5_vision`, `target: text`). Confirmed with `python -m tools.artifact.inspect <artifact>` — both `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` and `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-ninfer-v3.ninfer` list `'vision': {...}` in their `components`.
- Vision is turned on by passing `--vision` in the card overlay's `extra_args` (`options.cpp:495`). No separate mmproj file. `--spec dflash2` + `--vision` is a supported combo (`docs/cli.md:209`).
- **Gotcha:** without `--vision`, an image request is rejected at prepare with `HTTP 400 | vision disabled` (`openai_common.cpp`). The AC client then retries against other models, so a single image causes LRU churn (evicting one model after another until one with vision on answers). Add `--vision` to NInfer cards that should accept images.

### Session persistence: MTP OK, DFlash/DFlash2 NOT supported (engine-side)
- NInfer's `/slots` save returns `{"code":"slot_save_failed","message":"session persistence does not support the DFlash/DFlash2 backends"}` — hardcoded in `session_snapshot.cpp` (both the save and restore paths, `:437` and `:698` at this tree). So **`auto_save_state` must be false/omitted on DFlash2 cards**; the autoloader cannot work around it.
- A `--spec mtp` card (e.g. GSQ-RCO) DOES support persistence. A `slot_save_failed` / "slot holds no retained session" on those just means nothing was retained to save (e.g. the only requests were rejected image calls), not an engine limitation.

### Autoloader side (server.py / app.js) — what the bridge needs to drive NInfer slots
- Slot save/restore URL must carry the model: `POST /slots/{id}?action=save&model=<id>`. NInfer's `lease_query_model` reads the `?model=` query param; without it `observe("")` → `ModelUnknown` → **HTTP 404**. (`server.py` `_perform_slot_action` now appends `&model={model_id}` when non-empty; llama-server ignores the extra param.)
- The `is_llama` gates on auto-save-on-unload and auto-restore-on-load were widened to `(cfg.is_llama or cfg.engine == "ninfer")`.
- `_make_engine_config` no longer force-sets `auto_save_state = False`; it stays False by default and NInfer MTP cards opt in via the overlay.
- **Frontend gate (static/app.js):** the "States" button is rendered only when `m.engine === 'llama' || m.engine === 'strata'` (`:182`) and the in-place update check mirrors it (`:202`). NInfer must be added to BOTH or the button never appears on NInfer cards.

### serve_args in the engine config JSON is dead metadata
- The `serve_args` key in `configs_dir/*.json` (the engine config, not the overlay) is **never read** by the autoloader — only `artifact` + `model_name` are consumed by `_make_engine_config`. All argv beyond the template comes from the overlay's `extra_args`. Keep engine configs minimal (`artifact` + `model_name`); put every flag in the overlay.
