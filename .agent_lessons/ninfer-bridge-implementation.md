---
tags: [llama-autoloader, bridge-engine, ninfer, native-exe, artifact-token, engine-validation]
aliases: [ninfer-bridge-implementation, artifact-token-implementation]
related: [[infer-bridge-engine-facts]], [[native-exe-engine-blockers]], [[strata-bridge-lifecycle-facts]], [[llama-autoloader-pytest-canonical-invocation]]
confidence: verified
author: coder_ninfer-bridge-coder_20261024_115317.jsonl
---

# NInfer bridge engine (Option B) implemented: {artifact} token + argv-conditional interpreter validation

Implemented on top of `N:\work\stuff\Beta\llama-autoloader` @ `b97491c` (working-tree edits, uncommitted). Realises the plan in [[infer-bridge-engine-facts]] / `N:\work\WD\AgentWorkspace\reports\ninfer-bridge-plan.md`. All launch/forward/teardown paths are reused unchanged; only these points changed:

- **`ModelConfig.artifact: str = ""`** added next to `engine_config` (the bridge-card fields block). It is a **persisted** field (NOT in `DERIVED_FIELDS = {"arch","model_type"}`), so it appears in saved card JSON via `asdict`-minus-derived. Harmless for engines that never reference `{artifact}` (strata/llama cards get `""` or the config_key value but never use the token).
- **`_make_engine_config`** forces `base.artifact = str(data[config_key])` in the same forced-after-overlay block as `engine`/`engine_config`/`auto_save_state`. The guard above (`config_key not in data or not str(data[config_key]).strip()`) guarantees the key is present AND non-empty, so this index cannot KeyError and a degenerate empty card is rejected. The per-card artifact path is the engine config's `config_key`-named value (for NInfer `config_key: artifact` → the `.ninfer` path; for strata `config_key: exe` → the exe value, unused). Single source of truth; a card overlay cannot repoint it.
- **`bridge_adapter._launch_bridge`** adds `"artifact": cfg.artifact` to the `tokens` dict. A template entry `{artifact}` expands to the per-card path at `argv[1]`.
- **`config.yaml`** `engines.ninfer` block (native exe, `argv` places `{artifact}` at index 1, `config_key: artifact`, `derive_config: false`, `ready_timeout_seconds: 600`; NO interpreter/serve_script/min_free_vram_flag/env_scrub). Stub card `N:\work\WD\ninfer-all\ninfer-cards\qwen38-27b.json` (model_id = file stem).

## Gotcha 1 — the validation dry-run token dict must know `artifact`

`_validate_engines` expands the argv template once with a **dummy values dict** so a bad `{token}` is a boot warning, not a launch crash. That dict enumerates the known tokens. If you add a new token to `_launch_bridge` but NOT to this dict, every engine using it is falsely flagged `Unknown template token {artifact}` at boot. **Any new launch token must be added to BOTH dicts** (`_launch_bridge` tokens AND the `_validate_engines` dummy dict). See [[verify-new-branch-reachable-through-public-entry-point]].

## Gotcha 2 — interpreter/serve_script relaxation must keep the argv-missing fallback

The requirement is now conditional, but ONLY skip it when argv is a **valid non-empty list that references neither** `{interpreter}` nor `{serve_script}`:

```python
argv_strs = [str(tok) for tok in argv] if isinstance(argv, list) and argv else None
for key in ("interpreter", "serve_script"):
    if argv_strs is not None and not any(f"{{{key}}}" in s for s in argv_strs):
        continue   # native-exe template references neither → not required
    # else: strict "is not set" / "path does not exist" check (unchanged)
```

If argv is missing/empty/not-a-list, `argv_strs is None` → fall back to the **strict** requirement (we cannot prove the tokens are unused). This is load-bearing: `tests/test_bridge_engines.py::test_boot_validation_collects_errors_not_raises` builds an engine with **no argv at all** and still asserts `"interpreter"`/`"serve_script"` in `_errors`. A naive "skip when not referenced" (treating missing-argv as not-referenced) breaks that test. The brace-delimited substring test `f"{{{key}}}" in s` is exact — matches `{interpreter}` but not `{interpreter_x}`.

## Test-suite reality on this box (do NOT trust absolute counts)

Canonical `python -m pytest tests/ test_server.py -q -n 0` here yields **400 passed** + 3 environmental errors (see below):
- 3 × `tests/test_real_model.py::TestRealModelLoad` **setup errors**: fixture asserts a `CUSTOM_BACKEND = "llama.cpp-win-x86_64-nvidia-cuda12-avx2-local"` binary that is absent from `backends/` (only cuda13 variants present). Purely environmental.
- 1–2 async **setup errors** that MOVE between runs (`test_safe_mode_slot_ordering` one run, `test_regression` another) — the Windows event-loop/socket-exhaustion flake in [[llama-autoloader-pytest-canonical-invocation]]; they pass in isolation.
- Added tests `test_validate_engines_native_exe_skips_interpreter` + `test_launch_bridge_artifact_token` both pass. The change introduced zero deterministic regressions.
