# Capability Flags: Capability-Set Derivation for Omni Models

**Date:** 2026-10-29
**Project:** llama-autoloader
**Related:** [[lfm3-omni-smoke-test-tts-realtime-pass-multimodal-flag-gap]], [[llama-autoloader-pytest-canonical-invocation]]

## What happened

The `/v1/models` route handler derived `multimodal` and `audio` flags from a single scalar
(`model_type`), so an omni model (vision-input AND speech-output) could never advertise both.
LFM3-Omni (arch not in `_ARCH_TO_TYPE`) classified as `text` → both flags `False`.

## Fix

Changed to capability-set derivation from existing fields (no new hardcoded list):

```python
"multimodal": (model_type == VISION or "omni" in tags or "vision" in tags or has_mmproj),
"audio": (model_type == AUDIO or "omni" in tags),
```

Key decisions:
- `audio` relies on **tags**, not arch names — avoids a second hardcoded list (spec constraint).
- `has_mmproj` is the projector-resolved flag already computed in `list_models()`.
- LFM3-Omni (tagged "omni", `has_mmproj=true`) now advertises **both** flags.

## Test gaps closed

- `test_server.py`: `test_capability_flags_present_and_correct` — 6 model types covering every OR-clause
  (vision-type, omni-tag, TTS-type, text, has_mmproj-only, vision-tag-only). Uses `TestClient` on the real route.
- `tests/test_tcp_forwarder.py`: `TestOmiIntegratedRouting` — 3 `_handle_connection`-driven tests
  (POST /tts → backend, GET /v1/realtime?model=X → backend, GET /v1/realtime → 400).
- `tests/test_tcp_forwarder.py`: 4 URL-decode tests in `TestExtractModelFromQuery`
  (%20, +, %2B, plain).

## Gotchas

- **Sidecar field is `type_override`, not `type`** — the `ModelConfig.load()` parser matches
  JSON keys to dataclass field names via `hasattr`. Using `"type"` silently falls through to
  the default (empty string → `text`).
- **mmproj file naming matters**: `mmproj-model.gguf` gets detected as an mmproj file AND
  paired via name-stripping. Use unambiguous names like `clip-model.gguf` + `mmproj-clip-model.gguf`.
- **`unquote_plus` vs `unquote + replace`**: `unquote("my%2Bmodel").replace("+"," ")` → `"my model"`
  (wrong). `unquote_plus("my%2Bmodel")` → `"my+model"` (correct). Always use `unquote_plus` for
  query-string values.
- **`_fwd()` has `model_manager=None`**: integration tests that drive `_handle_connection` through
  the POST/WS path need `f.model_manager = MagicMock()` with `queue_registry.get_or_create` mocked,
  or the code crashes at `self.model_manager.queue_registry.get_or_create(...)`.
