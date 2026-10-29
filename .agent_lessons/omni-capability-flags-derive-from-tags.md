---
tags: [llama-autoloader, omni, capabilities, v1-models, mmproj, tts, realtime]
aliases: [omni-capability-flags-design]
related: [[raw-tcp-forwarder-architecture]], [[mmproj-pairing-heuristic-and-provenance]], [[model-type-derived-vs-override-split]]
confidence: verified
author: Maine
---

# Omni capability flags in `/v1/models` are derived from a capability SET, not `model_type`

The `/v1/models` payload advertises `multimodal` and `audio` capability flags so clients can
discover an omni model's abilities. They are derived from a capability SET (single source of
truth, reusing existing fields — NOT a second hardcoded list, NOT a new `omni` model_type):

- `multimodal` = (model_type == VISION) OR ("omni" in tags) OR ("vision" in tags) OR has_mmproj
- `audio`      = (model_type == AUDIO)  OR ("omni" in tags)

## Deliberate limitation (NOT a defect)
`audio` keys off the `"omni"` tag. A vision-classified omni variant (e.g. `qwen-omni`) that does
NOT carry the `"omni"` tag intentionally gets `audio=False`. This is by design: it avoids a second
hardcoded arch list and keeps the user's sidecar `tags` the authoritative capability declaration.
LFM3-Omni-2B (arch `lfm3`, not in `_ARCH_TO_TYPE`, so it classifies as `text`) advertises BOTH
flags because its sidecar has `tags: ["omni"]` and `has_mmproj` is true. If a new omni model is
added, ensure its sidecar carries the `omni` tag (or an appropriate `type_override`).

## Routing (verified live 2026-10-29)
The omni endpoints are routed to the per-model llama-server by `RawTCPForwarder`:
- model-scoped POSTs `/tts`, `/v1/audio/transcriptions`, `/v1/messages` → piped, model from body.
- `WS /v1/realtime` → piped (raw pipe carries the WS upgrade + frames), model from `?model=` query
  (URL-decoded via `unquote_plus`); missing model → 400. WS branch is guarded on `method == "GET"`.
Live smoke test (LFM3-Omni-2B-Q8_0 + mmproj-LFM3-Omni-2B-F16): image input, TTS (WAV out), and
realtime WS all PASS end-to-end.
