---
tags: [lfm3-omni, smoke-test, tts, realtime-websocket, mmproj, capabilities]
aliases: [omni-smoke-test, lfm3-tts-realtime-verified, multimodal-flag-gap]
related: [[autoloader-1234-is-agents-own-llm-control-plane]], [[start-bat-port-wait-relaunch]]
confidence: verified
author: coder_omni-live-smoke_20261029_092439.jsonl
---

# LFM3-Omni smoke test: /tts + /v1/realtime routing PASS; `multimodal` flag false (classifier gap)

**Tree pin:** llama-autoloader @ HEAD `ee02b40c82e45233a7384a2b5033f91b5bde810c` + **dirty**
working tree — the `/tts` and `/v1/realtime` routing in `tcp_forwarder.py`
(`_is_llm_endpoint` / `_is_ws_model_endpoint`, ~line 1280-1320) is **uncommitted**
(`git status`: server.py, tcp_forwarder.py, test_server.py, tests/test_tcp_forwarder.py modified).

## Live results (2026-10-29 09:33, autoloader PID 29044, model LFM3-Omni-2B-Q8_0 on port 9008)
- **Load:** PASS. Launch argv carries `--mmproj N:\work\stuff\Beta\LFM\LFM3-Omni-2B\mmproj-LFM3-Omni-2B-F16.gguf`;
  llama-server log: `load_model: loaded multimodal model, '...mmproj-LFM3-Omni-2B-F16.gguf'`.
- **Multi-modal input:** PASS. `POST /v1/chat/completions` with base64 PNG content part →
  200, answer correctly described the red test image ("solid, uniform field of bright red").
- **TTS:** PASS. `POST /tts` (no `voice` field needed) → 200, 122,924 bytes, `Content-Type: audio/wav`, RIFF/WAVE magic.
- **Realtime WS:** PASS. Raw `GET /v1/realtime?model=LFM3-Omni-2B-Q8_0` → `HTTP/1.1 101 Switching Protocols`,
  first frame in 0.01 s: `{"type":"session.created","session":{...,"model":"LFM3-Omni-2B-Q8_0.gguf",...}}`.
  The raw TCP pipe carries the WS upgrade + frames verbatim to llama-server.
- **Capabilities:** PARTIAL. `/v1/models` entry advertises `use_mmproj:true`, `has_mmproj:true`,
  `tags:["omni"]` — but `multimodal:false` and `audio:false`.

## The `multimodal` flag gap (root cause)
`multimodal` is derived as `model_type == MODEL_TYPE_VISION` (server.py, `_models_list`
payload builder, symbol-anchored: the `"multimodal": m["model_type"] == MODEL_TYPE_VISION`
line). LFM3's GGUF architecture is **not in `_ARCH_TO_TYPE`** (the table knows
`qwen-omni` → vision, not LFM3), its sidecar `type_override` is `""`, and the filename has
no audio/ASR/embedding markers → `_classify_model_type` falls back to `text` → flag false.

**Fix (one word, unapplied):** set the sidecar
(`N:\work\stuff\Beta\LFM\LFM3-Omni-2B\LFM3-Omni-2B-Q8_0.gguf.json`) `"type_override": "vision"`.
This is a config decision (it moves the model into the WebUI "Vision" group), so it was
reported, not applied.

## Other facts worth keeping
- `/v1/models` ids carry the **`.gguf` suffix** (`LFM3-Omni-2B-Q8_0.gguf`); match on both forms.
- Test harness: `temp/omni_smoke_test.py` (section-filterable: `python temp\omni_smoke_test.py ws caps`),
  artifact `temp/omni_smoke_results.json`. Each endpoint hit exactly once (GPU-tight box).
- Cleanup verified: `POST /v1/models/LFM3-Omni-2B-Q8_0/unload` → llama-server pid gone,
  autoloader left running (it is the control plane — see [[autoloader-1234-is-agents-own-llm-control-plane]]).
