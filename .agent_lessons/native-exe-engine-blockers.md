---
tags: [llama-autoloader, bridge-engine, native-exe, koboldcpp, audio.cpp, asr, stt, readiness, health-path]
aliases: [native-exe-engine-blockers]
related: [[stt-integrates-as-bridge-engine]], [[bridge-generic-plan-dig]], [[strata-bridge-lifecycle-facts]], [[koboldcpp-oneshot-tts-probe]]
confidence: verified
author: researcher_stt_loader_dig_20261022_144828.jsonl
---

# Native-.exe bridge engines hit TWO loader blockers (both narrow) — no-code path is NOT achievable

Adding a native engine binary (koboldcpp.exe / audiocpp_server.exe) as a loader bridge engine is
blocked by exactly two things in the loader; both are small, and neither can be worked around by config
alone. (Loader @ `b97491c`.)

## Blocker 3 — `config_key` + engine_config-file structural gate (not just validation)
- A bridge card's `engine_config` must EXIST on disk (`_do_load_model` raises 400 if missing, `server.py:2690`),
  parse as JSON when `derive_config` is on (default), and contain a key literally named `config_key`
  (default `"exe"`, `_make_engine_config:1485` → returns None at `:1493-1595` otherwise).
- audio.cpp server configs have NO `exe` key (keys are host/port/backend/models[]) → set
  `engines.audiocpp.config_key` to `models`/`backend`/`host`, else every card is silently skipped.
- CLI-style engines with no config concept (koboldcpp): ship a STUB config JSON containing `config_key`
  and set `derive_config: false` (`bridge_adapter.py:119` passes the path verbatim; file must still exist).
- **VRAM safety net is not portable:** `min_free_vram_mib` only works if the engine exposes a `min_free_vram_flag`
  (`bridge_adapter.py:129`); neither engine does. Use the engine's own knob instead: audio.cpp config JSON
  `min_free_memory_mb` (`app/server/example.json:10`, via `config_overlay`); koboldcpp `--maxvram`/`--gpulayers`.
- **Teardown is engine-agnostic — NOT a blocker:** `_kill_proc` (`server.py:2968-3069`) snapshots
  `psutil.Process(pid).children(recursive=True)`, terminates children→parent, escalates to kill, Windows
  `Popen.kill()`/POSIX `killpg`. Correctly drains a native .exe with GPU-worker children. The ~15s CUDA release
  is inherent, not a bug.

## Blocker 1 — native-exe engines fail engine validation
- `_validate_engines` (`server.py:1078-1085`) REQUIRES `engines.<name>.interpreter` AND `serve_script`
  to be set AND their paths to EXIST. A native `.exe` has neither → `_errors` set → `GET /v1/engines`
  reports it and `_do_load_model` raises 400 (`:2688-2689`).
- The LAUNCH path does NOT require them: `_launch_bridge` (`bridge_adapter.py:101-137`) only substitutes
  `{interpreter}`/`{serve_script}` tokens IF the argv references them. An argv like
  `["backends/koboldcpp/koboldcpp.exe","--port","{port}",...]` references neither → launches fine.
- **Minimal fix:** gate the interpreter/serve_script requirement on whether the argv template actually
  uses those tokens. Do NOT touch `_launch_bridge`/`_do_load_model`.
- The built-in llama engine resolves its native binary via a SEPARATE path `resolve_binary`
  (`server.py:1195-1228`) = `backends_dir/<backend>/llama-server[.exe]` (hardcoded name). Bridge engines
  do NOT use it — their binary is whatever `argv[0]` points to. There is no `backends/<name>/` auto-resolve
  for bridge engines.

## Blocker 2 — readiness probe is hardcoded to GET /health
- `_wait_until_ready` polls `http://127.0.0.1:<port>/health` (`server.py:2094`), per-engine
  `ready_timeout_seconds`. No per-engine health path exists.
- **audio.cpp server HAS `GET /health`** (`audio.cpp/app/server/README.md:303`) → default works as-is.
- **koboldcpp has NO `/health`** (confirmed against its full endpoint reference). Readiness must use
  `GET /api/extra/version` (no auth, 200 once up) or `GET /props`. → add an optional `health_path`
  field to the engine spec (default `/health`).

## Model routing is the card's `engine` field, NOT model_type
- Every bridge card MUST point at an `engine_config` JSON containing the engine's `config_key` or it is
  skipped (`_register_card_file` `:1583-1585`; `_make_engine_config` `:1486-1495`). Model id = config
  file stem (`:1587`). `_make_engine_config` forces `engine`/`engine_config` after overlay (`:1505-1506`).
- `MODEL_TYPE_ASR` (`:257/281`) + GGUF arch classifier (`:302-305`) are DISCOVERY/UI only — they do NOT
  select an engine. A bridge card defaults to `MODEL_TYPE_TEXT` unless it carries `type_override`/`model_type`
  (`:1509-1513`). "How does MODEL_TYPE_ASR pick the engine" → it doesn't; set `engine:` on the card.
- `_scan_bridge_cards` NEVER touches GGUF entries (`:1517`): a discovered whisper GGUF is NOT auto-bound to
  an engine; it must be an authored card. Otherwise it lists as an ASR-typed GGUF with no engine and fails to load.

## Engine ASR contracts (confirmed)
- **audio.cpp server** (`audiocpp_server`, NOT the CLI): binds `--host/--port` (`docs/docker.md:123`),
  `GET /health` (`app/server/README.md:303`), ASR via `POST /v1/audio/transcriptions` (multipart `-F model=<id>
  -F file=@audio.wav` → `{text,timing}`, `:376`), details route `:414`. Its server config JSON (`docs/asr.md:471-490`,
  keys host/port/backend/models[]) is a natural `engine_config` (config_key e.g. `models`). Models from
  `huggingface.co/audio-cpp/audio.cpp-gguf`; install `audiocpp_model_manager install qwen3_asr_0_6b_q8_0`
  (`README:202`). Windows CUDA needs the `cudart` archive beside the binary (`README:254`).
- **audio.cpp CLI** (`audiocpp_cli --task asr --family qwen3_asr --model <gguf> --audio <wav> --text-out`)
  is one-shot (no port/health/resident process) → does NOT fit the loader's server lifecycle. User added ONLY
  `audiocpp_cli.exe`; `audiocpp_server.exe` must be added.
- **koboldcpp v1.123**: `--whispermodel <gguf> --port N` (native exe; `--help` confirms `--whispermodel`,
  `--port` default 5001). Transcribe: native `POST /api/extra/transcribe` AND OpenAI-compat `POST /v1/audio/transcriptions`
  (JSON `audio_data` or multipart file). Capability via `GET /api/extra/version` ("active features"). Models from
  `huggingface.co/koboldcpp` (whisper repo). NO min-free-VRAM flag (has --maxvram/--gpulayers) → VRAM bounded
  only by single-slot scheduling.

## Convergence (high value)
Both engines expose an OpenAI-compatible `POST /v1/audio/transcriptions` (multipart file upload) → ONE media_gen
`voice_to_text` backend can target that path through the loader's `raw_passthrough` (`server.py:4316`, NO JIT —
caller must `POST /v1/models/<id>/load` first). Only the model id (+ audio.cpp inner `model=` field) differ.

## VRAM
Keep `max_loaded_models:1` (loader default) = serialized single slot; neither new engine self-limits VRAM.
ASR load evicts the LRU LLM (cold-start on switch). Co-residency is an explicit opt-in, riskier.

## Teardown (from [[koboldcpp-oneshot-tts-probe]], unverified in loader)
Correct Windows kill = `taskkill /T /F` as SOLE primary (do NOT `proc.kill()` first), then poll `tasklist`
for true tree-exit (≥20s; ~15-18s CUDA release) before treating VRAM free. Confirm the loader's teardown
(`server.py:960-976` per memory) already does this.

Full plan: `N:\work\WD\AgentWorkspace\reports\stt-loader-koboldcpp-audiocpp-plan.md`.
