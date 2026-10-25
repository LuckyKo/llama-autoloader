---
tags: [llama-autoloader, bridge-card, engines, asr, stt, whisper, dig-phase, integration-plan]
aliases: [stt-integrates-as-bridge-engine]
related: [[bridge-generic-plan-dig]], [[strata-bridge-lifecycle-facts]], [[strata-save-restore-blocked-by-405-gate]], [[autoloader-registry-is-gguf-shaped]]
confidence: verified
author: researcher_stt_loader_dig_20261022_144828.jsonl
---

# STT/ASR does NOT need a new loader modality — it registers as a bridge engine

Integrating a Whisper/faster-whisper STT backend into llama-autoloader is **not** a "generalize the
llama-only server" task. The loader already ships a generalized, config-driven **bridge-engine**
abstraction, with Strata as a live second engine, AND already has an `asr` model type. The work is:
write a thin ASR serve script + add an `engines.whisper` block. No loader-core lifecycle edits.

## Loader facts (at `llama-autoloader` @ `b97491c`)

- **Bridge engine = a Python serve script over a config argv template.** `bridge_adapter._launch_bridge`
  (`bridge_adapter.py:101-137`) renders `engines.<name>.argv` with tokens `{interpreter} {serve_script}
  --engine {engine} --config {derived_config} --port {port} --host {host}`, optionally writes a derived
  config (read-only overlay, cleaned on unload), scrubs env (DELETES, not blanks), sets cwd, and appends
  `<min_free_vram_flag> <min_free_vram_mib>` only when `min_free_vram_mib>0`. "Adding an engine is a pure
  config change; no code here is engine-specific."
- **`{interpreter}` is a dedicated venv python** (Strata: `…/Strata/.venv/Scripts/python.exe`) → native
  deps (cuDNN/CTranslate2/torch) live in the engine's venv, isolated from the loader. This is the answer
  to the "Windows native deps" risk.
- **Engine dispatch in the load path:** `_do_load_model` (`server.py:2683-2697`): `if cfg.is_bridge:
  _launch_bridge(...) else _launch_llama(...)`. `_launch_llama` stays byte-identical; llama stays OUT of
  the engine registry. Readiness = poll `GET /health` 200 (`_wait_until_ready`, `server.py:2094`), per-engine
  `ready_timeout_seconds`.
- **Model type `asr` already exists:** `MODEL_TYPE_ASR="asr"` label "Speech to Text" (`server.py:257/281`);
  GGUF arch `whisper/w2v2/mmspeech/moonshine` → ASR (`server.py:302-305`). No new type code needed.
- **API surface:** `GET /v1/engines` (`server.py:4530`), `POST /v1/models/engine` add_engine_card_ep
  (`:4484`), discovery via `_scan_bridge_cards` (`:1516`). **`raw_passthrough` `POST /v1/raw/<model>/<path>`**
  (`:4316`) proxies any request to a RESIDENT engine — this is how a `/transcribe` call reaches the ASR
  engine. It does **NOT** JIT-load (503 if not resident) → caller must `POST /v1/models/<id>/load` first.
- **VRAM coexistence is a scheduler knob, not a manager:** `max_loaded_models:1` + LRU eviction
  (`_evict_lru_if_needed`, `:2115`) = serialized single slot. Loading ASR evicts the LRU LLM (cold-start).
  Co-residency requires raising `max_loaded_models` + `min_free_vram_mib` (enforced by the engine via
  its `--min-free-vram-mib` flag). GPU truth = `nvidia-smi` (`gpus()` `:3257`).
- **Bridge cards force `auto_save_state=False`** (`server.py:1507`). Nuance (reviewer, same tree): only the
  *automatic* save/restore path is llama-gated (`and cfg.is_llama` at `:2761`/`:2937`); manual `/slots` via
  `/v1/raw` is open to bridge engines. `_require_llama_engine` (`:2029`) has one call site (vision toggle,
  `:4477`). ASR has no session state, so this is correct here.
- **ASR typed two ways:** GGUF arch map (`:302-305`) AND filename markers `_FILENAME_ASR_MARKERS` (`:349`,
  applied `:399-400`); the filename path omits `mmspeech`. Irrelevant to a bridge engine (card sets
  `model_type` explicitly).
- **VoiceStudio PyTorch-fallback default is `_PYTORCH_ASR_DEFAULT` (`asr_backend.py:3861`)** — a *separate
  constant* from `omnivoice.py:310 _DEFAULT_ASR_MODEL` (same value, different symbol). Cite the asr_backend
  one.

## Addendum (reviewed 2026-10-22 by reviewer_stt_plan_reviewer @ loader b97491c)
9/11 claims CONFIRMED with accurate anchors; corrections above applied (claim 7 nuance, claim 9 constant
identity, claim 4 second ASR-typing path). Reviewer registered skill `same-value-constants-defer-identity-verification`.

## VoiceStudio STT facts (at `VoiceStudio` @ HEAD, `backend/`)

- The `/transcribe` router is **`api/routers/capture.py:49`** (job-free: POST audio → text). Request:
  multipart `audio` + form `language/model/mode/refine/dictation`. Response:
  `{text, refined_text?, segments[], language, duration_s, transcription_time_s, engine}` (`:81-90`).
  200MB cap on the MCP lane (`mcp_server.py:683`); typed 409 when weights missing (`capture.py:106-128`).
- Core engine = `FasterWhisperBackend` (CTranslate2), `asr_backend.py:1116-1302`; default model
  **`Systran/faster-whisper-large-v3`** (`_FASTER_WHISPER_DEFAULT`, `:3678`) — NOT the `omnivoice.py:310`
  `_DEFAULT_ASR_MODEL="openai/whisper-large-v3-turbo"`, which is the PyTorch/HF fallback path.
- **A crash-isolated ASR sidecar already exists:** `services/subprocess_asr.py` (`faster-whisper-isolated`)
  — spawns a child over a stdin/stdout length-prefixed JSON protocol, reusing the parent venv. Note: that
  transport is **NOT** wire-compatible with the loader's bridge (HTTP + `/health`), so a fresh serve script
  is required regardless of reuse.
- Windows native gates to carry over: `_ctranslate2_cudnn_ok` (CT2 needs **cuDNN 8** DLLs; missing →
  `__fastfail` 0xC0000409, `:270`), `_ctranslate2_execstack_ok` (ctranslate2 ≤4.4.0 RWE stack, `:294`),
  `_ctranslate2_cuda_ok` (not `torch.cuda.is_available` — ROCm caveat, `:1163`).
- Deps (`pyproject.toml`): `torch>=2.4` (→2.8.0, cu128), `whisperx>=3.1.0`, `faster-whisper>=1.0.0`,
  CTranslate2 transitive, numpy/soundfile/setuptools, cuDNN 8 at runtime.

## Extractable unit vs entanglement
Reusable: the `FasterWhisperBackend` core, compute-type/cuDNN/exec-stack gates, `_decode_audio_16k_mono`,
result-shape normalization. NOT reusable (VoiceStudio-entangled): GPU pool + `run_transcribe_guarded`,
`model_manager`, `model_acceptance` licensing, `segmentation`, `dictation`, `prefs`. The extracted artifact
should be a ~100–200 LOC FastAPI ASR server, not a port of `asr_backend.py`.

## Open decisions (supervisor sign-off)
Single-slot vs co-resident VRAM · which Whisper checkpoint · dedicated ASR venv vs reuse · reuse sidecar
vs fresh serve script (transport rewrite needed either way) · HF cache vs loader-managed configs_dir.

Full plan: `N:\work\WD\AgentWorkspace\reports\stt-loader-integration-plan.md`.
