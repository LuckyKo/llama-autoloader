# llama-autoloader: Comprehensive Technical Documentation

## Autoloader Overview

**What it is:**
llama-autoloader is a Just-In-Time (JIT) model loader and OpenAI-compatible proxy for llama.cpp. It automates the discovery, loading, and management of `.gguf` model files, spawning `llama-server` subprocesses on demand and providing a unified API endpoint for inference.

**Problem solved:**
- Eliminates manual model loading/unloading
- Provides a single entry point for multiple models
- Manages GPU memory automatically via LRU eviction
- Handles port allocation and process lifecycle
- Enables session state persistence across model switches

**Architecture:**
1. **Model Discovery**: Scans a configured root directory recursively for `*.gguf` files
2. **Configuration Loading**: For each model, loads an optional sidecar JSON file (`model.gguf.json`) for custom settings
3. **Process Management**: Spawns `llama-server` subprocesses on demand (JIT) with automatic port allocation
4. **Request Routing**: Proxies OpenAI-style requests to the appropriate llama-server instance based on model ID
5. **Health Monitoring**: Checks server readiness and manages process lifecycle
6. **WebUI Dashboard**: Provides a browser-based interface for monitoring and management

**Key Features:**
- **JIT Loading**: Models load only when first requested
- **Auto-Unload**: Models are unloaded after idle timeout to free resources
- **LRU Eviction**: Automatically unloads least-recently-used models when reaching `max_loaded_models` limit
- **Port Management**: Assigns a stable per-model port (`base_port` + position of the model id in the sorted registry), falling back to a dynamic free-port scan; keeps each card on the same port across reloads and restarts
- **Session State Persistence**: Can auto-save/restore KV cache state across load/unload cycles
- **Vision Module Auto-Pairing**: Automatically detects and associates `mmproj` vision projector files with models
- **Multi-Backend Support**: Supports different llama.cpp builds (e.g., CUDA, AVX2) with per-model overrides
- **OpenAI-Compatible API**: Full `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` endpoints

---

## Configuration

### Running Multiple Instances

The process binds three things; each instance needs its own set:

| What | Default | Control |
|------|---------|---------|
| Main client-facing port (raw TCP forwarder) | `launcher.port` (9124) | `--port` |
| Internal FastAPI / uvicorn | `$AUTOLOADER_INTERNAL_PORT` or 1235 | `--internal-port` |
| First engine port (one per model) | `launcher.base_port` (9001) | `--base-port` |

`--port` alone is enough to start a second instance: the internal port auto-advances past
anything already listening and the engine range self-heals by scanning upward.

- `_port_in_use(port, host)` — connect probe (not a bind probe; on Windows
  `SO_REUSEADDR` does not cleanly separate "free" from "owned by another instance").
- `_resolve_internal_port(desired, explicit, host, max_tries)` — returns
  `(port, auto_advanced_from)`. Busy + **explicit** raises `SystemExit` (the user named
  that port; moving it would hide the collision). Busy + **implicit** advances to the
  next free port, so a second instance starts cleanly. Exhausting `max_tries` raises.
- The main port being busy raises `SystemExit` with a readable message. Previously this
  surfaced as a bare `WinError 10048` from uvicorn *after* the forwarder had already
  bound, leaving a half-initialized app.
- Engine-port cross-instance safety comes from `_is_port_available` probing the real
  socket: `_allocate_port` only knows about its own manager's `loaded` set.

**Not instance-scoped:** `models.save_state_dir`, `engine_cards/` and `temp/`. Notably
`temp/<engine>/<model>.json` derived configs are written at load and unlinked at unload,
so one instance unloading a bridge card deletes the config another instance is using.

### config.yaml Structure

```yaml
gpu:
  poll_interval_seconds: 2

launcher:
  base_port: 9001
  host: 127.0.0.1
  port: 9124

llama_server:
  backends_dir: ./backends
  binary: llama-server
  default_args: --no-webui --parallel 1 --jinja --cache-ram 16384 --kv-unified
  selected_backend: llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.0

models:
  auto_save_state: false
  idle_timeout_seconds: 3600
  max_loaded_models: 1
  root_dir: n:/work/stuff/Beta
  save_state_dir: ./states
```

**Section Details:**

- **gpu.poll_interval_seconds**: How often to query GPU status via `nvidia-smi` (default: 2)
- **launcher.base_port**: First port allocated for llama-server instances (default: 9001)
- **launcher.host**: Host address for the autoloader itself (default: 127.0.0.1)
- **launcher.port**: Port for the autoloader's REST API and proxy endpoints (default: 9124)
- **llama_server.backends_dir**: Directory containing llama-server builds (subdirectories)
- **llama_server.binary**: Fallback binary name/path if no backend is selected
- **llama_server.default_args**: Space-separated string of arguments passed to every llama-server instance
- **llama_server.selected_backend**: Global default backend folder name in `backends_dir`
- **models.auto_save_state**: Global default for auto-save/restore (can be overridden per-model)
- **models.idle_timeout_seconds**: Seconds of inactivity before a model is unloaded (default: 3600)
- **models.max_loaded_models**: Maximum number of models that can be loaded simultaneously (default: 1)
- **models.root_dir**: Directory to scan for `.gguf` files (recursive)
- **models.save_state_dir**: Directory where KV cache state files are saved

### Model Sidecar JSON Format

Each model can have an optional sidecar config file named `model.gguf.json` placed alongside the `.gguf` file.

```json
{
  "name": "My Model 7B",
  "description": "A fine-tuned chat model",
  "args": "--temp 0.7 --top-k 40",
  "ctx_size": 8192,
  "n_gpu_layers": 999,
  "default": false,
  "pinned": false,
  "auto_save_state": false,
  "backend": "",
  "use_mmproj": true,
  "mmproj_file": "mmproj-MyModel-F16.gguf",
  "estimated_vram_mb": 5000,
  "tags": ["chat", "7B"]
}
```

**Field Descriptions:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `""` | Human-readable display name |
| `description` | string | `""` | Model description |
| `args` | string | `""` | Extra CLI arguments appended to llama-server command line |
| `ctx_size` | int | `8192` | Context window size (`--ctx-size`) |
| `n_gpu_layers` | int | `999` | Number of layers to offload to GPU (`--n-gpu-layers`) |
| `default` | bool | `false` | Informational default marker; LLM proxy requests must still name a model |
| `pinned` | bool | `false` | Never auto-unload this model (exempt from LRU eviction) |
| `auto_save_state` | bool | `false` | Auto save/restore KV cache state on model unload/load |
| `backend` | string | `""` | Force a specific backend build for this model (folder name in `backends/`) |
| `use_mmproj` | bool | `true` | Enable vision projector if an `mmproj` file is detected |
| `mmproj_file` | string | `""` | Explicit vision projector filename (auto-detected if empty) |
| `estimated_vram_mb` | int | `0` | Estimated VRAM usage hint (display only) |
| `tags` | string[] | `[]` | Tags for filtering in the WebUI |
| `type_override` | string | `""` | Pin the card to a model type; empty = auto-detect |

> `arch` and `model_type` also exist on `ModelConfig` but are **derived**: they are
> recomputed from the GGUF header on every scan and are deliberately NOT written to the
> sidecar (`ModelConfig.DERIVED_FIELDS`). Only `type_override` is persisted, so an
> auto-detected type can never be mistaken for a deliberate user pin.

### Model Type Classification

`_classify_model_type(path, info, override)` in `server.py` returns one of
`text`, `vision`, `audio`, `asr`, `embedding`, `draft`, `codec`, `mmproj`.
First match wins:

1. `override` — the sidecar's `type_override`, if it names a valid type.
2. Vision projector — `general.type == "mmproj"`, else an `mmproj` filename marker.
3. Auxiliary component — filename markers for codecs (`vae`, `wavtokenizer`, `codec`,
   `vocoder`, `tokenizer-`) and drafts (`mtp-`, `draft-`, `speculative`). **Checked
   before** the architecture table because some codecs share an architecture with their
   host model (`yue2-vae` and `yue2-3b` are both `audiocpp`).
4. `general.architecture` via `_ARCH_TO_TYPE`. Any `<arch>-assistant` suffix is a draft.
5. Filename family (`outetts`, `kokoro`, `tts`, `whisper`, `bge-`, …) for generic arches.
6. `text` — a plain decoder.

Only the GGUF KV block is read (`_read_gguf_info`); tensor data is never touched.
The same reader also supplies `general.name`, `*.context_length` and `general.architecture`,
so `scan()` opens each file exactly once.

`_is_mmproj(path, info)` drives the registry partition. It trusts `general.type` when
present and falls back to the filename marker, which preserves the original behaviour for
every file that already worked.

### Vision Projector Pairing Rules

`_pair_mmproj(p, cfg, mmproj_paths, *, eligible_siblings=None, eligible_others=None)`
resolves a model's projector. Order:

1. **Auxiliary types are never auto-paired.** A `draft`/MTP model or a `codec` has no
   vision tower, so a projector is always wrong for them. An *explicit* sidecar
   `mmproj_file` is still honoured — that is a user decision, not inference.
2. **Explicit `mmproj_file`** — as a path, else by filename in the model's directory,
   else in `root_dir`.
3. **Sibling projectors** — exactly one in the model's directory, *provided this is the
   only eligible model there*. If several eligible models share the directory, the lone
   projector is contested and is only handed over on a clear, **unique** name match
   (`_mmproj_affinity`). Two or more siblings fall back to the positional-prefix
   heuristic.
4. **Single projector anywhere in the tree** — same uniqueness rule, scoped tree-wide.

The ambiguity rules exist because both "lone sibling" and "single projector in the tree"
previously attached a projector to **every** model regardless of name; rule 4 bypassed
rule 3's directory scoping entirely. `_mmproj_affinity` normalizes the `mmproj-` prefix
and scores exact/containment matches, because the old positional heuristic scores every
projector above zero (index 2 of `mmproj-*` is always `p`).

Rule 3's multi-projector branch uses `_best_mmproj_by_name`, which prefers
`_mmproj_affinity` whenever **any** candidate has a name relationship, and falls back to
the positional heuristic only when none does. Affinity is strictly more trustworthy, and
the fallback preserves the historical answer for real libraries whose projectors carry no
model name at all (e.g. `mmproj-F16` / `mmproj-F32` beside a single model). The purely
positional heuristic gets this visibly wrong: for `alpha.gguf` in a directory holding
`mmproj-alpha.gguf` and `orphan-mmproj.gguf` it picked **orphan**-mmproj, because it
compares characters by index and the `mmproj-` prefix shifts every position (`alpha` vs
`mmpro` scores 1; `alpha` vs `orpha` scores 3 on coincidence). Affinity scores that
1000 vs 0.

An explicit `mmproj_file` is `.strip()`ed before resolution. A sidecar edited by hand or
filled by copy/paste can carry a trailing space (`"mmproj-F16.gguf "`); pathlib preserves
it, so the resolved `Path` would not equal the canonical one from `rglob` — silently
breaking every path comparison and emitting a `--mmproj` launch arg with a stray space
inside the quotes. It only appeared to work because Windows file APIs tolerate trailing
spaces; it would fail outright on Linux.

`eligible_siblings` / `eligible_others` default to `None`, preserving the historical
unconditional behaviour for `update_config()` and existing tests; `scan()` supplies
them. Pairing runs **after** classification in `scan()` so auxiliary types are known.

### Vision Projector Provenance

A projector pairing is a two-way fact, so both directions are exposed:

- `mmproj_paths[mid]` → the model (via `GET /v1/mmproj`, which lists every projector and
  its `owner`, counting any projector with no owner in `unpaired`)
- model → the projector (via `mmproj_resolved` + `mmproj_source` on `GET /v1/models`)

`mmproj_source` is `explicit` or `auto`. It is read **before** the pairing loop writes an
inferred name back into `cfg.mmproj_file`, because after that write the two are
indistinguishable forever — which is the bug this fixes: `mmproj_file` alone could not
tell a deliberate pairing from a guessed one.

Detection is a default, not a guarantee: `OuteTTS` reports `olmo` with no audio tensors
(filename only), and `Laya` reports `modern-bert` despite being multimodal. Both are
correctable via `type_override`.

### How default_args Combine with Model-Specific Args

The final command line for llama-server is constructed as follows:

1. Base arguments from `to_launch_args()`: model path, host, port, ctx-size, n-gpu-layers
2. If `model_id` is provided: `--alias model_id`
3. If vision projector is found and `use_mmproj` is true: `--mmproj mmproj_path`
4. If `slot_save_dir` is set: `--slot-save-path slot_save_dir`
5. **Global default_args** (split by spaces): from `llama_server.default_args`
6. **Model-specific args**: from `config.args` (split by spaces)

This means model-specific args are appended after global defaults, allowing them to override or extend the base configuration.

---

## API Reference

**Base URL:** `http://127.0.0.1:9124`

All endpoints return JSON unless otherwise noted. Models are automatically loaded (JIT) on first inference request if not already loaded.

### Model Management Endpoints

#### `GET /v1/models`

List all discovered models with their status.

**Response Schema:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "model.gguf",
      "name": "My Model",
      "description": "A model description",
      "tags": ["chat"],
      "default": false,
      "pinned": false,
      "auto_save_state": false,
      "backend": "",
      "resolved_binary": "/path/to/backend/llama-server.exe",
      "has_mmproj": true,
      "use_mmproj": true,
      "mmproj_file": "mmproj-model.gguf",
      "ctx_size": 8192,
      "n_gpu_layers": 999,
      "estimated_vram_mb": 5000,
      "args": "--temp 0.7",
      "gguf_name": "",
      "size_mb": 4567.8,
      "loaded": false,
      "ready": false,
      "port": null,
      "pid": null,
      "last_used": null,
      "state_path": null,
      "model_type": "text",
      "model_type_override": "",
      "model_type_label": "Text Generation",
      "arch": "llama",
      "auxiliary": false
    }
  ]
}
```

Classification fields:

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | string | Effective type; drives the WebUI group, badge and hint |
| `model_type_override` | string | The user's pin (`""` = auto-detected) |
| `model_type_label` | string | Display name for `model_type` |
| `arch` | string | Raw `general.architecture`, `""` when absent |
| `auxiliary` | bool | True for component types (`mmproj`, `codec`, `draft`) |

#### `GET /v1/models/{model_id}`

Get detailed information about a specific model.

**Response Schema:**
```json
{
  "id": "model.gguf",
  "object": "model",
  "config": { /* full ModelConfig as dict */ },
  "loaded": false,
  "ready": false
}
```

**Error Responses:**
- `404 Not Found`: Model ID not recognized.

#### `GET /v1/models/{model_id}/port`

Return the raw TCP port of a model's resident `llama-server` so a client can bypass the
autoloader's proxy data path. Useful for direct binary/large streaming access.

**Response Schema:**
```json
{
  "port": 9001,
  "model_id": "model.gguf"
}
```

**Error Responses:**
- `404 Not Found`: Model ID not recognized.
- `503 Service Unavailable`: Model known but not currently loaded.

#### `PUT /v1/models/{model_id}/config`

Update a model's configuration. Accepts partial updates.

**Request Body:** Any subset of ModelConfig fields.

**Type pinning:** `model_type` accepts any value in `VALID_MODEL_TYPES`. An empty string
clears the pin and returns the card to auto-detection (the effective type is re-derived
immediately, without waiting for a rescan). An unknown value is rejected with **400**.

**Response Schema:** Updated ModelConfig as dict.

**Error Responses:**
- `400 Bad Request`: unknown `model_type`, or an invalid `engine_config` on a bridge card.
- `404 Not Found`: Model ID not recognized.

#### `POST /v1/models/{model_id}/load`

Load a model on demand (JIT). Returns immediately when the process starts, not necessarily when ready.

**Response Schema:**
```json
{
  "id": "model.gguf",
  "port": 9001,
  "pid": 12345,
  "ready": true
}
```

**Error Responses:**
- `404 Not Found`: Model ID not recognized.
- `503 Service Unavailable`: Failed to start llama-server process (port conflict, binary error, etc.).

#### `POST /v1/models/{model_id}/unload`

Unload a model. Forces the llama-server process to terminate. Returns **500** if
the process could not be verifiably terminated; the model then stays tracked as
loaded so the unload can be retried.

**Response Schema:**
```json
{
  "id": "model.gguf",
  "unloaded": true
}
```

**Error Responses:**
- `404 Not Found`: Model ID not recognized.
- `500 Internal Server Error`: Process termination could not be verified. Body:
  `{"id": "...", "unloaded": false, "pid": 1234, "reason": "..."}`.

#### `POST /v1/models/{model_id}/interrupt`

Best-effort in-flight generation cancel for an active (loaded + ready) model.

Soft mode (default) closes the active upstream socket(s) to the resident `llama-server`.
Recent builds stop generating when the client connection drops and keep the model loaded;
older builds/states may continue. Returns HTTP 200 with `"status": "not-loaded"` when the
model is not currently loaded (not an error).

**Query Parameters:**
- `hard` (boolean, default `false`) — when `true`, kill and unload the resident process to
  guarantee the GPU stops (costs a fresh JIT load next time).

**Response Schema:**
```json
{
  "status": "interrupted",
  "model_id": "model.gguf",
  "mode": "soft",
  "cancelled_requests": 1
}
```

**Error Responses:**
- `404 Not Found`: Model ID not recognized.
- `500 Internal Server Error`: Hard interrupt (unload) failed.

#### `DELETE /v1/models/{model_id}`

Delete an engine bridge card (user-added). Unloads first if loaded.

**Response Schema:**
```json
{ "id": "card-id", "deleted": true }
```

**Error Responses:**
- `400 Bad Request`: Model is not a bridge/decoder card.
- `404 Not Found`: Model ID not recognized.

#### `POST /v1/models/{model_id}/vision/toggle`

Toggle vision module support for a model.

**Response Schema:**
```json
{
  "id": "model.gguf",
  "use_mmproj": true,
  "mmproj_file": "mmproj-model.gguf"
}
```

**Error Responses:**
- `404 Not Found`: Model ID not recognized.

#### `POST /v1/scan`

Rescan the models directory for new `.gguf` files.

**Response Schema:**
```json
{
  "models": ["model1.gguf", "model2.gguf"]
}
```

#### `POST /v1/unload_all`

Unload all currently loaded models.

**Response Schema:**
```json
{
  "unloaded_all": true,
  "unloaded": ["model-a.gguf"],
  "failed": []
}
```

If any engine process survives termination, the endpoint returns **500** with
`unloaded_all: false` and one `failed` entry per surviving model:

```json
{
  "unloaded_all": false,
  "unloaded": [],
  "failed": [
    {"model": "model-a.gguf", "pid": 1234, "reason": "process 1234 is still running"}
  ]
}
```

Surviving models remain tracked as loaded (`/v1/models` shows `loaded: true`),
so callers can retry the unload rather than assuming the memory was freed.

### State Management Endpoints

#### `POST /v1/models/{model_id}/state/save`

Save slot 0 KV cache state to a file. State files are stored in `save_state_dir` with naming convention `{sanitized_model_id}.{sanitized_label}.bin`. Path separators and `..` are stripped from both model_id and label for safety.

**Request Body:**
```json
{
  "label": "mystate"
}
```

**Response Schema:**
```json
{
  "id": "model.gguf",
  "label": "mystate",
  "path": "/path/to/states/model.gguf.mystate.bin"
}
```

**Error Responses:**
- `400 Bad Request`: Model not loaded.
- `500 Internal Server Error`: Save operation failed (e.g., llama-server unresponsive).

#### `POST /v1/models/{model_id}/state/load`

Restore slot 0 KV cache state from a saved file. Restored state preserves conversation context for resumed sessions, allowing llama.cpp's prompt cache to skip reprocessing tokens on subsequent requests.

**Request Body:**
```json
{
  "label": "mystate"
}
```

**Response Schema:**
```json
{
  "id": "model.gguf",
  "label": "mystate",
  "path": "/path/to/states/model.gguf.mystate.bin"
}
```

**Error Responses:**
- `400 Bad Request`: Model not loaded.
- `500 Internal Server Error`: Restore operation failed or state file not found.

#### `GET /v1/models/{model_id}/state`

List all saved states for a model. Scans `save_state_dir` for files matching `{sanitized_model_id}.*.bin`.

**Response Schema:**
```json
{
  "id": "model.gguf",
  "labels": ["mystate1", "mystate2"]
}
```

#### `DELETE /v1/models/{model_id}/state/{label}`

Delete a saved state file for the given label.

**Response Schema:**
```json
{ "id": "model.gguf", "label": "mystate", "deleted": true }
```

**Error Responses:**
- `404 Not Found`: Model or state label not found.
- `500 Internal Server Error`: Delete failed.

### OpenAI-Compatible Proxy Endpoints

These endpoints forward requests to the loaded model's llama-server.

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`

Also available without `/v1` prefix:
- `POST /chat/completions`
- `POST /completions`
- `POST /embeddings`

**Request Body:** Standard OpenAI API format with `model` field specifying which model to use.

### Direct llama-server Slot Endpoints (via Proxy)

For advanced state management, you can access the underlying llama-server's native slot
endpoints through the autoloader's proxy route `/v1/raw/{model_id}/{path}`. These
requests are forwarded directly to the loaded model's llama-server process.

> **Verified 2026-10-18.** This endpoint was documented here but never actually
> registered — every call 404'd. It now works. **The old examples were also wrong:**
> llama-server takes `action` in the **query string** but `filename` in the **JSON body**.
> Putting `filename` in the query returns **HTTP 400**, which is exactly what the
> previous examples did. `_perform_slot_action()` has always used the correct shape.

**Slot-specific endpoints:**
- `GET/POST /v1/raw/{model_id}/slots/0?action=save` + body `{"filename": "state.bin"}`
- `GET/POST /v1/raw/{model_id}/slots/0?action=restore` + body `{"filename": "state.bin"}`

**Multi-slot endpoints (specify slot via query param):**
- `GET/POST /v1/raw/{model_id}/slots?action=save&id_slot=0` + body `{"filename": "state.bin"}`
- `GET/POST /v1/raw/{model_id}/slots?action=restore&id_slot=0` + body `{"filename": "state.bin"}`

Both query parameters and JSON body formats are supported, matching llama-server's native
API. Examples:

```bash
# Via autoloader proxy (correct shape: action in query, filename in body)
curl -X POST 'http://127.0.0.1:9124/v1/raw/mymodel/slots/0?action=save' \
  -H "Content-Type: application/json" \
  -d '{"filename":"test.bin"}'

# WRONG -> HTTP 400: filename belongs in the body, not the query string
curl "http://127.0.0.1:9124/v1/raw/mymodel/slots/0?action=save&filename=test.bin"

# Direct llama-server (if you know the port)
curl -X POST 'http://127.0.0.1:9001/slots/0?action=save' \
  -H "Content-Type: application/json" \
  -d '{"filename":"test.bin"}'
```

The model must already be **loaded**: `/v1/raw` returns **503** rather than JIT-loading
it, because silently spending a minute loading a multi-GB model on a typo is worse than a
clear error. Call `POST /v1/models/{id}/load` first.

Unlike `/v1/models/{id}/state/save`, this route writes wherever llama-server's
`--slot-save-path` points and does **not** register the file in the autoloader's state
list — the two paths are independent.

### Other Endpoints

#### `GET /v1/engines`

List configured bridge engines with readiness status and errors.

Response per engine: `name`, `label`, `enabled`, `ready`, `error`, `cards_dir`,
`has_configs_dir`. `ready=false` means a load attempt would return a 400, and `error`
carries that exact reason.

The WebUI uses this both to populate the add-engine dropdown and to render the per-card
engine badge (`⚙ <label>`, dashed, coloured from a hash of the engine name). A
misconfigured engine gets a red `!` on the badge plus an inline warning naming `error`,
so the reason is visible on the card instead of surfacing as an opaque 400 on Load. It is
fetched once at startup and after add/delete — not on the per-tick refresh, since it is
static config.

#### `POST /v1/models/engine`

Register a bridge card for any configured engine.
Request body: `{ "engine": "...", "engine_config": "/path/to/config.json", "name": "optional display name" }`.
Returns `{"id": "<model_id>"}`.

**Error Responses:**
- `400 Bad Request`: Unknown/misconfigured engine or invalid engine config.
- `409 Conflict`: Model id already exists.

#### `POST /v1/models/strata`

**Deprecated** alias of `POST /v1/models/engine` with `engine="strata"`.
Request body: `{ "strata_config": "/path/to/config.json", "name": "optional" }`.

#### `GET /v1/models/{model_id}/port`

Return the raw TCP port of a resident `llama-server` (see also interrupt above location).

**Response Schema:**
```json
{ "port": 9001, "model_id": "model.gguf" }
```

**Error Responses:**
- `404 Not Found`: Model ID not recognized.
- `503 Service Unavailable`: Model known but not loaded.

#### `GET /v1/mmproj`

Every vision projector under `root_dir`, and which model owns it. Added 2026-10-18.
Projectors are not model cards, so this is the only place they are visible at all —
without it a projector paired to nothing was undetectable from the UI.

**Response:**
```json
{ "count": 2, "unpaired": 1,
  "projectors": [
    { "name": "mmproj-F16.gguf", "path": "...", "dir": "...", "size_mb": 1143.0,
      "owner": "Qwen3.8-27B-UD-Q4_K_XL.gguf", "source": "explicit", "enabled": true },
    { "name": "mmproj-F32.gguf", "path": "...", "dir": "...", "size_mb": 1584.4,
      "owner": "", "source": "", "enabled": false }
  ] }
```

`owner` is `""` when unpaired; `source` is `explicit` (sidecar) or `auto` (inferred at
scan); `enabled` is whether the owner will actually pass `--mmproj`. `count` and
`unpaired` let a caller assert nothing is orphaned without walking the list.

#### `GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS /v1/raw/{model_id}/{path}`

Pass any request through to a loaded model's resident `llama-server`. Registered
2026-10-18 — it was documented before but never actually routed, so every call 404'd.

Does **not** JIT-load: returns **503** if the model is not resident (load it first).
Upstream failures surface as **502**. The response is streamed, and hop-by-hop / framing
headers are recomputed rather than replayed. Query strings, request bodies and content
type are forwarded verbatim.

**Error Responses:**
- `404 Not Found`: Model ID not recognized.
- `502 Bad Gateway`: The engine connection failed.
- `503 Service Unavailable`: Model known but not loaded.

> **Security:** this exposes the engine's full HTTP surface, including slot
> save/restore which writes files next to it. It inherits the autoloader's trust model —
> bound to `launcher.host` (127.0.0.1 by default), no authentication. Do not bind the
> launcher to a shared interface if you care about this.

#### `GET /`

WebUI dashboard.

#### `GET /health`, `GET /v1/health`

Simple health check. Returns status and model count.

#### `GET /v1/backends`

List available backend builds.

#### `POST /v1/backend/select`

Select global backend. Request body: `{ "backend": "<folder_name>" }`.

#### `GET /v1/status`

Detailed system status including GPU/RAM info, loaded models, and process states.

#### `GET /v1/settings`

Get current autoloader runtime settings.

**Response Schema:**
```json
{
  "idle_timeout_seconds": 3600,
  "max_loaded_models": 1,
  "auto_save_state": false,
  "poll_interval_seconds": 2,
  "base_port": 9001,
  "host": "127.0.0.1",
  "port": 9124,
  "default_args": "--no-webui --parallel 1 --jinja",
  "selected_backend": "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.0"
}
```

#### `PUT /v1/settings`

Update autoloader runtime settings (partial update, persists to `config.yaml`). Only the fields you include are changed.

**Request Body:** Any subset of the following:
| Field | Type | Constraint | Notes |
|-------|------|------------|-------|
| `idle_timeout_seconds` | number | > 0 | Seconds before idle model is unloaded |
| `max_loaded_models` | integer | >= 1 | Max concurrent loaded models |
| `auto_save_state` | boolean | — | Global default for auto-save/restore |
| `poll_interval_seconds` | number | > 0 | GPU polling interval |
| `default_args` | string | — | Default llama-server CLI args |
| `selected_backend` | string | — | Backend folder name in `backends_dir` |
| `base_port` | integer | 1024–65535 | First port for llama-server instances |

**Error Responses:**
- `400 Bad Request`: Invalid field value (e.g., negative timeout).

**Note:** Changes to `host` or `port` require a restart to take effect. Response returns the updated settings.

#### `WebSocket /ws`

Connect for live status updates. The server pushes a full status snapshot (same format as `/v1/status`) every 2 seconds. Useful for real-time monitoring dashboards.

**Usage:**
```javascript
const ws = new WebSocket("ws://127.0.0.1:9124/ws");
ws.onmessage = (event) => {
  const status = JSON.parse(event.data);
  console.log(status.loaded_models, status.gpu_info);
};
```

---

## Slot State Save/Restore

### What is Slot State?

Slot state refers to the **KV cache** (key-value cache) maintained by llama.cpp during generation. This cache stores the computed keys and values for all tokens processed so far, enabling efficient continuation of conversations without reprocessing previous tokens. The slot state is saved to a binary file and can be restored to a new llama-server process.

### When to Use vs. When Not to Use

**Use slot state persistence when:**
- You want to maintain conversation context across model unload/load cycles
- Running long conversations that exceed idle timeout
- Need to free GPU memory while preserving conversation state
- Testing or debugging stateful interactions

**Do NOT rely on slot state for:**
- **Cross-model compatibility**: State files are model-specific and cannot be restored to a different model.
- **Guaranteed persistence**: If save fails during unload, state may be lost.

### Enabling Auto-Save/Restore Per Model

Set `auto_save_state: true` in the model's sidecar JSON file:

```json
{
  "name": "My Model",
  "auto_save_state": true
}
```

Alternatively, enable globally in `config.yaml`:

```yaml
models:
  auto_save_state: true
```

Global setting can be overridden by per-model config.

### SWA/Hybrid-Memory Models (e.g., Qwen3.6-27B)

For models with Sliding Window Attention (SWA), enabling `auto_save_state` alone is NOT sufficient to get cache reuse after restore. Due to how llama.cpp manages SWA checkpoints in RAM only (not persisted to slot state files), the model will fully reprocess prompts on every request after restore, even though the KV cache was restored.

To fix this for SWA models, add `--swa-full` to the model's args:

```json
{
  "name": "Qwen3.6-27B",
  "auto_save_state": true,
  "args": "--swa-full"
}
```

This allocates a full-size SWA cache instead of the compact windowed variant, bypassing the checkpoint requirement that would otherwise force full reprocessing. Trade-offs:
- **Pro**: KV cache reuse works correctly after state restore (no more 108s reprocessing for 81K token prompts)
- **Con**: Uses more VRAM; negligible impact on per-token speed in practice

This is only needed for SWA/hybrid-memory models; regular models don't need it.

### Manual Save/Restore via Direct llama-server Slot API

You can manually save and restore slot state using the autoloader's proxy endpoints or direct llama-server calls.

**Via autoloader:**
```bash
# Save state
curl -X POST http://127.0.0.1:9124/v1/models/mymodel/state/save \
  -H "Content-Type: application/json" \
  -d '{"label": "mysave"}'

# Load state
curl -X POST http://127.0.0.1:9124/v1/models/mymodel/state/load \
  -H "Content-Type: application/json" \
  -d '{"label": "mysave"}'
```

**Via direct llama-server (after getting port from `/v1/models/{id}`):**
```bash
# Save - action in the query, filename in the BODY
curl -X POST "http://127.0.0.1:9001/slots/0?action=save" \
  -H "Content-Type: application/json" -d '{"filename":"mysave.bin"}'

# Restore
curl -X POST "http://127.0.0.1:9001/slots/0?action=restore" \
  -H "Content-Type: application/json" -d '{"filename":"mysave.bin"}'
```

### Important Limitations

1. **Same-model only**: State files are only compatible with the exact same model architecture and configuration.
2. **Slot 0 only**: Currently only slot 0 (first conversation) is saved/restored.
3. **Path sanitization**: Model IDs are sanitized (path separators removed) for file naming.

Note: If the model is not loaded when restoring state, it will be automatically loaded first.

### Step-by-Step Example: Full Save/Restore Cycle

The following example demonstrates the complete lifecycle. All autoloader endpoints use port **9124**. Direct llama-server calls use the dynamic port returned by the load endpoint (shown as `$PORT`).

**Step 0: Ensure clean state**
```bash
curl -X POST http://127.0.0.1:9124/v1/models/mymodel/unload
```

**Step 1: Load model via autoloader**
```bash
RESP=$(curl -s -X POST http://127.0.0.1:9124/v1/models/mymodel/load)
PORT=$(echo $RESP | jq -r '.port')
echo "Loaded on port $PORT"
```

**Step 2: Reset slot via direct llama-server (optional)**
```bash
curl "http://127.0.0.1:$PORT/slots/0?action=reset"
```

**Step 3: Send initial prompt via autoloader proxy**
```bash
curl -X POST http://127.0.0.1:9124/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mymodel",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 50
  }'
```

**Step 4: Save slot state (via autoloader)**
```bash
curl -X POST http://127.0.0.1:9124/v1/models/mymodel/state/save \
  -H "Content-Type: application/json" \
  -d '{"label": "mysave"}'
```

This saves the KV cache to `save_state_dir` as `{sanitized_model_id}.{sanitized_label}.bin`. In this example: `mymodel.mysave.bin`.

**Alternative: via direct llama-server slot endpoint (using port from load response)**
```bash
curl -X POST "http://127.0.0.1:$PORT/slots/0?action=save" \
  -H "Content-Type: application/json" -d '{"filename":"mymodel.state.bin"}'
```

**Step 5: Unload model**
```bash
curl -X POST http://127.0.0.1:9124/v1/models/mymodel/unload
```

**Step 6: Reload model**
```bash
RESP=$(curl -s -X POST http://127.0.0.1:9124/v1/models/mymodel/load)
PORT=$(echo $RESP | jq -r '.port')
```

**Step 7: Restore slot state (via autoloader)**
```bash
curl -X POST http://127.0.0.1:9124/v1/models/mymodel/state/load \
  -H "Content-Type: application/json" \
  -d '{"label": "mysave"}'
```

After restore, the slot's KV cache contains the previous conversation context. Subsequent requests with matching prompts will leverage llama.cpp's prompt cache to skip reprocessing tokens.

**Alternative: via direct llama-server slot endpoint**
```bash
curl -X POST "http://127.0.0.1:$PORT/slots/0?action=restore" \
  -H "Content-Type: application/json" -d '{"filename":"mymodel.state.bin"}'
```

**Step 8: Continue conversation via autoloader (should use restored KV cache)**
```bash
curl -X POST http://127.0.0.1:9124/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mymodel",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"},
      {"role": "assistant", "content": "I am an AI assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50
  }'
```

The response should show a high number of cached tokens, indicating the KV cache was successfully restored and is being used for the conversation history.

---

## Integration with Agent Cascade

### How Agent Cascade Uses the Autoloader

Agent Cascade (AC) integrates with llama-autoloader as its LLM backend manager. The autoloader provides:

1. **Unified API endpoint** for all agent LLM requests
2. **Automatic model loading** when an agent needs to use a model
3. **Stateless operation** by default (no auto-save/restore) for multi-agent scenarios
4. **LRU eviction** to prevent GPU memory exhaustion when multiple agents compete for models

### Best Practices for Managing Models in Multi-Agent Environment

1. **Set `max_loaded_models` appropriately**:
   - For single-agent use: keep at 1
   - For multi-agent with different model requirements: increase to number of concurrent agents needing different models

2. **Use `pinned: true` for critical models**:
   - Mark frequently-used models as pinned to prevent eviction
   - Useful for primary assistant models that need to stay resident

3. **Configure `idle_timeout_seconds` based on agent activity patterns**:
   - Short timeouts (60-300s) for bursty usage
   - Longer timeouts (1800-3600s) for sustained activity

4. **Consider `auto_save_state: false` globally** in multi-agent setups unless conversation persistence is explicitly needed per agent.

5. **Use model tags** for filtering and discovery in the WebUI.

### Handling Model Switching Between Agents

When agents need to switch models:

1. **Direct API calls**: Agents can call `/v1/models/{id}/load` to ensure a specific model is loaded before making inference requests.

2. **Automatic loading**: Most agents simply specify the model in their request; the autoloader handles JIT loading transparently.

3. **Load ordering**: The autoloader loads models serially to avoid port conflicts, but other requests are queued.

4. **Eviction awareness**: If `max_loaded_models` is reached, the least-recently-used unpinned model will be evicted to make room.

5. **Best practice**: Design agents to use a consistent primary model when possible to reduce switching overhead. Use pinned models for critical paths.

### Example Agent Configuration

```yaml
# config.yaml for multi-agent setup
models:
  max_loaded_models: 2
  idle_timeout_seconds: 1800
  auto_save_state: false

# Model sidecar for primary assistant
primary-model.gguf.json:
  name: "Primary Assistant"
  pinned: true
  default: true

# Model sidecar for secondary specialized model
specialist-model.gguf.json:
  name: "Specialist Model"
  pinned: false
```

---

## Summary

llama-autoloader provides a robust, production-ready solution for managing multiple llama.cpp models with minimal operational overhead. Its JIT loading, automatic state persistence, and OpenAI-compatible API make it ideal for both single-agent and multi-agent deployments. The comprehensive configuration options and detailed WebUI dashboard provide visibility and control over the entire system.