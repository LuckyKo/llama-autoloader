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
- **Port Management**: Dynamically assigns available ports starting from `base_port`
- **Session State Persistence**: Can auto-save/restore KV cache state across load/unload cycles
- **Vision Module Auto-Pairing**: Automatically detects and associates `mmproj` vision projector files with models
- **Multi-Backend Support**: Supports different llama.cpp builds (e.g., CUDA, AVX2) with per-model overrides
- **OpenAI-Compatible API**: Full `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` endpoints

---

## Configuration

### config.yaml Structure

```yaml
gpu:
  poll_interval_seconds: 2

launcher:
  base_port: 9001
  host: 127.0.0.1
  port: 9123

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
- **launcher.port**: Port for the autoloader's REST API and proxy endpoints (default: 9123)
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
| `default` | bool | `false` | Use this model when API requests omit the `model` field |
| `pinned` | bool | `false` | Never auto-unload this model (exempt from LRU eviction) |
| `auto_save_state` | bool | `false` | Auto save/restore KV cache state on model unload/load |
| `backend` | string | `""` | Force a specific backend build for this model (folder name in `backends/`) |
| `use_mmproj` | bool | `true` | Enable vision projector if an `mmproj` file is detected |
| `mmproj_file` | string | `""` | Explicit vision projector filename (auto-detected if empty) |
| `estimated_vram_mb` | int | `0` | Estimated VRAM usage hint (display only) |
| `tags` | string[] | `[]` | Tags for filtering in the WebUI |

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

**Base URL:** `http://127.0.0.1:9123`

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
      "state_path": null
    }
  ]
}
```

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

#### `PUT /v1/models/{model_id}/config`

Update a model's configuration. Accepts partial updates.

**Request Body:** Any subset of ModelConfig fields.

**Response Schema:** Updated ModelConfig as dict.

**Error Responses:**
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

Unload a model. Forces the llama-server process to terminate.

**Response Schema:**
```json
{
  "id": "model.gguf",
  "unloaded": true
}
```

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
  "unloaded_all": true
}
```

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

Restore slot 0 KV cache state from a saved file. After restore, the autoloader sends a minimal request (`_warm_slot_cache()`) to prime the checkpoint cache — this is required for llama.cpp v2.26.0+.

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

For advanced state management, you can access the underlying llama-server's native slot endpoints through the autoloader's proxy route `/v1/raw/{model_id}/{path}`. These requests are forwarded directly to the loaded model's llama-server process.

**Slot-specific endpoints:**
- `GET/POST /v1/raw/{model_id}/slots/0?action=save&filename=state.bin`
- `GET/POST /v1/raw/{model_id}/slots/0?action=restore&filename=state.bin`

**Multi-slot endpoints (specify slot via query param):**
- `GET/POST /v1/raw/{model_id}/slots?action=save&id_slot=0&filename=state.bin`
- `GET/POST /v1/raw/{model_id}/slots?action=restore&id_slot=0&filename=state.bin`

Both query parameters and JSON body formats are supported, matching llama-server's native API. Examples:

```bash
# Via autoloader proxy (query param format)
curl "http://127.0.0.1:9123/v1/raw/mymodel/slots/0?action=save&filename=test.bin"

# Via autoloader proxy (JSON body format — POST only)
curl -X POST http://127.0.0.1:9123/v1/raw/mymodel/slots/0 \
  -H "Content-Type: application/json" \
  -d '{"action":"save","filename":"test.bin"}'

# Direct llama-server (if you know the port)
curl "http://127.0.0.1:9001/slots/0?action=save&filename=test.bin"
```

### Other Endpoints

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
  "port": 9123,
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
const ws = new WebSocket("ws://127.0.0.1:9123/ws");
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
- **Prompt cache population**: Restored KV cache does NOT populate the separate RAM-based prompt cache (`cache_prompt`). Resending the exact same prompt after restore will reprocess tokens.
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

### Manual Save/Restore via Direct llama-server Slot API

You can manually save and restore slot state using the autoloader's proxy endpoints or direct llama-server calls.

**Via autoloader:**
```bash
# Save state
curl -X POST http://127.0.0.1:9123/v1/models/mymodel/state/save \
  -H "Content-Type: application/json" \
  -d '{"label": "mysave"}'

# Load state
curl -X POST http://127.0.0.1:9123/v1/models/mymodel/state/load \
  -H "Content-Type: application/json" \
  -d '{"label": "mysave"}'
```

**Via direct llama-server (after getting port from `/v1/models/{id}`):**
```bash
# Save
curl -X POST "http://127.0.0.1:9001/slots/0?action=save&filename=mysave.bin"

# Restore
curl -X POST "http://127.0.0.1:9001/slots/0?action=restore&filename=mysave.bin"
```

### Important Limitations

1. **Same-model only**: State files are only compatible with the exact same model architecture and configuration.
2. **No prompt cache**: Restored KV cache does not populate RAM-based prompt cache. The first generation after restore will use the KV cache for history, but resending the exact same prompt will reprocess tokens.
3. **Slot 0 only**: Currently only slot 0 (first conversation) is saved/restored.
4. **Model must be loaded**: State can only be restored to a running llama-server instance.
5. **Path sanitization**: Model IDs are sanitized (path separators removed) for file naming.

### Step-by-Step Example: Full Save/Restore Cycle

The following example demonstrates the complete lifecycle. All autoloader endpoints use port **9123**. Direct llama-server calls use the dynamic port returned by the load endpoint (shown as `$PORT`).

**Step 0: Ensure clean state**
```bash
curl -X POST http://127.0.0.1:9123/v1/models/mymodel/unload
```

**Step 1: Load model via autoloader**
```bash
RESP=$(curl -s -X POST http://127.0.0.1:9123/v1/models/mymodel/load)
PORT=$(echo $RESP | jq -r '.port')
echo "Loaded on port $PORT"
```

**Step 2: Reset slot via direct llama-server (optional)**
```bash
curl "http://127.0.0.1:$PORT/slots/0?action=reset"
```

**Step 3: Send initial prompt via autoloader proxy**
```bash
curl -X POST http://127.0.0.1:9123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mymodel",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 50
  }'
```

**Step 4: Save slot state (via autoloader)**
```bash
curl -X POST http://127.0.0.1:9123/v1/models/mymodel/state/save \
  -H "Content-Type: application/json" \
  -d '{"label": "mysave"}'
```

This saves the KV cache to `save_state_dir` as `{sanitized_model_id}.{sanitized_label}.bin`. In this example: `mymodel.mysave.bin`.

**Alternative: via direct llama-server slot endpoint (using port from load response)**
```bash
curl -X POST "http://127.0.0.1:$PORT/slots/0?action=save&filename=mymodel.state.bin"
```

**Step 5: Unload model**
```bash
curl -X POST http://127.0.0.1:9123/v1/models/mymodel/unload
```

**Step 6: Reload model**
```bash
RESP=$(curl -s -X POST http://127.0.0.1:9123/v1/models/mymodel/load)
PORT=$(echo $RESP | jq -r '.port')
```

**Step 7: Restore slot state (via autoloader)**
```bash
curl -X POST http://127.0.0.1:9123/v1/models/mymodel/state/load \
  -H "Content-Type: application/json" \
  -d '{"label": "mysave"}'
```

After restore, the autoloader sends a minimal request (`_warm_slot_cache()`) to prime the checkpoint cache — this is required for llama.cpp v2.26.0+.

**Alternative: via direct llama-server slot endpoint**
```bash
curl -X POST "http://127.0.0.1:$PORT/slots/0?action=restore&filename=mymodel.state.bin"
```

**Step 8: Continue conversation via autoloader (should use restored KV cache)**
```bash
curl -X POST http://127.0.0.1:9123/v1/chat/completions \
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