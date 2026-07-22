# llama-autoloader

A **JIT (Just-In-Time) model loader** and **OpenAI-compatible proxy** for [llama.cpp](https://github.com/ggerganov/llama.cpp). Drop your `.gguf` files into a directory and llama-autoloader will discover them, spawn `llama-server` processes on demand, manage GPU memory, and expose a unified OpenAI-style API — all from a single endpoint.

## Features

- **JIT Model Loading** — Models are loaded on first API request and auto-unloaded after idle timeout
- **OpenAI-Compatible API** — Drop-in replacement for `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- **Multi-Backend Support** — Dynamically switch between different `llama-server` builds with per-model overrides
- **Vision Module Auto-Pairing** — Automatically detects and pairs `mmproj` vision projector files with their models
- **Session State Persistence** — Auto-save and restore KV cache state across model switches
- **LRU Auto-Eviction** — Automatically unloads least-recently-used models to stay within GPU memory limits
- **Live WebUI Dashboard** — Real-time GPU/RAM monitoring, model management, and one-click controls
- **Per-Model Configuration** — Sidecar `.gguf.json` files for custom launch args, context size, GPU layers, etc.

---

## Quick Start

### Prerequisites

- Python 3.10+
- One or more [llama.cpp](https://github.com/ggerganov/llama.cpp) builds (place in `./backends/`)
- NVIDIA GPU with `nvidia-smi` (optional, for GPU monitoring)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml`:

```yaml
launcher:
  host: 127.0.0.1
  port: 9123
  base_port: 9001          # first port assigned to spawned llama-servers

llama_server:
  binary: llama-server     # fallback binary name/path
  default_args: "--no-webui --parallel 1 --jinja"
  backends_dir: ./backends
  selected_backend: ""     # global backend folder name (empty = auto-detect)

models:
  root_dir: "/path/to/your/models"   # directory to scan recursively for *.gguf
  idle_timeout_seconds: 300
  max_loaded_models: 1               # LRU eviction threshold
  auto_save_state: false             # auto save/restore session state on switch
  save_state_dir: ./states

gpu:
  poll_interval_seconds: 2
```

### Running

```bash
python server.py
# or
python server.py --host 127.0.0.1 --port 9123
```

Open **http://127.0.0.1:9123** in your browser for the WebUI dashboard.

---

## Directory Structure

```
llama-autoloader/
├── server.py            # Main server application
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── start.bat            # Windows quick-start script
├── backends/            # llama-server builds (auto-scanned)
│   ├── llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.21.0/
│   │   └── llama-server.exe
│   └── llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.23.1/
│       └── llama-server.exe
├── states/              # Saved KV cache session states
├── static/
│   └── index.html       # WebUI dashboard
└── /path/to/models/     # Your .gguf model files (configured in config.yaml)
    ├── MyModel-Q4_K_M.gguf
    ├── MyModel-Q4_K_M.gguf.json    # Optional sidecar config
    └── mmproj-MyModel-F16.gguf     # Vision projector (auto-paired)
```

---

## Per-Model Sidecar Config

Each model can have an optional sidecar JSON file (`<model>.gguf.json`) placed next to the `.gguf` file:

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

| Field              | Type     | Default | Description                                                                 |
|--------------------|----------|---------|-----------------------------------------------------------------------------|
| `name`             | string   | `""`    | Human-readable display name                                                 |
| `description`      | string   | `""`    | Model description                                                           |
| `args`             | string   | `""`    | Extra CLI arguments appended to `llama-server`                              |
| `ctx_size`         | int      | `8192`  | Context window size (`--ctx-size`)                                          |
| `n_gpu_layers`     | int      | `999`   | GPU layers to offload (`--n-gpu-layers`)                                    |
| `default`          | bool     | `false` | Use this model when API requests omit the `model` field                     |
| `pinned`           | bool     | `false` | Never auto-unload this model (exempt from LRU eviction)                     |
| `auto_save_state`  | bool     | `false` | Auto save/restore KV cache state on model unload/load                       |
| `backend`          | string   | `""`    | Force a specific backend build for this model (folder name in `backends/`)   |
| `use_mmproj`       | bool     | `true`  | Enable vision projector if an `mmproj` file is detected                     |
| `mmproj_file`      | string   | `""`    | Explicit vision projector filename (auto-detected if empty)                 |
| `estimated_vram_mb`| int      | `0`     | Estimated VRAM usage hint (for display only)                                |
| `tags`             | string[] | `[]`    | Tags for filtering in the WebUI                                             |

---

## API Reference

Base URL: `http://127.0.0.1:9123`

All endpoints return JSON unless otherwise noted. Models are automatically loaded (JIT) on first inference request if not already loaded.

---

### OpenAI-Compatible Endpoints

These endpoints are fully compatible with the OpenAI API format. Point any OpenAI-compatible client at `http://127.0.0.1:9123` and it will work.

#### `POST /v1/chat/completions`

Chat completion (streaming and non-streaming). Auto-loads the requested model if needed.

```bash
curl http://127.0.0.1:9123/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MyModel-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

> **Note:** If `model` is omitted, the model marked as `default: true` will be used.

#### `POST /v1/completions`

Text completion (OpenAI legacy format).

```bash
curl http://127.0.0.1:9123/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MyModel-Q4_K_M.gguf",
    "prompt": "Once upon a time",
    "max_tokens": 100
  }'
```

#### `POST /v1/embeddings`

Generate embeddings for input text.

```bash
curl http://127.0.0.1:9123/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MyModel-Q4_K_M.gguf",
    "input": "The quick brown fox"
  }'
```

---

### Model Management

#### `GET /v1/models`

List all discovered models with their current status.

```bash
curl http://127.0.0.1:9123/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "MyModel-Q4_K_M.gguf",
      "object": "model",
      "created": 1753182000,
      "owned_by": "llama-autoloader",
      "loaded": true,
      "ready": true,
      "port": 9001,
      "tags": ["chat", "7B"],
      "size_mb": 4200.5,
      "default": false
    }
  ]
}
```

#### `GET /v1/models/{model_id}`

Get detailed info and full config for a specific model.

```bash
curl http://127.0.0.1:9123/v1/models/MyModel-Q4_K_M.gguf
```

**Response:**
```json
{
  "id": "MyModel-Q4_K_M.gguf",
  "object": "model",
  "config": {
    "name": "My Model 7B",
    "ctx_size": 8192,
    "n_gpu_layers": 999,
    "backend": "",
    "use_mmproj": true,
    "mmproj_file": "mmproj-MyModel-F16.gguf",
    "..."
  },
  "loaded": true
}
```

#### `PUT /v1/models/{model_id}/config`

Update a model's sidecar config. Only include the fields you want to change.

```bash
curl -X PUT http://127.0.0.1:9123/v1/models/MyModel-Q4_K_M.gguf/config \
  -H 'Content-Type: application/json' \
  -d '{
    "ctx_size": 16384,
    "n_gpu_layers": 40,
    "tags": ["chat", "large-ctx"],
    "pinned": true
  }'
```

**Updatable fields:** `name`, `description`, `args`, `ctx_size`, `n_gpu_layers`, `default`, `pinned`, `auto_save_state`, `backend`, `use_mmproj`, `mmproj_file`, `estimated_vram_mb`, `tags`

---

### Model Loading / Unloading

#### `POST /v1/models/{model_id}/load`

Explicitly load (pre-warm) a model without sending an inference request.

```bash
curl -X POST http://127.0.0.1:9123/v1/models/MyModel-Q4_K_M.gguf/load
```

**Response:**
```json
{
  "id": "MyModel-Q4_K_M.gguf",
  "port": 9001,
  "pid": 12345,
  "ready": true
}
```

#### `POST /v1/models/{model_id}/unload`

Unload a model and terminate its `llama-server` process.

```bash
curl -X POST http://127.0.0.1:9123/v1/models/MyModel-Q4_K_M.gguf/unload
```

#### `POST /v1/unload_all`

Unload all currently loaded models.

```bash
curl -X POST http://127.0.0.1:9123/v1/unload_all
```

#### `POST /v1/scan`

Re-scan the models directory for new/removed `.gguf` files.

```bash
curl -X POST http://127.0.0.1:9123/v1/scan
```

---

### Vision Module Toggle

#### `POST /v1/models/{model_id}/vision/toggle`

Toggle the vision projector (`--mmproj`) on or off for a model.

```bash
curl -X POST http://127.0.0.1:9123/v1/models/MyModel-Q4_K_M.gguf/vision/toggle
```

**Response:**
```json
{
  "id": "MyModel-Q4_K_M.gguf",
  "use_mmproj": false,
  "mmproj_file": "mmproj-MyModel-F16.gguf"
}
```

> **Note:** The model must be reloaded for this change to take effect on a running instance.

---

### Session State Management

Save and restore KV cache state (slot 0) for fast context resumption.

#### `POST /v1/models/{model_id}/state/save`

```bash
curl -X POST http://127.0.0.1:9123/v1/models/MyModel-Q4_K_M.gguf/state/save \
  -H 'Content-Type: application/json' \
  -d '{"label": "my-session"}'
```

#### `POST /v1/models/{model_id}/state/load`

```bash
curl -X POST http://127.0.0.1:9123/v1/models/MyModel-Q4_K_M.gguf/state/load \
  -H 'Content-Type: application/json' \
  -d '{"label": "my-session"}'
```

#### `GET /v1/models/{model_id}/state`

List all saved state labels for a model.

```bash
curl http://127.0.0.1:9123/v1/models/MyModel-Q4_K_M.gguf/state
```

**Response:**
```json
{
  "id": "MyModel-Q4_K_M.gguf",
  "labels": ["default", "my-session"]
}
```

---

### Backend Management

#### `GET /v1/backends`

List all available `llama-server` builds detected in the `backends/` directory.

```bash
curl http://127.0.0.1:9123/v1/backends
```

**Response:**
```json
{
  "backends": [
    {
      "id": "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.21.0",
      "name": "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.21.0",
      "path": "N:\\...\\backends\\llama.cpp-...-2.21.0\\llama-server.exe"
    },
    {
      "id": "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.23.1",
      "name": "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.23.1",
      "path": "N:\\...\\backends\\llama.cpp-...-2.23.1\\llama-server.exe"
    }
  ],
  "selected_backend": "",
  "default_binary": "llama-server",
  "resolved_global_binary": "N:\\...\\backends\\llama.cpp-...-2.21.0\\llama-server.exe"
}
```

#### `POST /v1/backend/select`

Set the global active backend. All models without a per-model override will use this build.

```bash
curl -X POST http://127.0.0.1:9123/v1/backend/select \
  -H 'Content-Type: application/json' \
  -d '{"backend": "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.23.1"}'
```

Pass `{"backend": ""}` to reset to auto-detect.

**Backend Resolution Order:**
1. Per-model `backend` field (from sidecar config)
2. Global `selected_backend` (set via this endpoint or `config.yaml`)
3. First available build scanned from `backends/`
4. Fallback `binary` value from `config.yaml`

---

### System Status & Health

#### `GET /health` · `GET /v1/health`

Simple health check.

```bash
curl http://127.0.0.1:9123/health
```

**Response:**
```json
{
  "status": "ok",
  "models_loaded": 1
}
```

#### `GET /v1/status`

Full system status including GPU info, RAM, loaded models, backends, and per-model resource usage.

```bash
curl http://127.0.0.1:9123/v1/status
```

**Response:**
```json
{
  "launcher": {
    "host": "127.0.0.1",
    "port": 9123,
    "binary": "N:\\...\\llama-server.exe",
    "selected_backend": "",
    "backends": [...]
  },
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "total_mb": 24564,
      "used_mb": 5120,
      "free_mb": 19444,
      "utilization_pct": 12.0
    }
  ],
  "ram": {
    "total_mb": 32768,
    "used_mb": 12000,
    "free_mb": 20768,
    "pct": 36.6
  },
  "models_loaded": 1,
  "models_total": 26,
  "idle_timeout": 300,
  "max_loaded_models": 1,
  "per_model": {
    "MyModel-Q4_K_M.gguf": {"ram_mb": 350, "vram_mb": 4800}
  },
  "uptime_models": [
    {
      "id": "MyModel-Q4_K_M.gguf",
      "port": 9001,
      "pid": 12345,
      "uptime_s": 142.5,
      "last_used_s_ago": 3.2,
      "ready": true
    }
  ]
}
```

---

### Raw Pass-Through

#### `{METHOD} /v1/raw/{model_id}/{path}`

Forward any request directly to a loaded model's `llama-server` instance. Useful for accessing native llama.cpp endpoints like `/slots`, `/tokenize`, `/detokenize`.

```bash
# Tokenize text
curl -X POST http://127.0.0.1:9123/v1/raw/MyModel-Q4_K_M.gguf/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"content": "Hello world"}'

# Get slot info
curl http://127.0.0.1:9123/v1/raw/MyModel-Q4_K_M.gguf/slots
```

---

### WebSocket Live Updates

#### `WS /ws`

Connect via WebSocket for real-time status updates (pushed every 1 second). The payload is identical to `GET /v1/status`.

```javascript
const ws = new WebSocket('ws://127.0.0.1:9123/ws');
ws.onmessage = (ev) => {
  const status = JSON.parse(ev.data);
  console.log('GPU usage:', status.gpus[0]?.used_mb, 'MB');
  console.log('Models loaded:', status.models_loaded);
};
```

---

## Integration Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9123/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="MyModel-Q4_K_M.gguf",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### SillyTavern / Text Generation WebUI

Set the API endpoint to:
```
http://127.0.0.1:9123
```
API type: **OpenAI-compatible**. No API key required.

### Continue.dev (VS Code)

```json
{
  "models": [{
    "title": "Local LLM",
    "provider": "openai",
    "model": "MyModel-Q4_K_M.gguf",
    "apiBase": "http://127.0.0.1:9123/v1",
    "apiKey": "not-needed"
  }]
}
```

---

## How It Works

1. **Scan** — On startup, recursively scans `models.root_dir` for `.gguf` files (excluding `mmproj` vision modules)
2. **JIT Load** — When an API request arrives for a model that isn't loaded, spawns a `llama-server` subprocess on an available port
3. **Proxy** — Forwards the request to the model's `llama-server` process and streams the response back
4. **Idle Reaper** — A background task checks every 10 seconds and unloads models that have been idle longer than `idle_timeout_seconds`
5. **LRU Eviction** — If `max_loaded_models` is reached, the least-recently-used unpinned model is unloaded before loading a new one

---

## License

MIT
