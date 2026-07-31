"""
llama-autoloader: a JIT model loader + OpenAI-compatible proxy for llama.cpp.

- Scans a root directory for *.gguf models.
- Each model may have a sidecar *.gguf.json with config (args, ctx, etc.).
- Spawns llama-server subprocesses on demand (JIT) and unloads when idle.
- Proxies OpenAI-style requests to the right subprocess based on `model`.
- Exposes LM-Studio-like API plus extensions (load/unload/save-state/load-state).
- Sleek WebUI dashboard at /.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import time
import logging
import platform
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx
import psutil
import yaml
from fastapi import (
    FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import gguf

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Enable uvicorn access logging so all requests are visible in the console
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
log = logging.getLogger("autoloader")

def _safe_int(val: str, default: int = 0) -> int:
    try:
        cleaned = val.strip().split()[0] if val else ""
        return int(cleaned)
    except Exception:
        return default

def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        cleaned = val.strip().split()[0] if val else ""
        return float(cleaned)
    except Exception:
        return default

def _read_gguf_name(gguf_path) -> Optional[str]:
    """Read general.name from GGUF file metadata. Returns None on failure."""
    try:
        reader = gguf.GGUFReader(str(gguf_path))
        field = reader.get_field('general.name')
        if field is None:
            return None
        return field.contents()
    except Exception as e:
        log.debug(f"Failed to read GGUF name from {gguf_path}: {e}")
        return None

# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Sidecar JSON config stored next to each .gguf file."""
    name: str = ""                  # human-readable name
    description: str = ""
    args: str = ""                  # extra args appended to llama-server
    ctx_size: int = 8192
    n_gpu_layers: int = 999
    default: bool = False
    pinned: bool = False            # never auto-unload
    auto_save_state: bool = False   # auto save/restore slot 0 session state
    backend: str = ""               # backend build override folder name (empty = use global default)
    use_mmproj: bool = True         # enable --mmproj vision projector if present
    mmproj_file: str = ""           # path or filename of associated mmproj file
    estimated_vram_mb: int = 0      # hint; 0 = unknown
    tags: List[str] = field(default_factory=list)
    gguf_name: str = ""             # embedded name from GGUF metadata

    @classmethod
    def default_for(cls, gguf_path: Path) -> "ModelConfig":
        return cls(
            name=gguf_path.stem,
            description="",
            args="",
            ctx_size=8192,
            n_gpu_layers=999,
            default=False,
            pinned=False,
            auto_save_state=False,
            backend="",
            use_mmproj=True,
            mmproj_file="",
            estimated_vram_mb=0,
            tags=[],
            gguf_name="",
        )

    @classmethod
    def load(cls, gguf_path: Path) -> "ModelConfig":
        sidecar = gguf_path.with_suffix(gguf_path.suffix + ".json")
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text())
                # Merge with defaults so missing fields don't break things.
                base = cls.default_for(gguf_path)
                for k, v in data.items():
                    if hasattr(base, k):
                        setattr(base, k, v)
                return base
            except Exception as e:
                log.warning(f"Failed to read sidecar {sidecar}: {e}")
        return cls.default_for(gguf_path)

    def save(self, gguf_path: Path) -> None:
        sidecar = gguf_path.with_suffix(gguf_path.suffix + ".json")
        sidecar.write_text(json.dumps(asdict(self), indent=2))

    def to_launch_args(self, model_path: Path, port: int, extra_default_args: str, mmproj_full_path: Optional[Path] = None, slot_save_dir: Optional[Path] = None, model_id: Optional[str] = None) -> List[str]:
        """Build argv for llama-server."""
        argv = [
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
        ]
        if model_id:
            argv += ["--alias", model_id]
        if self.use_mmproj and mmproj_full_path and mmproj_full_path.exists():
            argv += ["--mmproj", str(mmproj_full_path)]
        if slot_save_dir:
            argv += ["--slot-save-path", str(slot_save_dir)]
        if extra_default_args:
            argv += extra_default_args.split()
        if self.args:
            argv += self.args.split()
        return argv


@dataclass
class LoadedModel:
    model_id: str
    gguf_path: Path
    config: ModelConfig
    port: int
    process: subprocess.Popen
    last_used: float = field(default_factory=time.time)
    starting_at: float = field(default_factory=time.time)
    ready: bool = False
    pid: int = 0
    state_path: Optional[Path] = None

    def touch(self):
        self.last_used = time.time()


@dataclass
class GPUInfo:
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_pct: float


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------
class ModelManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.root_dir = Path(cfg["models"]["root_dir"]).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.save_state_dir = Path(cfg["models"].get("save_state_dir", "./states")).resolve()
        self.save_state_dir.mkdir(parents=True, exist_ok=True)
        self.idle_timeout = cfg["models"].get("idle_timeout_seconds", 300)
        self.max_loaded_models = cfg["models"].get("max_loaded_models", 1)
        self.default_auto_save_state = cfg["models"].get("auto_save_state", False)
        self.binary = cfg["llama_server"]["binary"]
        self.backends_dir = Path(cfg["llama_server"].get("backends_dir", "./backends")).resolve()
        self.selected_backend = cfg["llama_server"].get("selected_backend", "")
        self.default_args = cfg["llama_server"].get("default_args", "--cache-ram 16384 --kv-unified")
        self.host = cfg["launcher"].get("host", "127.0.0.1")
        self.port = cfg["launcher"].get("port", 9123)
        self.base_port = cfg["launcher"].get("base_port", 9001)

        # Warn if system RAM is below safe threshold for default cache size
        try:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            if total_ram_gb < 32:
                log.warning(f"System has {total_ram_gb:.1f} GB RAM. Default prompt cache of 16 GiB may cause memory pressure. Consider reducing --cache-ram in Global Settings.")
        except Exception:
            pass
        self.next_port = self.base_port

        self.models: Dict[str, ModelConfig] = {}      # id -> config
        self.gguf_paths: Dict[str, Path] = {}         # id -> path
        self.mmproj_paths: Dict[str, Path] = {}       # id -> mmproj path
        self.loaded: Dict[str, LoadedModel] = {}      # id -> loaded instance
        self._lock = asyncio.Lock()
        self._loading_tasks: Dict[str, asyncio.Task] = {}
        self._stop = False
        self._bg_task: Optional[asyncio.Task] = None
        self._model_sizes: Dict[str, float] = {}

        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        self._poll_interval = cfg.get("gpu", {}).get("poll_interval_seconds", 2)
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_time: float = 0.0

    # ---------------- config validation ----------------
    @staticmethod
    def validate_config(cfg: Dict[str, Any]) -> None:
        """Validate config at startup to catch issues early."""
        # launcher section
        launcher = cfg.get("launcher", {})
        if not launcher:
            raise ValueError("Config missing 'launcher' section")
        host = launcher.get("host")
        port = launcher.get("port")
        if not host or not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError(f"Config 'launcher' requires valid host and port (1-65535), got host={host!r} port={port!r}")
        base_port = launcher.get("base_port", 9001)
        if not isinstance(base_port, int) or base_port < 1:
            raise ValueError(f"Config 'launcher.base_port' must be a positive int, got {base_port!r}")

        # models section
        models = cfg.get("models", {})
        if not models:
            raise ValueError("Config missing 'models' section")
        root_dir = models.get("root_dir")
        if not root_dir:
            raise ValueError("Config 'models.root_dir' is required")
        idle_timeout = models.get("idle_timeout_seconds", 300)
        if not isinstance(idle_timeout, (int, float)) or (isinstance(idle_timeout, float) and math.isnan(idle_timeout)) or idle_timeout <= 0:
            raise ValueError(f"Config 'models.idle_timeout_seconds' must be positive, got {idle_timeout!r}")
        max_loaded = models.get("max_loaded_models", 1)
        if not isinstance(max_loaded, int) or max_loaded < 1:
            raise ValueError(f"Config 'models.max_loaded_models' must be >= 1, got {max_loaded!r}")

        # gpu section
        gpu = cfg.get("gpu", {})
        poll_interval = gpu.get("poll_interval_seconds", 2)
        if not isinstance(poll_interval, (int, float)) or poll_interval <= 0:
            raise ValueError(f"Config 'gpu.poll_interval_seconds' must be positive, got {poll_interval!r}")

        # llama_server section
        ls = cfg.get("llama_server", {})
        if not ls:
            raise ValueError("Config missing 'llama_server' section")
        if not ls.get("binary"):
            raise ValueError("Config 'llama_server.binary' is required")

    # ---------------- backend discovery & resolution ----------------
    def list_backends(self) -> List[Dict[str, Any]]:
        out = []
        if self.backends_dir.exists() and self.backends_dir.is_dir():
            for p in sorted(self.backends_dir.iterdir()):
                if p.is_dir():
                    exe = p / "llama-server.exe"
                    if not exe.exists():
                        exe = p / "llama-server"
                    if exe.exists():
                        out.append({
                            "id": p.name,
                            "name": p.name,
                            "path": str(exe),
                        })
        return out

    def resolve_binary(self, model_cfg: Optional[ModelConfig] = None) -> str:
        # Priority 1: Per-model backend override
        if model_cfg and model_cfg.backend:
            b_dir = self.backends_dir / model_cfg.backend
            exe = b_dir / "llama-server.exe"
            if not exe.exists():
                exe = b_dir / "llama-server"
            if exe.exists():
                return str(exe)

        # Priority 2: Global selected backend
        if self.selected_backend:
            b_dir = self.backends_dir / self.selected_backend
            exe = b_dir / "llama-server.exe"
            if not exe.exists():
                exe = b_dir / "llama-server"
            if exe.exists():
                return str(exe)

        # Priority 3: First available scanned backend
        scanned = self.list_backends()
        if scanned:
            return scanned[0]["path"]

        # Priority 4: Fallback binary setting
        return self.binary

    # ---------------- model discovery & resolution ----------------
    def scan(self) -> List[str]:
        """Re-scan root_dir for .gguf files, filtering out mmproj files and pairing them as vision modules."""
        all_ggufs = sorted(self.root_dir.rglob("*.gguf"))
        model_paths = []
        mmproj_paths = []

        for p in all_ggufs:
            if "mmproj" in p.name.lower():
                mmproj_paths.append(p)
            else:
                model_paths.append(p)

        found = {}
        self.gguf_paths = {}
        self.mmproj_paths = {}
        self._model_sizes = {}

        for p in model_paths:
            mid = p.name  # use filename as id
            cfg = ModelConfig.load(p)
            found[mid] = cfg
            self.gguf_paths[mid] = p
            if mid in self.loaded:
                self.loaded[mid].config = cfg

        # Auto-pair mmproj files with sibling models
        for mid, p in self.gguf_paths.items():
            cfg = found[mid]
            paired_mmproj = None

            # 1. Check explicit mmproj_file setting
            if cfg.mmproj_file:
                candidate = Path(cfg.mmproj_file)
                if candidate.exists() and candidate.is_file():
                    paired_mmproj = candidate
                else:
                    # Fall back to filename-only lookup in model dir and root dir
                    mmproj_name = candidate.name
                    target_p = p.parent / mmproj_name
                    if target_p.exists() and target_p.is_file():
                        paired_mmproj = target_p
                    else:
                        target_p2 = self.root_dir / mmproj_name
                        if target_p2.exists() and target_p2.is_file():
                            paired_mmproj = target_p2

            # 2. Look for sibling mmproj files in the same directory
            if not paired_mmproj:
                sibling_mmprojs = [mp for mp in mmproj_paths if mp.parent == p.parent]
                if len(sibling_mmprojs) == 1:
                    paired_mmproj = sibling_mmprojs[0]
                elif len(sibling_mmprojs) > 1:
                    p_stem = p.stem.lower()
                    best_match = None
                    best_score = -1
                    for mp in sibling_mmprojs:
                        mp_stem = mp.stem.lower()
                        common_len = sum(1 for a, b in zip(p_stem, mp_stem) if a == b)
                        if common_len > best_score:
                            best_score = common_len
                            best_match = mp
                    paired_mmproj = best_match

            # 3. Fallback: single mmproj in root_dir
            if not paired_mmproj and len(mmproj_paths) == 1:
                paired_mmproj = mmproj_paths[0]

            if paired_mmproj:
                self.mmproj_paths[mid] = paired_mmproj
                if not cfg.mmproj_file:
                    cfg.mmproj_file = paired_mmproj.name

        self.models = found
        # Cache model file sizes
        for mid in self.gguf_paths:
            path = self.gguf_paths[mid]
            try:
                self._model_sizes[mid] = path.stat().st_size / 1024 / 1024
            except Exception:
                self._model_sizes[mid] = 0.0
        log.info(f"Scan complete: {len(self.models)} models found (filtered {len(mmproj_paths)} mmproj vision modules)")
        return list(self.models.keys())

    async def resolve_model_id(self, model_id_or_alias: Optional[str]) -> Optional[str]:
        async with self._lock:
            if model_id_or_alias:
                if model_id_or_alias in self.models:
                    return model_id_or_alias
                # Match without .gguf suffix, stem, or lowercased name
                target_lower = model_id_or_alias.lower()
                for mid, cfg in self.models.items():
                    stem = Path(mid).stem.lower()
                    if target_lower == stem or target_lower == mid.lower():
                        return mid
                    if cfg.name and target_lower == cfg.name.lower():
                        return mid
                    if cfg.gguf_name and target_lower == cfg.gguf_name.lower():
                        return mid

            # Fallback resolution (for generic model names, missing model param, or unmatched aliases)
            if self.loaded:
                ready_loaded = [m for m, lm in self.loaded.items() if lm.ready]
                if ready_loaded:
                    model_id = ready_loaded[0]
                    log.debug(f"Model '{model_id_or_alias}' not found in registry; falling back to '{model_id}'")
                    return model_id
                model_id = next(iter(self.loaded.keys()))
                log.debug(f"Model '{model_id_or_alias}' not found in registry; falling back to '{model_id}'")
                return model_id
            for mid, cfg in self.models.items():
                if cfg.default:
                    log.debug(f"Model '{model_id_or_alias}' not found in registry; falling back to default '{mid}'")
                    return mid
            if len(self.models) == 1:
                model_id = next(iter(self.models.keys()))
                log.debug(f"Model '{model_id_or_alias}' not found in registry; falling back to single model '{model_id}'")
                return model_id
            return None

    async def list_models(self) -> List[Dict[str, Any]]:
        async with self._lock:
            snapshots = list(self.models.items())
            loaded_snap = dict(self.loaded)
            sizes_snap = dict(self._model_sizes)
            mmproj_snap = dict(self.mmproj_paths)
        out = []
        for mid, cfg in snapshots:
            loaded = loaded_snap.get(mid)
            size_mb = sizes_snap.get(mid, 0.0)
            resolved_bin = self.resolve_binary(cfg)
            mmproj_p = mmproj_snap.get(mid)
            has_mmproj = mmproj_p is not None and mmproj_p.exists()
            out.append({
                "id": mid,
                "name": cfg.name or mid,
                "description": cfg.description,
                "tags": cfg.tags,
                "default": cfg.default,
                "pinned": cfg.pinned,
                "auto_save_state": cfg.auto_save_state or self.default_auto_save_state,
                "backend": cfg.backend,
                "resolved_binary": resolved_bin,
                "has_mmproj": has_mmproj,
                "use_mmproj": cfg.use_mmproj,
                "mmproj_file": cfg.mmproj_file,
                "ctx_size": cfg.ctx_size,
                "n_gpu_layers": cfg.n_gpu_layers,
                "estimated_vram_mb": cfg.estimated_vram_mb,
                "args": cfg.args,
                "gguf_name": cfg.gguf_name,
                "size_mb": round(size_mb, 1),
                "loaded": loaded is not None,
                "ready": loaded.ready if loaded else False,
                "port": loaded.port if loaded else None,
                "pid": loaded.pid if loaded else None,
                "last_used": loaded.last_used if loaded else None,
                "state_path": str(loaded.state_path) if loaded and loaded.state_path else None,
            })
        return out

    async def get_config(self, model_id_or_alias: str) -> Optional[ModelConfig]:
        mid = await self.resolve_model_id(model_id_or_alias)
        if not mid:
            return None
        async with self._lock:
            self._ensure_gguf_name(mid)
            return self.models.get(mid)

    def _ensure_gguf_name(self, mid: str):
        """Read gguf_name from GGUF file lazily, caching the result in config."""
        cfg = self.models.get(mid)
        if cfg and not cfg.gguf_name:
            path = self.gguf_paths.get(mid)
            if path:
                cfg.gguf_name = _read_gguf_name(path) or ""

    async def update_config(self, model_id_or_alias: str, new_cfg: ModelConfig) -> ModelConfig:
        mid = await self.resolve_model_id(model_id_or_alias)
        async with self._lock:
            if not mid or mid not in self.models:
                raise KeyError(model_id_or_alias)
            path = self.gguf_paths[mid]
        new_cfg.save(path)
        async with self._lock:
            self.models[mid] = new_cfg
            if mid in self.loaded:
                self.loaded[mid].config = new_cfg
        return new_cfg

    # ---------------- model ID sanitization ----------------
    @staticmethod
    def _sanitize_model_id(model_id: str) -> str:
        """Strip path separators and parent references from model IDs."""
        if not model_id:
            return model_id
        # Remove path separators and parent refs
        mid = model_id.replace("\\", "").replace("/", "").replace("..", "")
        return mid.strip()

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Strip path traversal chars from state labels."""
        if not label:
            return label
        return label.replace("\\", "").replace("/", "").replace("..", "").strip()

    # ---------------- process lifecycle & port allocation ----------------
    def _is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _allocate_port(self) -> int:
        used_ports = {lm.port for lm in self.loaded.values()}
        port = self.next_port
        for _ in range(500):
            if port not in used_ports and self._is_port_available(port):
                self.next_port = port + 1
                return port
            port += 1
        raise RuntimeError("No available ports found for llama-server")

    async def _wait_until_ready(self, port: int, proc: Optional[subprocess.Popen] = None, timeout: float = 120.0) -> bool:
        url = f"http://127.0.0.1:{port}/health"
        log.info(f"Waiting for llama-server on port {port} at {url}...")
        start = time.time()
        while time.time() - start < timeout:
            if proc and proc.poll() is not None:
                log.error(f"llama-server process exited prematurely with code {proc.poll()}")
                return False
            try:
                r = await self.client.get(url, timeout=2.0)
                if r.status_code == 200:
                    log.info(f"llama-server on port {port} is READY! (took {round(time.time() - start, 2)}s)")
                    return True
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(0.5)
        log.error(f"Timed out waiting for llama-server on port {port} after {timeout}s")
        return False

    async def _evict_lru_if_needed(self, target_model_id: str):
        """If loaded models count >= max_loaded_models, unload LRU unpinned model (caller must hold self._lock)."""
        if len(self.loaded) < self.max_loaded_models:
            return
        candidates = [
            (mid, lm) for mid, lm in self.loaded.items()
            if mid != target_model_id and not lm.config.pinned
        ]
        if not candidates:
            log.warning("Max loaded models limit reached, but all loaded models are pinned or target!")
            return
        candidates.sort(key=lambda item: item[1].last_used)
        lru_id, lru_lm = candidates[0]
        log.info(f"Auto-evicting LRU model {lru_id} (last used {round(time.time() - lru_lm.last_used, 1)}s ago) to free slot")
        await self._unload_unlocked(lru_id)

    async def load_model(self, model_id_input: str, force: bool = False) -> LoadedModel:
        """Load (JIT) a model with non-blocking concurrency for other requests."""
        model_id = await self.resolve_model_id(model_id_input)
        if not model_id:
            raise KeyError(f"Unknown model: {model_id_input}")

        async with self._lock:
            # Double-check under lock
            if model_id in self.loaded and self.loaded[model_id].ready and not force:
                self.loaded[model_id].touch()
                return self.loaded[model_id]

            if model_id in self._loading_tasks and not self._loading_tasks[model_id].done():
                task = self._loading_tasks[model_id]
            else:
                task = asyncio.create_task(self._do_load_model(model_id, force=force))
                self._loading_tasks[model_id] = task

        try:
            return await task
        finally:
            async with self._lock:
                if self._loading_tasks.get(model_id) is task:
                    self._loading_tasks.pop(model_id, None)

    async def _do_load_model(self, model_id: str, force: bool = False) -> LoadedModel:
        if not model_id or not model_id.strip():
            raise ValueError("Model ID must be a non-empty string")
        try:
            async with self._lock:
                if model_id in self.loaded and force:
                    await self._unload_unlocked(model_id)
                await self._evict_lru_if_needed(model_id)
                cfg = self.models[model_id]
                path = self.gguf_paths[model_id]
                port = self._allocate_port()
                binary_path = self.resolve_binary(cfg)
                mmproj_p = self.mmproj_paths.get(model_id)
                argv = [binary_path] + cfg.to_launch_args(path, port, self.default_args, mmproj_full_path=mmproj_p, slot_save_dir=self.save_state_dir, model_id=model_id)

                log.info(f"Launching llama-server ({binary_path}) for {model_id} on port {port}")
                log.info("argv: " + " ".join(argv))

                # Spawn process
                if platform.system() == "Windows":
                    proc = subprocess.Popen(
                        argv,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                else:
                    proc = subprocess.Popen(
                        argv,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )

                lm = LoadedModel(
                    model_id=model_id,
                    gguf_path=path,
                    config=cfg,
                    port=port,
                    process=proc,
                    pid=proc.pid,
                    starting_at=time.time(),
                )
                self.loaded[model_id] = lm
                self._start_log_thread(model_id, proc)
                self._status_cache = None  # Force status cache rebuild after load

            # Wait until ready OUTSIDE the lock!
            ready = await self._wait_until_ready(port, proc=proc)
            lm.ready = ready
            if not ready:
                log.error(f"Model {model_id} failed to become ready")
                async with self._lock:
                    await self._unload_unlocked(model_id)
                raise RuntimeError(f"Model {model_id} failed to start")

            log.info(f"Model {model_id} ready on port {port}")
            lm.touch()

            # Auto-restore session state if enabled and cached auto state exists
            if cfg.auto_save_state or self.default_auto_save_state:
                mid_clean = self._sanitize_model_id(model_id)
                state_path = self.save_state_dir / f"{mid_clean}.auto.bin"
                if state_path.exists():
                    try:
                        log.info(f"Auto-restoring session state for {model_id} from {state_path}...")
                        await self._perform_slot_restore(port, state_path, timeout=30.0)
                        await self._warm_slot_cache(port)
                        lm.state_path = state_path
                    except Exception as e:
                        log.warning(f"Auto-restore failed for {model_id}: {e}")

            return lm
        except asyncio.CancelledError:
            async with self._lock:
                await self._unload_unlocked(model_id)
            raise

    def _start_log_thread(self, model_id: str, proc: subprocess.Popen):
        def _worker():
            try:
                assert proc.stdout is not None
                for line in iter(proc.stdout.readline, b""):
                    try:
                        txt = line.decode("utf-8", errors="replace").rstrip()
                        if txt:
                            log.info(f"[{model_id}] {txt}")
                    except Exception:
                        pass
            except Exception as e:
                log.warning(f"log stream for {model_id} ended: {e}")

        t = threading.Thread(target=_worker, daemon=True, name=f"log-{model_id}")
        t.start()

    async def _unload(self, model_id: str):
        """Unload a model (acquires lock if not already held)."""
        async with self._lock:
            await self._unload_unlocked(model_id)

    async def _unload_unlocked(self, model_id: str):
        """Unload a model (caller must hold self._lock)."""
        lm = self.loaded.pop(model_id, None)
        if lm is None:
            return
        self._status_cache = None  # Force status cache rebuild on next request
        log.info(f"Unloading {model_id} (port {lm.port})")

        # Auto-save session state if enabled
        if lm.ready and (lm.config.auto_save_state or self.default_auto_save_state):
            try:
                mid_clean = self._sanitize_model_id(model_id)
                state_path = self.save_state_dir / f"{mid_clean}.auto.bin"
                log.info(f"Auto-saving session state for {model_id} to {state_path}...")
                await self._perform_slot_save(lm.port, state_path, timeout=60.0)
            except Exception as e:
                log.warning(f"Auto-save failed for {model_id}: {e}")

        # Kill entire process group reliably across platforms
        def _kill_proc():
            try:
                parent = psutil.Process(lm.pid)
                children = parent.children(recursive=True)
                # Terminate children first, then parent
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                parent.terminate()
                gone, alive = psutil.wait_procs(children + [parent], timeout=3)
                for p in alive:
                    try:
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                log.warning(f"Process termination failed for {model_id} (pid {lm.pid}): {e}")
                # Fallback: kill the process group directly
                try:
                    if platform.system() == "Windows":
                        # Send CTRL+C to the process group on Windows
                        os.kill(lm.pid, os.CTRL_C_EVENT)
                    else:
                        # Kill the entire process group on Unix
                        os.killpg(os.getpgid(lm.pid), signal.SIGKILL)
                    lm.process.kill()
                except Exception:
                    pass

        await asyncio.to_thread(_kill_proc)

    async def unload_model(self, model_id_input: str):
        mid = await self.resolve_model_id(model_id_input)
        if not mid:
            return
        async with self._lock:
            await self._unload_unlocked(mid)

    async def unload_all(self):
        async with self._lock:
            ids = list(self.loaded.keys())
            for mid in ids:
                await self._unload_unlocked(mid)

    # ---------------- GPU info ----------------
    def gpus(self) -> List[GPUInfo]:
        out: List[GPUInfo] = []
        if not shutil.which("nvidia-smi"):
            return out
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 6:
                    continue
                out.append(GPUInfo(
                    index=_safe_int(parts[0], 0),
                    name=parts[1],
                    total_mb=_safe_int(parts[2], 0),
                    used_mb=_safe_int(parts[3], 0),
                    free_mb=_safe_int(parts[4], 0),
                    utilization_pct=_safe_float(parts[5], 0.0),
                ))
        except Exception as e:
            log.warning(f"nvidia-smi failed: {e}")
        return out

    def system_ram(self) -> Dict[str, float]:
        m = psutil.virtual_memory()
        return {
            "total_mb": m.total / 1024 / 1024,
            "used_mb": m.used / 1024 / 1024,
            "free_mb": m.available / 1024 / 1024,
            "pct": m.percent,
        }

    def per_model_ram_vram(self, loaded_items: Optional[List[tuple]] = None) -> Dict[str, Dict[str, int]]:
        """Best-effort per-model RAM/VRAM accounting via process info."""
        out: Dict[str, Dict[str, int]] = {}
        try:
            apps = subprocess.run(
                ["nvidia-smi",
                 "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            pid_to_vram: Dict[int, int] = {}
            for line in apps.stdout.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit():
                    pid_to_vram[int(parts[0])] = _safe_int(parts[1], 0)
            items = loaded_items if loaded_items is not None else list(self.loaded.items())
            for mid, lm in items:
                try:
                    p = psutil.Process(lm.pid)
                    ram = p.memory_info().rss / 1024 / 1024
                except Exception:
                    ram = 0
                vram = pid_to_vram.get(lm.pid, 0)
                out[mid] = {"ram_mb": int(ram), "vram_mb": int(vram)}
        except Exception as e:
            log.warning(f"per_model_ram_vram failed: {e}")
        return out

    # ---------------- proxy ----------------
    async def proxy(self, model_id_input: str, path: str, request: Request) -> Response:
        """Proxy a request to the loaded model's llama-server."""
        log.info(f"--> Incoming {request.method} request to '{request.url.path}' (model param input: {model_id_input!r})")
        model_id = await self.resolve_model_id(model_id_input)
        if not model_id:
            log.error(f"Could not resolve model for input {model_id_input!r}")
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id_input}")

        if model_id_input and model_id != model_id_input:
            log.warning(f"Model '{model_id_input}' not found; falling back to '{model_id}'")
        log.info(f"Resolved model ID: '{model_id}'")
        async with self._lock:
            lm = self.loaded.get(model_id)
        if lm is None or not lm.ready:
            log.info(f"Model '{model_id}' not loaded or not ready. Loading model JIT...")
            try:
                lm = await self.load_model(model_id)
            except Exception as e:
                log.error(f"Failed to load model '{model_id}': {e}")
                raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")
        lm.touch()
        log.info(f"Forwarding {request.method} {request.url.path} -> llama-server on port {lm.port}")

        url = f"http://127.0.0.1:{lm.port}{path}"
        body = await request.body()
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
        req_method = request.method

        # Streaming?
        try:
            parsed = json.loads(body) if body else {}
        except Exception:
            parsed = {}
        stream = parsed.get("stream", False) if isinstance(parsed, dict) else False

        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }

        if stream and req_method in ("POST", "PUT"):
            try:
                req = self.client.build_request(req_method, url, content=body, headers=headers)
                r = await self.client.send(req, stream=True)
                if r.status_code >= 400:
                    err_bytes = await r.aread()
                    await r.aclose()
                    return Response(content=err_bytes, status_code=r.status_code,
                                    media_type=r.headers.get("content-type", "application/json"),
                                    headers=cors_headers)

                async def gen() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in r.aiter_bytes():
                            yield chunk
                    except (httpx.ReadError, httpx.ReadTimeout):
                        pass
                    finally:
                        await r.aclose()

                return StreamingResponse(gen(), status_code=r.status_code, media_type="text/event-stream", headers=cors_headers)
            except httpx.ConnectError:
                raise HTTPException(status_code=502, detail="Model server unreachable")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Proxy error: {e}")

        try:
            r = await self.client.request(req_method, url, content=body, headers=headers)
            return Response(content=r.content, status_code=r.status_code,
                            media_type=r.headers.get("content-type", "application/json"),
                            headers=cors_headers)
        except (httpx.ReadError, httpx.ReadTimeout):
            raise HTTPException(status_code=502, detail="Model server read error")
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Model server unreachable")

    # ---------------- state save/load ----------------
    async def _perform_slot_action(self, action: str, port: int, state_path: Path, timeout: float = 120.0) -> bool:
        """Attempt a slot action (save/restore) trying supported endpoint formats."""
        endpoints = [
            (f"http://127.0.0.1:{port}/slots/0?action={action}", {"filename": state_path.name}),
            (f"http://127.0.0.1:{port}/slots?action={action}", {"id_slot": 0, "filename": state_path.name}),
            (f"http://127.0.0.1:{port}/slots", {"action": action, "id_slot": 0, "filename": state_path.name}),
        ]

        last_err = None
        for url, payload in endpoints:
            try:
                r = await self.client.post(url, json=payload, timeout=timeout)
                if r.status_code == 200:
                    return True
                else:
                    last_err = f"HTTP {r.status_code}: {r.text}"
            except Exception as e:
                last_err = str(e)

        raise RuntimeError(last_err or f"{action} failed across all slot endpoints")

    async def _perform_slot_save(self, port: int, state_path: Path, timeout: float = 120.0) -> bool:
        """Attempt to save slot state trying supported endpoint formats."""
        state_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving slot 0 state to file: {state_path}")
        return await self._perform_slot_action("save", port, state_path, timeout)

    async def _perform_slot_restore(self, port: int, state_path: Path, timeout: float = 120.0) -> bool:
        """Attempt to restore slot state trying supported endpoint formats."""
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        log.info(f"Restoring slot 0 state from file: {state_path}")
        return await self._perform_slot_action("restore", port, state_path, timeout)

    async def _warm_slot_cache(self, port: int, timeout: float = 10.0) -> bool:
        """Send a minimal request to warm checkpoint cache after slot restore (llama.cpp v2.26.0 workaround)."""
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
        }
        try:
            r = await self.client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            return r.status_code == 200
        except Exception:
            return False

    async def save_state(self, model_id_input: str, label: str = "default") -> Path:
        model_id = await self.resolve_model_id(model_id_input)
        if not model_id:
            raise HTTPException(400, "Model not loaded")
        async with self._lock:
            if model_id not in self.loaded:
                raise HTTPException(400, "Model not loaded")
            lm = self.loaded[model_id]
        mid_clean = self._sanitize_model_id(model_id)
        label_clean = self._sanitize_label(label)
        state_path = self.save_state_dir / f"{mid_clean}.{label_clean}.bin"

        try:
            await self._perform_slot_save(lm.port, state_path, timeout=120.0)
        except Exception as e:
            raise HTTPException(500, f"save failed: {e}")

        lm.state_path = state_path
        return state_path

    async def load_state(self, model_id_input: str, label: str = "default") -> Path:
        model_id = await self.resolve_model_id(model_id_input)
        if not model_id:
            raise HTTPException(400, "Model not loaded")
        async with self._lock:
            if model_id not in self.loaded:
                raise HTTPException(400, "Model not loaded")
            lm = self.loaded[model_id]
        mid_clean = self._sanitize_model_id(model_id)
        label_clean = self._sanitize_label(label)
        state_path = self.save_state_dir / f"{mid_clean}.{label_clean}.bin"

        try:
            await self._perform_slot_restore(lm.port, state_path, timeout=120.0)
            await self._warm_slot_cache(lm.port)
        except Exception as e:
            raise HTTPException(500, f"restore failed: {e}")

        lm.state_path = state_path
        return state_path

    async def list_states(self, model_id_input: str) -> List[str]:
        model_id = await self.resolve_model_id(model_id_input) or model_id_input
        mid_clean = self._sanitize_model_id(model_id)
        out = []
        for p in self.save_state_dir.glob(f"{mid_clean}.*.bin"):
            label = p.stem.split(".", 1)[1]
            out.append(label)
        return out

    # ---------------- status cache ----------------
    async def _build_status(self) -> Dict[str, Any]:
        """Build a full status snapshot (cached to avoid redundant nvidia-smi calls)."""
        gpus = [asdict(g) for g in self.gpus()]
        backends = self.list_backends()
        async with self._lock:
            loaded_snap = list(self.loaded.items())
            total = len(self.models)
        per = self.per_model_ram_vram(loaded_items=loaded_snap)
        return {
            "launcher": {
                "host": self.host,
                "port": self.port,
                "binary": self.resolve_binary(None),
                "selected_backend": self.selected_backend,
                "backends": backends,
            },
            "gpus": gpus,
            "ram": self.system_ram(),
            "models_loaded": len(loaded_snap),
            "models_total": total,
            "idle_timeout": self.idle_timeout,
            "max_loaded_models": self.max_loaded_models,
            "per_model": per,
            "uptime_models": [
                {
                    "id": mid,
                    "port": lm.port,
                    "pid": lm.pid,
                    "uptime_s": round(time.time() - lm.starting_at, 1),
                    "last_used_s_ago": round(time.time() - lm.last_used, 1),
                    "ready": lm.ready,
                } for mid, lm in loaded_snap
            ],
        }

    async def _status_cache_updater(self) -> None:
        """Background task that refreshes status cache at poll_interval intervals."""
        while not self._stop:
            try:
                status = await self._build_status()
                async with self._lock:
                    self._status_cache_time = time.time()
                    self._status_cache = status
            except Exception as e:
                log.warning(f"status_cache_updater error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def get_cached_status(self) -> Dict[str, Any]:
        """Return cached status, refreshing if stale."""
        async with self._lock:
            if self._status_cache is None or (time.time() - self._status_cache_time) > self._poll_interval:
                needs_refresh = True
            else:
                needs_refresh = False
            cached = self._status_cache.copy() if self._status_cache else {}
        if needs_refresh:
            status = await self._build_status()
            async with self._lock:
                self._status_cache_time = time.time()
                self._status_cache = status
            return status
        return cached

    # ---------------- background idle reaper ----------------
    async def idle_reaper(self) -> None:
        while not self._stop:
            try:
                async with self._lock:
                    now = time.time()
                    to_unload = []
                    for mid, lm in list(self.loaded.items()):
                        if lm.config.pinned or not lm.ready:
                            continue
                        if now - lm.last_used > self.idle_timeout:
                            to_unload.append(mid)
                    for mid in to_unload:
                        log.info(f"Idle-unloading {mid}")
                        await self._unload_unlocked(mid)
            except Exception as e:
                log.warning(f"idle_reaper error: {e}")
            await asyncio.sleep(10)

    async def start(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self.scan)
        self._bg_task = asyncio.create_task(self.idle_reaper())
        self._status_task: Optional[asyncio.Task] = asyncio.create_task(self._status_cache_updater())

    async def stop(self) -> None:
        self._stop = True
        bg_tasks = []
        if self._bg_task:
            self._bg_task.cancel()
            bg_tasks.append(self._bg_task)
        if getattr(self, "_status_task", None):
            self._status_task.cancel()
            bg_tasks.append(self._status_task)
        if bg_tasks:
            await asyncio.gather(*bg_tasks, return_exceptions=True)
        # Cancel any in-flight loading tasks and wait for them
        async with self._lock:
            tasks = list(self._loading_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.unload_all()
        await self.client.aclose()


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("AUTOLOADER_CONFIG", "config.yaml")
try:
    with open(CONFIG_PATH) as f:
        CFG = yaml.safe_load(f)
    ModelManager.validate_config(CFG)
except Exception as e:
    raise RuntimeError(f"Failed to load config from '{CONFIG_PATH}': {e}") from e
manager = ModelManager(CFG)

app = FastAPI(title="llama-autoloader", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# ---------- models payloads ----------
class ModelConfigUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    args: Optional[str] = None
    ctx_size: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    default: Optional[bool] = None
    pinned: Optional[bool] = None
    auto_save_state: Optional[bool] = None
    backend: Optional[str] = None
    use_mmproj: Optional[bool] = None
    mmproj_file: Optional[str] = None
    estimated_vram_mb: Optional[int] = None
    tags: Optional[List[str]] = None
    gguf_name: Optional[str] = None

class SelectBackendPayload(BaseModel):
    backend: str = ""

class StateLabel(BaseModel):
    label: str = "default"

# ---------- status / dashboard / health ----------
@app.get("/")
async def index():
    p = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(p.read_text(encoding="utf-8"))

@app.get("/health")
@app.get("/v1/health")
async def health():
    async with manager._lock:
        count = len(manager.loaded)
    return {"status": "ok", "models_loaded": count}

@app.get("/v1/backends")
async def get_backends():
    return {
        "backends": manager.list_backends(),
        "selected_backend": manager.selected_backend,
        "default_binary": manager.binary,
        "resolved_global_binary": manager.resolve_binary(None),
    }

@app.post("/v1/backend/select")
async def select_backend(body: SelectBackendPayload):
    if body.backend:
        backends = manager.list_backends()
        backend_ids = [b["id"] for b in backends]
        if body.backend not in backend_ids:
            raise HTTPException(400, f"Unknown backend: '{body.backend}'. Available: {backend_ids}")
    manager.selected_backend = body.backend
    # Persist to config.yaml
    cfg_yaml = manager.cfg
    cfg_yaml["llama_server"]["selected_backend"] = body.backend
    try:
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg_yaml, f, default_flow_style=False)
    except OSError as e:
        log.warning(f"Failed to persist backend selection to {CONFIG_PATH}: {e}")
    log.info(f"Global backend selected: '{body.backend}' -> resolved binary: '{manager.resolve_binary(None)}'")
    return {
        "selected_backend": manager.selected_backend,
        "resolved_binary": manager.resolve_binary(None),
    }

@app.get("/v1/status")
async def status():
    return await manager.get_cached_status()

# ---------- global settings ----------
@app.get("/v1/settings")
async def get_settings():
    return {
        "idle_timeout_seconds": manager.idle_timeout,
        "max_loaded_models": manager.max_loaded_models,
        "auto_save_state": manager.default_auto_save_state,
        "poll_interval_seconds": manager._poll_interval,
        "base_port": manager.base_port,
        "host": manager.host,
        "port": manager.port,
        "default_args": manager.default_args,
        "selected_backend": manager.selected_backend,
    }

@app.put("/v1/settings")
async def update_settings(request: Request):
    body = await request.json()
    errors = []

    if "idle_timeout_seconds" in body:
        v = body["idle_timeout_seconds"]
        if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)) or v <= 0:
            errors.append("'idle_timeout_seconds' must be > 0")
    if "max_loaded_models" in body:
        v = body["max_loaded_models"]
        if not isinstance(v, int) or v < 1:
            errors.append("'max_loaded_models' must be >= 1")
    if "poll_interval_seconds" in body:
        v = body["poll_interval_seconds"]
        if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)) or v <= 0:
            errors.append("'poll_interval_seconds' must be > 0")
    if "base_port" in body:
        v = body["base_port"]
        if not isinstance(v, int) or v < 1024 or v > 65535:
            errors.append("'base_port' must be 1024-65535")
    if "auto_save_state" in body:
        v = body["auto_save_state"]
        if not isinstance(v, bool):
            errors.append("'auto_save_state' must be a boolean")

    if errors:
        raise HTTPException(400, "; ".join(errors))

    changed = []
    if "idle_timeout_seconds" in body:
        manager.idle_timeout = body["idle_timeout_seconds"]
        changed.append("idle_timeout_seconds")
    if "max_loaded_models" in body:
        manager.max_loaded_models = body["max_loaded_models"]
        changed.append("max_loaded_models")
    if "auto_save_state" in body:
        manager.default_auto_save_state = body["auto_save_state"]
        changed.append("auto_save_state")
    if "poll_interval_seconds" in body:
        manager._poll_interval = body["poll_interval_seconds"]
        changed.append("poll_interval_seconds")
    if "default_args" in body:
        manager.default_args = body["default_args"]
        changed.append("default_args")
    if "selected_backend" in body:
        manager.selected_backend = body["selected_backend"]
        changed.append("selected_backend")
    if "base_port" in body:
        manager.base_port = body["base_port"]
        manager.next_port = max(manager.base_port, manager.next_port)
        changed.append("base_port")

    # Persist changes to config.yaml
    if changed:
        cfg = manager.cfg
        if "idle_timeout_seconds" in body:
            cfg["models"]["idle_timeout_seconds"] = body["idle_timeout_seconds"]
        if "max_loaded_models" in body:
            cfg["models"]["max_loaded_models"] = body["max_loaded_models"]
        if "auto_save_state" in body:
            cfg["models"]["auto_save_state"] = body["auto_save_state"]
        if "poll_interval_seconds" in body:
            cfg.setdefault("gpu", {})["poll_interval_seconds"] = body["poll_interval_seconds"]
        if "default_args" in body:
            cfg["llama_server"]["default_args"] = body["default_args"]
        if "selected_backend" in body:
            cfg["llama_server"]["selected_backend"] = body["selected_backend"]
        if "base_port" in body:
            cfg["launcher"]["base_port"] = body["base_port"]
        if "host" in body:
            cfg["launcher"]["host"] = body["host"]
        if "port" in body:
            cfg["launcher"]["port"] = body["port"]
        try:
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
        except OSError as e:
            log.warning(f"Failed to persist settings to {CONFIG_PATH}: {e}")

    notes = []
    if "host" in body or "port" in body:
        notes.append("host/port require restart to take effect")

    if changed:
        log.info(f"Settings updated: {', '.join(changed)}")
    if notes:
        log.info(f"Note: {'; '.join(notes)}")

    return await get_settings()

# ---------- model list / config (LM Studio compatible) ----------
@app.get("/v1/models")
async def list_models():
    data = []
    for m in await manager.list_models():
        data.append({
            "id": m["id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llama-autoloader",
            "name": m["name"],
            "description": m["description"],
            "tags": m["tags"],
            "size_mb": m["size_mb"],
            "loaded": m["loaded"],
            "ready": m["ready"],
            "port": m["port"],
            "pid": m["pid"],
            "default": m["default"],
            "pinned": m["pinned"],
            "auto_save_state": m["auto_save_state"],
            "backend": m["backend"],
            "use_mmproj": m["use_mmproj"],
            "mmproj_file": m["mmproj_file"],
            "has_mmproj": m["has_mmproj"],
            "ctx_size": m["ctx_size"],
            "n_gpu_layers": m["n_gpu_layers"],
            "estimated_vram_mb": m["estimated_vram_mb"],
            "args": m["args"],
            "gguf_name": m["gguf_name"],
        })
    return {"object": "list", "data": data}

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    mid = await manager.resolve_model_id(model_id)
    if not mid:
        raise HTTPException(404, "Model not found")
    cfg = await manager.get_config(mid)
    async with manager._lock:
        loaded = mid in manager.loaded
    return {
        "id": mid,
        "object": "model",
        "config": asdict(cfg) if cfg else {},
        "loaded": loaded,
    }

@app.put("/v1/models/{model_id}/config")
async def update_model_config(model_id: str, body: ModelConfigUpdate):
    mid = await manager.resolve_model_id(model_id)
    if not mid:
        raise HTTPException(404, "Unknown model")
    cfg = await manager.get_config(mid)
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(cfg, k, v)
    await manager.update_config(mid, cfg)
    return asdict(cfg)

@app.post("/v1/models/{model_id}/vision/toggle")
async def toggle_vision_ep(model_id: str):
    mid = await manager.resolve_model_id(model_id)
    if not mid:
        raise HTTPException(404, "Unknown model")
    cfg = await manager.get_config(mid)
    cfg.use_mmproj = not cfg.use_mmproj
    await manager.update_config(mid, cfg)
    return {"id": mid, "use_mmproj": cfg.use_mmproj, "mmproj_file": cfg.mmproj_file}

@app.post("/v1/scan")
async def rescan():
    async with manager._lock:
        ids = await asyncio.to_thread(manager.scan)
    return {"models": ids}

# ---------- load / unload ----------
@app.post("/v1/models/{model_id}/load")
async def load_model_ep(model_id: str):
    try:
        lm = await manager.load_model(model_id, force=False)
        return {"id": model_id, "port": lm.port, "pid": lm.pid, "ready": lm.ready}
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(503, str(e))

@app.post("/v1/models/{model_id}/unload")
async def unload_model_ep(model_id: str):
    await manager.unload_model(model_id)
    return {"id": model_id, "unloaded": True}

@app.post("/v1/unload_all")
async def unload_all_ep():
    await manager.unload_all()
    return {"unloaded_all": True}

# ---------- state save / load ----------
@app.post("/v1/models/{model_id}/state/save")
async def save_state_ep(model_id: str, body: StateLabel):
    p = await manager.save_state(model_id, body.label)
    return {"id": model_id, "label": body.label, "path": str(p)}

@app.post("/v1/models/{model_id}/state/load")
async def load_state_ep(model_id: str, body: StateLabel):
    p = await manager.load_state(model_id, body.label)
    return {"id": model_id, "label": body.label, "path": str(p)}

@app.get("/v1/models/{model_id}/state")
async def list_states_ep(model_id: str):
    return {"id": model_id, "labels": await manager.list_states(model_id)}

# ---------- OpenAI-compatible proxy endpoints ----------
async def _resolve_proxy_model(body: bytes) -> str:
    """Extract model ID from request body, with fallback to loaded/default/single model."""
    model_id = None
    if body:
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                model_id = data.get("model")
        except Exception:
            pass

    resolved = await manager.resolve_model_id(model_id)
    if not resolved:
        raise HTTPException(404, "No matching model found and no loaded/default model available")
    return resolved

@app.api_route("/v1/chat/completions", methods=["POST"])
async def proxy_chat(request: Request):
    body = await request.body()
    model_id = await _resolve_proxy_model(body)
    return await manager.proxy(model_id, "/v1/chat/completions", request)

@app.api_route("/v1/completions", methods=["POST"])
async def proxy_completions(request: Request):
    body = await request.body()
    model_id = await _resolve_proxy_model(body)
    return await manager.proxy(model_id, "/v1/completions", request)

@app.api_route("/v1/embeddings", methods=["POST"])
async def proxy_embeddings(request: Request):
    body = await request.body()
    model_id = await _resolve_proxy_model(body)
    return await manager.proxy(model_id, "/v1/embeddings", request)

# Top-level non-/v1 routes (for clients connecting without /v1 prefix)
@app.api_route("/chat/completions", methods=["POST"])
async def proxy_chat_top(request: Request):
    body = await request.body()
    model_id = await _resolve_proxy_model(body)
    return await manager.proxy(model_id, "/v1/chat/completions", request)

@app.api_route("/completions", methods=["POST"])
async def proxy_completions_top(request: Request):
    body = await request.body()
    model_id = await _resolve_proxy_model(body)
    return await manager.proxy(model_id, "/v1/completions", request)

@app.api_route("/embeddings", methods=["POST"])
async def proxy_embeddings_top(request: Request):
    body = await request.body()
    model_id = await _resolve_proxy_model(body)
    return await manager.proxy(model_id, "/v1/embeddings", request)

# Pass-through other llama-server endpoints (e.g. /slots, /tokenize, /detokenize)
@app.api_route("/v1/raw/{model_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_raw(model_id: str, path: str, request: Request):
    return await manager.proxy(model_id, "/" + path, request)

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_catchall(path: str, request: Request):
    body = await request.body()
    model_id = await _resolve_proxy_model(body)
    return await manager.proxy(model_id, "/v1/" + path, request)

# ---------- WebSocket for live updates ----------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(2.0)
            await websocket.send_json(await manager.get_cached_status())
    except asyncio.CancelledError:
        return
    except WebSocketDisconnect:
        return
    except (RuntimeError, ConnectionError) as e:
        log.warning(f"WebSocket error: {e}")
        return

# ---------- lifecycle ----------
@app.on_event("startup")
async def _startup():
    await manager.start()

@app.on_event("shutdown")
async def _shutdown():
    await manager.stop()


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=CFG["launcher"]["host"])
    ap.add_argument("--port", type=int, default=CFG["launcher"]["port"])
    args = ap.parse_args()
    manager.host = args.host
    manager.port = args.port
    uvicorn.run(app, host=args.host, port=args.port, reload=False)