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
import struct
import subprocess
import time
import logging
import platform
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import psutil
import yaml
from fastapi import (
    FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
import uvicorn

# Console-log-to-file logging (see logger.py). init_logging() is called in the
# __main__ block below, BEFORE uvicorn.run(), so uvicorn's dictConfig handlers
# bind to our capturing stdout/stderr streams and land in the log file.
from logger import init_logging

# Engine launch seam (see bridge_adapter.py). _launch_llama reproduces today's
# llama-server argv byte-for-byte (env/cwd are None); _launch_bridge renders an
# engine's config-supplied argv template, derived config and scrubbed env.
from bridge_adapter import _launch_llama, _launch_bridge, _expand_token

# Per-model FIFO ticket queue: orders LLM requests (TCP forwarder) against state
# save/restore (FastAPI) so a restore enqueued first runs before the next request.
from model_queue import ModelQueueRegistry

# Legacy launch template used ONLY when migrating a pre-registry ``strata:`` section into
# ``engines.strata``. New engines must supply their own ``argv`` (there is no generic default).
_DEFAULT_ARGV = ["{interpreter}", "{serve_script}", "--engine", "{engine}",
                 "--config", "{derived_config}", "--port", "{port}", "--host", "127.0.0.1"]

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
# NOTE: We intentionally do NOT call logging.basicConfig() at module level. The
# dedicated "autoloader" logger is fully configured by init_logging() (console +
# rotating file handlers). Keeping basicConfig off here avoids a root-level
# handler double-printing autoloader lines once stdout is tee'd to the file.
# Tests that import this module without calling init_logging still get output via
# Python's default lastResort handler / uvicorn's own logging config.

# Enable uvicorn access logging so all requests are visible in the console
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
log = logging.getLogger("autoloader")

# --------------------------------------------------------------------------
# Streaming relay pacing (Round 10 follow-up: write-drain-starvation test)
# --------------------------------------------------------------------------
# On this Windows host the relay gen() (which reads a live TCP upstream via
# httpx aiter_bytes while yielding into a StreamingResponse) collapses to an
# all-at-once end-of-stream burst, even though a trivial in-process generator
# streams incrementally. `await asyncio.sleep(0)` per chunk (round 5) did NOT
# fix it — one loop turn may be too little for the downstream socket's
# "writable" callback to fire and DRAIN uvicorn's transport write-buffer before
# gen() re-arms the next upstream read. This env-gated knob lets the operator
# inject a REAL per-chunk sleep to test whether wall-clock time lets the
# downstream write drain (cheapest decisive test). Default 0 = current
# behavior (sleep(0)), so unset/0 is byte-for-byte identical to before.
_chunk_sleep_ms = float(os.environ.get("AUTOLOADER_CHUNK_SLEEP_MS", "0"))
# Round-10 follow-up 5: env-gated one-line live debug. When AUTOLOADER_DEBUG_RESP=1, log the
# response object's type + whether it already carries a content-length header at construction
# time (right before `return StreamingResponse(...)`). A true Starlette StreamingResponse NEVER
# sets content-length; if this line shows content-length=True or a non-StreamingResponse type,
# that localizes the live burst to response-construction-under-load. Default off = no logging.
_debug_resp = os.environ.get("AUTOLOADER_DEBUG_RESP", "0") == "1"
# Round-10 follow-up 8: env-gated relay mode for A/B testing the sync/threadpool fix (Fix 2).
#   "async" (default) = current async def gen() + aiter_bytes relay (with keepalive pings, Fix 1).
#   "sync"            = synchronous def gen() reading via a SYNC httpx.Client in FastAPI's threadpool
#                       (documented alternative for blocking relays — keeps the event loop free).
# NOT the default yet: Fix 1 (headers + keepalive) is tried first. Set AUTOLOADER_STREAM_MODE=sync
# to A/B it live without another code change. Any other value falls back to "async".
_stream_mode = os.environ.get("AUTOLOADER_STREAM_MODE", "async").strip().lower()
if _stream_mode not in ("async", "sync"):
    _stream_mode = "async"

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


# Fixed-width GGUF value types -> byte width. Used to skip large numeric arrays with a
# single seek instead of one seek per element (a 150k-entry float array is one seek, not
# 150k). String arrays (type 8) are variable-width and still walked element by element,
# but skipped without decoding.
_GGUF_FIXED_WIDTH = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}


def _read_gguf_info(gguf_path) -> Dict[str, Any]:
    """Read the GGUF header keys we care about, using fast binary stream parsing.

    Returns a dict with:
      name      -- ``general.name``                (display only; NOT a retrieval alias)
      max_ctx   -- the first ``*.context_length``  (max context the arch supports)
      arch      -- ``general.architecture``        (e.g. "llama", "qwen35", "clip", "whisper")
      gguf_type -- ``general.type``                (e.g. "model", "mmproj")

    Bypasses np.memmap and tensor allocations. Only the KV block is walked; values are
    skipped by seek and the tensor-info/tensor-data blocks are never touched, so this
    stays in the sub-millisecond-per-file range even for a 150k-token vocabulary.

    Every failure mode (missing file, bad magic, corrupt header, unknown value type)
    returns an all-None dict rather than raising - callers treat "no metadata" and
    "unreadable metadata" identically.
    """
    empty = {"name": None, "max_ctx": None, "arch": None, "gguf_type": None}
    name: Optional[str] = None
    max_ctx: Optional[int] = None
    arch: Optional[str] = None
    gguf_type: Optional[str] = None
    try:
        with open(gguf_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return empty
            ver, tensors, kvs = struct.unpack('<IQQ', f.read(20))
            if kvs > 1_000_000:
                return empty

            def read_str():
                l = struct.unpack('<Q', f.read(8))[0]
                return f.read(l).decode('utf-8', errors='replace')

            def skip_val(vtype):
                if vtype in _GGUF_FIXED_WIDTH:
                    f.seek(_GGUF_FIXED_WIDTH[vtype], 1)
                elif vtype == 8:
                    l = struct.unpack('<Q', f.read(8))[0]
                    f.seek(l, 1)
                elif vtype == 9:
                    itype, ilen = struct.unpack('<IQ', f.read(12))
                    if itype in _GGUF_FIXED_WIDTH:
                        f.seek(ilen * _GGUF_FIXED_WIDTH[itype], 1)   # one seek, not ilen
                    else:
                        for _ in range(ilen):
                            skip_val(itype)
                else:
                    # Unknown/corrupt value type — can't safely skip; abort parse.
                    raise ValueError(f"Unknown GGUF value type {vtype}")

            for examined in range(kvs):
                key = read_str()
                vtype = struct.unpack('<I', f.read(4))[0]
                if key == 'general.name' and vtype == 8:
                    name = read_str()
                elif key == 'general.architecture' and vtype == 8:
                    arch = read_str()
                elif key == 'general.type' and vtype == 8:
                    gguf_type = read_str()
                elif key.endswith('.context_length'):
                    if vtype == 4: max_ctx = struct.unpack('<I', f.read(4))[0]
                    elif vtype == 10: max_ctx = struct.unpack('<Q', f.read(8))[0]
                    elif vtype == 5: max_ctx = struct.unpack('<i', f.read(4))[0]
                    elif vtype == 11: max_ctx = struct.unpack('<q', f.read(8))[0]
                    else: skip_val(vtype)
                else:
                    skip_val(vtype)
                # Stop as soon as we have everything the callers use. ``gguf_type`` is
                # best-effort: plenty of files omit ``general.type`` entirely, so after a
                # bounded number of keys we stop looking rather than walking a 150k-entry
                # tokenizer array. mmproj detection has a filename fallback, so a missing
                # ``general.type`` never costs us a correct classification.
                if name and max_ctx and arch is not None and (gguf_type is not None or examined >= 128):
                    break
    except Exception as e:
        log.debug(f"Fast GGUF header read failed for {gguf_path}: {e}")
        return empty
    return {"name": name, "max_ctx": max_ctx, "arch": arch, "gguf_type": gguf_type}


def _read_gguf_metadata(gguf_path) -> Tuple[Optional[str], Optional[int]]:
    """Read (general.name, context_length) from GGUF file metadata.

    Thin wrapper over :func:`_read_gguf_info` kept for its existing callers/tests, which
    only need the name/context pair.
    """
    info = _read_gguf_info(gguf_path)
    return info["name"], info["max_ctx"]

def _read_gguf_name(gguf_path) -> Optional[str]:
    """Read general.name from GGUF file metadata. Returns None on failure."""
    name, _ = _read_gguf_metadata(gguf_path)
    return name


# --------------------------------------------------------------------------
# Model type taxonomy
# --------------------------------------------------------------------------
# Every card gets exactly one *model type*. It drives three separate things:
#   1. the WebUI grouping / filter chips,
#   2. the type badge on each card,
#   3. the "not directly servable" hint shown on auxiliary types.
#
# It is deliberately NOT the same concept as the existing `vision` tag: that tag means
# "this text model has a paired mmproj", which is orthogonal to what the weights are.
# A Qwen3.8 text LLM with an mmproj stays in the Text Generation group and keeps its
# `vision` tag; a Qwen3-VL model is natively vision and gets the Vision group.
#
# Types that are components or sidecars rather than standalone servers:
#   mmproj - vision projector, attached to a host model via --mmproj
#   codec  - audio codec / VAE / tokenizer, consumed by a TTS host model
#   draft  - speculative-decoding draft (MTP / draft model), used via --model-draft
# `embedding` is standalone-servable but only through /v1/embeddings, not chat.
MODEL_TYPE_TEXT = "text"
MODEL_TYPE_VISION = "vision"
MODEL_TYPE_AUDIO = "audio"
MODEL_TYPE_ASR = "asr"
MODEL_TYPE_EMBEDDING = "embedding"
MODEL_TYPE_DRAFT = "draft"
MODEL_TYPE_CODEC = "codec"
MODEL_TYPE_MMPROJ = "mmproj"

# Canonical display order for the WebUI groups (first match wins; sorted, not frequency).
MODEL_TYPE_ORDER = [
    MODEL_TYPE_TEXT,
    MODEL_TYPE_VISION,
    MODEL_TYPE_AUDIO,
    MODEL_TYPE_ASR,
    MODEL_TYPE_EMBEDDING,
    MODEL_TYPE_DRAFT,
    MODEL_TYPE_CODEC,
    MODEL_TYPE_MMPROJ,
]
VALID_MODEL_TYPES = set(MODEL_TYPE_ORDER)

# Human-readable labels, shared with the WebUI via /v1/models so both sides agree.
MODEL_TYPE_LABELS = {
    MODEL_TYPE_TEXT: "Text Generation",
    MODEL_TYPE_VISION: "Vision",
    MODEL_TYPE_AUDIO: "Audio Generation",
    MODEL_TYPE_ASR: "Speech to Text",
    MODEL_TYPE_EMBEDDING: "Embeddings",
    MODEL_TYPE_DRAFT: "Draft / Speculative",
    MODEL_TYPE_CODEC: "Audio Codec",
    MODEL_TYPE_MMPROJ: "Vision Projector",
}

# Component types: they are loaded by a host model, never served on their own. The WebUI
# shows an explanatory hint on these instead of blocking the Load button.
MODEL_TYPE_AUXILIARY = {MODEL_TYPE_MMPROJ, MODEL_TYPE_CODEC, MODEL_TYPE_DRAFT}

# general.architecture -> model type. Only architectures whose family unambiguously
# implies a modality are listed; a plain decoder (llama/qwen2/qwen3/qwen35/gemma4/...)
# is deliberately absent and falls through to the default TEXT.
_ARCH_TO_TYPE = {
    # Vision projectors (llama.cpp loads these as --mmproj).
    "clip": MODEL_TYPE_MMPROJ,
    "vit": MODEL_TYPE_MMPROJ,
    "siglip": MODEL_TYPE_MMPROJ,
    "mamba-clip": MODEL_TYPE_MMPROJ,
    # Speech-to-text.
    "whisper": MODEL_TYPE_ASR,
    "w2v2": MODEL_TYPE_ASR,
    "mmspeech": MODEL_TYPE_ASR,
    "moonshine": MODEL_TYPE_ASR,
    # Audio generation / codecs.
    "kokoro": MODEL_TYPE_AUDIO,
    "parler_tts": MODEL_TYPE_AUDIO,
    "outetts": MODEL_TYPE_AUDIO,
    "vits": MODEL_TYPE_AUDIO,
    # NOTE: audiocpp covers BOTH the Yue TTS LLM and its VAE half, so the filename codec
    # markers above (which run first) are what tell `yue2-vae-f32` from `yue2-3b-q8_0`.
    # Both report arch `audiocpp`; only the name separates them.
    "audiocpp": MODEL_TYPE_AUDIO,
    "snac": MODEL_TYPE_CODEC,
    "encodec": MODEL_TYPE_CODEC,
    "mimi": MODEL_TYPE_CODEC,
    "dac": MODEL_TYPE_CODEC,
    "wavtokenizer-dec": MODEL_TYPE_CODEC,
    "wavtokenizer-enc": MODEL_TYPE_CODEC,
    # Native multimodal LLMs (vision baked into the weights, no mmproj needed).
    "qwen2vl": MODEL_TYPE_VISION,
    "qwen3vl": MODEL_TYPE_VISION,
    "qwen-omni": MODEL_TYPE_VISION,
    "minicpmv": MODEL_TYPE_VISION,
    "internvl": MODEL_TYPE_VISION,
    "llava": MODEL_TYPE_VISION,
    "mllama": MODEL_TYPE_VISION,
    "pixtral": MODEL_TYPE_VISION,
    "moondream": MODEL_TYPE_VISION,
    "smolvlm": MODEL_TYPE_VISION,
    # Encoder-only backbones: serve embeddings, not chat.
    "bert": MODEL_TYPE_EMBEDDING,
    "nomic-bert": MODEL_TYPE_EMBEDDING,
    "jina-bert-v2": MODEL_TYPE_EMBEDDING,
    "modern-bert": MODEL_TYPE_EMBEDDING,
    "roberta": MODEL_TYPE_EMBEDDING,
    "xlm-roberta": MODEL_TYPE_EMBEDDING,
    "gte": MODEL_TYPE_EMBEDDING,
}

# Filename markers for auxiliary components. Checked BEFORE the architecture table on
# purpose: several codecs ship under a shared architecture with their host model
# (yue2-vae and yue2-3b are both `audiocpp`), so only the filename separates them.
_FILENAME_CODEC_MARKERS = ("vae", "wavtokenizer", "vocoder", "codec", "tokenizer-")
_FILENAME_DRAFT_MARKERS = ("mtp-", "-mtp", "draft-", "-draft", "speculative")
# Filename families that disambiguate an otherwise generic architecture.
_FILENAME_AUDIO_MARKERS = ("outetts", "kokoro", "tts")
_FILENAME_ASR_MARKERS = ("whisper", "w2v2", "moonshine")
_FILENAME_EMBEDDING_MARKERS = ("bge-", "-bge", "gte-", "e5-", "minilm", "nomic-embed", "embed-")


def _classify_model_type(gguf_path: Optional[Path], info: Dict[str, Any],
                         override: str = "") -> str:
    """Return the model type for a GGUF file.

    ``override`` (the sidecar's ``type_override``) always wins when it names a valid type —
    that is the escape hatch for weights whose metadata lies or is silent (OuteTTS
    reports ``olmo`` and carries no audio tensors; Laya reports ``modern-bert``).

    Precedence:
      1. explicit sidecar override
      2. vision projector  (general.type == "mmproj", else the filename marker)
      3. auxiliary codec / draft   (filename markers - see _FILENAME_CODEC_MARKERS)
      4. general.architecture      (the _ARCH_TO_TYPE table)
      5. filename family           (a generic arch like "olmo" with "outetts" in its name)
      6. text                      (a plain decoder - the overwhelmingly common case)

    Never raises: a missing file or empty info yields ``text``.
    """
    if override and override in VALID_MODEL_TYPES:
        return override

    name = (gguf_path.name if gguf_path is not None else "").lower()
    stem = (gguf_path.stem if gguf_path is not None else "").lower()
    arch = ((info or {}).get("arch") or "").strip().lower()
    gguf_type = ((info or {}).get("gguf_type") or "").strip().lower()

    # 2. Vision projector. general.type is authoritative when present; the filename marker
    #    catches older conversions that omit the key (and is what the original scan used).
    if gguf_type == "mmproj" or "mmproj" in name:
        return MODEL_TYPE_MMPROJ

    # 3. Auxiliary components, before the arch table (shared-arch codecs live here).
    if any(m in name for m in _FILENAME_CODEC_MARKERS):
        return MODEL_TYPE_CODEC
    if any(m in stem for m in _FILENAME_DRAFT_MARKERS):
        return MODEL_TYPE_DRAFT

    # 4. Architecture. Some engines name the draft head "<arch>-assistant".
    if arch.endswith("-assistant"):
        return MODEL_TYPE_DRAFT
    if arch in _ARCH_TO_TYPE:
        return _ARCH_TO_TYPE[arch]

    # 5. Filename families, for generic architectures.
    if any(m in stem for m in _FILENAME_AUDIO_MARKERS):
        return MODEL_TYPE_AUDIO
    if any(m in stem for m in _FILENAME_ASR_MARKERS):
        return MODEL_TYPE_ASR
    if any(m in stem for m in _FILENAME_EMBEDDING_MARKERS):
        return MODEL_TYPE_EMBEDDING

    # 6. Default: a plain text decoder.
    return MODEL_TYPE_TEXT

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
    max_ctx_size: Optional[int] = None   # max context window extracted from GGUF metadata
    n_gpu_layers: Optional[int] = 999   # 999 = offload all layers to GPU by default
    default: bool = False
    pinned: bool = False            # never auto-unload
    auto_save_state: bool = False   # auto save/restore slot 0 session state
    backend: str = ""               # backend build override folder name (empty = use global default)
    use_mmproj: bool = True         # enable --mmproj vision projector if present
    mmproj_file: str = ""           # path or filename of associated mmproj file
    estimated_vram_mb: int = 0      # hint; 0 = unknown
    tags: List[str] = field(default_factory=list)
    gguf_name: str = ""             # embedded name from GGUF metadata
    # ---- DERIVED classification (computed, never persisted) ----
    arch: str = ""                 # general.architecture from GGUF metadata (e.g. "qwen35")
    model_type: str = ""           # EFFECTIVE type the UI groups on (see _classify_model_type)
    # ---- USER choice (persisted) ----
    # Empty = follow auto-detection. Set it to pin a card to a type when detection guesses
    # wrong (OuteTTS reports arch "olmo"; Laya reports "modern-bert"). This is deliberately a
    # SEPARATE field from ``model_type``: conflating them meant a card whose type was merely
    # auto-detected got persisted on the next save and then reported back as a pinned
    # override, permanently freezing the classification.
    type_override: str = ""
    # --- Bridge-card fields (defaulted; llama cards ignore them) ---
    engine: str = "llama"           # "llama" (default) or a key of the config's engines: section
    engine_config: str = ""         # absolute path to the engine config JSON (bridge cards only)
    min_free_vram_mib: int = 0      # 0 = off; >0 → the engine's min_free_vram_flag (VRAM safety net)
    extra_args: List[str] = field(default_factory=list)  # appended verbatim to the serve argv

    @property
    def is_bridge(self) -> bool:
        return (self.engine or "llama") != "llama"

    @property
    def is_llama(self) -> bool:
        return (self.engine or "llama") == "llama"

    @classmethod
    def default_for(cls, gguf_path: Path) -> "ModelConfig":
        return cls(
            name=gguf_path.stem,
            description="",
            args="",
            ctx_size=8192,
            max_ctx_size=None,
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
            arch="",
            model_type="",
            type_override="",
            engine="llama",
            engine_config="",
            min_free_vram_mib=0,
            extra_args=[],
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

    # Fields recomputed on every scan from the file itself. Writing them into the sidecar
    # would freeze a stale classification and, worse, turn an auto-detected value into what
    # ``type_override`` claims is a deliberate user choice.
    DERIVED_FIELDS = frozenset({"arch", "model_type"})

    def save(self, gguf_path: Path) -> None:
        sidecar = gguf_path.with_suffix(gguf_path.suffix + ".json")
        data = {k: v for k, v in asdict(self).items() if k not in self.DERIVED_FIELDS}
        sidecar.write_text(json.dumps(data, indent=2))

    def save_engine_card(self, card_path: Path) -> None:
        """Write the editable card-overlay fields to a bridge card file.

        Persists only the fields a user can edit in the WebUI, plus ``engine`` and
        ``engine_config``. Those two are forced at scan time anyway; recording them
        here is what lets ONE shared cards_dir hold cards of different engines
        (they can no longer be derived from the card's location).
        """
        card = {
            "name": self.name,
            "description": self.description,
            "pinned": self.pinned,
            "default": self.default,
            "tags": self.tags,
            "min_free_vram_mib": self.min_free_vram_mib,
            "extra_args": self.extra_args,
            "type_override": self.type_override,
            "engine": self.engine or "llama",
            # Self-describing: lets each card file record its engine config path.
            "engine_config": self.engine_config,
        }
        card_path.write_text(json.dumps(card, indent=2))

    def to_launch_args(self, model_path: Path, port: int, extra_default_args: str, mmproj_full_path: Optional[Path] = None, slot_save_dir: Optional[Path] = None, model_id: Optional[str] = None) -> List[str]:
        """Build argv for llama-server."""
        argv = [
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(self.ctx_size),
        ]
        if self.n_gpu_layers is not None:
            argv += ["--n-gpu-layers", str(self.n_gpu_layers)]
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
# Engine registry
# --------------------------------------------------------------------------
# Defaults applied to a legacy `strata:` section when it is migrated to `engines.strata`.
# Strata's original hardcoded launcher behaviour is pure data here, so a second engine
# needs no code change. The argv template is _DEFAULT_ARGV (defined above).
_LEGACY_STRATA_OVERLAY = {"cors_origins": ["*"], "lazy_load": False, "idle_unload_s": 0}
_LEGACY_STRATA_ENV_SCRUB = ["STRATA_API_KEY", "STRATA_ALLOWED_HOSTS"]

# Pre-registry card directories, kept READ-ONLY as a migration aid: a card written
# before the engine registry (no "engine" key) must keep showing. Nothing is ever
# written here any more — new cards go to ./engine_cards (or an engine's cards_dir).
LEGACY_CARDS_DIRS = ("./strata_cards",)


def _normalize_engines(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``engines:`` registry, migrating a legacy ``strata:`` section.

    Migration is one dict build plus one warning. The shipped config.yaml has no
    ``strata:`` section, so this only fires for existing user setups.
    """
    engines = cfg.get("engines")
    if isinstance(engines, dict) and engines:
        return dict(engines)
    legacy = cfg.get("strata")
    if not legacy:
        return {}
    spec = dict(legacy)
    spec.update({
        "enabled": True,
        "label": "Strata",
        "argv": list(_DEFAULT_ARGV),
        "interpreter": legacy.get("venv_python", ""),
        "serve_script": legacy.get("serve_script", ""),
        "cwd": legacy.get("root_dir", ""),
        "derive_config": True,
        "config_overlay": dict(_LEGACY_STRATA_OVERLAY),
        "config_overlay_dynamic": {"aliases": ["{model_id}"]},
        "env_scrub": list(_LEGACY_STRATA_ENV_SCRUB),
        "min_free_vram_flag": "--min-free-vram-mib",
        "config_key": "exe",
        "ready_timeout_seconds": 600,
    })
    log.warning("legacy 'strata:' section migrated to 'engines.strata'")
    return {"strata": spec}


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
        self.save_state_max_keep = cfg["models"].get("save_state_max_keep", 5)
        self.default_auto_save_state = cfg["models"].get("auto_save_state", False)
        self.binary = cfg["llama_server"]["binary"]
        self.backends_dir = Path(cfg["llama_server"].get("backends_dir", "./backends")).resolve()
        self.selected_backend = cfg["llama_server"].get("selected_backend", "")
        self.default_args = cfg["llama_server"].get("default_args", "--cache-ram 16384 --kv-unified")
        # Read global default for n_gpu_layers; None=auto-detect, 999=all layers to GPU
        self.default_n_gpu_layers: Optional[int] = cfg["llama_server"].get("default_n_gpu_layers", 999)
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
        # id -> "explicit" | "auto": how the pairing above came to be. Must be recorded
        # at pairing time, because scan() then writes an inferred name back into
        # cfg.mmproj_file - after which the two cases are indistinguishable forever.
        self.mmproj_source: Dict[str, str] = {}
        # Every vision projector found by the last scan. Seeded here (and fingerprinted in
        # the loop below) so _list_mmproj_files() validates cheaply instead of re-reading
        # every header on the update_config path.
        self._mmproj_paths: List[Path] = []
        self._mmproj_sig: Tuple[Tuple[str, int, int], ...] = ()
        self.loaded: Dict[str, LoadedModel] = {}      # id -> loaded instance
        # Bridge-card engine registry: {} => feature fully inert (no engines: section).
        self.engines = _normalize_engines(cfg)
        self.cards_dir = Path(cfg.get("cards_dir", "./engine_cards"))
        self.temp_dir = Path(__file__).resolve().parent / "temp"
        self._lock = asyncio.Lock()
        self._loading_tasks: Dict[str, asyncio.Task] = {}
        # Per-model FIFO ticket queues (keyed by engine port). The TCP forwarder and
        # the state save/restore paths both acquire a ticket here so operations for a
        # model run in strict FIFO order. See model_queue.py.
        self.queue_registry = ModelQueueRegistry()
        self._stop = False
        self._bg_task: Optional[asyncio.Task] = None
        self._model_sizes: Dict[str, float] = {}

        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        self._poll_interval = cfg.get("gpu", {}).get("poll_interval_seconds", 2)
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_time: float = 0.0
        # Generation counter guarding the status cache against a stale commit.
        # _build_status() snapshots self.loaded and then awaits slow nvidia-smi
        # subprocesses on worker threads (100-500ms). A load/unload can land in that
        # window and invalidate the cache, after which the in-flight builder would write
        # its pre-mutation snapshot back with a FRESH timestamp — so the next /v1/status
        # serves stale data until the following poll (a just-unloaded model still shows
        # as loaded). Builders stamp the generation they read and only commit if it is
        # unchanged; every mutation bumps it via _invalidate_status().
        self._status_gen: int = 0

        # Reference to the MAIN running event loop, captured at startup so worker threads
        # (e.g. the raw TCP forwarder) can schedule async-only lookups onto it via
        # asyncio.run_coroutine_threadsafe. Never a fresh loop — see tcp_forwarder.py.
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Validate selected backend exists at startup
        self._validate_backend()
        # Collect (never raise) per-engine config errors so they can be surfaced by
        # GET /v1/engines and returned as a 400 at load time.
        self._validate_engines()

    def _validate_engines(self) -> None:
        """Validate each engine entry into ``spec["_errors"]``.

        WARN-ONLY by design: a bad bridge engine config must NEVER stop llama from
        booting. Errors are stored on the spec, logged once per engine, exposed via
        ``GET /v1/engines`` (so the WebUI can say why) and raised as a 400 in
        ``_do_load_model``.
        """
        for name, spec in self.engines.items():
            errors: List[str] = []
            if not spec.get("interpreter"):
                errors.append(f"engines.{name}.interpreter is not set")
            if not spec.get("serve_script"):
                errors.append(f"engines.{name}.serve_script is not set")
            for key in ("interpreter", "serve_script"):
                val = spec.get(key)
                if val and not Path(val).exists():
                    errors.append(f"engines.{name}.{key} path does not exist: {val}")
            argv = spec.get("argv")
            if not argv:
                errors.append(f"engines.{name}.argv is required (a launch template; there is no default)")
            elif not isinstance(argv, list):
                errors.append(f"engines.{name}.argv must be a list")
            else:
                # Expand the template once with dummy values so a bad {token} is a
                # boot warning instead of a launch-time crash.
                for token in (argv or []):
                    try:
                        _expand_token(str(token), {
                            "engine": name, "model_id": "model", "port": 0,
                            "host": self.host, "derived_config": "derived.json",
                            "interpreter": spec.get("interpreter", ""),
                            "serve_script": spec.get("serve_script", "")})
                    except ValueError as e:
                        errors.append(f"engines.{name}.argv: {e}")
            cfg_dir = spec.get("configs_dir")
            if cfg_dir:
                if not Path(cfg_dir).is_dir():
                    errors.append(f"engines.{name}.configs_dir is not a directory: {cfg_dir}")
                if not spec.get("config_key"):
                    errors.append(f"engines.{name}.config_key is required when configs_dir is set")
            if errors:
                spec["_errors"] = "; ".join(errors)
                log.warning(f"Engine '{name}' is misconfigured: {spec['_errors']}")
            else:
                spec.pop("_errors", None)

    def _validate_backend(self) -> None:
        """Validate that the selected_backend directory exists and contains llama-server binary."""
        if not self.selected_backend:
            return
        b_dir = self.backends_dir / self.selected_backend
        exe = b_dir / "llama-server.exe"
        if not exe.exists():
            exe = b_dir / "llama-server"
        if not exe.exists():
            log.warning(
                f"Selected backend '{self.selected_backend}' is configured but the binary was not found at: {exe}. "
                f"Model loading will fail until this is fixed or another backend is selected."
            )

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
        save_state_max_keep = models.get("save_state_max_keep", 5)
        if not isinstance(save_state_max_keep, int) or save_state_max_keep < 1:
            raise ValueError(f"Config 'models.save_state_max_keep' must be >= 1, got {save_state_max_keep!r}")

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

        # NOTE: the `engines:` section is deliberately NOT validated here — a broken
        # bridge engine must never block llama boot. See ModelManager._validate_engines.

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
    def _unique_display_name(self, desired: str, owner_mid: str, stem: str) -> str:
        """Return ``desired`` if it is unique (case-insensitive) among registered models.

        Because the editable ``name`` is the retrieval key, two models must never share
        one. When a collision is detected we disambiguate deterministically by appending
        the owner's filename stem (then a short numeric suffix if still needed), so names
        stay unique and re-scans are stable. Logs a warning whenever we disambiguate.
        """
        def collides(name: str) -> bool:
            n = name.lower()
            return any(
                o_mid != owner_mid and o_cfg.name and o_cfg.name.lower() == n
                for o_mid, o_cfg in self.models.items()
            )

        if not desired:
            desired = stem  # no usable metadata name -> fall back to the filename stem

        if not collides(desired):
            return desired

        candidate = f"{desired} ({stem})"
        i = 2
        while collides(candidate):
            candidate = f"{desired} ({stem}-{i})"
            i += 1
        if candidate != desired:
            log.warning(
                f"Model name '{desired}' already in use; disambiguating to '{candidate}' "
                f"for {owner_mid} (names must be unique as retrieval keys)."
            )
        return candidate

    def _list_mmproj_files(self) -> List[Path]:
        """Every vision projector under root_dir, sorted for determinism.

        ONE detection rule, shared with :meth:`scan`: ``_is_mmproj`` (metadata first,
        filename as fallback). This used to be a filename-only ``"mmproj" in name`` scan,
        which meant a projector not named ``mmproj*`` paired correctly at scan time and
        then silently lost its pairing the next time the user saved that model's config -
        two rules for one concept.

        Results are cached against a (path, mtime_ns, size) fingerprint of the .gguf tree.
        Re-validating costs one rglob + a stat per file (~70ms here) versus re-reading
        every GGUF header (~200ms), while still noticing a projector that was just dropped
        in - which ``update_config`` relies on. A fingerprint mismatch recomputes, so this
        can never serve a stale list.
        """
        files = sorted(self.root_dir.rglob("*.gguf"))
        sig: List[Tuple[str, int, int]] = []
        for p in files:
            try:
                st = p.stat()
            except OSError:
                continue
            sig.append((str(p), st.st_mtime_ns, st.st_size))
        fingerprint = tuple(sig)
        if fingerprint != self._mmproj_sig:
            self._mmproj_paths = [p for p in files if self._is_mmproj(p, _read_gguf_info(p))]
            self._mmproj_sig = fingerprint
        return list(self._mmproj_paths)

    def _best_mmproj_by_name(self, model_stem: str, candidates: List[Path]) -> Optional[Path]:
        """Pick the projector most likely to belong to a model, out of several.

        Two signals, in order:

        1. :meth:`_mmproj_affinity` - a normalized name relationship (exact / containment
           after stripping the ``mmproj-`` prefix). Where this finds any signal at all it
           is strictly more trustworthy than the positional heuristic, so it wins.
        2. Positional character agreement - only when NO candidate has a name
           relationship. This preserves the historical behaviour for real libraries whose
           projectors carry no model name at all (e.g. mmproj-F16 / mmproj-F32 beside a
           single model), where there is no better information and changing the choice
           would alter a working setup for no gain.

        The positional heuristic on its own gets things plainly wrong: pairing "alpha"
        with a directory containing mmproj-alpha.gguf and orphan-mmproj.gguf picked
        ORPHAN-mmproj, because it compares characters by index and the "mmproj-" prefix
        shifts every position ("alpha" vs "mmpro" scores 1, "alpha" vs "orpha" scores 3
        on pure coincidence). Normalized affinity scores that pair 1000 vs 0.
        """
        if not candidates:
            return None
        # key= avoids comparing Paths, which are not orderable.
        best_affinity = max(self._mmproj_affinity(model_stem, mp.stem) for mp in candidates)
        if best_affinity > 0:
            return max(candidates, key=lambda mp: self._mmproj_affinity(model_stem, mp.stem))
        p_stem = model_stem.lower()
        best_match = None
        best_score = -1
        for mp in candidates:
            mp_stem = mp.stem.lower()
            common_prefix_len = sum(1 for a, b in zip(p_stem, mp_stem) if a == b)
            if common_prefix_len > best_score:
                best_score = common_prefix_len
                best_match = mp
        return best_match

    @staticmethod
    def _strip_projector_prefix(stem: str) -> str:
        """``mmproj-alpha-f16`` -> ``alpha-f16``. Projector stems carry a role prefix
        that the model's own stem never has, so it must go before comparing."""
        s = stem.lower()
        for prefix in ("mmproj-", "mmproj_", "mmprojection-", "mmprojection_"):
            if s.startswith(prefix):
                return s[len(prefix):]
        return s

    def _mmproj_affinity(self, model_stem: str, proj_stem: str) -> int:
        """How strongly a projector belongs to a model, by normalized stem overlap.

        Returns 0 when there is no real relationship. Used to decide whether attaching a
        directory's lone projector to one of several models is defensible or a guess -
        the positional prefix heuristic cannot do this, because "mmproj-" itself creates
        spurious character collisions (index 2 is 'p' for every projector, so any model
        with 'p' at index 2 scored above zero).
        """
        m = self._strip_projector_prefix(model_stem)
        p = self._strip_projector_prefix(proj_stem)
        if not m or not p:
            return 0
        if m == p:
            return 1000
        if p in m or m in p:
            # Longer overlap wins, so a better match beats a mere substring hit.
            return 500 + len(p if p in m else m)
        return 0

    def _pair_mmproj(self, p: Path, cfg: "ModelConfig", mmproj_paths: List[Path],
                     *, eligible_siblings: Optional[List[str]] = None,
                     eligible_others: Optional[List[str]] = None) -> Optional[Path]:
        """Resolve the vision (mmproj) module for a single model.

        Single source of truth for the pairing rules; shared by scan() and
        update_config(). Returns the resolved Path or None. Does NOT mutate
        self.mmproj_paths or cfg - callers apply the mutation so both behave
        identically.

        Both ``eligible_*`` arguments are the stems of the OTHER models that could
        legitimately own a projector (i.e. not auxiliary components):

        ``eligible_siblings`` - same directory. Lets rule 2 detect that a directory's lone
            projector is contested instead of handing it to every model present.
        ``eligible_others``   - anywhere in the tree. Lets rule 3 do the same for the
            "exactly one projector exists" fallback.

        Both default to None, which preserves the historical unconditional behaviour.
        scan() supplies them; tests and update_config() do not.
        """
        # Auxiliary components are never AUTO-paired. A draft/MTP model or an audio codec
        # has no vision tower, so attaching a projector is always wrong - it produced a
        # real mispairing here: mtp-gemma-4-31B-it.gguf sits next to gemma-4-31B and
        # inherited its projector, gaining a misleading "vision" tag and a --mmproj it
        # cannot use. An explicitly configured mmproj_file is still honoured, because that
        # is the user's decision, not our inference.
        if (cfg.model_type or MODEL_TYPE_TEXT) in MODEL_TYPE_AUXILIARY and not cfg.mmproj_file:
            return None

        def _contested(lone: Path, rivals_stems: List[str]) -> Optional[Path]:
            """Return ``lone`` only if this model is its clear, unique namesake."""
            mine = self._mmproj_affinity(p.stem, lone.stem)
            if mine <= 0:
                return None
            rivals = [self._mmproj_affinity(s, lone.stem) for s in rivals_stems]
            return lone if not rivals or mine > max(rivals) else None

        paired_mmproj = None

        # 1. Check explicit mmproj_file setting
        if cfg.mmproj_file:
            # Strip: a sidecar edited by hand or filled by a copy/paste can carry a
            # trailing space ("mmproj-F16.gguf "). pathlib keeps it, so the resolved
            # Path would NOT equal the canonical one from rglob - silently breaking every
            # path comparison, and putting a stray space inside the --mmproj launch arg.
            # It only "worked" because Windows file APIs tolerate trailing spaces.
            candidate = Path(cfg.mmproj_file.strip())
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
                lone = sibling_mmprojs[0]
                if not eligible_siblings:
                    # Either this is the only eligible model here (the normal case), or
                    # the caller supplied no sibling context at all (tests/update_config).
                    paired_mmproj = lone
                else:
                    # Contested by other models in this directory: only a clear, unique
                    # name match justifies handing the projector over.
                    paired_mmproj = _contested(lone, eligible_siblings)
            elif len(sibling_mmprojs) > 1:
                paired_mmproj = self._best_mmproj_by_name(p.stem, sibling_mmprojs)

        # 3. Fallback: a single projector anywhere in the tree.
        #    Tree-wide rule, so its ambiguity check must be tree-wide too - previously it
        #    attached that one projector to EVERY model, bypassing rule 2's directory
        #    scoping entirely. That was the broadest form of the mispairing bug.
        if not paired_mmproj and len(mmproj_paths) == 1:
            lone = mmproj_paths[0]
            if not eligible_others:
                paired_mmproj = lone
            else:
                paired_mmproj = _contested(lone, eligible_others)

        return paired_mmproj

    def _cards_dir_for(self, engine_name: str) -> Path:
        """Card directory for an engine: its own ``cards_dir`` override, else the shared dir."""
        spec = self.engines.get(engine_name) or {}
        return Path(spec.get("cards_dir") or self.cards_dir)

    def _make_engine_config(self, cfg_path: Path, engine_name: str,
                            card_path: Optional[Path]) -> Optional[ModelConfig]:
        """Build a bridge-card ModelConfig from an engine config (single source of truth).

        Returns None when ``cfg_path`` is not a valid engine config (missing, unparseable,
        or lacking the engine's ``config_key``). The card overlay (if any) is applied FIRST,
        then engine / engine_config / auto_save_state are forced so a hand-edited card can
        never flip the engine or override the config path.
        """
        config_key = (self.engines.get(engine_name) or {}).get("config_key", "exe")
        if not cfg_path.is_file():
            return None
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        except Exception as e:
            log.warning(f"Skipping {engine_name} config {cfg_path}: {e}")
            return None
        if not isinstance(data, dict) or config_key not in data:
            log.warning(f"Skipping {engine_name} config {cfg_path}: missing '{config_key}' key (not a valid {engine_name} engine config)")
            return None                                   # not an engine config
        base = ModelConfig.default_for(cfg_path)
        if card_path is not None and card_path.exists():
            try:
                for k, v in json.loads(card_path.read_text(encoding="utf-8-sig")).items():
                    if hasattr(base, k):
                        setattr(base, k, v)
            except Exception as e:
                log.warning(f"Bad {engine_name} card {card_path}: {e}")
        # Force AFTER the overlay: a card file must not be able to re-engine the card.
        base.engine = engine_name
        base.engine_config = str(cfg_path)
        base.auto_save_state = False                       # slot save/restore is llama-only
        base.name = data.get("model_name") or base.name or cfg_path.stem
        # A bridge card has no GGUF to classify, so default it to text generation: every
        # engine configured today (llama, Strata) serves text tokens. The card overlay
        # applied above may already carry an override.
        if not base.model_type:
            base.model_type = base.type_override or MODEL_TYPE_TEXT
        return base

    def _scan_bridge_cards(self) -> None:
        """Additive: register engine configs as bridge cards. Never touches gguf entries.

        Inert when there is no enabled engine. Idempotent across rescans: a gguf card of
        the same id wins (inserted earlier in ``scan()``), and bridge ids are only added
        where free. No auto-seed: card files are only ever READ here (created by
        ``POST /v1/models/engine``), so a discovered card vanishes when its config file
        is deleted while a user-added card persists.
        """
        for name, spec in self.engines.items():
            if not spec.get("enabled"):
                continue
            cards_dir = self._cards_dir_for(name)
            # Loop 1 — discovered cards from configs_dir (only if configured).
            cfg_dir = spec.get("configs_dir")
            if cfg_dir:
                cfg_dir = Path(cfg_dir)
                if cfg_dir.is_dir():
                    for p in sorted(cfg_dir.glob("*.json")):
                        if p.name.endswith(".shared-settings.json"):
                            continue
                        cfg = self._make_engine_config(p, name, cards_dir / f"{p.stem}.json")
                        if cfg and p.stem not in self.models:
                            self.models[p.stem] = cfg
                            log.info(f"Discovered {name} bridge card: {p.stem} -> {p}")
                else:
                    log.warning(f"engines.{name}.configs_dir {cfg_dir} is not a directory; {name} cards disabled")
            # Loop 2 — user-added card files (persist across rescans). The legacy
            # pre-registry dirs are READ-ONLY fallbacks so existing cards keep showing;
            # all new saves go to the engine's cards_dir.
            for card_dir in self._card_dirs_to_read(name):
                if not card_dir.is_dir():
                    continue
                for card in sorted(card_dir.glob("*.json")):
                    if card.name.endswith(".shared-settings.json"):
                        continue
                    mid = self._register_card_file(card)
                    if mid:
                        log.info(f"Loaded {name} bridge card: {mid} -> {card}")

    def _card_dirs_to_read(self, engine_name: str) -> List[Path]:
        """Card dirs to scan for ``engine_name``: its own, plus the legacy pre-registry ones."""
        dirs = [self._cards_dir_for(engine_name)]
        dirs += [Path(d) for d in LEGACY_CARDS_DIRS]
        return dirs

    def _register_card_file(self, card: Path) -> Optional[str]:
        """Register one card file (read-only). Returns the model id, or None when skipped.

        Backward compatible with pre-registry card files: ``strata_config`` and a missing
        ``engine`` are accepted, and such a file is rewritten in the new shape on the next
        save/add (a scan never mutates user files).
        """
        try:
            data = json.loads(card.read_text(encoding="utf-8-sig")) or {}
        except Exception as e:
            log.warning(f"Bad bridge card {card}: {e}")
            return None
        engine = data.get("engine") or ("strata" if data.get("strata_config") else None)
        if not engine:
            log.warning(f"Bridge card {card} has no engine/engine_config; skipping")
            return None
        if engine not in self.engines:
            log.warning(f"Bridge card {card} names unknown engine '{engine}'; skipping "
                        f"(add an engines.{engine} section to config.yaml)")
            return None
        ec = data.get("engine_config") or data.get("strata_config")
        if not ec:
            log.warning(f"Bridge card {card} has no engine_config; skipping")
            return None
        cfg = self._make_engine_config(Path(ec), engine, card)
        mid = self._sanitize_model_id(Path(ec).stem)
        if cfg and mid not in self.models:
            self.models[mid] = cfg
            return mid
        return None

    def _is_mmproj(self, path: Path, info: Dict[str, Any]) -> bool:
        """True when a .gguf is a vision projector rather than a servable model.

        ``general.type == "mmproj"`` is authoritative when the converter wrote it; the
        filename marker is the fallback for files that omit the key (and preserves the
        original scan behaviour exactly for every file that already worked).
        """
        return (info.get("gguf_type") or "").strip().lower() == "mmproj" or "mmproj" in path.name.lower()

    def _apply_gguf_info(self, cfg: "ModelConfig", path: Path, mid: str,
                         info: Dict[str, Any]) -> None:
        """Fill display + classification fields on ``cfg`` from parsed GGUF metadata.

        Single source of truth shared by :meth:`scan` and :meth:`_ensure_gguf_name` so the
        two can never drift on how a model is named, sized or typed. Mutates ``cfg``.

        ``type_override`` is the user's explicit choice (or empty); ``model_type`` is
        always the effective type, so it is recomputed here rather than trusted from the
        sidecar. This is what lets "Auto-detect" in the WebUI take effect immediately.
        """
        cfg.model_type = _classify_model_type(path, info, cfg.type_override)
        cfg.arch = (info.get("arch") or "").strip()
        name = info.get("name")
        max_ctx = info.get("max_ctx")
        if name and not cfg.gguf_name:
            cfg.gguf_name = name
            if not cfg.name or cfg.name == path.stem:
                # Promote the GGUF internal name to display name, but keep it
                # unique (it is the retrieval key). Two quants of the same base
                # model share the same gguf_name, so disambiguate on collision.
                cfg.name = self._unique_display_name(name, mid, path.stem)
        if max_ctx and cfg.max_ctx_size is None:
            cfg.max_ctx_size = max_ctx

    def scan(self) -> List[str]:
        """Re-scan root_dir for .gguf files, partitioning out vision projectors and pairing them.

        Each file's header is read exactly once here: the same parse feeds the mmproj
        partition, the display name, the max-context figure and the model type.
        """
        all_ggufs = sorted(self.root_dir.rglob("*.gguf"))
        model_paths = []
        mmproj_paths = []
        # path -> parsed header, so the metadata pass below never re-opens the file.
        gguf_info: Dict[Path, Dict[str, Any]] = {}
        sig: List[Tuple[str, int, int]] = []
        for p in all_ggufs:
            info = _read_gguf_info(p)
            gguf_info[p] = info
            try:
                st = p.stat()
                sig.append((str(p), st.st_mtime_ns, st.st_size))
            except OSError:
                pass
            if self._is_mmproj(p, info):
                mmproj_paths.append(p)
            else:
                model_paths.append(p)
        # Publish the projector set AND its fingerprint, so _list_mmproj_files() (the
        # update_config path) starts warm and only recomputes when the tree changes.
        self._mmproj_paths = list(mmproj_paths)
        self._mmproj_sig = tuple(sig)

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

        self.models = found
        # Apply global default_n_gpu_layers to models that have n_gpu_layers=None
        for cfg in self.models.values():
            if cfg.n_gpu_layers is None:
                cfg.n_gpu_layers = self.default_n_gpu_layers
        # Cache model file sizes and apply GGUF metadata (name, context, arch, type)
        for mid in self.gguf_paths:
            path = self.gguf_paths[mid]
            try:
                self._model_sizes[mid] = path.stat().st_size / 1024 / 1024
            except Exception:
                self._model_sizes[mid] = 0.0
            # Reuse the header already parsed during the partition above.
            self._apply_gguf_info(self.models[mid], path, mid, gguf_info.get(path) or {})

        # Auto-pair vision projectors with sibling models. Runs AFTER classification so
        # _pair_mmproj knows which models are auxiliary components that must never be
        # paired, and so it can see the other candidates in each directory.
        eligible_by_dir: Dict[Path, List[str]] = {}
        eligible_all: List[str] = []
        for mid, p in self.gguf_paths.items():
            if (self.models[mid].model_type or MODEL_TYPE_TEXT) not in MODEL_TYPE_AUXILIARY:
                eligible_by_dir.setdefault(p.parent, []).append(p.stem)
                eligible_all.append(p.stem)
        self.mmproj_source = {}
        for mid, p in self.gguf_paths.items():
            cfg = self.models[mid]
            ctx = {
                "eligible_siblings": [s for s in eligible_by_dir.get(p.parent, []) if s != p.stem],
                "eligible_others": [s for s in eligible_all if s != p.stem],
            }
            # Read provenance BEFORE pairing: the write two lines below stamps an inferred
            # name into cfg.mmproj_file, which would make "auto" look "explicit" forever.
            explicit = bool((cfg.mmproj_file or "").strip())
            paired_mmproj = self._pair_mmproj(p, cfg, mmproj_paths, **ctx)
            if paired_mmproj:
                self.mmproj_paths[mid] = paired_mmproj
                self.mmproj_source[mid] = "explicit" if explicit else "auto"
                if not cfg.mmproj_file:
                    cfg.mmproj_file = paired_mmproj.name
            else:
                # Drop any pairing a previous scan inferred, so a model that has become
                # ineligible (or lost its projector) does not keep a stale --mmproj.
                self.mmproj_paths.pop(mid, None)

        # Additive bridge-card pass (inert with no engines: section). Runs AFTER the
        # gguf metadata loop; bridge ids never enter self.gguf_paths, so _read_gguf_info
        # is never called on a JSON file (R5). Bridge cards have no GGUF of their own, so
        # they carry a fixed type: every engine configured today serves text tokens.
        self._scan_bridge_cards()
        counts: Dict[str, int] = {}
        for cfg in self.models.values():
            counts[cfg.model_type or MODEL_TYPE_TEXT] = counts.get(cfg.model_type or MODEL_TYPE_TEXT, 0) + 1
        log.info(f"Scan complete: {len(self.models)} models found "
                 f"(filtered {len(mmproj_paths)} mmproj vision modules); "
                 f"types: {counts}")
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
                    # Order: exact filename -> stem/full id -> editable display name.
                    # gguf_name is intentionally NOT a request alias here: it is the
                    # GGUF's internal general.name, shared across quants of the same
                    # base model, so using it would let one model silently resolve to
                    # another (cross-model collision). It stays display-only metadata.
                    if target_lower == stem or target_lower == mid.lower():
                        return mid
                    if cfg.name and target_lower == cfg.name.lower():
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
            mmproj_src_snap = dict(self.mmproj_source)

        out = []
        for mid, cfg in snapshots:
            loaded = loaded_snap.get(mid)
            size_mb = sizes_snap.get(mid, 0.0)
            # Bridge cards have no llama binary; surface an empty resolved_binary for them.
            resolved_bin = self.resolve_binary(cfg) if cfg.is_llama else ""
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
                "engine": cfg.engine or "llama",
                "engine_config": cfg.engine_config,
                "min_free_vram_mib": cfg.min_free_vram_mib,
                "extra_args": cfg.extra_args,
                "resolved_binary": resolved_bin,
                # Classification. ``model_type`` is the effective type the UI groups and
                # badges on; ``model_type_override`` is the raw sidecar value, so the UI can
                # show "auto-detected" vs "pinned by you" and offer a reset. ``arch`` is the
                # raw general.architecture, useful context for an auto-detected type.
                "model_type": cfg.model_type or MODEL_TYPE_TEXT,
                "model_type_override": cfg.type_override,
                "arch": cfg.arch,
                "auxiliary": (cfg.model_type or MODEL_TYPE_TEXT) in MODEL_TYPE_AUXILIARY,
                "model_type_label": MODEL_TYPE_LABELS.get(
                    cfg.model_type or MODEL_TYPE_TEXT, "Text Generation"),
                "has_mmproj": has_mmproj,
                "use_mmproj": cfg.use_mmproj,
                "mmproj_file": cfg.mmproj_file,
                # Vision provenance. ``mmproj_file`` is only the CONFIGURED name and is
                # empty when a pairing was inferred; ``mmproj_resolved`` is the file that
                # actually reaches --mmproj. ``mmproj_source`` says which produced it, so
                # the UI can distinguish "you set this" from "we guessed this" - the two
                # were indistinguishable on the card, which is how a wrong pairing stayed
                # invisible.
                "mmproj_resolved": mmproj_p.name if has_mmproj else "",
                "mmproj_source": mmproj_src_snap.get(mid, ""),
                "ctx_size": cfg.ctx_size,
                "max_ctx_size": cfg.max_ctx_size,
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
        await self._ensure_gguf_name(mid)
        async with self._lock:
            return self.models.get(mid)

    async def _ensure_gguf_name(self, mid: str):
        """Read gguf_name, max_ctx_size, arch and model_type from GGUF metadata lazily.

        Caches the result on the config. Delegates to :meth:`_apply_gguf_info` so the lazy
        path and the scan path can never disagree about a model's name, context or type.
        """
        # Fast path: check cache without holding the lock across I/O.
        # ``model_type`` is the completion marker because classification ALWAYS assigns a
        # non-empty value (defaulting to "text"), whereas ``arch`` legitimately stays "" for
        # files that carry no general.architecture — gating on it would re-read forever.
        async with self._lock:
            cfg = self.models.get(mid)
            path = self.gguf_paths.get(mid)
            if not cfg or not path or (cfg.gguf_name and cfg.max_ctx_size is not None
                                      and cfg.model_type):
                return  # Already cached

        # Perform the (now-fast) blocking read outside the lock so we don't
        # serialize other lock users during file I/O.
        info = await asyncio.to_thread(_read_gguf_info, path)
        if not any(info.values()):
            return

        async with self._lock:
            cfg = self.models.get(mid)
            if not cfg:
                return
            self._apply_gguf_info(cfg, path, mid, info)

    async def update_config(self, model_id_or_alias: str, new_cfg: ModelConfig) -> ModelConfig:
        mid = await self.resolve_model_id(model_id_or_alias)
        if not mid:
            raise KeyError(model_id_or_alias)
        async with self._lock:
            if mid not in self.models:
                raise KeyError(model_id_or_alias)
            cfg = self.models[mid]
            is_bridge = cfg.is_bridge
            path = None if is_bridge else self.gguf_paths[mid]
            # Uniqueness guard: `name` is now the retrieval key, so it must stay
            # unique (case-insensitive). Reject a name already used by ANOTHER model.
            if new_cfg.name:
                target_name = new_cfg.name.lower()
                for other_mid, other_cfg in self.models.items():
                    if other_mid == mid:
                        continue
                    if other_cfg.name and other_cfg.name.lower() == target_name:
                        raise ValueError(
                            f"Model name '{new_cfg.name}' is already used by "
                            f"'{other_cfg.name}' (model ID '{other_mid}'); names must be unique (they are the retrieval key)."
                        )

        paired = None
        if is_bridge:
            # Bridge cards have no gguf path / mmproj pairing; persist only the editable
            # card-overlay fields (plus engine/engine_config) to the engine's cards_dir.
            save_path = self._cards_dir_for(cfg.engine) / f"{self._sanitize_model_id(mid)}.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            new_cfg.model_type = new_cfg.type_override or MODEL_TYPE_TEXT
            await asyncio.to_thread(new_cfg.save_engine_card, save_path)
        else:
            # Pair mmproj OUTSIDE the lock BEFORE saving, so an auto-filled
            # mmproj_file is written to the sidecar (matches scan() behavior).
            mmprojs = self._list_mmproj_files()
            # Provenance before the fill below, same reason as in scan().
            explicit = bool((new_cfg.mmproj_file or "").strip())
            paired = self._pair_mmproj(path, new_cfg, mmprojs)
            paired_source = ("explicit" if explicit else "auto") if paired is not None else ""
            if paired is not None and not new_cfg.mmproj_file:
                new_cfg.mmproj_file = paired.name
            # Re-derive the effective type so a just-changed override (or a reset back to
            # "Auto-detect") lands on the card without waiting for the next rescan.
            info = await asyncio.to_thread(_read_gguf_info, path)
            new_cfg.model_type = _classify_model_type(path, info, new_cfg.type_override)
            await asyncio.to_thread(new_cfg.save, path)

        async with self._lock:
            self.models[mid] = new_cfg
            if mid in self.loaded:
                self.loaded[mid].config = new_cfg
            # Sync the mmproj pairing computed above so --mmproj is correct at launch.
            if paired is not None:
                self.mmproj_paths[mid] = paired
                self.mmproj_source[mid] = paired_source
            else:
                # Stale/cleared: drop any previous pairing so --mmproj is dropped at launch.
                self.mmproj_paths.pop(mid, None)
                self.mmproj_source.pop(mid, None)
        return new_cfg

    # ---------------- sanitization ----------------
    @staticmethod
    def sanitize_safe_string(s: str) -> str:
        """Strip path separators and parent references from user-supplied strings."""
        if not s:
            return s
        return s.replace("\\", "").replace("/", "").replace("..", "").strip()

    # Alias for backwards compatibility with existing call sites
    @staticmethod
    def _sanitize_model_id(model_id: str) -> str:
        return ModelManager.sanitize_safe_string(model_id)

    @staticmethod
    def _sanitize_label(label: str) -> str:
        return ModelManager.sanitize_safe_string(label)

    def _require_llama_engine(self, cfg) -> None:
        """405 gate for llama-only operations (vision toggle).

        Only blocks a *known* non-llama card; an unknown id (cfg is None) falls through
        to the normal downstream 404/500 handling, matching the original inline behaviour.
        Note: state save/restore is NOT gated here — Strata's serve exposes the same
        /slots/0?action=save|restore endpoint as llama-server.
        """
        if cfg is not None and not cfg.is_llama:
            raise HTTPException(405, "Operation not supported for this engine")

    def _derived_config_path(self, engine: str, model_id: str) -> Path:
        """Deterministic path of an engine's derived config (write at launch, unlink at unload).

        Keyed on the engine name, so a card of a previously-installed engine keeps the
        exact same path (``temp/strata/<mid>.json``) and stale files still get cleaned.
        """
        return self.temp_dir / self._sanitize_model_id(engine) / f"{self._sanitize_model_id(model_id)}.json"

    def ready_timeout_for(self, model_id: str) -> float:
        """Ready timeout in seconds for a card: the engine's value for bridge cards, else 120s.

        Single source of truth for BRIDGE cards — the JIT path in tcp_forwarder.py uses this
        too, so the "bridge models load slowly" rule cannot drift. (Llama keeps its own
        default: 120s here vs the env-overridable _JIT_LOAD_TIMEOUT in the forwarder; that
        divergence is intentional — do not "unify" it.)
        """
        cfg = self.models.get(model_id)
        if cfg is None or cfg.is_llama:
            return 120.0
        return float(self.engines.get(cfg.engine, {}).get("ready_timeout_seconds", 600))

    # ---------------- process lifecycle & port allocation ----------------
    def _is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _allocate_port(self, model_id: Optional[str] = None) -> int:
        used_ports = {lm.port for lm in self.loaded.values()}
        # Stable per-model port: base_port + position of the model id in the sorted
        # model registry. A given card therefore tends to reuse the same port across
        # load/unload cycles and restarts, which keeps derived configs and manual
        # connections stable. Fall back to the monotonically-increasing pointer for
        # unknown models (preserves the old behaviour and the exhaustion tests).
        if model_id is not None and model_id in self.models:
            idx = sorted(self.models.keys()).index(model_id)
            port = self.base_port + idx
            if port > 65535:
                log.warning(f"Stable port {port} for {model_id} exceeds 65535; falling back to dynamic scan")
                port = self.next_port
        else:
            port = self.next_port
        for _ in range(500):
            if port not in used_ports and self._is_port_available(port):
                if model_id is None:
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
                port = self._allocate_port(model_id)
                # Engine dispatch: bridge cards build their own argv/cwd/env from the engine
                # registry; the llama path is byte-identical to before (env/cwd are None →
                # inherit). Every bridge failure mode below is a clean 400, never a
                # traceback — a misconfigured engine can never reach the launcher's
                # subscript, so the old KeyError('venv_python') class is dead.
                if cfg.is_bridge:
                    spec = self.engines.get(cfg.engine)
                    if spec is None:
                        raise HTTPException(400, f"Unknown engine '{cfg.engine}'. "
                                                f"Add an engines.{cfg.engine} section to config.yaml")
                    if spec.get("_errors"):
                        raise HTTPException(400, f"Engine '{cfg.engine}' is misconfigured: {spec['_errors']}")
                    if not Path(cfg.engine_config).is_file():
                        raise HTTPException(400, f"Engine config file missing for "
                                                f"'{cfg.engine}': {cfg.engine_config}")
                    artifact = Path(cfg.engine_config)
                    argv, cwd, env = _launch_bridge(self, cfg, cfg.engine, model_id, port)
                else:
                    artifact = self.gguf_paths[model_id]
                    argv, cwd, env = _launch_llama(self, cfg, model_id, artifact, port)

                log.info(f"Launching {cfg.engine or 'llama'} for {model_id} on port {port}")
                log.info("argv: " + " ".join(argv))

                # Spawn process (env/cwd are None for llama → inherit, byte-identical).
                if platform.system() == "Windows":
                    proc = subprocess.Popen(
                        argv,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                        env=env,
                        cwd=cwd,
                    )
                else:
                    proc = subprocess.Popen(
                        argv,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=env,
                        cwd=cwd,
                    )

                lm = LoadedModel(
                    model_id=model_id,
                    gguf_path=artifact,
                    config=cfg,
                    port=port,
                    process=proc,
                    pid=proc.pid,
                    starting_at=time.time(),
                )
                self.loaded[model_id] = lm
                self._start_log_thread(model_id, proc)
                self._invalidate_status()  # Force status cache rebuild after load

            # Wait until ready OUTSIDE the lock! Bridge first loads can be slow (multi-GPU),
            # so the timeout comes from the engine entry; llama keeps the original 120s.
            ready = await self._wait_until_ready(port, proc=proc,
                                                 timeout=self.ready_timeout_for(model_id))
            lm.ready = ready
            if not ready:
                log.error(f"Model {model_id} failed to become ready")
                async with self._lock:
                    await self._unload_unlocked(model_id)
                raise RuntimeError(f"Model {model_id} failed to start")

            log.info(f"Model {model_id} ready on port {port}")
            lm.touch()

            # Start this model's FIFO queue now that it is ready, so the auto-restore
            # below and any concurrent LLM request are ordered through one queue.
            queue = self.queue_registry.get_or_create(port, model_id)

            # Auto-restore session state if enabled and cached auto state exists.
            # Slot save/restore is llama-only: a bridge card must never trigger it, even
            # if the global default_auto_save_state is True.
            if (cfg.auto_save_state or self.default_auto_save_state) and cfg.is_llama:
                mid_clean = self._sanitize_model_id(model_id)
                state_path = self.save_state_dir / f"{mid_clean}.auto.bin"
                if state_path.exists():
                    try:
                        log.info(f"Auto-restoring session state for {model_id} from {state_path}...")
                        # Hold the FIFO ticket so the restore runs before any LLM request.
                        ticket = await asyncio.to_thread(queue.acquire_ticket, timeout=60.0)
                        try:
                            await self._perform_slot_restore(port, state_path, timeout=30.0)
                            lm.state_path = state_path
                        finally:
                            queue.release_ticket(ticket)
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
        self._invalidate_status()  # Force status cache rebuild on next request
        log.info(f"Unloading {model_id} (port {lm.port})")

        # Auto-save session state if enabled. Slot save/restore is llama-only: a strata
        # card must never trigger it (even if the global default_auto_save_state is True).
        # NOTE: intentionally NOT routed through the FIFO ticket queue. This is a teardown
        # path — no later LLM request depends on this save — and the engine's own FIFO
        # already serializes it against any in-flight generation. Ticketing it here would
        # hold the asyncio lock across a potentially long ticket wait for no ordering gain.
        if lm.ready and (lm.config.auto_save_state or self.default_auto_save_state) and lm.config.is_llama:
            try:
                mid_clean = self._sanitize_model_id(model_id)
                state_path = self.save_state_dir / f"{mid_clean}.auto.bin"
                log.info(f"Auto-saving session state for {model_id} to {state_path}...")
                await self._perform_slot_save(lm.port, state_path, timeout=60.0)
            except Exception as e:
                log.warning(f"Auto-save failed for {model_id}: {e}")

        # Stop and drop this model's FIFO queue before killing the process, so no new
        # ticket is granted to a port that is about to die and any waiters are woken.
        self.queue_registry.remove(lm.port)

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

        # Bridge cards only: remove the derived config we wrote at launch. The path is
        # deterministic, so this also covers LRU-evict, force-reload, and timeout-failure
        # unloads (all route through here). The user's engine JSON is never touched.
        if lm.config.is_bridge:
            try:
                self._derived_config_path(lm.config.engine, model_id).unlink(missing_ok=True)
            except Exception as e:
                log.warning(f"Failed to delete derived engine config for {model_id}: {e}")

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

    # ---------------- state save/load ----------------
    async def _perform_slot_action(self, action: str, port: int, state_path: Path, timeout: float = 120.0) -> bool:
        """Attempt a slot action (save/restore) trying supported endpoint formats."""
        base = f"http://127.0.0.1:{port}"
        attempts = [
            ("/slots/0", {"filename": state_path.name}, f"URL param: ?action={action}"),
            ("/slots", {"id_slot": 0, "filename": state_path.name}, f"URL param: ?action={action}"),
            ("/slots", {"action": action, "id_slot": 0, "filename": state_path.name}, "JSON body params"),
        ]

        last_err = None
        for path, payload, desc in attempts:
            try:
                url = f"{base}{path}?action={action}"
                r = await self.client.post(url, json=payload, timeout=timeout)
                if r.status_code == 200:
                    log.debug(f"Slot {action} succeeded via {desc}")
                    return True
                log.warning(f"Slot {action} attempt failed [{desc}]: HTTP {r.status_code} — {r.text[:500]}")
                last_err = f"[{desc}] HTTP {r.status_code}: {r.text}"
                # 5xx = server is reachable but can't do the operation (e.g. OOM).
                # Retrying with a different URL format will hit the same server and
                # either get the same 5xx or a misleading 404 (endpoint doesn't exist).
                if r.status_code >= 500:
                    break
            except Exception as e:
                log.warning(f"Slot {action} attempt failed [{desc}]: {e}")
                last_err = f"[{desc}] {e}"

        raise RuntimeError(f"{action} failed across all slot endpoints. Last error: {last_err}")

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
            queue = self.queue_registry.get_or_create(lm.port, model_id)
            ticket = await asyncio.to_thread(queue.acquire_ticket, timeout=120.0)
            try:
                await self._perform_slot_save(lm.port, state_path, timeout=120.0)
            finally:
                queue.release_ticket(ticket)
        except Exception as e:
            raise HTTPException(500, f"save failed: {e}")

        lm.state_path = state_path

        # Clean up old states for this model (best-effort)
        try:
            await self._cleanup_old_states(model_id)
        except Exception as e:
            log.debug(f"State cleanup failed (non-critical): {e}")

        return state_path

    async def load_state(self, model_id_input: str, label: str = "default") -> Path:
        model_id = await self.resolve_model_id(model_id_input)
        if not model_id:
            raise HTTPException(404, f"Unknown model: {model_id_input}")

        async with self._lock:
            lm = self.loaded.get(model_id)

        if lm is None or not lm.ready:
            log.info(f"Auto-loading model '{model_id}' before state restore")
            try:
                lm = await asyncio.wait_for(self.load_model(model_id), timeout=60.0)
            except asyncio.TimeoutError:
                raise HTTPException(504, f"Timeout loading model '{model_id}' for state restore")
            except Exception as e:
                log.error(f"Failed to load model '{model_id}' before state restore: {e}")
                raise HTTPException(503, f"Failed to load model before state restore: {e}")

        async with self._lock:
            lm = self.loaded[model_id]
        mid_clean = self._sanitize_model_id(model_id)
        label_clean = self._sanitize_label(label)
        state_path = self.save_state_dir / f"{mid_clean}.{label_clean}.bin"

        try:
            queue = self.queue_registry.get_or_create(lm.port, model_id)
            ticket = await asyncio.to_thread(queue.acquire_ticket, timeout=120.0)
            try:
                await self._perform_slot_restore(lm.port, state_path, timeout=120.0)
            finally:
                queue.release_ticket(ticket)
        except Exception as e:
            raise HTTPException(500, f"restore failed: {e}")

        lm.state_path = state_path
        
        return state_path

    async def list_states(self, model_id_input: str) -> List[str]:
        model_id = await self.resolve_model_id(model_id_input) or model_id_input
        mid_clean = self._sanitize_model_id(model_id)
        out = []
        for p in self.save_state_dir.glob(f"{mid_clean}.*.bin"):
            label = p.stem[len(mid_clean) + 1:]   # Strip the known model-id prefix and its trailing dot.
            out.append(label)
        return out

    async def delete_state(self, model_id_input: str, label: str) -> None:
        """Delete a state file for the given model and label."""
        model_id = await self.resolve_model_id(model_id_input) or model_id_input
        mid_clean = self._sanitize_model_id(model_id)
        label_clean = self._sanitize_label(label)
        state_path = self.save_state_dir / f"{mid_clean}.{label_clean}.bin"

        if not state_path.exists():
            raise HTTPException(404, f"State file not found: {state_path.name}")

        try:
            state_path.unlink()
            log.info(f"Deleted state file: {state_path.name}")
        except Exception as e:
            raise HTTPException(500, f"Failed to delete state file: {e}")

    async def _cleanup_old_states(self, model_id_input: str) -> None:
        """Delete old state files for a model, keeping only the latest N files.

        Cleanup is per-model (not per-agent), so all agents sharing the same model
        share the state file pool. Keeps the latest N (models.save_state_max_keep,
        default 5) state files by mtime.
        """
        max_states = max(1, self.save_state_max_keep)  # Ensure minimum of 1 file kept
        mid_clean = self._sanitize_model_id(model_id_input)

        if not mid_clean:
            log.debug("Skipping cleanup: empty sanitized model ID")
            return

        try:
            # Find all state files for this model using prefix/suffix matching
            # instead of glob to avoid pattern injection issues with special chars
            state_files = [
                p for p in self.save_state_dir.iterdir()
                if (p.is_file() and 
                    p.name.startswith(f"{mid_clean}.") and 
                    p.name.endswith(".bin"))
            ]

            if len(state_files) <= max_states:
                return

            # Sort by modification time (newest first)
            state_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Delete files beyond the limit
            for old_file in state_files[max_states:]:
                try:
                    size_mb = old_file.stat().st_size / (1024 * 1024)
                    old_file.unlink()
                    log.info(f"Cleaned up old state file: {old_file.name} ({size_mb:.1f} MB)")
                except Exception as e:
                    log.warning(f"Failed to delete old state file {old_file.absolute()} for model={model_id_input}: {e}")

        except Exception as e:
            # Best-effort cleanup - don't fail the save if cleanup fails
            log.debug(f"State cleanup failed for model={model_id_input}: {e}")

    # ---------------- status cache ----------------
    async def _build_status(self) -> Dict[str, Any]:
        """Build a full status snapshot (cached to avoid redundant nvidia-smi calls)."""
        # Round-8 fix: gpus() and per_model_ram_vram() each run a SYNCHRONOUS
        # subprocess.run(["nvidia-smi", ...]) (timeout=5). Called from the main event
        # loop by _status_cache_updater every poll_interval (default 2s), that blocking
        # call froze the whole loop for as long as nvidia-smi took (100-500ms+ under
        # GPU/driver contention on this dual-GPU Windows box). During the freeze the SSE
        # streaming gen() could not run, so upstream chunks piled up in the socket buffer
        # and were yielded all at once when the loop unblocked -> end-of-stream burst.
        # Move both blocking subprocess calls onto a worker thread via asyncio.to_thread
        # (same args/timeout => byte-identical behavior) so the loop stays free. The sync
        # method bodies are left untouched; this is their ONLY caller, so nothing else
        # needs to change and there is no "coroutine never awaited" risk.
        backends = self.list_backends()
        async with self._lock:
            loaded_snap = list(self.loaded.items())
            total = len(self.models)
        # Run the two blocking nvidia-smi calls concurrently on worker threads.
        gpus_list, per = await asyncio.gather(
            asyncio.to_thread(self.gpus),
            asyncio.to_thread(self.per_model_ram_vram, loaded_items=loaded_snap),
        )
        gpus = [asdict(g) for g in gpus_list]
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

    # ---------------- status cache ----------------
    def _invalidate_status(self) -> None:
        """Drop the cached status and bump the generation.

        Call under ``self._lock`` whenever ``self.loaded`` changes. The bump is what
        stops an in-flight :meth:`_build_status` from committing a pre-mutation snapshot
        back into the cache after we have just cleared it.
        """
        self._status_gen += 1
        self._status_cache = None

    async def _status_cache_updater(self) -> None:
        """Background task that refreshes status cache at poll_interval intervals."""
        while not self._stop:
            try:
                async with self._lock:
                    gen = self._status_gen
                status = await self._build_status()
                async with self._lock:
                    # Discard the result if anything mutated the loaded set while we were
                    # building — otherwise we'd resurrect the invalidated snapshot.
                    if gen == self._status_gen:
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
            gen = self._status_gen
        if needs_refresh:
            status = await self._build_status()
            async with self._lock:
                if gen == self._status_gen:
                    self._status_cache_time = time.time()
                    self._status_cache = status
            # A concurrent mutation means this snapshot is already outdated; return what we
            # built for this caller (it answered from a consistent point in time) but leave
            # the cache invalidated so the next reader rebuilds.
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
        # Defensive: stop any FIFO queues not already removed by unload_all and wake
        # any waiters so shutdown cannot hang on a held ticket.
        self.queue_registry.stop_all()
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

# ---------------------------------------------------------------------------
# CORS — ROUND-9 FIX (SSE end-of-stream burst root cause)
# ---------------------------------------------------------------------------
# The LIVE wire probe (round 9, _probe_wire_headers.py) showed a genuine stream:true
# response through the proxy as: content-type=text/event-stream BUT content-length set
# + NO transfer-encoding:chunked + ALL bytes in ONE read after full prefill. A proper
# streaming response must be chunked with no content-length and incremental per-chunk
# arrival. The one structural difference between that live path and the smooth direct
# llama-server path is the global `CORSMiddleware` (a Starlette BaseHTTPMiddleware
# subclass) sitting in front of the StreamingResponse.
#
# Mechanism caveat (verified against starlette 0.46.2 source + an isolated loopback A/B,
# see _sanity_cors_middleware.py): BaseHTTPMiddleware routes the inner response through
# an anyio memory-stream and re-wraps it; on this host/version that wrapper is what
# produced the single buffered Content-Length body (the exact version-sensitive behavior
# that breaks incremental SSE). It did NOT reproduce in a bare loopback test, so the
# trigger is host/version-specific — but removing the BaseHTTPMiddleware wrapper removes
# the entire class of response-streaming interaction at once. That is why TCP_NODELAY /
# nvidia-smi-to_thread / sleep(0) all failed: none of them touch whole-body framing.
#
# The proxied /v1/* endpoints ALREADY attach manual `cors_headers` (see the proxy
# handler, ~L1086), so the global middleware is redundant for those routes. We now use a
# minimal pure-ASGI CORS middleware that injects headers into the single
# `http.response.start` message and passes every `http.response.body` message through
# UNTOUCHED — it never consumes/collects the body, so StreamingResponse stays chunked
# and incremental. Preflight OPTIONS is short-circuited (200 + CORS headers, no body).
#
# Env toggles (all default to the NEW non-buffering behavior):
#   AUTOLOADER_DISABLE_CORS_MIDDLEWARE=1  -> NO global CORS middleware at all.
#       CONFIRM TEST: proves the old CORSMiddleware was the buffering culprit. The
#       proxied /v1/* routes still carry their manual cors_headers, so a browser can
#       still consume them; only non-proxied routes lose CORS during this test.
#   AUTOLOADER_CORS=starlette             -> restore the old (buffering) CORSMiddleware
#       for A/B comparison against the new one.
#   AUTOLOADER_CORS_CUSTOM=0              -> use manual cors_headers only (no global).
# ---------------------------------------------------------------------------

class _NonBufferingCORSMiddleware:
    """Pure-ASGI CORS that adds headers WITHOUT wrapping the response stream.

    Why not Starlette's CORSMiddleware: it subclasses BaseHTTPMiddleware, which routes
    the inner response through an anyio memory-stream and re-wraps it before sending.
    On this host/starlette version that wrapper is what turned the StreamingResponse
    into a single buffered Content-Length body (see the round-9 wire probe + the
    isolated A/B in _sanity_cors_middleware.py). Here we only touch the
    `http.response.start` message; every `http.response.body` message is forwarded
    verbatim, so uvicorn keeps `transfer-encoding: chunked` and streams each SSE frame
    as it arrives — no response re-wrapping at all.

    Header policy (deliberately mirrors the old config's *effective* browser behavior):
      - Allow-Origin: request Origin if present, else "*". We do NOT emit the
        invalid `Allow-Origin: *` + `Allow-Credentials: true` combo; credentialed
        clients get an echoed Origin, non-credentialed get "*".
      - Preflight OPTIONS: 200 + Allow-Methods/Headers/Max-Age, no body.
    """

    _ALL_METHODS = "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        origin = headers.get("origin")

        # Echo the Origin only if it is ASCII-safe (HTTP header values must be ASCII);
        # otherwise fall back to "*" rather than emitting a malformed/non-ASCII value.
        safe_origin = origin
        if safe_origin is not None:
            try:
                safe_origin.encode("ascii")
            except UnicodeEncodeError:
                safe_origin = None
        allow_origin = (safe_origin or "*").encode("ascii")

        # --- Preflight: answer immediately, never touch the body. ---
        if scope["method"] == "OPTIONS" and "access-control-request-method" in headers:
            resp_headers = [
                (b"access-control-allow-origin", allow_origin),
                (b"access-control-allow-methods", self._ALL_METHODS.encode()),
                (b"access-control-allow-headers", b"*"),
                (b"access-control-max-age", b"600"),
            ]
            if safe_origin is not None:
                resp_headers.append((b"vary", b"Origin"))
            await send({"type": "http.response.start", "status": 200, "headers": resp_headers})
            await send({"type": "http.response.body", "body": b""})
            return

        # --- Simple / actual response: inject headers into the start message only. ---
        # IMPORTANT: some routes (the proxied /v1/* handler, ~L1086) already set their own
        # manual `Access-Control-Allow-Origin`. A duplicate ACAO header is rejected by
        # browsers, so we REPLACE any existing ACAO (and Vary:Origin) instead of appending.

        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers") or [])
                # Drop pre-existing ACAO / Vary-Origin so we never emit duplicates.
                keep = []
                for name, value in hdrs:
                    ln = name.lower()
                    if ln == b"access-control-allow-origin":
                        continue
                    if ln == b"vary" and b"origin" in value.lower():
                        # Keep a Vary that also lists other fields (e.g. "Accept-Encoding, Origin").
                        parts = [p.strip().lower() for p in value.split(b",")]
                        if len(parts) > 1:
                            keep.append((name, value))
                        continue
                    keep.append((name, value))
                keep.append((b"access-control-allow-origin", allow_origin))
                if safe_origin is not None:
                    keep.append((b"vary", b"Origin"))
                message["headers"] = keep
            await send(message)

        await self.app(scope, receive, send_with_cors)


def _install_cors():
    """Install the global CORS middleware per env toggles (see block above)."""
    if os.environ.get("AUTOLOADER_DISABLE_CORS_MIDDLEWARE", "0") == "1":
        # Confirm test: no global CORS. Proxied /v1/* still send manual cors_headers.
        log.info("[CORS] AUTOLOADER_DISABLE_CORS_MIDDLEWARE=1 -> NO global CORS middleware (confirm test)")
        return
    mode = os.environ.get("AUTOLOADER_CORS", "custom").lower()
    if mode == "starlette":
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        log.info("[CORS] AUTOLOADER_CORS=starlette -> legacy buffering CORSMiddleware (A/B only)")
    elif os.environ.get("AUTOLOADER_CORS_CUSTOM", "1") != "0":
        app.add_middleware(_NonBufferingCORSMiddleware)
        log.info("[CORS] non-buffering ASGI CORS middleware installed (Round-9 fix)")
    else:
        log.info("[CORS] AUTOLOADER_CORS_CUSTOM=0 -> no global CORS (manual cors_headers only)")


_install_cors()

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# ---------- models payloads ----------
class ModelConfigUpdate(BaseModel):
    # `model_type` collides with pydantic's protected `model_` namespace. The field name is
    # load-bearing (it is the sidecar key and the WebUI field id), so opt out explicitly
    # instead of renaming it.
    model_config = ConfigDict(protected_namespaces=())

    name: Optional[str] = None
    description: Optional[str] = None
    args: Optional[str] = None
    ctx_size: Optional[int] = None
    max_ctx_size: Optional[int] = None
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
    # Model type override. Accepts any of VALID_MODEL_TYPES; an empty string RESETS the
    # card back to auto-detection (rather than storing an invalid type). An unknown value
    # is rejected with a 400 instead of being silently dropped.
    model_type: Optional[str] = None
    # Bridge-card fields (llama cards ignore them). The engine NAME is intentionally
    # ABSENT: a card must not be able to re-engine itself (see _make_engine_config).
    engine_config: Optional[str] = None
    min_free_vram_mib: Optional[int] = None
    extra_args: Optional[List[str]] = None

class EngineCardCreate(BaseModel):
    engine: str
    engine_config: str
    name: Optional[str] = None

class StrataCardCreate(BaseModel):
    """DEPRECATED body schema for POST /v1/models/strata. Use EngineCardCreate."""
    strata_config: str
    name: Optional[str] = None

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
        "save_state_max_keep": manager.save_state_max_keep,
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
    if "save_state_max_keep" in body:
        v = body["save_state_max_keep"]
        if not isinstance(v, int) or v < 1:
            errors.append("'save_state_max_keep' must be >= 1")
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
    if "save_state_max_keep" in body:
        manager.save_state_max_keep = body["save_state_max_keep"]
        changed.append("save_state_max_keep")
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
        if "save_state_max_keep" in body:
            cfg["models"]["save_state_max_keep"] = body["save_state_max_keep"]
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
            # Vision provenance — the resolved file and how the link was decided.
            "mmproj_resolved": m["mmproj_resolved"],
            "mmproj_source": m["mmproj_source"],
            "ctx_size": m["ctx_size"],
            "max_ctx_size": m["max_ctx_size"],
            "n_gpu_layers": m["n_gpu_layers"],
            "estimated_vram_mb": m["estimated_vram_mb"],
            "args": m["args"],
            "gguf_name": m["gguf_name"],
            # Classification (see _classify_model_type). Required by the WebUI field
            # whitelist, which drives the type chips, the group headers and the card badge.
            "model_type": m["model_type"],
            "model_type_override": m["model_type_override"],
            "model_type_label": m["model_type_label"],
            "arch": m["arch"],
            "auxiliary": m["auxiliary"],
            # Bridge-card fields (defaulted for llama cards). Pass-through only —
            # required by the WebUI field whitelist (F1); the llama payload shape is unchanged.
            "engine": m["engine"],
            "engine_config": m["engine_config"],
            "resolved_binary": m["resolved_binary"],
            "min_free_vram_mib": m["min_free_vram_mib"],
            "extra_args": m["extra_args"],
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

@app.get("/v1/models/{model_id}/port")
async def get_model_port(model_id: str):
    """Return the llama-server port for a model so clients can connect directly.

    Lets AC bypass the proxy data path entirely (no double async relay) by
    querying this quick JSON lookup, then streaming straight to llama-server.
    Unknown model -> 404. Known but not loaded -> 503 (client should fall back
    to the proxy path, which triggers a JIT load).
    """
    mid = await manager.resolve_model_id(model_id)
    if not mid:
        raise HTTPException(404, f"Model not found: {model_id}")
    async with manager._lock:
        lm = manager.loaded.get(mid)
        ready = bool(lm and lm.ready)
        port = lm.port if lm else None
    if not ready:
        raise HTTPException(503, f"Model '{mid}' is not loaded")
    return {"port": port, "model_id": mid}


# Headers that describe THIS hop's framing or routing and must not be replayed
# upstream: the engine sees its own connection, and its own content-length.
_RAW_STRIP_HEADERS = {
    "host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
    "content-encoding", "accept-encoding",
}


@app.api_route("/v1/raw/{model_id}/{path:path}",
               methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def raw_passthrough(model_id: str, path: str, request: Request):
    """Proxy any request straight to a loaded engine's HTTP port.

    Documented in both READMEs (and used there for manual slot save/restore) but never
    actually registered - every call 404'd. This is that endpoint.

    Reach for it for native llama.cpp endpoints the autoloader does not wrap:
    ``/tokenize``, ``/detokenize``, ``/slots``, ``/props``, ``/lora-adapters``...

        GET  /v1/raw/<model>/slots
        POST /v1/raw/<model>/slots/0?action=save&filename=state.bin

    Unlike ``/v1/models/{id}/load`` this does NOT JIT-load: the model must already be
    resident, because a proxy that silently spends a minute loading a multi-GB model on a
    typo is worse than a clear 503 telling you to Load first.

    Security: this exposes the engine's full HTTP surface, including slot save/restore
    that writes files next to the engine. It inherits the autoloader's trust model - bound
    to launcher.host (127.0.0.1 by default), no auth - so do not expose it on a shared
    interface. Any request reaching it has already passed the TCP forwarder.
    """
    mid = await manager.resolve_model_id(model_id)
    if not mid:
        raise HTTPException(404, f"Model not found: {model_id}")
    async with manager._lock:
        lm = manager.loaded.get(mid)
        ready = bool(lm and lm.ready)
        port = lm.port if lm else None
    if not ready or port is None:
        raise HTTPException(503, f"Model '{mid}' is not loaded - POST /v1/models/{mid}/load first")

    url = f"http://127.0.0.1:{port}/{path}"
    if request.url.query:
        url += "?" + request.url.query

    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _RAW_STRIP_HEADERS}

    try:
        # Generous timeout: /slots?action=save can take tens of seconds on a big model.
        upstream = await manager.client.request(
            request.method, url, content=body or None, headers=headers,
            timeout=httpx.Timeout(300.0, connect=10.0))
    except Exception as e:
        log.warning(f"raw passthrough to {url} failed: {e}")
        raise HTTPException(502, f"Engine request failed: {e}")

    async def _relay():
        # Stream so a large body (a saved slot, a props dump) never sits in memory twice.
        if upstream.request.method == "HEAD":
            return
        async for chunk in upstream.aiter_bytes():
            yield chunk

    resp_headers = {k: v for k, v in upstream.headers.items()
                    if k.lower() not in _RAW_STRIP_HEADERS}
    return StreamingResponse(
        _relay(), status_code=upstream.status_code, headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


@app.post("/v1/models/{model_id}/interrupt")
async def interrupt_model(model_id: str, hard: bool = False):
    """Best-effort in-flight generation interrupt for an active model.

    Preferred (soft) path: closes the active upstream connection(s) to that model's resident
    llama-server. Modern llama-server builds honor a dropped client connection and stop
    generating ("srv stop: cancel task") while keeping the model loaded, so the next request
    does not pay a JIT reload. This is best-effort: on some builds/states generation may
    continue until it naturally completes.

    With ``?hard=true`` (or when no forwarder/sockets are available) it falls back to the old
    behaviour of unloading (killing + reclaiming) the resident process, guaranteeing the GPU
    stops at the cost of a fresh JIT load next time.
    """
    mid = await manager.resolve_model_id(model_id)
    if not mid:
        raise HTTPException(404, f"Model not found: {model_id}")

    async with manager._lock:
        lm = manager.loaded.get(mid)
        if lm is None or not lm.ready:
            return {"status": "not-loaded", "model_id": mid, "cancelled_requests": 0}

        cancelled = 0
        fw = globals().get("_forwarder")
        if fw is not None:
            try:
                cancelled = fw.cancel_active(lm.port)
            except Exception as e:
                log.warning(f"Soft interrupt failed for {model_id}: {e}")

        if hard:
            try:
                await manager._unload_unlocked(mid)
                return {"status": "interrupted", "model_id": mid, "mode": "hard",
                        "cancelled_requests": cancelled}
            except Exception as e:
                log.warning(f"Interrupt (hard) failed for {model_id}: {e}")
                raise HTTPException(500, f"Interrupt failed for {model_id}: {e}")

        return {"status": "interrupted", "model_id": mid, "mode": "soft",
                "cancelled_requests": cancelled}

@app.put("/v1/models/{model_id}/config")
async def update_model_config(model_id: str, body: ModelConfigUpdate):
    mid = await manager.resolve_model_id(model_id)
    if not mid:
        raise HTTPException(404, "Unknown model")
    cfg = await manager.get_config(mid)
    for k, v in body.model_dump(exclude_none=True).items():
        if k == "model_type":
            # Maps the API/WebUI field onto the PERSISTED override, not onto the derived
            # effective type. "" is the explicit "reset to auto-detection" value and must
            # survive; anything else must name a real type.
            v = (v or "").strip().lower()
            if v and v not in VALID_MODEL_TYPES:
                raise HTTPException(
                    400, f"Unknown model_type '{v}'. Valid: {', '.join(sorted(VALID_MODEL_TYPES))}. "
                         f"Use an empty string to reset to auto-detection.")
            cfg.type_override = v
            continue   # never setattr(cfg, "model_type", ...) — that field is derived
        if k == "engine_config" and cfg.is_bridge:
            # Pointing a bridge card at a new engine config must be validated exactly the
            # way the add path does it, or the card is broken until the next rescan.
            # (The engine NAME stays immutable — a card must not re-engine itself.)
            if await asyncio.to_thread(manager._make_engine_config, Path(v), cfg.engine, None) is None:
                raise HTTPException(400, f"not a valid {cfg.engine} engine config: {v}")
        setattr(cfg, k, v)
    if cfg.max_ctx_size and cfg.ctx_size and cfg.ctx_size > cfg.max_ctx_size:
        log.info(f"Capping ctx_size ({cfg.ctx_size}) to max_ctx_size ({cfg.max_ctx_size}) for model {mid}")
        cfg.ctx_size = cfg.max_ctx_size
    await manager.update_config(mid, cfg)
    return asdict(cfg)

@app.post("/v1/models/{model_id}/vision/toggle")
async def toggle_vision_ep(model_id: str):
    mid = await manager.resolve_model_id(model_id)
    if not mid:
        raise HTTPException(404, "Unknown model")
    cfg = await manager.get_config(mid)
    manager._require_llama_engine(cfg)
    cfg.use_mmproj = not cfg.use_mmproj
    await manager.update_config(mid, cfg)
    return {"id": mid, "use_mmproj": cfg.use_mmproj, "mmproj_file": cfg.mmproj_file}

@app.post("/v1/models/engine")
async def add_engine_card_ep(body: EngineCardCreate):
    """Register a bridge card for any configured engine.

    Resolves the engine (400 unknown/misconfigured), validates the engine config
    (is_file + parses to dict + has the engine's ``config_key``), writes
    ``engine_cards/<mid>.json`` (self-describing via ``engine`` + ``engine_config``)
    and registers the card. 400 on invalid config, 409 on id collision.

    NOTE: assumes a single-instance server. The lock is deliberately split (card-file
    write outside the lock, registration inside) to avoid holding it during disk I/O; a
    concurrent same-id add is benign because the card-file write is idempotent.
    """
    spec = manager.engines.get(body.engine)
    if spec is None:
        raise HTTPException(400, f"Unknown engine '{body.engine}'. "
                                 f"Add an engines.{body.engine} section to config.yaml")
    if spec.get("_errors"):
        raise HTTPException(400, f"Engine '{body.engine}' is misconfigured: {spec['_errors']}")
    if not spec.get("enabled"):
        raise HTTPException(400, f"Engine '{body.engine}' is disabled in config.yaml")
    cfg_path = Path(body.engine_config)
    cfg = await asyncio.to_thread(manager._make_engine_config, cfg_path, body.engine, None)
    if cfg is None:
        raise HTTPException(400, f"not a valid {body.engine} engine config: {body.engine_config}")
    mid = manager._sanitize_model_id(cfg_path.stem)
    async with manager._lock:
        if mid in manager.models:
            raise HTTPException(409, f"Model id '{mid}' already exists")
    if body.name:
        cfg.name = body.name
    cards_dir = manager._cards_dir_for(body.engine)
    cards_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(cfg.save_engine_card, cards_dir / f"{mid}.json")
    async with manager._lock:
        manager.models[mid] = cfg
    log.info(f"Added {body.engine} bridge card: {mid} -> {cfg_path}")
    return {"id": mid}

@app.get("/v1/engines")
async def list_engines():
    """Configured bridge engines, sorted by name.

    ``ready`` = enabled and free of boot-time validation errors; ``error`` carries the
    same message that a load attempt would return as a 400, so the WebUI can explain
    why an engine is unusable instead of letting the user discover it at load time.
    """
    out = []
    for name in sorted(manager.engines):
        spec = manager.engines[name]
        error = spec.get("_errors")
        if not error and not spec.get("enabled"):
            error = "engine is disabled in config.yaml"
        out.append({
            "name": name,
            "label": spec.get("label") or name,
            "enabled": bool(spec.get("enabled")),
            "ready": bool(spec.get("enabled")) and not error,
            "error": error or "",
            "cards_dir": str(manager._cards_dir_for(name)),
            "has_configs_dir": bool(spec.get("configs_dir")),
        })
    return out

@app.get("/v1/mmproj")
async def list_projectors():
    """Every vision projector under root_dir, and which model owns it.

    Projectors are deliberately not model cards, so this is the ONLY place they are
    visible at all. It exists because a pairing is otherwise a one-way, invisible fact:
    a card shows "vision" without saying which file provides it, and a projector attached
    to nothing is not shown anywhere - so an orphan (exactly the mispairing class fixed
    in _pair_mmproj) was undetectable from the UI.

    Fields per projector:
      ``name``     filename, e.g. "mmproj-F16.gguf"
      ``path``     absolute path
      ``size_mb``  on-disk size
      ``owner``    model id it is paired to, or "" when unpaired
      ``source``   "explicit" (set in a sidecar) | "auto" (inferred at scan)
      ``enabled``  whether the owner will actually pass --mmproj at launch
      ``dir``      parent directory, to spot a projector shared from elsewhere

    ``count``/``unpaired`` let a caller assert "nothing is orphaned" without walking the
    list.
    """
    async with manager._lock:
        pairs = dict(manager.mmproj_paths)
        sources = dict(manager.mmproj_source)
        models = dict(manager.models)
    owner_of: Dict[Path, str] = {p: mid for mid, p in pairs.items()}
    out = []
    for p in sorted(manager._list_mmproj_files()):
        mid = owner_of.get(p, "")
        try:
            size_mb = p.stat().st_size / 1024 / 1024
        except OSError:
            size_mb = 0.0
        out.append({
            "name": p.name,
            "path": str(p),
            "dir": str(p.parent),
            "size_mb": round(size_mb, 1),
            "owner": mid,
            "source": sources.get(mid, "") if mid else "",
            "enabled": bool(mid and models.get(mid) and models[mid].use_mmproj),
        })
    return {
        "count": len(out),
        "unpaired": sum(1 for p in out if not p["owner"]),
        "projectors": out,
    }

@app.post("/v1/models/strata")
async def add_strata_card_ep(b: StrataCardCreate):
    """DEPRECATED alias for ``POST /v1/models/engine`` with engine="strata". Kept working
    for existing scripts; prefer the generic endpoint."""
    return await add_engine_card_ep(
        EngineCardCreate(engine="strata", engine_config=b.strata_config, name=b.name))

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
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(503, str(e))

@app.post("/v1/models/{model_id}/unload")
async def unload_model_ep(model_id: str):
    await manager.unload_model(model_id)
    return {"id": model_id, "unloaded": True}

@app.delete("/v1/models/{model_id}")
async def delete_model_ep(model_id: str):
    """Delete an engine bridge card (user-added). Llama/GGUF models and auto-discovered
    engine cards cannot be deleted through this endpoint (remove their source file instead)."""
    mid = await manager.resolve_model_id(model_id)
    if not mid:
        raise HTTPException(404, f"Model not found: {model_id}")

    # Claim the card atomically: guard check + in-memory removal under ONE lock, so a
    # concurrent duplicate delete sees cfg is None and returns idempotently instead of
    # both passing the is_bridge check (a re-add can't slip in between check and remove).
    async with manager._lock:
        cfg = manager.models.get(mid)
        if cfg is None:
            # Already gone (e.g. a concurrent delete won the race) — idempotent no-op.
            return {"id": mid, "deleted": True}
        if not cfg.is_bridge:
            raise HTTPException(400, "Only engine bridge cards can be deleted via this endpoint")
        engine = cfg.engine
        manager.models.pop(mid)
        manager._model_sizes.pop(mid, None)
        manager.mmproj_paths.pop(mid, None)

    # State is claimed; do the blocking/disk side effects OUTSIDE the lock.
    # Make sure the resident process (if any) is stopped and its derived config removed.
    try:
        await manager.unload_model(mid)
    except Exception as e:
        log.warning(f"Unload before delete failed for {mid}: {e}")

    # Remove the card file (and any legacy copy) so a rescan won't re-add it.
    cards_dir = manager._cards_dir_for(engine)
    for d in [cards_dir] + [Path(p) for p in getattr(manager, "LEGACY_CARDS_DIRS", ())]:
        try:
            (d / f"{mid}.json").unlink(missing_ok=True)
        except OSError as e:
            log.warning(f"Failed to remove card file for {mid} in {d}: {e}")
    try:
        manager._derived_config_path(engine, mid).unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"Derived-config cleanup failed for {mid}: {e}")

    log.info(f"Deleted engine bridge card: {mid}")
    return {"id": mid, "deleted": True}

@app.post("/v1/unload_all")
async def unload_all_ep():
    await manager.unload_all()
    return {"unloaded_all": True}

# ---------- state save / load ----------
@app.post("/v1/models/{model_id}/state/save")
async def save_state_ep(model_id: str, body: StateLabel):
    mid = await manager.resolve_model_id(model_id)
    cfg = await manager.get_config(mid) if mid else None
    # State save/restore is supported by both llama.cpp and Strata engines.
    # (Previously gated to llama-only; Strata's serve exposes the same /slots/0?action=save endpoint.)
    try:
        p = await manager.save_state(model_id, body.label)
        size = p.stat().st_size
        log.info(f"State saved for model={model_id} label={body.label} path={p.name} size={size} bytes")
        return {"id": model_id, "label": body.label, "path": str(p)}
    except HTTPException as e:
        if e.status_code == 400:
            log.info(f"State save skipped for model={model_id} label={body.label}: {e.detail}")
        raise

@app.post("/v1/models/{model_id}/state/load")
async def load_state_ep(model_id: str, body: StateLabel):
    mid = await manager.resolve_model_id(model_id)
    cfg = await manager.get_config(mid) if mid else None
    # State save/restore is supported by both llama.cpp and Strata engines.
    try:
        p = await manager.load_state(model_id, body.label)
        log.info(f"State restored for model={model_id} label={body.label} path={p.name}")
        return {"id": model_id, "label": body.label, "path": str(p)}
    except HTTPException as e:
        if "restore failed" in e.detail and "state file not found" in e.detail.lower():
            log.info(f"State restore skipped for model={model_id} label={body.label}: state file missing")
        raise

@app.get("/v1/models/{model_id}/state")
async def list_states_ep(model_id: str):
    labels = await manager.list_states(model_id)
    log.info(f"List states requested for model={model_id}, found {len(labels)} state(s)")
    return {"id": model_id, "labels": labels}

@app.delete("/v1/models/{model_id}/state/{label}")
async def delete_state_ep(model_id: str, label: str):
    try:
        await manager.delete_state(model_id, label)
        log.info(f"State deleted for model={model_id} label={label}")
        return {"id": model_id, "label": label, "deleted": True}
    except HTTPException as e:
        if e.status_code == 404:
            log.info(f"State delete skipped for model={model_id} label={label}: {e.detail}")
        raise

# NOTE: The OpenAI-compatible LLM data path (/v1/chat/completions, /completions,
# /embeddings, and other llama-server endpoints) is served by the RawTCPForwarder on
# the MAIN port (see tcp_forwarder.py), NOT by FastAPI. This app now exposes only
# management/status/state routes plus the /ws WebSocket; LLM requests that are not
# already loaded fall through to FastAPI which JIT-loads them and returns an error.

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
# Raw TCP forwarder: owns the MAIN client-facing port. Streaming requests are piped
# straight to llama-server (no ASGI/uvicorn/httpx relay); everything else falls through
# to this FastAPI app, which now runs on an INTERNAL port. See tcp_forwarder.py.
from tcp_forwarder import RawTCPForwarder  # noqa: E402

# The app binds THREE things: the main client-facing port (forwarder), an internal
# FastAPI/uvicorn port, and a base port for spawned engines. All three must be distinct
# per instance, which is what lets several autoloaders run side by side.
_DEFAULT_INTERNAL_PORT = 1235
INTERNAL_API_PORT = int(os.environ.get("AUTOLOADER_INTERNAL_PORT", _DEFAULT_INTERNAL_PORT))
_forwarder: Optional[RawTCPForwarder] = None


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True when something is ACCEPTING connections on ``host:port``.

    A connect probe rather than a bind probe: on Windows ``SO_REUSEADDR`` does not
    cleanly distinguish "free" from "another instance owns it", whereas a refused
    connection is unambiguous.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


def _resolve_internal_port(desired: int, explicit: bool, host: str = "127.0.0.1",
                           max_tries: int = 64) -> Tuple[int, Optional[int]]:
    """Return ``(port, auto_advanced_from)`` for the internal FastAPI port.

    uvicorn binds the internal port AFTER the forwarder has already claimed the main
    port, so a collision used to surface as a bare ``WinError 10048`` and take the whole
    process down with a half-initialized forwarder behind it.

    - Explicitly requested (``--internal-port``) and busy -> fail fast with a readable
      message. The user named that exact port; silently moving it would only hide the
      collision they are trying to diagnose.
    - Implicit (env var / default) and busy -> advance to the next free port. Another
      instance is almost certainly what is holding it, and starting alongside it is the
      whole point.
    """
    if not _port_in_use(desired, host):
        return desired, None
    if explicit:
        raise SystemExit(
            f"[start] --internal-port {desired} is already in use on {host}. "
            f"Choose a different one, or omit --internal-port to auto-select.")
    for port in range(desired + 1, desired + 1 + max_tries):
        if not _port_in_use(port, host):
            return port, desired
    raise SystemExit(
        f"[start] No free internal port in {desired + 1}..{desired + max_tries} "
        f"on {host}. Pass --internal-port explicitly.")


@app.on_event("startup")
async def _startup():
    await manager.start()
    # Capture the main running loop for cross-thread async lookups (forwarder threads).
    manager._main_loop = asyncio.get_running_loop()
    # Start the raw TCP forwarder on the MAIN port. It must come up before uvicorn is
    # asked to bind, but we run uvicorn on the INTERNAL port so there is no collision.
    global _forwarder
    _forwarder = RawTCPForwarder(manager.port, INTERNAL_API_PORT, manager)
    _forwarder.start()

@app.on_event("shutdown")
async def _shutdown():
    if _forwarder is not None:
        _forwarder.stop()
    await manager.stop()


# --------------------------------------------------------------------------
# TCP_NODELAY on the downstream (server -> client) socket
# --------------------------------------------------------------------------
# Round 7: an external review proposed that Nagle + delayed-ACK on Windows loopback
# coalesces uvicorn's per-chunk transport.write() into one end-of-stream batch, and
# that setting TCP_NODELAY on the accepted downstream socket fixes the SSE burst.
# This is a MINIMAL, REVERSIBLE patch of httptools' HttpToolsProtocol.connection_made
# (the only place uvicorn has the accepted-connection transport). It:
#   - applies TCP_NODELAY to each accepted connection's socket (no-op on failure),
#   - is guarded so it NEVER breaks non-streaming or other models,
#   - can be disabled with env AUTOLOADER_TCP_NODELAY=0.
# It does NOT touch the global _socket.socket class (the review's "Option 3" hack).
# See N:\work\WD\AgentWorkspace\_findings_autoloader_stream_burst.md (Round 7).
def install_tcp_nodelay() -> None:
    """Patch httptools' HttpToolsProtocol.connection_made to set TCP_NODELAY.

    Idempotent and safe: if the protocol class or its socket is unavailable, it
    falls back to the original behavior unchanged. Call once before uvicorn.run().
    """
    if os.environ.get("AUTOLOADER_TCP_NODELAY", "1") == "0":
        log.info("TCP_NODELAY patch disabled via AUTOLOADER_TCP_NODELAY=0")
        return
    try:
        from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
    except Exception as e:  # noqa: BLE001 — never let this break startup
        log.warning(f"TCP_NODELAY patch skipped (httptools protocol unavailable): {e}")
        return

    if getattr(HttpToolsProtocol, "_autoloader_nodelay_patched", False):
        return  # already patched (idempotent)

    _orig_cm = HttpToolsProtocol.connection_made

    def connection_made(self, transport):  # type: ignore[override]
        try:
            sock = transport.get_extra_info("socket")
            if sock is not None and hasattr(sock, "setsockopt"):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:  # noqa: BLE001 — best-effort; never break the connection
            pass
        _orig_cm(self, transport)

    HttpToolsProtocol.connection_made = connection_made
    HttpToolsProtocol._autoloader_nodelay_patched = True
    log.info("TCP_NODELAY enabled on downstream (server->client) sockets")


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="llama-autoloader - JIT model loader and OpenAI-compatible proxy. "
                    "Run several instances side by side by giving each one a distinct "
                    "--port / --internal-port / --base-port.")
    ap.add_argument("--host", default=CFG["launcher"]["host"],
                    help="bind address (default: config launcher.host)")
    ap.add_argument("--port", type=int, default=CFG["launcher"]["port"],
                    help="main client-facing port, owned by the raw TCP forwarder "
                         "(default: config launcher.port)")
    ap.add_argument("--internal-port", type=int, default=None,
                    help=f"port for the internal FastAPI/uvicorn app, which the forwarder "
                         f"falls back to for management + /ws traffic "
                         f"(default: $AUTOLOADER_INTERNAL_PORT, else {_DEFAULT_INTERNAL_PORT}). "
                         f"If that default is already taken it auto-advances to the next free "
                         f"port so a second instance can start.")
    ap.add_argument("--base-port", type=int, default=None,
                    help="first port allocated to spawned engines "
                         "(default: config launcher.base_port)")
    args = ap.parse_args()
    manager.host = args.host
    manager.port = args.port

    # We pass use_colors=False to uvicorn.run() below so the log file stays free of
    # ANSI escape codes (uvicorn ignores NO_COLOR and auto-detects TTY).

    # Set up console-log-to-file logging BEFORE uvicorn.run() so that uvicorn's
    # dictConfig handlers bind to our capturing stdout/stderr streams (see logger.py).
    _lg = CFG.get("logging", {})
    if _lg.get("enabled", True):
        _base = Path(__file__).resolve().parent
        _ldir = _lg.get("log_dir", "./logs")
        if not os.path.isabs(_ldir):
            _ldir = str(_base / _ldir)
        init_logging(
            level=getattr(logging, str(_lg.get("level", "INFO")).upper(), logging.INFO),
            log_dir=_ldir,
            filename=_lg.get("filename", "console.log"),
            max_bytes=_safe_int(_lg.get("max_bytes", 10 * 1024 * 1024), 10 * 1024 * 1024),
            backup_count=_safe_int(_lg.get("backup_count", 5), 5),
        )

    # Windows: force the Selector event loop instead of the default Proactor loop.
    # NOTE (round 5): this does NOT fix the SSE burst — live instrumented runs proved
    # the proxy already runs on _WindowsSelectorEventLoop and still bursts, and a
    # real-uvicorn-over-TCP repro is SMOOTH under heavy concurrent load with Selector.
    # Kept because it's harmless (matches Linux/macOS defaults) and may help other
    # I/O paths; see N:\work\WD\AgentWorkspace\_findings_autoloader_stream_burst.md.
    if platform.system() == "Windows" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Round 7: set TCP_NODELAY on downstream sockets (env-disableable; no-op on failure).
    install_tcp_nodelay()

    # Round 10: test whether h11's write cadence avoids the relay-path buffering (httptools is default).
    _http_backend = os.environ.get("AUTOLOADER_HTTP", "auto").lower()  # auto|httptools|h11

    # ------------------------------------------------------------------
    # Multi-instance port resolution
    # ------------------------------------------------------------------
    # The forwarder owns the MAIN port; uvicorn owns the INTERNAL port; spawned engines
    # take base_port+. All three must be clear of any other running instance. This runs
    # AFTER init_logging() so the diagnostics below land in logs/console.log too.
    if args.base_port is not None:
        manager.base_port = args.base_port

    _wanted_internal = (args.internal_port if args.internal_port is not None
                        else int(os.environ.get("AUTOLOADER_INTERNAL_PORT", _DEFAULT_INTERNAL_PORT)))
    INTERNAL_API_PORT, _auto_advanced_from = _resolve_internal_port(
        _wanted_internal, explicit=args.internal_port is not None, host=args.host)

    # Fail fast and legibly on the one collision we must NOT paper over: if the main port
    # is taken, this instance has nothing to offer. Raw bind errors here are cryptic.
    if _port_in_use(manager.port, args.host):
        raise SystemExit(
            f"[start] Main port {manager.port} is already in use on {args.host}. "
            f"Another autoloader instance is probably running - pass --port <free port> "
            f"(and --internal-port / --base-port so this instance stays independent).")
    if _auto_advanced_from is not None:
        log.warning(
            f"Internal port {_auto_advanced_from} is already in use on {args.host} "
            f"(another instance?) - using {INTERNAL_API_PORT} instead. "
            f"Pin it for reproducible restarts with --internal-port {INTERNAL_API_PORT}.")
    if _port_in_use(manager.base_port, args.host):
        log.warning(
            f"Engine base port {manager.base_port} is already in use. Per-model ports "
            f"self-heal by scanning upward, but pass --base-port <n> to keep this "
            f"instance's engines in a predictable range.")

    log.info("=" * 72)
    log.info("llama-autoloader instance")
    log.info(f"  WebUI + OpenAI API : http://{args.host}:{manager.port}   (raw TCP forwarder)")
    log.info(f"  Internal FastAPI    : {INTERNAL_API_PORT}   (forwarder falls back here)")
    log.info(f"  Engine port range   : {manager.base_port}+   (one port per model)")
    log.info(f"  Models root         : {manager.root_dir}")
    log.info(f"  Shared on disk      : states={manager.save_state_dir}  cards={manager.cards_dir}  temp={manager.temp_dir}")
    log.info("  NOTE: states/, engine_cards/ and temp/ are NOT instance-scoped, so two")
    log.info("        instances share them (incl. temp/<engine>/<model>.json derived")
    log.info("        configs, which one instance deletes when it unloads a bridge card).")
    log.info("=" * 72)

    # The MAIN client-facing port (args.port) is owned by the RawTCPForwarder (started in
    # _startup). uvicorn/FastAPI runs on an INTERNAL port for management + non-streaming
    # requests only. This removes the ASGI layer from the streaming data path entirely.
    uvicorn.run(app, host=args.host, port=INTERNAL_API_PORT, reload=False, use_colors=False, http=_http_backend)