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
import copy
import json
import math
import os
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time
import logging
import platform
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import psutil
import yaml
from fastapi import (
    FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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
from tcp_forwarder import (LEGACY_GENERATION, RawTCPForwarder,
                          failure_record_applies, gen_matches)

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
                length = struct.unpack('<Q', f.read(8))[0]
                return f.read(length).decode('utf-8', errors='replace')

            def skip_val(vtype):
                if vtype in _GGUF_FIXED_WIDTH:
                    f.seek(_GGUF_FIXED_WIDTH[vtype], 1)
                elif vtype == 8:
                    length = struct.unpack('<Q', f.read(8))[0]
                    f.seek(length, 1)
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
                    if vtype == 4:
                        max_ctx = struct.unpack('<I', f.read(4))[0]
                    elif vtype == 10:
                        max_ctx = struct.unpack('<Q', f.read(8))[0]
                    elif vtype == 5:
                        max_ctx = struct.unpack('<i', f.read(4))[0]
                    elif vtype == 11:
                        max_ctx = struct.unpack('<q', f.read(8))[0]
                    else:
                        skip_val(vtype)
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
# Small filesystem / argv helpers
# --------------------------------------------------------------------------
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (temp file + os.replace).

    Config, sidecar and card writes used to truncate the live file before the
    new content was durable; a crash or a full disk mid-write could leave an
    unreadable file behind. The temp file lives in the same directory so
    ``os.replace`` stays atomic on every platform we support.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # Some network/special filesystems do not implement fsync.
                # os.replace below is still enough to avoid partial files.
                pass
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _split_launch_args(raw: str) -> List[str]:
    """Split a user-supplied argv string without breaking quoted Windows paths.

    ``str.split()`` turned ``--lora "C:\\models\\my lora.gguf"`` into two
    arguments. shlex with ``posix=False`` preserves backslashes but keeps the
    surrounding quotes, so strip one matching quote pair from each token.
    """
    if not raw or not raw.strip():
        return []
    if os.name != "nt":
        return shlex.split(raw)
    try:
        parts = shlex.split(raw, posix=False)
    except ValueError:
        # Unbalanced quotes: fall back to whitespace splitting rather than
        # refusing to launch the engine entirely.
        return raw.split()
    out: List[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in ("'", '"'):
            part = part[1:-1]
        out.append(part)
    return out


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
                # Explicit UTF-8: the Windows default is cp1252, which either
                # mojibakes or rejects hand-edited/foreign non-ASCII sidecars.
                data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
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
        _atomic_write_text(sidecar, json.dumps(data, indent=2, ensure_ascii=False))

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
        _atomic_write_text(
            card_path,
            json.dumps(card, indent=2, ensure_ascii=False),
        )

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
            argv += _split_launch_args(extra_default_args)
        if self.args:
            argv += _split_launch_args(self.args)
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
    # In-flight LLM requests currently being relayed to this engine. Used by
    # both the idle reaper and LRU eviction so a long generation is never
    # killed just because another model wants the slot.
    active_requests: int = 0
    # Monotonic identity of THIS engine instance, assigned by ModelManager at launch.
    # Ports are stable per model and are reused after unload/reload, so a bare port
    # cannot distinguish a dead generation from its replacement. Recovery and active-
    # request accounting compare this token instead.
    generation: int = 0

    def touch(self):
        self.last_used = time.time()


@dataclass
class UnloadResult:
    """Outcome of one unload request.

    Unloading is only successful when the engine process is actually gone; a
    termination that cannot be verified is a failure, because every caller
    (safe-mode, VRAM seams, LRU eviction) treats a successful unload as a
    guarantee that the memory was released.
    """
    model_id: str
    unloaded: bool
    pid: int = 0
    reason: str = ""

    def failure(self) -> Dict[str, Any]:
        return {"model": self.model_id, "pid": self.pid, "reason": self.reason}


# BUG_0007: a failed dead-backend recovery is remembered per model so a client retry
# loop cannot spawn a full load attempt on every request. The delay grows exponentially
# from BASE up to MAX; a manual/force load (or a successful load/unload/config edit)
# clears it. These are env-overridable for tests and operators.
_RECOVERY_BACKOFF_BASE_S = max(
    0.1, float(os.environ.get("AUTOLOADER_RECOVERY_BACKOFF_BASE", "5")))
_RECOVERY_BACKOFF_MAX_S = max(
    _RECOVERY_BACKOFF_BASE_S,
    float(os.environ.get("AUTOLOADER_RECOVERY_BACKOFF_MAX", "300")))


@dataclass
class RecoveryFailure:
    """One model's failed-reload circuit-breaker state (BUG_0007)."""
    model_id: str
    dead_port: int
    generation: int
    attempts: int
    reason: str
    failed_at: float          # wall clock, for status display
    retry_at: float           # time.monotonic() deadline for the next attempt


class RecoveryCooldownError(RuntimeError):
    """A load/recovery was suppressed by the per-model failed-reload cooldown."""

    def __init__(self, model_id: str, retry_after: float):
        self.model_id = model_id
        self.retry_after = max(0.0, float(retry_after))
        super().__init__(
            f"Model '{model_id}' recovery is cooling down; retry in "
            f"{int(self.retry_after + 0.999)}s")


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

# --------------------------------------------------------------------------
# Slot-action sequencing bounds and state-temp cleanup (safe-mode ordering contract)
# --------------------------------------------------------------------------
# Bounded wait for an in-flight slot action (save/restore) that holds a granted
# FIFO ticket to finish before an unload drops the queue and kills its engine.
SLOT_ACTION_DRAIN_TIMEOUT_S = 15.0
# A state/load landing WHILE a deliberate unload is in progress is refused (409)
# instead of silently cold-reloading the just-released model. The guard is the
# unload itself, not a wall-clock window: the safe-mode contract (save -> unload
# -> command -> restore) legitimately lands its restore right AFTER the unload
# returns 200, and any window long enough to cover a slow command would also
# refuse that restore.
# Age past which the engine's interrupted-save scratch files (*.bin.<hash>.tmp)
# are swept from states/. Nothing else ever reclaims them.
STATE_TEMP_MAX_AGE_S = 3600.0

# Exceptions meaning the engine is unreachable AT ALL (connection refused /
# closed), as opposed to an HTTP error it actually returned. Retrying the other
# endpoint formats against a dead socket only reproduces the same error.
# httpx.NetworkError is the category covering exactly Connect/Read/Write/Close,
# so it also catches a peer that dies mid-response (httpx.ReadError('') raised
# once response headers have already arrived).
# Deliberately NOT httpx.TransportError: a ReadTimeout means the engine was
# reachable and may answer a differently-shaped request. ConnectTimeout is kept
# explicitly because it subclasses TimeoutException, NOT NetworkError — a failed
# connect is a dead socket, not a slow one.
_UNREACHABLE_ERRORS = (httpx.NetworkError, httpx.ConnectTimeout, httpx.RemoteProtocolError)


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
        # Per-model serialization for load vs. dead-backend teardown. Recovery must
        # drain/kill OUTSIDE the global manager lock (BUG_0005), which means a load
        # could otherwise spawn a replacement on the same stable port while the old
        # process is still being killed. This lock is per model (never global), so
        # recoveries for different models still proceed concurrently.
        self._model_locks: Dict[str, asyncio.Lock] = {}
        # Monotonic counter for LoadedModel.generation. Never reset at runtime; a
        # generation token must remain unique across unload/reload cycles so a delayed
        # event cannot be mistaken for the replacement that reused the same port.
        self._engine_generation = 0
        # Dead-backend recoveries currently in flight, keyed by (dead port, expected
        # engine generation). The TCP forwarder can have many worker threads hit the
        # same refused port at once; the first one creates the recovery task and the
        # rest await it. The generation is part of the key so a delayed recovery for
        # an OLD instance can never attach to a healthy replacement on the same port.
        self._recovery_tasks: Dict[Tuple[int, int], asyncio.Task] = {}
        # Failed dead-backend reload state, keyed by resolved model id (BUG_0007).
        # A cooldown suppresses repeated spawn/readiness attempts after a deterministic
        # failure (missing binary, bad config, GPU allocation failure, ...).
        self._recovery_failures: Dict[str, RecoveryFailure] = {}
        # Per-model FIFO ticket queues (keyed by engine port). The TCP forwarder and
        # the state save/restore paths both acquire a ticket here so operations for a
        # model run in strict FIFO order. See model_queue.py.
        self.queue_registry = ModelQueueRegistry()
        self._stop = False
        self._bg_task: Optional[asyncio.Task] = None
        self._model_sizes: Dict[str, float] = {}
        # In-flight slot-action bookkeeping: engine port -> number of save/restore
        # operations currently holding a GRANTED ticket against that engine. An unload
        # drains these (bounded) before killing the process, so a save is never cut
        # mid-write. Plain dict, no extra lock: mutated only in single bytecode steps
        # and only ever polled by the drain — it is never awaited on, so it cannot
        # participate in a lock cycle.
        self._inflight_slot_actions: Dict[int, int] = {}
        # model_ids whose deliberate unload is IN PROGRESS: armed at the start of
        # _unload_unlocked and cleared only once the queue drop and process kill
        # complete. A state/load landing during that window is refused with 409
        # instead of cold-reloading the model the caller just released; a restore
        # landing after the unload finished is the legitimate safe-mode one and
        # is allowed through.
        self._unload_in_progress: Set[str] = set()

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
            log.warning(
                "Model backend '%s' was configured but not found under %s; "
                "falling back to the global/default backend",
                model_cfg.backend, self.backends_dir)

        # Priority 2: Global selected backend
        if self.selected_backend:
            b_dir = self.backends_dir / self.selected_backend
            exe = b_dir / "llama-server.exe"
            if not exe.exists():
                exe = b_dir / "llama-server"
            if exe.exists():
                return str(exe)
            log.warning(
                "selected_backend '%s' was configured but not found under %s; "
                "falling back to the first available/default binary",
                self.selected_backend, self.backends_dir)

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

    def _mmproj_pairing_context(self, mid: str, p: Path,
                                model_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Eligibility context for :meth:`_pair_mmproj` (single source for scan/update).

        A projector is contested when several non-auxiliary models could claim it;
        passing the siblings/others lets ``_pair_mmproj`` refuse an ambiguous match.
        """
        eligible_by_dir: Dict[Path, List[str]] = {}
        eligible_all: List[str] = []
        for other_mid, other_path in self.gguf_paths.items():
            other_type = (model_type if other_mid == mid else
                          (self.models[other_mid].model_type or MODEL_TYPE_TEXT))
            if other_type not in MODEL_TYPE_AUXILIARY:
                eligible_by_dir.setdefault(other_path.parent, []).append(other_path.stem)
                eligible_all.append(other_path.stem)
        return {
            "eligible_siblings": [s for s in eligible_by_dir.get(p.parent, []) if s != p.stem],
            "eligible_others": [s for s in eligible_all if s != p.stem],
        }

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
        scan() and update_config() both supply them via
        :meth:`_mmproj_pairing_context`; tests may omit them.
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
            if mid in found:
                log.warning(
                    "Duplicate model filename '%s': keeping %s and skipping %s "
                    "(model ids are filenames; rename one file to expose both)",
                    mid, self.gguf_paths.get(mid), p)
                continue
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
        self.mmproj_source = {}
        for mid, p in self.gguf_paths.items():
            cfg = self.models[mid]
            # Read provenance BEFORE pairing: the write two lines below stamps an inferred
            # name into cfg.mmproj_file, which would make "auto" look "explicit" forever.
            explicit = bool((cfg.mmproj_file or "").strip())
            paired_mmproj = self._pair_mmproj(
                p, cfg, mmproj_paths,
                **self._mmproj_pairing_context(mid, p))
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

    def _lookup_model_locked(self, model_id_or_alias: Optional[str]) -> Optional[str]:
        """Exact id / stem / display-name lookup. Caller must hold ``self._lock``."""
        if not model_id_or_alias:
            return None
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
        return None

    async def resolve_model_id_strict(self, model_id_or_alias: Optional[str]) -> Optional[str]:
        """Resolve an EXPLICIT model reference, or return None.

        Destructive/management endpoints and explicit ``model`` fields must use
        this variant: silently substituting a different model for a typo is how
        DELETE/raw/port/state calls ended up acting on an unrelated resident
        model. Only :meth:`resolve_model_id` keeps the "omitted model" fallback.
        """
        async with self._lock:
            return self._lookup_model_locked(model_id_or_alias)

    async def resolve_request_model(self, model_id_or_alias: Optional[str]) -> Optional[str]:
        """Resolve the model for an INFERENCE request, tolerating unknown names.

        Inference is not destructive, so an unrecognised ``model`` value (a provider
        alias, a display name we never learned, or a typo) must NOT fail the request
        while a model is already resident: route it to that loaded model, which is
        the historical behaviour clients rely on. Only *loaded* models qualify as a
        fallback - we never JIT-load a model the request did not name.

        Management/destructive endpoints must keep using
        :meth:`resolve_model_id_strict`; their explicit references may not be
        silently redirected to an unrelated resident model.
        """
        async with self._lock:
            match = self._lookup_model_locked(model_id_or_alias)
            if match:
                return match
            if not self.loaded:
                return None

            # Prefer a ready model; fall back to any resident one so the caller can
            # reuse its port rather than spawning a second engine.
            ready_loaded = [m for m, lm in self.loaded.items() if lm.ready]
            model_id = ready_loaded[0] if ready_loaded else next(iter(self.loaded.keys()))
            log.debug(f"Model '{model_id_or_alias}' not found in registry; "
                      f"serving request with loaded model '{model_id}'")
            return model_id

    async def resolve_model_id(self, model_id_or_alias: Optional[str]) -> Optional[str]:
        """Resolve a model, with a default fallback for an OMITTED reference.

        Use this only when ``None``/empty means "pick a sensible default" (for
        example an OpenAI request without a ``model`` field). Explicit endpoint
        paths must use :meth:`resolve_model_id_strict`.
        """
        async with self._lock:
            match = self._lookup_model_locked(model_id_or_alias)
            if match:
                return match

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
        mid = await self.resolve_model_id_strict(model_id_or_alias)
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
        mid = await self.resolve_model_id_strict(model_id_or_alias)
        if not mid:
            raise KeyError(model_id_or_alias)
        async with self._lock:
            if mid not in self.models:
                raise KeyError(model_id_or_alias)
            cfg = self.models[mid]
            is_bridge = cfg.is_bridge
            path = None if is_bridge else self.gguf_paths[mid]
            # Provenance must be read BEFORE we overwrite the config: the previous
            # scan stamps an inferred filename into mmproj_file, so inferring
            # "explicit" from that field alone made every edit flip auto -> explicit.
            previous_source = self.mmproj_source.get(mid, "")
            previous_file = (cfg.mmproj_file or "").strip()
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
            # Re-derive the effective type first: _pair_mmproj uses model_type to
            # refuse auxiliary components (draft/MTP/audio), exactly like scan().
            info = await asyncio.to_thread(_read_gguf_info, path)
            new_cfg.model_type = _classify_model_type(path, info, new_cfg.type_override)
            # Pair mmproj OUTSIDE the lock BEFORE saving, so an auto-filled
            # mmproj_file is written to the sidecar (matches scan() behavior).
            mmprojs = await asyncio.to_thread(self._list_mmproj_files)
            requested_file = (new_cfg.mmproj_file or "").strip()
            # An auto-filled field is not an explicit choice. Treat it as explicit
            # only when the caller actually changed it, or when the previous
            # provenance was already explicit.
            explicit = bool(requested_file) and (
                previous_source == "explicit"
                or previous_source == ""
                or requested_file != previous_file
            )
            paired = self._pair_mmproj(
                path, new_cfg, mmprojs,
                **self._mmproj_pairing_context(mid, path, new_cfg.model_type))
            paired_source = ("explicit" if explicit else "auto") if paired is not None else ""
            if paired is not None and not new_cfg.mmproj_file:
                new_cfg.mmproj_file = paired.name
            await asyncio.to_thread(new_cfg.save, path)

        async with self._lock:
            self.models[mid] = new_cfg
            if mid in self.loaded:
                self.loaded[mid].config = new_cfg
            # A configuration edit plausibly fixes a failed reload (bad args, engine
            # path, GPU settings), so allow the next attempt immediately (BUG_0007).
            self._clear_recovery_failure(mid)
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
        """Strip path separators/parent refs and make the result safe as a filename.

        Used for state-file names and derived engine configs. Windows rejects
        ``<>:"|?*`` and control characters, ignores trailing dots/spaces, and
        treats names like ``CON``/``NUL`` as devices; an empty result used to
        produce paths such as ``.json``.
        """
        if s is None:
            return "_"
        cleaned = str(s).replace("\\", "").replace("/", "").replace("..", "").strip()
        cleaned = re.sub(r'[<>:"|?*\x00-\x1f]', "_", cleaned)
        cleaned = cleaned.rstrip(" .")
        if not cleaned:
            return "_"
        if cleaned.upper().split(".")[0] in _WINDOWS_RESERVED_NAMES:
            cleaned = "_" + cleaned
        return cleaned

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

    async def _evict_lru_if_needed(self, target_model_id: str) -> Optional[UnloadResult]:
        """If loaded models count >= max_loaded_models, unload the LRU evictable model
        (caller must hold self._lock).

        A model is an eviction candidate only when it is not the target, not pinned,
        has no active requests, and is not mid-load (an incomplete task in
        ``_loading_tasks`` — evicting one causes a load/evict ping-pong cascade).

        Returns the ``UnloadResult`` of the eviction, or ``None`` when no eviction
        was needed or no candidate existed. Callers MUST check the result: a
        non-``None`` result with ``unloaded`` False means the slot was NOT freed
        (through this path the process is still running), and the load must be
        refused. ``_unload_unlocked`` may also return ``None`` (the entry vanished
        mid-call) or raise (e.g. cancellation); those propagate to the caller's
        existing error handling.
        When no candidate exists the load proceeds over the cap by design —
        refusing or killing a just-started engine is the ping-pong bug this
        predicate exists to prevent.
        """
        if len(self.loaded) < self.max_loaded_models:
            return None
        candidates = [
            (mid, lm) for mid, lm in self.loaded.items()
            if mid != target_model_id and not lm.config.pinned
            and lm.active_requests == 0
            and not self._is_loading(mid)
        ]
        if not candidates:
            log.warning(
                "Max loaded models limit reached, but every other model is pinned, "
                "busy (active request), mid-load, or the target itself; "
                "loading without evicting")
            return None
        candidates.sort(key=lambda item: item[1].last_used)
        lru_id, lru_lm = candidates[0]
        log.info(f"Auto-evicting LRU model {lru_id} (last used {round(time.time() - lru_lm.last_used, 1)}s ago) to free slot")
        return await self._unload_unlocked(lru_id)

    # ---- active-request tracking (called from forwarder worker threads) ----
    def _gen_matches(self, recorded: int, expected_generation: int) -> bool:
        """True if the recorded engine token matches, or the caller's expectation is legacy (<= 0).

        Thin wrapper over the module-level :func:`gen_matches` (tcp_forwarder.py) so manager-side
        engine-identity checks share one definition with the forwarder. NOT used for RecoveryFailure
        records -- those go through the deliberately broader failure_record_applies, whose own
        token may be unresolvable and must still guard the model.
        """
        return gen_matches(recorded, expected_generation)

    def note_request_start(self, port: int,
                           expected_generation: int = LEGACY_GENERATION) -> Optional[LoadedModel]:
        """Mark one in-flight request and return the EXACT engine instance.

        ``expected_generation`` narrows the match to the instance the forwarder routed to.
        The returned object is the end() token: a late request from a dead engine
        must decrement its own instance, never a replacement that reused the port.

        The legacy 0 form is the backwards-compatible fake/legacy form: match by
        port only. Real engines always carry a manager-assigned positive generation.
        """
        # Snapshot: this runs on a forwarder thread while the event loop may
        # unload a model. Iterating the live dict could raise "changed size".
        for lm in list(self.loaded.values()):
            if lm.port != port:
                continue
            if not self._gen_matches(lm.generation, expected_generation):
                continue
            lm.active_requests += 1
            lm.touch()
            return lm
        return None

    def note_request_end(self, engine: Optional[LoadedModel]) -> None:
        """Release an in-flight request and refresh ``last_used`` on completion.

        Takes the LoadedModel token returned by note_request_start and decrements
        THAT object directly. A late completion from a dead generation can therefore
        never decrement a replacement that reused the same port.

        Touching at the END matters for long generations: otherwise a request
        that started just before the idle timeout could be reaped while it is
        still streaming.
        """
        if engine is None:
            return
        engine.active_requests = max(0, engine.active_requests - 1)
        engine.touch()

    def _model_lock(self, model_id: str) -> asyncio.Lock:
        """Per-model lock serializing loads against recovery teardown.

        Access is from the main event loop only. Keeping this lock separate from
        ``self._lock`` is the point of BUG_0005: a slow recovery for model A never
        blocks routing/status for model B.
        """
        lock = self._model_locks.get(model_id)
        if lock is None:
            lock = asyncio.Lock()
            self._model_locks[model_id] = lock
        return lock

    def _is_loading(self, model_id: str) -> bool:
        """True if a load task for this model is in flight (caller must hold ``_lock``).

        A finished-but-unpruned task entry is not "loading": the finally clause
        pops it, and a stale entry must never gate eviction or unload marking.
        """
        task = self._loading_tasks.get(model_id)
        return task is not None and not task.done()

    async def load_model(self, model_id_input: str, force: bool = False,
                         respect_recovery_cooldown: bool = True) -> LoadedModel:
        """Load (JIT) a model with non-blocking concurrency for other requests.

        BUG_0007: when a dead-backend reload has just failed, ordinary JIT loading is
        suppressed for the per-model cooldown so a retry loop cannot spawn a doomed
        engine on every request. A manual "load now" endpoint passes
        ``respect_recovery_cooldown=False`` (and ``force=True`` also bypasses it).
        """
        model_id = await self.resolve_model_id_strict(model_id_input)
        if not model_id:
            raise KeyError(f"Unknown model: {model_id_input}")
        if not force and respect_recovery_cooldown:
            remaining = self.recovery_retry_after(model_id)
            if remaining is not None:
                log.warning(f"JIT load of '{model_id}' suppressed: recovery cooldown "
                            f"has {remaining:.1f}s remaining")
                raise RecoveryCooldownError(model_id, remaining)
        if force or not respect_recovery_cooldown:
            self._clear_recovery_failure(model_id)
        had_failure = model_id in self._recovery_failures
        try:
            async with self._model_lock(model_id):
                # Re-check now that concurrent loads are serialized: the first waiter
                # may have just failed and armed a fresh cooldown. Without this, a
                # post-cooldown burst would each attempt its own doomed load.
                if not force and respect_recovery_cooldown:
                    remaining = self.recovery_retry_after(model_id)
                    if remaining is not None:
                        raise RecoveryCooldownError(model_id, remaining)
                had_failure = model_id in self._recovery_failures
                return await self._load_model_serialized(model_id, force)
        except asyncio.CancelledError:
            raise
        except RecoveryCooldownError:
            # A concurrent waiter armed the cooldown after this call passed its first
            # check. Suppression is not another failed load attempt.
            raise
        except Exception as e:  # noqa: BLE001 - extend an existing dead-recovery backoff
            if had_failure and respect_recovery_cooldown and not force:
                current = self.loaded.get(model_id)
                dead_port = int(getattr(current, "port", 0) or 0) if current else 0
                self._record_recovery_failure(
                    model_id, dead_port,
                    int(getattr(current, "generation", 0) or 0) if current else 0,
                    f"JIT reload failed: {e}")
            raise

    async def _load_model_serialized(self, model_id: str,
                                     force: bool = False) -> LoadedModel:
        async with self._lock:
            # Double-check under lock
            if model_id in self.loaded and self.loaded[model_id].ready and not force:
                self.loaded[model_id].touch()
                return self.loaded[model_id]

            # A load in flight cancels the "unloading" refusal: a state/load
            # landing during this load must join it through the dedup below,
            # not be told the endpoint was deliberately released.
            self._unload_in_progress.discard(model_id)

            if self._is_loading(model_id):
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

    def _clear_recovery_failure(self, model_id: str) -> None:
        """Forget a model's failed-reload cooldown (success/manual action/config edit)."""
        if model_id and self._recovery_failures.pop(model_id, None) is not None:
            log.info(f"Cleared failed-recovery backoff for '{model_id}'")

    def _record_recovery_failure(self, model_id: str, dead_port: int,
                                 expected_generation: int = LEGACY_GENERATION,
                                 reason: str = "") -> RecoveryFailure:
        """Record one failed recovery and arm exponential backoff (BUG_0007).

        Runs on the main event loop. Existing state is never replaced by a *shorter*
        cooldown: repeated deterministic failures grow the delay up to the cap.

        ``expected_generation`` is the engine the caller was recovering. With the
        legacy 0 form the token is resolved from the registry entry instead, so the
        stored record carries a concrete generation whenever one is knowable.
        """
        previous = self._recovery_failures.get(model_id)
        attempts = (previous.attempts + 1) if previous else 1
        exponent = min(attempts - 1, 20)
        delay = min(_RECOVERY_BACKOFF_MAX_S, _RECOVERY_BACKOFF_BASE_S * (2 ** exponent))
        now_mono = time.monotonic()
        generation = expected_generation
        if expected_generation <= LEGACY_GENERATION:
            current = self.loaded.get(model_id)
            generation = int(getattr(current, "generation", 0) or 0) if current else 0
        failure = RecoveryFailure(
            model_id=model_id,
            dead_port=dead_port,
            generation=generation,
            attempts=attempts,
            reason=reason,
            failed_at=time.time(),
            retry_at=now_mono + delay,
        )
        self._recovery_failures[model_id] = failure
        log.error(f"Dead-backend recovery for '{model_id}' failed "
                  f"(attempt {attempts}); suppressing retries for {delay:.1f}s: {reason}")
        return failure

    def recovery_retry_after(self, model_id_hint: str = "",
                             dead_port: Optional[int] = None,
                             expected_generation: int = LEGACY_GENERATION) -> Optional[float]:
        """Seconds remaining on a model's failed-recovery cooldown, or None.

        Safe to call from forwarder worker threads: it only snapshots dict values and
        never awaits/takes the asyncio lock. ``dead_port`` matches the exact failed
        engine; otherwise the model id (including its normal alias forms) is used.
        """
        now = time.monotonic()
        best: Optional[float] = None
        resolved_hint = None
        if model_id_hint and model_id_hint not in self._recovery_failures:
            try:
                resolved_hint = self._lookup_model_locked(model_id_hint)
            except Exception:  # noqa: BLE001 - concurrent registry mutation
                resolved_hint = None
        # Deliberately the broader record rule, not _gen_matches: a failure recorded
        # without a resolvable token still guards this model (see failure_record_applies).
        for rec in list(self._recovery_failures.values()):
            if not failure_record_applies(rec.generation, expected_generation):
                continue
            matched = dead_port is not None and rec.dead_port == dead_port
            if not matched and model_id_hint:
                matched = (rec.model_id == model_id_hint
                           or rec.model_id == resolved_hint)
            if not matched:
                continue
            remaining = rec.retry_at - now
            if remaining > 0 and (best is None or remaining > best):
                best = remaining
        return best

    def recovery_failures_snapshot(self) -> List[Dict[str, Any]]:
        """Serializable failed-recovery state for ``/v1/status`` observability."""
        now = time.monotonic()
        out = []
        for rec in list(self._recovery_failures.values()):
            out.append({
                "model": rec.model_id,
                "port": rec.dead_port,
                "generation": rec.generation,
                "attempts": rec.attempts,
                "reason": rec.reason,
                "failed_at": round(rec.failed_at, 1),
                "retry_after_s": round(max(0.0, rec.retry_at - now), 1),
            })
        out.sort(key=lambda item: item["model"])
        return out

    async def recover_dead_model(self, model_id_hint: str, dead_port: int,
                                 expected_generation: int = LEGACY_GENERATION,
                                 force: bool = False) -> Optional[LoadedModel]:
        """Evict and reload the resident model whose engine refused a connection.

        This is the recovery half of the TCP forwarder's dead-backend detection: the
        forwarder has already observed a connection-refused error on ``dead_port``.
        Recoveries are deduplicated by ``(dead_port, expected_generation)``, so a burst
        of concurrent requests against one dead engine performs a single unload/reload
        cycle and every caller awaits the same result — while a delayed recovery for an
        OLD instance cannot attach to a healthy replacement that reused the port.

        ``expected_generation`` is the LoadedModel.generation captured when the request
        was routed. If the generation currently on that port is newer, recovery returns
        the replacement unchanged instead of evicting it. A value of 0 keeps the legacy
        port-only behavior for fakes/older callers.

        Returns the ready replacement, or ``None`` when no recovery was possible. A
        killed-but-unverifiable process is left tracked (ready=False) exactly like any
        other failed unload; we never start a second engine on top of a process that
        may still be running.

        BUG_0007: after a failed reload the model enters an exponential cooldown and
        subsequent non-forced calls return ``None`` immediately instead of spawning
        again. ``force=True`` (manual action) bypasses the cooldown.
        """
        lock_key = model_id_hint
        if not force:
            lock_key = await self._recovery_lock_key(model_id_hint, dead_port)

        task_key = (dead_port, expected_generation)
        async with self._lock:
            if not force:
                remaining = self.recovery_retry_after(
                    lock_key, dead_port, expected_generation)
                if remaining is None:
                    remaining = self.recovery_retry_after(lock_key)
                if remaining is not None:
                    log.warning(
                        f"Recovery for '{lock_key}' on port {dead_port} suppressed: "
                        f"failed-reload cooldown has {remaining:.1f}s remaining")
                    return None
            task = self._recovery_tasks.get(task_key)
            if task is None or task.done():
                task = asyncio.create_task(
                    self._recover_dead_model_impl(
                        model_id_hint, dead_port, expected_generation))
                self._recovery_tasks[task_key] = task
                task.add_done_callback(
                    lambda t, key=task_key: self._forget_recovery_task(key, t))
        # Shield: one request's cancellation (client disconnect, shutdown of its worker)
        # must not cancel the shared recovery the other requests are waiting on.
        return await asyncio.shield(task)

    def _forget_recovery_task(self, task_key: Tuple[int, int],
                              task: asyncio.Task) -> None:
        if self._recovery_tasks.get(task_key) is task:
            self._recovery_tasks.pop(task_key, None)

    def recovery_in_progress(self, dead_port: int,
                             expected_generation: int = LEGACY_GENERATION) -> bool:
        """True while a dead-backend recovery owns ``dead_port``.

        With a concrete ``expected_generation``, only that engine generation counts;
        with the legacy 0 form, any active recovery for the port counts. Called by
        forwarder worker threads: ``dict.get`` and ``Task.done()`` are safe reads from
        another thread; no manager lock is needed (and cannot be taken here).
        """
        if expected_generation != LEGACY_GENERATION:
            task = self._recovery_tasks.get((dead_port, expected_generation))
            return task is not None and not task.done()
        return any(
            key[0] == dead_port and not task.done()
            for key, task in list(self._recovery_tasks.items())
        )

    async def _recovery_lock_key(self, model_id_hint: str, dead_port: int) -> str:
        """Resolve the per-model lock key for a recovery.

        Prefer the model currently owning ``dead_port`` (the exact engine being
        recovered), then the hinted id/alias, then the inference fallback. Taking
        ``self._lock`` here is intentionally brief; the slow teardown happens later
        under the per-model lock only.
        """
        async with self._lock:
            for mid, lm in self.loaded.items():
                if lm.port == dead_port:
                    return mid
            resolved = self._lookup_model_locked(model_id_hint)
            if resolved:
                return resolved
            ready_loaded = [mid for mid, lm in self.loaded.items() if lm.ready]
            if ready_loaded:
                return ready_loaded[0]
        return model_id_hint

    async def _recover_dead_model_impl(self, model_id_hint: str, dead_port: int,
                                       expected_generation: int = LEGACY_GENERATION) -> Optional[LoadedModel]:
        """Do one dead-backend eviction/reload (``recover_dead_model`` dedups calls).

        BUG_0005: the slow drain/kill runs OUTSIDE ``self._lock``. Only the fast
        validate-and-detach phase and the final registry update hold the global
        lock, so routing/status for unrelated models stays responsive. A per-model
        lock prevents a load from spawning a replacement on the same stable port
        while the old process is still being torn down.
        """
        lock_key = await self._recovery_lock_key(model_id_hint, dead_port)
        reload_mid: Optional[str] = None
        async with self._model_lock(lock_key):
            detached_lm: Optional[LoadedModel] = None
            deliberate = False
            try:
                async with self._lock:
                    hinted = self.loaded.get(model_id_hint)

                    # A concurrent recovery may already have replaced this model on a new
                    # port. Reuse that instance instead of cycling it a second time.
                    if hinted is not None and hinted.port != dead_port:
                        if hinted.ready:
                            self._clear_recovery_failure(model_id_hint)
                            return hinted
                        reload_mid = model_id_hint  # join the in-flight load below

                    else:
                        lm = hinted if (hinted is not None and hinted.port == dead_port) else None
                        if lm is not None:
                            reload_mid = model_id_hint
                        else:
                            for mid, cand in self.loaded.items():
                                if cand.port == dead_port:
                                    lm, reload_mid = cand, mid
                                    break

                        if lm is None:
                            # The dead instance is already gone (possibly recovered and
                            # then unloaded). Resolve the hint so this request can still be
                            # served; never invent a model for an unknown name.
                            resolved = self._lookup_model_locked(model_id_hint)
                            if not resolved:
                                # Unknown inference name: match resolve_request_model's
                                # accepted fallback and reuse a ready resident model if one
                                # exists. Never invent a JIT load for the unknown name.
                                ready_loaded = [cand for cand in self.loaded.values()
                                                if cand.ready]
                                if ready_loaded:
                                    self._clear_recovery_failure(
                                        getattr(ready_loaded[0], "model_id", ""))
                                    return ready_loaded[0]
                                log.warning(
                                    f"Dead backend on port {dead_port} is gone and "
                                    f"'{model_id_hint}' no longer resolves; cannot recover")
                                return None
                            reload_mid = resolved
                            current = self.loaded.get(resolved)
                            if current is not None:
                                if current.ready:
                                    self._clear_recovery_failure(reload_mid)
                                    return current
                                # A load is already in flight; fall through to join it.
                        else:
                            # Only evict the instance that actually refused the connection.
                            # A newer generation on the same (stable) port is a healthy
                            # replacement from a concurrent load/restore; unloading it here
                            # would kill the fix instead of the failure.
                            if not self._gen_matches(lm.generation, expected_generation):
                                log.info(
                                    f"Recovery for {reload_mid} on port {dead_port} skipped: "
                                    f"port now has engine generation {lm.generation} "
                                    f"(expected {expected_generation})")
                                if lm.ready:
                                    self._clear_recovery_failure(reload_mid)
                                    return lm
                                # The replacement is still loading; join it below.
                            else:
                                state = ("refused a connection" if lm.ready
                                         else "is already marked unready")
                                log.error(f"Engine for {reload_mid} on port {dead_port} "
                                          f"{state}; evicting and reloading")
                                # Auto-save would hang against a dead engine and capture
                                # nothing useful. Mark unready first so _unload_finish skips it.
                                lm.ready = False
                                self._invalidate_status()
                                # Detach under the lock, then release it for the slow part.
                                begin = self._begin_unload_locked(reload_mid)
                                if begin is not None:
                                    detached_lm, deliberate = begin

                if detached_lm is not None:
                    try:
                        killed, kill_reason = await self._unload_finish(
                            detached_lm, reload_mid)
                    except BaseException as exc:
                        # BUG_0009: the teardown above runs OUTSIDE self._lock, so the
                        # pop made by _begin_unload_locked must be undone if it aborts:
                        # routing, LRU eviction and the next recovery pass all key off
                        # self.loaded. The entry comes back with ready=False, so idle_reaper
                        # will not reclaim it (it skips not-ready entries) -- that is
                        # deliberate; only the next recovery/unload pass can.
                        # Unconditional restore (unlike _unload_unlocked's liveness-gated
                        # handler): ready was forced False above, so this entry can't route.
                        # The flag is cleared before re-inserting on purpose: a cancelled
                        # lock acquire must never leave _unload_in_progress set forever.
                        if deliberate:
                            self._unload_in_progress.discard(reload_mid)
                        async with self._lock:
                            self.loaded.setdefault(reload_mid, detached_lm)
                            self._invalidate_status()
                        log.error(
                            f"Recovery teardown for {reload_mid} aborted by "
                            f"{type(exc).__name__}; model restored to registry")
                        raise
                    async with self._lock:
                        result = self._finalize_unload_locked(
                            detached_lm, reload_mid, deliberate, killed, kill_reason)
                    if result is not None and not result.unloaded:
                        log.error(
                            f"Recovery of {reload_mid} aborted: engine process "
                            f"{result.pid} is still running ({result.reason}); "
                            f"keeping it tracked and refusing to start a duplicate")
                        self._record_recovery_failure(
                            reload_mid, dead_port, expected_generation,
                            f"engine process {result.pid} is still running "
                            f"({result.reason})")
                        return None
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - recovery must never escape into a worker thread
                log.error(f"Dead-backend recovery for '{model_id_hint}' on port {dead_port} "
                          f"failed during eviction: {e}")
                self._record_recovery_failure(
                    reload_mid or lock_key, dead_port, expected_generation,
                    f"eviction failed: {e}")
                return None

            if reload_mid is None:
                return None
            try:
                # Hold the per-model lock across the reload too. A racing explicit
                # load joins this task through _loading_tasks; an explicit unload
                # waits here instead of observing the detached gap and returning
                # success before the replacement engine exists.
                lm = await self._load_model_serialized(reload_mid)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - reported to the forwarder as a 503
                log.error(f"Dead-backend recovery reload of '{reload_mid}' failed: {e}")
                self._record_recovery_failure(
                    reload_mid, dead_port, expected_generation,
                    f"reload failed: {e}")
                return None
            self._clear_recovery_failure(reload_mid)
            return lm

    def _require_engine_gone(self, result: Optional[UnloadResult],
                             context: str) -> None:
        """Refuse a load when a preceding unload/eviction left its process running.

        Shared by the force-reload and LRU-eviction guards in _do_load_model: both
        must abort rather than spawn a replacement engine into a slot whose previous
        process was not verifiably killed (same contract as unload). ``context`` is
        the caller's own prefix; a ``None`` result means nothing was attempted and
        is never an error.
        """
        if result is not None and not result.unloaded:
            raise RuntimeError(
                f"{context}: process {result.pid} is still running ({result.reason})")

    async def _do_load_model(self, model_id: str, force: bool = False) -> LoadedModel:
        if not model_id or not model_id.strip():
            raise ValueError("Model ID must be a non-empty string")
        try:
            async with self._lock:
                existing = self.loaded.get(model_id)
                if existing is not None and (force or not existing.ready):
                    # A dead/not-ready entry must be evicted before spawning a replacement;
                    # otherwise the old process would be orphaned while the new one takes
                    # the registry slot. If the kill cannot be verified, refuse the load
                    # rather than leaking a second engine (same contract as unload).
                    result = await self._unload_unlocked(model_id)
                    self._require_engine_gone(
                        result, f"Cannot reload {model_id} after unloading "
                        f"the previous engine")
                evict_result = await self._evict_lru_if_needed(model_id)
                self._require_engine_gone(
                    evict_result,
                    f"Cannot load {model_id}: LRU eviction of "
                    f"{evict_result.model_id if evict_result else model_id} failed")
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

                self._engine_generation += 1
                lm = LoadedModel(
                    model_id=model_id,
                    gguf_path=artifact,
                    config=cfg,
                    port=port,
                    process=proc,
                    pid=proc.pid,
                    starting_at=time.time(),
                    generation=self._engine_generation,
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
            # Any successful load (manual, JIT, or recovery reload) clears the
            # failed-recovery circuit breaker for this model (BUG_0007).
            self._clear_recovery_failure(model_id)

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
                    self._begin_slot_action(port)
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
                    finally:
                        # Runs on every path so the drain counter cannot leak.
                        self._end_slot_action(port)

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

    def _begin_unload_locked(self, model_id: str) -> Optional[Tuple[LoadedModel, bool]]:
        """Detach an engine for teardown (caller must hold ``self._lock``).

        Fast phase of an unload: remove the model from the routing/registry view
        and arm the in-progress flag. The slow drain/kill happens in
        ``_unload_finish`` and may run without the global lock (recovery path).
        Returns ``(lm, deliberate)`` or None when the model was not resident.
        """
        lm = self.loaded.pop(model_id, None)
        if lm is None:
            return None
        self._invalidate_status()  # Force status cache rebuild on next request
        log.info(f"Unloading {model_id} (port {lm.port})")
        # Mark the release as IN PROGRESS: a state/load landing while this
        # method runs would cold-reload the endpoint the caller just released —
        # a full model load plus a full prompt reprocess. manager._lock spans
        # this whole teardown on the ordinary path, so no caller can observe the
        # flag mid-unload: this is a race backstop, not the mechanism that
        # closed BUG_0089 (see .agent_lessons/unload-in-progress-flag-replaces-recent-unload-window.md).
        # Only DELIBERATE releases set the flag. _unload_unlocked is also
        # reached from a load in flight (force-reload, failed-load and
        # cancelled-load cleanup); a state/load landing during such a load
        # must join it through the load path, not be refused.
        deliberate = not self._is_loading(model_id)
        if deliberate:
            self._unload_in_progress.add(model_id)
        return lm, deliberate

    def _finalize_unload_locked(self, lm: LoadedModel, model_id: str,
                                deliberate: bool, killed: bool,
                                kill_reason: str) -> UnloadResult:
        """Finalize registry state after teardown (caller must hold ``_lock``)."""
        try:
            if not killed:
                self.loaded.setdefault(model_id, lm)
                # The queue was dropped before the kill; restore it so the still-
                # resident engine keeps its FIFO ordering for future requests.
                try:
                    self.queue_registry.get_or_create(lm.port, model_id)
                except Exception as e:  # noqa: BLE001 - tracking correctness comes first
                    log.warning(f"Could not restore FIFO queue for {model_id} after failed unload: {e}")
                self._invalidate_status()
                log.error(f"Unload of {model_id} failed: process {lm.pid} is still "
                          f"running ({kill_reason}); keeping the model tracked")
                return UnloadResult(model_id, False, lm.pid, kill_reason)
            return UnloadResult(model_id, True, lm.pid)
        finally:
            # Clear the flag only once _unload_finish has run to completion —
            # queue dropped and process killed. Past that point a state/load is
            # the legitimate post-command restore of the safe-mode contract and
            # must be allowed, so this is the discriminator, not a timer.
            # try/finally: an exception part-way through the teardown must not
            # leave the flag set forever and refuse every later restore.
            if deliberate:
                self._unload_in_progress.discard(model_id)

    async def _unload_unlocked(self, model_id: str) -> Optional[UnloadResult]:
        """Unload a model (caller must hold self._lock).

        Returns ``None`` when the model was not resident. Otherwise the result's
        ``unloaded`` flag says whether the engine process is verifiably gone. On a
        failed kill the model is put back into ``self.loaded``, and on an aborted
        teardown it is restored only while its process may still be running: the
        registry must never claim an engine was released while its process is still
        running, and must never resurrect one it has verified dead. Callers can
        retry the unload.
        """
        begin = self._begin_unload_locked(model_id)
        if begin is None:
            return None
        lm, deliberate = begin
        try:
            killed, kill_reason = await self._unload_finish(lm, model_id)
        except BaseException as exc:
            if deliberate:
                self._unload_in_progress.discard(model_id)
            # BUG_0009: _begin_unload_locked popped the model. Decide from the
            # process, not from the exception type: a CancelledError can arrive
            # either before the kill (process alive, must stay visible to routing
            # and LRU) or at _unload_finish's own re-raise after a verified
            # kill (dead, must NOT come back as a routable ready=True entry).
            # Unlike the recovery handler (which forces ready=False before detaching),
            # a normal unload can abort with the model still ready=True — hence this gate.
            # The caller already holds self._lock (contract), so the restore runs
            # directly — re-acquiring would deadlock. setdefault matches the
            # sibling restores in _finalize_unload_locked / recovery.
            try:
                if lm.process is not None:
                    still_running = lm.process.poll() is None
                else:
                    # Same "no real pid means nothing was ever running" rule as
                    # _kill_proc uses for fake/legacy entries. Unlike the Popen
                    # probe (handle-based), this PID probe cannot tell a zombie or
                    # a recycled PID from a live engine, so it errs toward keeping
                    # the model tracked. Reachable today only from tests: every
                    # real LoadedModel carries its Popen.
                    still_running = bool(lm.pid and lm.pid > 0
                                         and psutil.pid_exists(lm.pid))
            except Exception:  # noqa: BLE001 - a failed probe must not mask exc
                still_running = True  # doubt resolves toward the invariant
            if still_running:
                self.loaded.setdefault(model_id, lm)
                self._invalidate_status()
                log.error(
                    f"Unload of {model_id} aborted by {type(exc).__name__}; "
                    f"process {lm.pid} may still be running, model restored "
                    f"to registry")
            else:
                log.error(
                    f"Unload of {model_id} aborted by {type(exc).__name__}; "
                    f"process {lm.pid} is gone, not re-registered")
            raise
        return self._finalize_unload_locked(
            lm, model_id, deliberate, killed, kill_reason)

    async def _unload_finish(self, lm: LoadedModel, model_id: str) -> Tuple[bool, str]:
        """Complete an unload: auto-save, drain, drop the queue, kill, clean up.

        Returns ``(killed, reason)``. ``killed`` is False when the engine process
        could not be verified dead; every other cleanup step is best-effort and
        never changes that verdict.
        """
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

        # A save/restore holding a ticket on this port is mid-operation against
        # this very engine; killing it now truncates the write (that is what left
        # the orphan ``*.bin.<hash>.tmp``). Wait for it to finish first — bounded,
        # and deadlock-free (see _drain_inflight_slot_action).
        # A cancellation of the WAIT must not skip the queue drop and the kill:
        # releasing memory is the point of this call, so the drain is only a
        # courtesy. Fall through and re-raise once the cleanup has run.
        cancelled = False
        try:
            await self._drain_inflight_slot_action(lm.port)
        except asyncio.CancelledError:
            cancelled = True
            log.warning(f"Unload of {model_id} cancelled while draining; dropping the queue and killing anyway")

        # Stop and drop this model's FIFO queue before killing the process, so no new
        # ticket is granted to a port that is about to die and any waiters are woken.
        self.queue_registry.remove(lm.port)

        # Kill entire process group reliably across platforms. This function
        # VERIFIES the outcome instead of trusting a successful call: a refusal
        # (AccessDenied) or a process that survives kill must surface as a
        # failure, never as a silent no-op.
        def _kill_proc() -> Tuple[bool, str]:
            # Fake/legacy entries may not carry a real pid. There is nothing to
            # kill, and probing pid 0/system processes is unsafe.
            if not lm.pid or lm.pid <= 0:
                return True, ""

            primary_error: Optional[BaseException] = None
            parent = None
            children = []
            try:
                parent = psutil.Process(lm.pid)
            except psutil.NoSuchProcess:
                return True, ""  # already gone
            except Exception as e:  # noqa: BLE001 - fallback below still gets a chance
                primary_error = e

            if parent is not None:
                try:
                    children = parent.children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                except Exception as e:  # noqa: BLE001
                    # One inaccessible grandchild must not abort the whole teardown.
                    # Try direct children so the parent and its accessible children
                    # are still terminated; the fallback below handles the rest.
                    primary_error = primary_error or e
                    try:
                        children = parent.children(recursive=False)
                    except psutil.NoSuchProcess:
                        children = []
                    except Exception as direct_error:  # noqa: BLE001
                        primary_error = primary_error or direct_error
                        children = []

            targets = ([parent] if parent is not None else []) + children

            # Terminate children first, then parent. Catch AccessDenied/ZombieProcess
            # per process: a single protected or already-exiting descendant should
            # not prevent the rest of the tree (especially the engine parent) from
            # being stopped.
            for proc in children:
                try:
                    proc.terminate()
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:  # noqa: BLE001
                    primary_error = primary_error or e
            if parent is not None:
                try:
                    parent.terminate()
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:  # noqa: BLE001
                    primary_error = primary_error or e

            alive = []
            if targets:
                try:
                    _gone, alive = psutil.wait_procs(targets, timeout=3)
                except Exception as e:  # noqa: BLE001
                    primary_error = primary_error or e
                    alive = list(targets)
                for proc in alive:
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass
                    except Exception as e:  # noqa: BLE001
                        primary_error = primary_error or e
                try:
                    _gone, alive = psutil.wait_procs(alive, timeout=3)
                except Exception as e:  # noqa: BLE001
                    primary_error = primary_error or e

            # Force-kill when the normal path left a process alive or when psutil
            # could not even open the parent. On Windows Popen.kill() is
            # TerminateProcess; the old ``os.CTRL_C_EVENT`` fallback did not exist
            # (it is a ``signal`` constant) and therefore never ran.
            fallback_error: Optional[BaseException] = None
            if alive or parent is None:
                if primary_error is not None:
                    log.warning(f"Process termination needed a fallback for {model_id} "
                                f"(pid {lm.pid}): {primary_error}")
                if platform.system() == "Windows":
                    try:
                        lm.process.kill()
                    except Exception as e:  # noqa: BLE001 - reported via verification
                        fallback_error = e
                else:
                    try:
                        os.killpg(os.getpgid(lm.pid), signal.SIGKILL)
                    except Exception as e:  # noqa: BLE001 - Popen.kill is the backstop
                        fallback_error = e
                    try:
                        lm.process.kill()
                    except Exception as e:  # noqa: BLE001
                        fallback_error = fallback_error or e

            # Verify the PARENT is gone — that's what holds GPU memory and the port.
            # Children (e.g. strata workers) are best-effort: they don't hold engine
            # resources, so lingering children don't block the swap.
            verify_error: Optional[BaseException] = None
            if parent is not None:
                try:
                    parent_alive = parent.is_running() and parent.status() != psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    parent_alive = False
                except Exception as e:  # noqa: BLE001
                    verify_error = e
                    parent_alive = True  # cannot verify → treat as alive
            else:
                # Never opened the parent; check by pid.
                try:
                    parent_alive = psutil.Process(lm.pid).is_running()
                except psutil.NoSuchProcess:
                    parent_alive = False
                except Exception as e:  # noqa: BLE001
                    verify_error = e
                    parent_alive = True

            if parent_alive or verify_error is not None:
                reason_parts = []
                if primary_error is not None:
                    reason_parts.append(str(primary_error))
                if parent_alive:
                    reason_parts.append(f"parent process {lm.pid} still running")
                if verify_error is not None:
                    reason_parts.append(f"could not verify termination: {verify_error}")
                if fallback_error is not None:
                    reason_parts.append(f"fallback error: {fallback_error}")
                return False, "; ".join(reason_parts) or "process still running"

            # Parent is gone. Note any lingering children for diagnostics.
            lingering = []
            for proc in children:
                try:
                    if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                        lingering.append(proc.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if lingering:
                log.warning(
                    f"Child process(es) {', '.join(str(p) for p in lingering)} "
                    f"still running after unload of {model_id} (parent {lm.pid} gone); "
                    f"they will be reaped by the OS")
            return True, ""

        killed, kill_reason = await asyncio.to_thread(_kill_proc)
        if not killed:
            log.error(f"Process termination failed for {model_id} (pid {lm.pid}): {kill_reason}")

        # Bridge cards only: remove the derived config we wrote at launch. The path is
        # deterministic, so this also covers LRU-evict, force-reload, and timeout-failure
        # unloads (all route through here). The user's engine JSON is never touched.
        if lm.config.is_bridge:
            try:
                self._derived_config_path(lm.config.engine, model_id).unlink(missing_ok=True)
            except Exception as e:
                log.warning(f"Failed to delete derived engine config for {model_id}: {e}")

        if cancelled:
            # The queue is dropped, the process killed and the cleanup done, so the
            # unload itself succeeded; only the caller's cancellation is preserved.
            raise asyncio.CancelledError

        return killed, kill_reason

    async def unload_model(self, model_id_input: str) -> Optional[UnloadResult]:
        mid = await self.resolve_model_id_strict(model_id_input)
        if not mid:
            return None
        async with self._model_lock(mid):
            async with self._lock:
                result = await self._unload_unlocked(mid)
            if result is None or result.unloaded:
                self._clear_recovery_failure(mid)
            return result

    async def unload_model_exact(self, model_id: str) -> Optional[UnloadResult]:
        """Unload a specific registry id without name/stem resolution.

        Used by delete after the card has already been claimed/removed from
        ``self.models``: calling the resolving variant at that point would find
        nothing and silently leave the engine process running.
        """
        async with self._model_lock(model_id):
            async with self._lock:
                result = await self._unload_unlocked(model_id)
            if result is None or result.unloaded:
                self._clear_recovery_failure(model_id)
            return result

    async def unload_all(self) -> Dict[str, Any]:
        """Unload every resident model and report which ones actually went away.

        A kill that cannot be verified is reported in ``failed`` (and the model
        remains tracked), so callers never mistake an attempted unload for a
        released one. Returns the same shape as the HTTP endpoint:
        ``{"unloaded_all": bool, "unloaded": [...], "failed": [{model, pid, reason}]}``.
        """
        async with self._lock:
            ids = list(self.loaded.keys())
            results = [await self._unload_unlocked(mid) for mid in ids]

        unloaded = [r.model_id for r in results if r is not None and r.unloaded]
        failed = [r.failure() for r in results if r is not None and not r.unloaded]
        for mid in ids:
            result = next((r for r in results if r is not None and r.model_id == mid), None)
            if result is None or result.unloaded:
                self._clear_recovery_failure(mid)
        return {"unloaded_all": not failed, "unloaded": unloaded, "failed": failed}

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
    def _begin_slot_action(self, port: int) -> None:
        """Count a slot action against ``port`` BEFORE it acquires its ticket.

        Counting from before acquire_ticket (not just while granted) is what lets
        the drain keep waiting through a queue hand-off: model_queue grants the
        next waiter the instant the drained holder releases.
        """
        self._inflight_slot_actions[port] = self._inflight_slot_actions.get(port, 0) + 1

    def _end_slot_action(self, port: int) -> None:
        """Undo one _begin_slot_action, from an outer finally (runs on every path)."""
        n = self._inflight_slot_actions.get(port, 0) - 1
        if n > 0:
            self._inflight_slot_actions[port] = n
        else:
            self._inflight_slot_actions.pop(port, None)

    async def _drain_inflight_slot_action(self, port: int) -> None:
        """Wait (bounded) for the slot actions counted against ``port`` to finish.

        An unload calls this before dropping the port's FIFO queue and killing
        the engine: otherwise a save mid-write is truncated, which is what left
        the orphan ``*.bin.<hash>.tmp`` in states/. Counted holders include
        actions still WAITING for a ticket — model_queue grants the next waiter
        the instant the drained holder releases, so an untracked queued holder
        would otherwise be killed mid-write right after the hand-off.

        Deadlock-free by construction: this polls a plain dict and never
        acquires a lock, and the operations it waits on (save_state /
        load_state / _do_load_model's auto-restore) never hold ``self._lock``
        across the slot I/O — they take it only for the brief membership lookup.

        Cost, deliberately: the wait runs under ``unload_all``'s lock, so other
        operations on the pool can stall up to the cap behind it. The cap keeps
        that bounded, and a holder that does not finish in time is simply left
        to fail (memory safety wins over preserving the write).
        """
        if self._inflight_slot_actions.get(port, 0) <= 0:
            return
        log.info(
            f"Unloading: waiting for in-flight slot action on port {port} before unload "
            f"(max {SLOT_ACTION_DRAIN_TIMEOUT_S:.0f}s)"
        )
        deadline = time.monotonic() + SLOT_ACTION_DRAIN_TIMEOUT_S
        while self._inflight_slot_actions.get(port, 0) > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning(
                    f"In-flight slot action on port {port} did not finish within {SLOT_ACTION_DRAIN_TIMEOUT_S:.0f}s; "
                    "proceeding with the kill anyway (memory safety wins; the save fails cleanly)"
                )
                return
            await asyncio.sleep(min(0.1, remaining))

    def _cleanup_state_temps(self) -> int:
        """Delete the engine's interrupted-save scratch files older than STATE_TEMP_MAX_AGE_S.

        A save killed mid-write leaves ``<state-name>.<hash>.tmp`` behind (the
        engine writes a dot-prefixed temp, e.g. ``.model.label.bin.08a653e446816a81.tmp``)
        and nothing else ever reclaims it, so multi-GB scratch files accumulate.
        Only names ending in ``.tmp`` with a ``.bin.`` component are touched —
        completed ``.bin`` state files never match.
        """
        cutoff = time.time() - STATE_TEMP_MAX_AGE_S
        removed = 0
        try:
            for p in self.save_state_dir.iterdir():
                name = p.name
                if not name.endswith(".tmp") or ".bin." not in name:
                    continue
                try:
                    if not p.is_file():
                        continue
                    st = p.stat()
                    if st.st_mtime > cutoff:
                        continue
                    p.unlink()
                    removed += 1
                    log.info(
                        f"Removed orphan state temp file: {name} "
                        f"({st.st_size / (1024 * 1024):.1f} MB, older than {STATE_TEMP_MAX_AGE_S:.0f}s)"
                    )
                except Exception as e:
                    log.debug(f"Failed to remove state temp file {name}: {e}")
        except Exception as e:
            log.debug(f"State temp cleanup failed: {e}")
        return removed

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
                # str() of some httpx exceptions (e.g. ConnectError('')) is empty; repr()
                # always carries the class and its args, so the log/exception stay legible.
                detail = str(e) or repr(e)
                log.warning(f"Slot {action} attempt failed [{desc}]: {detail}")
                last_err = f"[{desc}] {detail}"
                if isinstance(e, _UNREACHABLE_ERRORS):
                    # The engine is unreachable at all (refused/closed) — not an
                    # HTTP error it returned. The remaining endpoint formats hit
                    # the same dead socket, so stop burning time on them.
                    log.warning(
                        f"Slot {action} aborted for port {port}: engine unreachable "
                        f"({type(e).__name__}), not retrying other formats"
                    )
                    break

        raise RuntimeError(f"{action} failed across all slot endpoints. Last error: {last_err}")

    async def _perform_slot_save(self, port: int, state_path: Path, timeout: float = 120.0) -> bool:
        """Attempt to save slot state trying supported endpoint formats."""
        state_path.parent.mkdir(parents=True, exist_ok=True)
        # A save killed mid-write leaves a multi-GB scratch file behind; reclaim
        # the stale ones right before writing another one (never touches .bin).
        self._cleanup_state_temps()
        log.info(f"Saving slot 0 state to file: {state_path}")
        return await self._perform_slot_action("save", port, state_path, timeout)

    async def _perform_slot_restore(self, port: int, state_path: Path, timeout: float = 120.0) -> bool:
        """Attempt to restore slot state trying supported endpoint formats."""
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        log.info(f"Restoring slot 0 state from file: {state_path}")
        return await self._perform_slot_action("restore", port, state_path, timeout)

    async def save_state(self, model_id_input: str, label: str = "default") -> Path:
        model_id = await self.resolve_model_id_strict(model_id_input)
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
            self._begin_slot_action(lm.port)
            try:
                queue = self.queue_registry.get_or_create(lm.port, model_id)
                ticket = await asyncio.to_thread(queue.acquire_ticket, timeout=120.0)
                try:
                    await self._perform_slot_save(lm.port, state_path, timeout=120.0)
                finally:
                    queue.release_ticket(ticket)
            finally:
                # Runs on every path (including a failed acquire), so the counter
                # can never leak and leave later unloads paying the full drain cap.
                self._end_slot_action(lm.port)
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
        model_id = await self.resolve_model_id_strict(model_id_input)
        if not model_id:
            raise HTTPException(404, f"Unknown model: {model_id_input}")

        async with self._lock:
            lm = self.loaded.get(model_id)

        if lm is None or not lm.ready:
            # This endpoint is being released right now. Silently cold-reloading
            # here is what made a racing/abandoned restore look like a success
            # while burning a full model load; refuse instead so the caller can
            # decide (409 = conflict with the release request). Once the unload
            # has completed the flag is cleared and the restore is allowed —
            # that is the legitimate post-command restore of the safe-mode
            # contract, which by design lands seconds after unload_all returns.
            if model_id in self._unload_in_progress:
                log.info(
                    f"Refusing state restore for '{model_id}' (label '{label}'): an unload of "
                    f"this endpoint is still in progress"
                )
                raise HTTPException(
                    409,
                    f"unload in progress: model '{model_id}' is being released; reload it "
                    f"explicitly if a state restore is intended",
                )
            log.info(f"Auto-loading model '{model_id}' before state restore")
            try:
                # Bridge engines legitimately take up to ready_timeout_seconds
                # (600s by default); the old fixed 60s timeout cancelled them.
                load_timeout = self.ready_timeout_for(model_id) + 30.0
                lm = await asyncio.wait_for(self.load_model(model_id), timeout=load_timeout)
            except asyncio.TimeoutError:
                raise HTTPException(504, f"Timeout loading model '{model_id}' for state restore")
            except Exception as e:
                log.error(f"Failed to load model '{model_id}' before state restore: {e}")
                raise HTTPException(503, f"Failed to load model before state restore: {e}")

        async with self._lock:
            lm = self.loaded.get(model_id)
        if lm is None or not lm.ready:
            raise HTTPException(503, f"Model '{model_id}' is no longer loaded")
        mid_clean = self._sanitize_model_id(model_id)
        label_clean = self._sanitize_label(label)
        state_path = self.save_state_dir / f"{mid_clean}.{label_clean}.bin"

        try:
            self._begin_slot_action(lm.port)
            try:
                queue = self.queue_registry.get_or_create(lm.port, model_id)
                ticket = await asyncio.to_thread(queue.acquire_ticket, timeout=120.0)
                try:
                    await self._perform_slot_restore(lm.port, state_path, timeout=120.0)
                finally:
                    queue.release_ticket(ticket)
            finally:
                self._end_slot_action(lm.port)
        except Exception as e:
            raise HTTPException(500, f"restore failed: {e}")

        lm.state_path = state_path
        
        return state_path

    async def list_states(self, model_id_input: str) -> List[str]:
        model_id = await self.resolve_model_id_strict(model_id_input)
        if not model_id:
            raise HTTPException(404, f"Unknown model: {model_id_input}")
        mid_clean = self._sanitize_model_id(model_id)
        out = []
        for p in self.save_state_dir.glob(f"{mid_clean}.*.bin"):
            label = p.stem[len(mid_clean) + 1:]   # Strip the known model-id prefix and its trailing dot.
            out.append(label)
        return out

    async def delete_state(self, model_id_input: str, label: str) -> None:
        """Delete a state file for the given model and label."""
        model_id = await self.resolve_model_id_strict(model_id_input)
        if not model_id:
            raise HTTPException(404, f"Unknown model: {model_id_input}")
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
        recovery_failures = self.recovery_failures_snapshot()
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
            "recovery_failures": recovery_failures,
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
                        if lm.config.pinned or not lm.ready or lm.active_requests > 0:
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
        # Cancel any in-flight loading and dead-backend recovery tasks and wait for them
        async with self._lock:
            tasks = list(self._loading_tasks.values()) + list(self._recovery_tasks.values())
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
    with open(CONFIG_PATH, encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
    ModelManager.validate_config(CFG)
except Exception as e:
    raise RuntimeError(f"Failed to load config from '{CONFIG_PATH}': {e}") from e
manager = ModelManager(CFG)

@asynccontextmanager
async def _lifespan(app):
    await manager.start()
    # Capture the main running loop for cross-thread async lookups (forwarder threads).
    manager._main_loop = asyncio.get_running_loop()
    # Start the raw TCP forwarder on the MAIN port. It must come up before uvicorn is
    # asked to bind, but we run uvicorn on the INTERNAL port so there is no collision.
    global _forwarder
    _forwarder = RawTCPForwarder(manager.port, INTERNAL_API_PORT, manager)
    _forwarder.start()
    try:
        yield
    finally:
        if _forwarder is not None:
            _forwarder.stop()
        await manager.stop()


app = FastAPI(title="llama-autoloader", version="0.1.0", lifespan=_lifespan)

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
        _atomic_write_text(
            CONFIG_PATH,
            yaml.safe_dump(cfg_yaml, default_flow_style=False, sort_keys=False,
                           allow_unicode=True),
        )
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
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)) or v <= 0:
            errors.append("'idle_timeout_seconds' must be > 0")
    if "max_loaded_models" in body:
        v = body["max_loaded_models"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            errors.append("'max_loaded_models' must be >= 1")
    if "save_state_max_keep" in body:
        v = body["save_state_max_keep"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            errors.append("'save_state_max_keep' must be >= 1")
    if "poll_interval_seconds" in body:
        v = body["poll_interval_seconds"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)) or v <= 0:
            errors.append("'poll_interval_seconds' must be > 0")
    if "base_port" in body:
        v = body["base_port"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1024 or v > 65535:
            errors.append("'base_port' must be 1024-65535")
    if "auto_save_state" in body:
        v = body["auto_save_state"]
        if not isinstance(v, bool):
            errors.append("'auto_save_state' must be a boolean")
    if "host" in body:
        v = body["host"]
        if not isinstance(v, str) or not v.strip() or any(c.isspace() for c in v.strip()):
            errors.append("'host' must be a non-empty hostname/IP without whitespace")
    if "port" in body:
        v = body["port"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1 or v > 65535:
            errors.append("'port' must be 1-65535")
    if "default_args" in body and not isinstance(body["default_args"], str):
        errors.append("'default_args' must be a string")
    if "selected_backend" in body:
        v = body["selected_backend"]
        if not isinstance(v, str):
            errors.append("'selected_backend' must be a string")
        elif v:
            backend_ids = [b["id"] for b in manager.list_backends()]
            if v not in backend_ids:
                errors.append(
                    f"'selected_backend' '{v}' is not an installed backend "
                    f"(available: {backend_ids})")

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

    # Persist changes to config.yaml. host/port are included even when they are
    # the ONLY fields supplied: they cannot be applied at runtime, so a restart
    # is the only way they take effect, and the old `if changed` gate silently
    # dropped them (the response said "restart required" but nothing was saved).
    if changed or "host" in body or "port" in body:
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
            cfg["launcher"]["host"] = body["host"].strip()
        if "port" in body:
            cfg["launcher"]["port"] = body["port"]
        try:
            _atomic_write_text(
                CONFIG_PATH,
                yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False,
                               allow_unicode=True),
            )
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
    mid = await manager.resolve_model_id_strict(model_id)
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
    mid = await manager.resolve_model_id_strict(model_id)
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
    mid = await manager.resolve_model_id_strict(model_id)
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
        # Build + send with stream=True so the response body is relayed as it
        # arrives. `client.request(...)` used to buffer the ENTIRE response
        # (including SSE and large slot dumps) before StreamingResponse saw it.
        upstream_req = manager.client.build_request(
            request.method, url, content=body or None, headers=headers,
            timeout=httpx.Timeout(300.0, connect=10.0))
        upstream = await manager.client.send(upstream_req, stream=True)
    except Exception as e:
        log.warning(f"raw passthrough to {url} failed: {e}")
        raise HTTPException(502, f"Engine request failed: {e}")

    async def _relay():
        try:
            if request.method != "HEAD":
                async for chunk in upstream.aiter_bytes():
                    yield chunk
        finally:
            await upstream.aclose()

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
    mid = await manager.resolve_model_id_strict(model_id)
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
                cancelled = fw.cancel_active(lm.port, lm.generation)
            except Exception as e:
                log.warning(f"Soft interrupt failed for {model_id}: {e}")

        if hard:
            try:
                result = await manager._unload_unlocked(mid)
                if result is not None and not result.unloaded:
                    raise HTTPException(
                        500, f"Interrupt failed for {model_id}: "
                             f"process {result.pid} is still running ({result.reason})")
                return {"status": "interrupted", "model_id": mid, "mode": "hard",
                        "cancelled_requests": cancelled}
            except HTTPException:
                raise
            except Exception as e:
                log.warning(f"Interrupt (hard) failed for {model_id}: {e}")
                raise HTTPException(500, f"Interrupt failed for {model_id}: {e}")

        return {"status": "interrupted", "model_id": mid, "mode": "soft",
                "cancelled_requests": cancelled}

@app.put("/v1/models/{model_id}/config")
async def update_model_config(model_id: str, body: ModelConfigUpdate):
    mid = await manager.resolve_model_id_strict(model_id)
    if not mid:
        raise HTTPException(404, "Unknown model")
    cfg_live = await manager.get_config(mid)
    if cfg_live is None:
        raise HTTPException(404, "Unknown model")
    # Validate/edit a COPY: the old code mutated the live config in place, so a
    # rejected field (e.g. a duplicate name) left the in-memory registry dirty
    # even though the request returned 500 and nothing was persisted.
    cfg = copy.deepcopy(cfg_live)
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
    saved = await manager.update_config(mid, cfg)
    return asdict(saved)

@app.post("/v1/models/{model_id}/vision/toggle")
async def toggle_vision_ep(model_id: str):
    mid = await manager.resolve_model_id_strict(model_id)
    if not mid:
        raise HTTPException(404, "Unknown model")
    cfg_live = await manager.get_config(mid)
    if cfg_live is None:
        raise HTTPException(404, "Unknown model")
    manager._require_llama_engine(cfg_live)
    cfg = copy.deepcopy(cfg_live)
    cfg.use_mmproj = not cfg.use_mmproj
    saved = await manager.update_config(mid, cfg)
    return {"id": mid, "use_mmproj": saved.use_mmproj, "mmproj_file": saved.mmproj_file}

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
    if not mid or mid == "_":
        raise HTTPException(400, "engine_config must have a non-empty file stem (model id)")
    if body.name:
        target_name = body.name.lower()
        for other_mid, other_cfg in manager.models.items():
            if other_cfg.name and other_cfg.name.lower() == target_name:
                raise HTTPException(
                    409, f"Model name '{body.name}' is already used by '{other_mid}'")
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
        # A manual "load now" is an explicit operator action: it bypasses the
        # failed-recovery cooldown, even though JIT loads respect it (BUG_0007).
        lm = await manager.load_model(
            model_id, force=False, respect_recovery_cooldown=False)
        return {"id": model_id, "port": lm.port, "pid": lm.pid, "ready": lm.ready}
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(503, str(e))

@app.post("/v1/models/{model_id}/unload")
async def unload_model_ep(model_id: str):
    mid = await manager.resolve_model_id_strict(model_id)
    if not mid:
        raise HTTPException(404, f"Model not found: {model_id}")
    result = await manager.unload_model_exact(mid)
    if result is not None and not result.unloaded:
        # The process survived termination: do NOT report success. The model stays
        # tracked so /v1/models still shows it and a retry can be issued.
        return JSONResponse(status_code=500, content={
            "id": mid,
            "unloaded": False,
            "pid": result.pid,
            "reason": result.reason,
        })
    return {"id": mid, "unloaded": True}

@app.delete("/v1/models/{model_id}")
async def delete_model_ep(model_id: str):
    """Delete an engine bridge card (user-added). Llama/GGUF models and auto-discovered
    engine cards cannot be deleted through this endpoint (remove their source file instead)."""
    mid = await manager.resolve_model_id_strict(model_id)
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
        await manager.unload_model_exact(mid)
    except Exception as e:
        log.warning(f"Unload before delete failed for {mid}: {e}")

    # Remove the card file (and any legacy copy) so a rescan won't re-add it.
    cards_dir = manager._cards_dir_for(engine)
    for d in [cards_dir] + [Path(p) for p in LEGACY_CARDS_DIRS]:
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
    result = await manager.unload_all()
    if result["failed"]:
        # 200 here used to mean "all engines evicted". Returning 200 with failed
        # entries would preserve that dangerous ambiguity for existing callers.
        return JSONResponse(status_code=500, content=result)
    return result

# ---------- state save / load ----------
@app.post("/v1/models/{model_id}/state/save")
async def save_state_ep(model_id: str, body: StateLabel):
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
            # Connecting to a wildcard address is meaningless on most stacks;
            # probe loopback instead. A service bound to 0.0.0.0 is still
            # reachable there, which is the common "another instance" case.
            probe_host = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
            s.settimeout(0.5)
            return s.connect_ex((probe_host, port)) == 0
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
    # The internal FastAPI/uvicorn port is an implementation detail: only the
    # loopback forwarder talks to it. Bind it to 127.0.0.1 even when --host is
    # 0.0.0.0, otherwise the unauthenticated management API (and /v1/raw) would
    # be exposed on every interface while the forwarder itself stays loopback.
    _internal_bind_host = "127.0.0.1"
    INTERNAL_API_PORT, _auto_advanced_from = _resolve_internal_port(
        _wanted_internal, explicit=args.internal_port is not None, host=_internal_bind_host)

    # Fail fast and legibly on the one collision we must NOT paper over: if the main port
    # is taken, this instance has nothing to offer. Raw bind errors here are cryptic.
    if _port_in_use(manager.port, args.host):
        raise SystemExit(
            f"[start] Main port {manager.port} is already in use on {args.host}. "
            f"Another autoloader instance is probably running - pass --port <free port> "
            f"(and --internal-port / --base-port so this instance stays independent).")
    if _auto_advanced_from is not None:
        log.warning(
            f"Internal port {_auto_advanced_from} is already in use on {_internal_bind_host} "
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
    log.info(f"  Internal FastAPI    : http://{_internal_bind_host}:{INTERNAL_API_PORT}   (loopback only)")
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
    uvicorn.run(app, host=_internal_bind_host, port=INTERNAL_API_PORT, reload=False, use_colors=False, http=_http_backend)