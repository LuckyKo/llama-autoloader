"""Byte-level GGUF header builders for testing the fast binary metadata parser.

These helpers emit a minimal-but-valid GGUF header so that
``server._read_gguf_metadata`` can be exercised without any real model files.
The byte layout mirrors what is already proven in ``test_server.py``:

    4 bytes   magic b"GGUF"
    4 bytes   version (uint32, little-endian)
    8 bytes   tensor count (uint64)
    8 bytes   KV count     (uint64)
    then N KV pairs, each:
        8 bytes key length (uint64) + key bytes
        4 bytes value type code (uint32)
        <value payload per type>

Value-type codes used by the parser (see server._read_gguf_metadata.skip_val):
    0 BOOL(1B), 1 UINT8(1B), 7 FLOAT64(8B)  -> 1 byte
    2 INT16, 3 UINT16                        -> 2 bytes
    4 UINT32, 5 INT32, 6 FLOAT32             -> 4 bytes
    10 UINT64, 11 INT64, 12 FLOAT64          -> 8 bytes
    8 STRING: 8B len + bytes
    9 ARRAY : 4B element type + 8B count + repeated elements

The parser reads ``general.name`` (STRING) and any key ending in
``.context_length`` (UINT32/INT32/UINT64/INT64). All other KVs are skipped.
"""
from __future__ import annotations

import struct
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Value-type codes (subset relevant to the parser)
# ---------------------------------------------------------------------------
BOOL = 0
UINT8 = 1
INT16 = 2
UINT16 = 3
UINT32 = 4
INT32 = 5
FLOAT32 = 6
FLOAT64 = 7
STRING = 8
ARRAY = 9
UINT64 = 10
INT64 = 11


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _pack_str(key) + struct.pack("<I", vtype) + payload


def _ctx_payload(value: int, ctx_vtype: int) -> bytes:
    if ctx_vtype == UINT32:
        return struct.pack("<I", 4) + struct.pack("<I", value)
    if ctx_vtype == INT32:
        return struct.pack("<I", 5) + struct.pack("<i", value)
    if ctx_vtype == UINT64:
        return struct.pack("<I", 10) + struct.pack("<Q", value)
    if ctx_vtype == INT64:
        return struct.pack("<I", 11) + struct.pack("<q", value)
    raise ValueError(f"unsupported ctx_vtype {ctx_vtype}")


def _make_header(kvs: int, version: int = 3, tensors: int = 0) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", version, tensors, kvs)


# ---------------------------------------------------------------------------
# Primary builders (mirror test_server.py helpers)
# ---------------------------------------------------------------------------

def make_dummy_gguf(name: Optional[str] = None, max_ctx: Optional[int] = None,
                    ctx_vtype: int = UINT32) -> bytes:
    """Build a minimal GGUF header with optional general.name and llama.context_length.

    ``ctx_vtype`` selects the value type used for context_length (default 4=UINT32);
    pass 10=UINT64 or 11=INT64 to exercise the wider integer branches in skip_val.
    """
    kvs = 0
    if name is not None:
        kvs += 1
    if max_ctx is not None:
        kvs += 1

    buf = bytearray(_make_header(kvs))
    if name is not None:
        val = name.encode("utf-8")
        buf.extend(_pack_kv("general.name", STRING, struct.pack("<Q", len(val)) + val))
    if max_ctx is not None:
        buf.extend(_pack_str("llama.context_length"))
        buf.extend(_ctx_payload(max_ctx, ctx_vtype))
    return bytes(buf)


def make_gguf_with_skips(name: Optional[str] = None, max_ctx: Optional[int] = None,
                         skip_fields: Optional[List[Tuple[str, int, bytes]]] = None) -> bytes:
    """Build a GGUF header with extra KV fields (to be skipped by the parser) placed
    before general.name and llama.context_length.

    Each entry in ``skip_fields`` is ``(key, vtype, payload_bytes)``. This exercises the
    skip_val path for FLOAT32 (6), BOOL (7), ARRAY (9) etc. value types.
    """
    skip_fields = skip_fields or []
    kvs = len(skip_fields)
    if name is not None:
        kvs += 1
    if max_ctx is not None:
        kvs += 1

    buf = bytearray(_make_header(kvs))
    for key, vtype, payload in skip_fields:
        buf.extend(_pack_kv(key, vtype, payload))
    if name is not None:
        val = name.encode("utf-8")
        buf.extend(_pack_kv("general.name", STRING, struct.pack("<Q", len(val)) + val))
    if max_ctx is not None:
        buf.extend(_pack_str("llama.context_length"))
        buf.extend(_ctx_payload(max_ctx, UINT32))
    return bytes(buf)


def make_gguf_unknown_vtype(unknown_vtype: int = 13) -> bytes:
    """Build a GGUF header whose first KV value has an unrecognized type code.

    The parser's skip_val should raise on it and the outer handler returns (None, None).
    """
    buf = bytearray(_make_header(1))
    buf.extend(_pack_kv("some.unknown", unknown_vtype, b"\x00" * 16))
    return bytes(buf)


# ---------------------------------------------------------------------------
# Convenience: write a named dummy GGUF into a directory
# ---------------------------------------------------------------------------

def write_gguf(path, name: Optional[str] = None, max_ctx: Optional[int] = None,
               ctx_vtype: int = UINT32, skip_fields: Optional[List[Tuple[str, int, bytes]]] = None) -> "path":
    """Write a dummy GGUF file at ``path`` and return the path for chaining."""
    if skip_fields is not None:
        data = make_gguf_with_skips(name=name, max_ctx=max_ctx, skip_fields=skip_fields)
    else:
        data = make_dummy_gguf(name=name, max_ctx=max_ctx, ctx_vtype=ctx_vtype)
    path.write_bytes(data)
    return path
