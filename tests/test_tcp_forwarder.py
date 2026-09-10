"""Unit tests for tcp_forwarder parsing helpers.

These exercise ONLY the pure request-parsing/routing logic (_extract_model /
_is_llm_endpoint) and the raw-header helper — no live sockets, no event loop, no network.
The pipe/accept paths are integration concerns and are deliberately not tested here (see
constraints: do NOT run live tests).
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

# Ensure repo root is importable regardless of cwd (matches tests/conftest.py approach).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tcp_forwarder import RawTCPForwarder  # noqa: E402


def _fwd():
    """A forwarder instance with no real manager — only the pure helpers are used."""
    return RawTCPForwarder(listen_port=19999, internal_api_port=19998, model_manager=None)


# --------------------------------------------------------------------------- _extract_model
class TestExtractModel:
    def test_full_json(self):
        assert _fwd()._extract_model(b'{"model": "qwen3", "stream": true}') == "qwen3"

    def test_model_first_in_prefix(self):
        # Body may arrive truncated; model is near the front.
        assert _fwd()._extract_model(b'{"model":"llama-8b","messages":[{"role') == "llama-8b"

    def test_truncated_json_falls_back_to_regex(self):
        # Incomplete JSON (no closing brace) -> regex path.
        assert _fwd()._extract_model(b'{"stream": true, "model": "mistral"') == "mistral"

    def test_no_model_field(self):
        assert _fwd()._extract_model(b'{"stream": false}') is None

    def test_empty_body(self):
        assert _fwd()._extract_model(b"") is None

    def test_non_string_model_ignored(self):
        # model must be a string; numeric/other types are not usable as an id.
        assert _fwd()._extract_model(b'{"model": 123}') is None

    def test_unicode_model(self):
        assert _fwd()._extract_model('{"model": "mod\\u00e9l"}'.encode("utf-8")) == "mod\u00e9l"


# --------------------------------------------------------------------------- _is_llm_endpoint
class TestIsLlmEndpoint:
    def test_v1_chat_completions(self):
        assert RawTCPForwarder._is_llm_endpoint("/v1/chat/completions") is True

    def test_bare_chat_completions(self):
        assert RawTCPForwarder._is_llm_endpoint("/chat/completions") is True

    def test_v1_completions_and_embeddings(self):
        assert RawTCPForwarder._is_llm_endpoint("/v1/completions") is True
        assert RawTCPForwarder._is_llm_endpoint("/v1/embeddings") is True

    def test_query_string_ignored(self):
        assert RawTCPForwarder._is_llm_endpoint("/v1/chat/completions?x=1") is True

    def test_management_paths_not_llm(self):
        for p in ("/v1/status", "/v1/models", "/v1/settings", "/ws", "/", "/v1/raw/m/x"):
            assert RawTCPForwarder._is_llm_endpoint(p) is False, p

    def test_no_false_prefix_match(self):
        # /v1/completionsx must NOT match /v1/completions (boundary enforced).
        assert RawTCPForwarder._is_llm_endpoint("/v1/completionsx") is False


# --------------------------------------------------------------------------- _header_value (raw bytes)
class TestHeaderValue:
    def test_case_insensitive(self):
        raw = b"GET /v1/x HTTP/1.1\r\nContent-Length: 42\r\nHost: a\r\n\r\n"
        assert RawTCPForwarder._header_value(raw, "content-length") == "42"

    def test_missing_header(self):
        raw = b"GET / HTTP/1.1\r\nHost: a\r\n\r\n"
        assert RawTCPForwarder._header_value(raw, "content-length") is None

    def test_preserves_original_casing_of_value(self):
        raw = b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        assert RawTCPForwarder._header_value(raw, "transfer-encoding") == "chunked"


# --------------------------------------------------------------------------- routing decision
class TestRoutingDecision:
    """The core branch in _handle_connection: LLM endpoint + loaded model -> backend pipe; else API."""

    def test_llm_endpoint_with_model_goes_to_backend(self):
        f = _fwd()
        body = b'{"model": "qwen3", "stream": true}'
        assert f._is_llm_endpoint("/v1/chat/completions") and f._extract_model(body) is not None

    def test_non_llm_path_goes_to_api(self):
        f = _fwd()
        body = b'{"model": "qwen3"}'
        # Management path: even with a model field, it is not an LLM endpoint.
        assert not f._is_llm_endpoint("/v1/models")

    def test_llm_path_without_model_goes_to_api(self):
        f = _fwd()
        assert f._is_llm_endpoint("/v1/chat/completions") and f._extract_model(b"") is None


# --------------------------------------------------------------------------- _read_more_for_model
class _FakeSock:
    """Minimal blocking-socket stand-in: serves pre-chunked body bytes, then EOF."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def recv(self, n):
        if not self._chunks:
            return b""  # EOF
        chunk = self._chunks.pop(0)
        return chunk[:n]


def _hdr(content_length):
    return (f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Content-Length: {content_length}\r\nHost: a\r\n\r\n").encode("ascii")


class TestReadMoreForModel:
    def test_model_found_after_extra_read(self):
        # Model field sits well beyond the initial prefix; must be found by reading more.
        f = _fwd()
        pad = b'"system":"' + b"x" * 8000 + b'",'
        body = b'{' + pad + b'"model": "qwen3"}'
        headers = _hdr(len(body))
        initial = body[:100]  # no model yet
        rest = [body[100:]]
        result = f._read_more_for_model(_FakeSock(rest), headers, initial)
        assert f._extract_model(result) == "qwen3"

    def test_stops_early_once_model_present(self):
        # Should not over-read past the point where the model is locatable.
        f = _fwd()
        body = b'{"model": "llama-8b", "messages": [' + b'"pad"' * 5000 + b']}'
        headers = _hdr(len(body))
        initial = body[:20]
        # Feed the remainder in one chunk; extraction succeeds on first extra read.
        result = f._read_more_for_model(_FakeSock([body[20:]]), headers, initial)
        assert f._extract_model(result) == "llama-8b"

    def test_chunked_body_model_past_256kb_now_found(self):
        # Regression for the live 400 bug: a chunked (no Content-Length) body whose "model" field
        # sits beyond byte 256KB used to be truncated by the old scan cap and wrongly rejected.
        # Now we read past it until found, then stop at EOF.
        f = _fwd()
        headers = b"POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\n\r\n"  # no content-length
        pad_len = 270_000  # push the model field well past the old 256KB cap
        body = (b'{"system":"' + b"x" * pad_len + b'","model":"qwen-27b"}')
        # Feed in 64KB chunks, then EOF.
        chunks = [body[i:i + 65536] for i in range(0, len(body), 65536)]
        result = f._read_more_for_model(_FakeSock(chunks), headers, b"")
        assert f._extract_model(result) == "qwen-27b"

    def test_no_model_eof_terminates_cleanly(self):
        # No Content-Length and no model field at all: must terminate at EOF (not hang).
        f = _fwd()
        headers = b"POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\n\r\n"
        body = b'{"stream": true, "messages": []}'  # no model field
        chunks = [body[i:i + 65536] for i in range(0, len(body), 65536)]
        result = f._read_more_for_model(_FakeSock(chunks), headers, b"")
        assert isinstance(result, bytes)
        assert f._extract_model(result) is None

    def test_eof_before_full_body(self):
        # Peer closes early; must return what we have without hanging.
        f = _fwd()
        headers = _hdr(100000)  # declares more than will arrive
        result = f._read_more_for_model(_FakeSock([b'{"partial": ']), headers, b"")
        assert isinstance(result, bytes)
        assert f._extract_model(result) is None

    def test_respects_declared_content_length(self):
        # Must not read beyond the declared content-length even if more is available.
        f = _fwd()
        body = b'{"model": "m1"}'
        headers = _hdr(len(body))
        result = f._read_more_for_model(_FakeSock([body, b"EXTRA-TRAILING"]), headers, b"")
        assert len(result) <= len(body)
        assert f._extract_model(result) == "m1"


# --------------------------------------------------------------------------- _resolve_target_port
class _FakeLM:
    """Minimal LoadedModel stand-in exposing only what _resolve_target_port reads."""

    def __init__(self, port=9001, ready=True):
        self.port = port
        self.ready = ready
        self.last_used = 0.0
        self.touch_count = 0

    def touch(self):
        self.last_used = 1.0  # any change is enough to prove it was called
        self.touch_count += 1


class _FakeManager:
    """Stands in for the model manager used by _resolve_target_port."""

    def __init__(self, loaded, resolved_mid="qwen3"):
        self.loaded = loaded
        self._resolved = resolved_mid

    async def resolve_model_id(self, model_id):
        return self._resolved


class TestResolveTargetPort:
    """Regression: a piped request for an already-loaded model must refresh last_used.

    Without this the idle reaper unloads a model that is still actively being served
    through the raw-TCP fast path (the "recently used but idle-unloaded" bug).
    """

    def test_ready_model_returns_port_and_touches(self):
        # _resolve_target_port schedules resolve_model_id onto the manager's main loop and
        # blocks until it returns. In production this runs on a worker thread, so we must
        # NOT call it from within that same loop (it would deadlock). We therefore run a
        # dedicated event loop in a background thread and invoke the method from THIS
        # thread — mirroring the real cross-thread usage.
        lm = _FakeLM(port=9042, ready=True)
        mgr = _FakeManager(loaded={"qwen3": lm})
        f = RawTCPForwarder(listen_port=19999, internal_api_port=19998, model_manager=mgr)

        loop_holder = {}

        def _run_loop():
            loop = asyncio.new_event_loop()
            loop_holder["loop"] = loop
            mgr._main_loop = loop  # what _resolve_target_port falls back to in worker threads
            try:
                loop.run_forever()
            finally:
                loop.close()

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()
        while "loop" not in loop_holder:  # wait for the loop to be up
            time.sleep(0.01)

        try:
            port = f._resolve_target_port("qwen3")
        finally:
            loop_holder["loop"].call_soon_threadsafe(loop_holder["loop"].stop)
            t.join(timeout=5)

        assert port == 9042
        assert lm.touch_count == 1

    def test_not_ready_returns_none_and_does_not_touch(self):
        lm = _FakeLM(port=9042, ready=False)
        mgr = _FakeManager(loaded={"qwen3": lm})
        f = RawTCPForwarder(listen_port=19999, internal_api_port=19998, model_manager=mgr)

        loop_holder = {}

        def _run_loop():
            loop = asyncio.new_event_loop()
            loop_holder["loop"] = loop
            mgr._main_loop = loop
            try:
                loop.run_forever()
            finally:
                loop.close()

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()
        while "loop" not in loop_holder:
            time.sleep(0.01)
        try:
            assert f._resolve_target_port("qwen3") is None
        finally:
            loop_holder["loop"].call_soon_threadsafe(loop_holder["loop"].stop)
            t.join(timeout=5)
        assert lm.touch_count == 0

    def test_unknown_model_returns_none(self):
        mgr = _FakeManager(loaded={}, resolved_mid=None)
        f = RawTCPForwarder(listen_port=19999, internal_api_port=19998, model_manager=mgr)
        assert f._resolve_target_port("nope") is None
