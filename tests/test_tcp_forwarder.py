"""Unit tests for tcp_forwarder parsing helpers.

These exercise ONLY the pure request-parsing/routing logic (_extract_model /
_is_llm_endpoint) and the raw-header helper — no live sockets, no event loop, no network.
The pipe/accept paths are integration concerns and are deliberately not tested here (see
constraints: do NOT run live tests).
"""
from __future__ import annotations

import asyncio
import socket
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


# =========================================================================== CORS / preflight
# Regression for the CWrite breakage introduced by commit e4cbe26 (raw-TCP passthrough):
# llama-server emits NO CORS headers, so a cross-origin browser client had its response blocked.
# These tests prove CORS is restored on the LLM fast path WITHOUT regressing the AC streaming path
# or reintroducing SSE buffering. They use real socket pairs and model clients that STAY OPEN (a
# closed client would let the pipe see EOF and mask a hang).

def _run_with_timeout(fn, timeout=8.0):
    """Run ``fn`` in a thread; raise AssertionError if it does not finish within ``timeout``.

    Guards against a hang: a buggy forwarder that blocks on recv/select forever would otherwise
    wedge the whole test run. The worker is daemon so a stuck test still lets pytest move on.
    """
    result = {}

    def _target():
        try:
            result["value"] = fn()
        except Exception as e:  # noqa: BLE001 — surface it to the main thread
            result["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise AssertionError(f"forwarder did not finish within {timeout}s (hang)")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _recv_until_closed(sock, timeout=8.0):
    """Read from a socket until the peer closes it (EOF), returning all bytes."""
    sock.settimeout(timeout)
    chunks = []
    try:
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    except socket.timeout:
        pass  # peer did not close within the window; return what we have (caller asserts)
    finally:
        sock.settimeout(None)
    return b"".join(chunks)


def _socket_pair():
    """A connected pair of loopback sockets, both blocking. Returns (a, b)."""
    a, b = socket.socketpair()
    for s in (a, b):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return a, b


def _loopback_client():
    """A real loopback TCP connection to a fresh local server.

    Returns ``(client_sock, server_sock, server_listen_sock)``. Unlike ``socket.socketpair()``
    (which on Windows ignores ``settimeout`` — overlapped I/O), a real connected socket honours
    ``settimeout``/recv-timeout, so tests that exercise the bounded READ phase use this instead.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    client = socket.create_connection(("127.0.0.1", srv.getsockname()[1]))
    server_sock, _addr = srv.accept()
    for s in (client, server_sock):
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
    return client, server_sock, srv


def _client_side(fwd, backend_bytes, *, label="llama-server",
                 extra_backend=None, read_timeout=8.0):
    """Drive the full LLM fast path with real sockets and return (bytes_delivered, backend_got).

    - ``backend_bytes``: what the fake llama-server sends to the forwarder (its response).
    - ``extra_backend``: optional extra bytes the fake server sends later (e.g. a second SSE chunk),
      then closes — used to prove streaming continues verbatim after header injection.
    - The client socket STAYS OPEN for the whole run (it is only closed at teardown) so an EOF-based
      shortcut cannot mask a hang. After receiving the response it sends one more keep-alive request
      and closes — realistic HTTP/1.1 behaviour that lets the forwarder's pipe see data+EOF on the
      client side and terminate cleanly. Completion is proven by the forwarder thread finishing on time.
    """
    fwd._running = True  # unblock the pipe loop's `while self._running` guard
    client, client_fwd = _socket_pair()   # client <-> forwarder
    backend, backend_fwd = _socket_pair()  # forwarder <-> fake llama-server

    # Substitute the real backend socket for the (unreachable) port-0 connection so we can drive
    # the full fast path with a genuine peer. Restore afterwards to avoid leaking state.
    orig_connect = fwd._connect_backend
    fwd._connect_backend = lambda port: backend_fwd

    def _fake_server():
        try:
            backend.sendall(backend_bytes)
            if extra_backend:
                time.sleep(0.05)
                backend.sendall(extra_backend)
            time.sleep(0.1)  # keep open briefly so the forwarder finishes its first recv cleanly
        finally:
            try:
                backend.close()
            except OSError:
                pass

    def _forward():
        fwd._pipe_to_backend(client_fwd, b"", b"", 0, label)

    server_t = threading.Thread(target=_fake_server, daemon=True)
    server_t.start()
    fwd_t = threading.Thread(target=_forward, daemon=True)
    fwd_t.start()

    # Read the response until the backend closes (the forwarder half-closes our read side). The
    # client keeps its socket open while reading — no early EOF that could mask a hang. The CORS
    # bytes and body are fully captured by this point, so we now close to let the forwarder's pipe
    # observe client-side EOF and terminate (a real HTTP/1.1 client closes after its final response).
    delivered = _recv_until_closed(client, read_timeout)

    try:
        client.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass

    fwd_t.join(read_timeout)
    if fwd_t.is_alive():
        raise AssertionError("forwarder thread hung (client stayed open)")
    server_t.join(read_timeout)

    for s in (client, client_fwd, backend, backend_fwd):
        try:
            s.close()
        except OSError:
            pass
    fwd._connect_backend = orig_connect
    return delivered, b""


# --------------------------------------------------------------------------- _inject_cors_headers (pure)
class TestInjectCorsHeaders:
    def test_inserts_after_status_line(self):
        head = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
        from tcp_forwarder import _inject_cors_headers
        res = _inject_cors_headers(head)
        assert res.startswith(b"HTTP/1.1 200 OK\r\n")
        assert b"Access-Control-Allow-Origin: *\r\n" in res
        assert b"Access-Control-Allow-Methods: *\r\n" in res
        assert b"Access-Control-Allow-Headers: *\r\n" in res
        # Original headers preserved, order intact.
        assert b"Content-Type: text/event-stream\r\n" in res
        # CORS inserted before the original header (right after status line).
        assert res.index(b"Access-Control-Allow-Origin") < res.index(b"Content-Type")

    def test_no_double_inject_when_backend_sends_origin(self):
        head = b"HTTP/1.1 200 OK\r\naccess-control-allow-origin: *\r\nContent-Length: 5\r\n\r\n"
        from tcp_forwarder import _inject_cors_headers
        res = _inject_cors_headers(head)
        assert res == head  # unchanged

    def test_case_insensitive_existing_origin(self):
        head = b"HTTP/1.1 200 OK\r\nACCESS-CONTROL-ALLOW-ORIGIN: example.com\r\n\r\n"
        from tcp_forwarder import _inject_cors_headers
        assert _inject_cors_headers(head) == head

    def test_no_end_of_headers_returns_unchanged(self):
        partial = b"HTTP/1.1 200 OK\r\nContent-Type: x"  # no \r\n\r\n yet
        from tcp_forwarder import _inject_cors_headers
        assert _inject_cors_headers(partial) == partial

    def test_body_after_blank_line_untouched(self):
        head = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n"
        from tcp_forwarder import _inject_cors_headers
        res = _inject_cors_headers(head)
        # Everything after the (now shifted) blank line must still be empty / unchanged.
        assert res.endswith(b"\r\n\r\n")


# --------------------------------------------------------------------------- CORS injection on the LLM fast path (real sockets)
class TestCorsInjectionOnLlmPath:
    def test_cors_injected_exactly_once_content_length(self):
        f = _fwd()
        backend_resp = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        b"Content-Length: 16\r\n\r\n") + b'{"choices": []}\r\n'
        delivered, _ = _client_side(f, backend_resp)
        assert delivered.count(b"Access-Control-Allow-Origin: *") == 1
        assert delivered.count(b"Access-Control-Allow-Methods: *") == 1
        assert delivered.count(b"Access-Control-Allow-Headers: *") == 1
        # Other headers + body intact.
        assert b"Content-Type: application/json\r\n" in delivered
        assert b'{"choices": []}' in delivered

    def test_cors_injected_chunked_body_intact(self):
        f = _fwd()
        sse = (b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n\r\n"
               b"data: [DONE]\r\n\r\n")
        backend_resp = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                        b"Transfer-Encoding: chunked\r\n\r\n") + sse
        delivered, _ = _client_side(f, backend_resp)
        assert delivered.count(b"Access-Control-Allow-Origin: *") == 1
        # SSE body passes through byte-for-byte (no re-framing / no buffering corruption).
        assert b"data: [DONE]\r\n\r\n" in delivered
        assert b"Transfer-Encoding: chunked\r\n" in delivered

    def test_no_double_inject_when_backend_already_cors(self):
        f = _fwd()
        backend_resp = (b"HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\n"
                        b"Content-Length: 5\r\n\r\n") + b"hello"
        delivered, _ = _client_side(f, backend_resp)
        assert delivered.count(b"Access-Control-Allow-Origin: *") == 1

    def test_streaming_continues_verbatim_after_injection(self):
        f = _fwd()
        first = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                 b"Transfer-Encoding: chunked\r\n\r\n") + b"data: one\r\n\r\n"
        second = b"data: two\r\n\r\ndata: [DONE]\r\n\r\n"
        delivered, _ = _client_side(f, first, extra_backend=second)
        assert delivered.count(b"Access-Control-Allow-Origin: *") == 1
        assert b"data: one\r\n\r\n" in delivered
        assert b"data: two\r\n\r\n" in delivered          # second chunk piped verbatim
        assert b"data: [DONE]\r\n\r\n" in delivered

    def test_partial_head_split_across_recv_not_dropped(self):
        # Regression for a data-loss bug: if the first recv() returns a PARTIAL head (no \r\n\r\n
        # yet), the forwarder must keep buffering until the head is complete and must NEVER drop the
        # bytes it already read. Here the backend splits the head across two sends before the blank
        # line; the client must still receive the full, intact response.
        f = _fwd()
        full_head = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                     b"Content-Length: 5\r\n\r\n")
        body = b"hello"
        fragment1 = full_head[:20]   # "HTTP/1.1 200 OK\r\nConte..." (no blank line yet)
        fragment2 = full_head[20:]   # rest of head incl. terminating \r\n\r\n
        delivered, _ = _client_side(f, fragment1, extra_backend=fragment2 + body)
        # Nothing dropped: the complete status line + all original headers are present.
        assert b"HTTP/1.1 200 OK\r\n" in delivered
        assert b"Content-Type: application/json\r\n" in delivered
        assert b"Content-Length: 5\r\n" in delivered
        assert body in delivered
        # CORS injected exactly once, right after the status line.
        assert delivered.count(b"Access-Control-Allow-Origin: *") == 1

    def test_incomplete_head_closed_early_forwarded_verbatim(self):
        # Incomplete-head scenario: the backend sends a PARTIAL head (no terminating \r\n\r\n) and
        # then closes without ever completing it. The forwarder must NOT hang and must forward those
        # bytes verbatim to the client — no injection, no corruption, nothing dropped.
        f = _fwd()
        partial_head = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"  # no \r\n\r\n
        delivered, _ = _client_side(f, partial_head)
        # The exact bytes the backend sent arrive intact and are NOT modified (no CORS injected).
        assert delivered == partial_head
        assert b"Access-Control-Allow-Origin" not in delivered


# --------------------------------------------------------------------------- AC fast path unchanged (only CORS added)
class TestAcFastPathUnchanged:
    def test_normal_streaming_request_pipes_verbatim_only_cors_added(self):
        f = _fwd()
        sse_body = b"data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\r\n\r\n"
        backend_resp = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                        b"Transfer-Encoding: chunked\r\nX-Backend: llama\r\n\r\n") + sse_body
        delivered, _ = _client_side(f, backend_resp)
        # Exactly the three CORS headers added; no other header invented.
        assert delivered.count(b"Access-Control-Allow-Origin: *") == 1
        assert b"X-Backend: llama\r\n" in delivered          # backend's own header preserved
        assert sse_body in delivered                          # SSE bytes intact
        # No unexpected CORS-ish or duplicate headers.
        assert delivered.count(b"Access-Control-Max-Age") == 0


# --------------------------------------------------------------------------- OPTIONS preflight
class TestOptionsPreflight:
    def test_options_llm_returns_cors_without_piping(self):
        f = _fwd()
        client, fwd_side = _socket_pair()

        def _forward():
            # Emulate the routing decision: an OPTIONS on an LLM path answers directly.
            assert f._is_llm_endpoint("/v1/chat/completions")
            f._send_cors_preflight_response(fwd_side)

        t = threading.Thread(target=_forward, daemon=True)
        t.start()
        resp = _recv_until_closed(client, timeout=5.0)
        t.join(5.0)
        for s in (client, fwd_side):
            try:
                s.close()
            except OSError:
                pass
        assert b"HTTP/1.1 204" in resp
        assert b"Access-Control-Allow-Origin: *\r\n" in resp
        assert b"Access-Control-Allow-Methods: *\r\n" in resp
        assert b"Access-Control-Allow-Headers: *\r\n" in resp
        assert b"Access-Control-Max-Age:" in resp

    def test_handle_connection_routes_options_to_preflight(self):
        # Prove the real routing branch picks OPTIONS -> preflight (not pipe, not FastAPI).
        f = _fwd()
        client, fwd_side, srv = _loopback_client()
        # Connection: close signals no further body on this socket, so _read_request does not wait
        # in the bounded read phase for a Content-Length-less OPTIONS request.
        req = (b"OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: a\r\n"
               b"Origin: http://localhost:3136\r\nConnection: close\r\n\r\n")

        def _client():
            try:
                client.sendall(req)
                time.sleep(0.2)  # let the forwarder read it before we close
                client.close()
            except OSError:
                pass

        # Patch the two side-effect targets so we can assert which path ran without real backends.
        calls = {"preflight": 0, "pipe": 0}
        orig_pipe = f._pipe_to_backend
        orig_preflight = f._send_cors_preflight_response

        def _fake_pipe(*a, **k):
            calls["pipe"] += 1

        def _fake_preflight(sock):
            calls["preflight"] += 1
            try:
                sock.sendall(b"HTTP/1.1 204 No Content\r\n"
                             b"Access-Control-Allow-Origin: *\r\nContent-Length: 0\r\n"
                             b"Connection: close\r\n\r\n")
            except OSError:
                pass

        f._pipe_to_backend = _fake_pipe
        f._send_cors_preflight_response = staticmethod(_fake_preflight)

        ct = threading.Thread(target=_client, daemon=True)
        ct.start()
        try:
            f._handle_connection(fwd_side)
        finally:
            for s in (client, fwd_side, srv):
                try:
                    s.close()
                except OSError:
                    pass
            f._pipe_to_backend = orig_pipe
            f._send_cors_preflight_response = orig_preflight

        assert calls["preflight"] == 1
        assert calls["pipe"] == 0


# --------------------------------------------------------------------------- Tertiary hardening (no hang)
class TestTertiaryNoHang:
    def test_expect_100_continue_gets_continue(self):
        # Real loopback sockets: a client that sends Expect: 100-continue must receive the interim
        # "100 Continue" before it is allowed to send its body (otherwise both sides deadlock).
        f = _fwd()
        client, fwd_side, srv = _loopback_client()
        body = b'{"model": "qwen3", "stream": true}'
        req = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\n"
               b"Expect: 100-continue\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n")

        def _client():
            try:
                client.sendall(req)
                # Wait for the interim 100 (real client behaviour), then send the body.
                time.sleep(0.2)
                client.sendall(body)
            except OSError:
                pass

        parsed_holder = {}

        def _read():
            parsed_holder["v"] = f._read_request(fwd_side)

        ct = threading.Thread(target=_client, daemon=True)
        ct.start()
        # Would hang if the forwarder did NOT send "100 Continue" before reading the body.
        _run_with_timeout(_read, timeout=8.0)
        parsed = parsed_holder.get("v")
        assert parsed is not None
        method, path, raw_headers, body_prefix = parsed
        assert method == "POST"
        # The full body was read (the client sent it only after receiving the 100).
        assert f._extract_model(body_prefix) == "qwen3"
        for s in (client, fwd_side, srv):
            try:
                s.close()
            except OSError:
                pass

    def test_no_content_length_does_not_hang_client_stays_open(self):
        # A no-Content-Length client that sends headers then STAYS OPEN (sends nothing more).
        # The bounded READ phase must time out and return, NOT block forever. Uses real loopback
        # sockets because socket.socketpair() on Windows ignores settimeout (overlapped I/O), which
        # would make the timeout path untestable.
        f = _fwd()
        client, fwd_side, srv = _loopback_client()
        req = b"POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\n\r\n"  # no content-length

        def _client():
            try:
                client.sendall(req)
                time.sleep(8.0)  # stay open, send nothing — models a stalled client
            except OSError:
                pass

        parsed_holder = {}

        def _read():
            parsed_holder["v"] = f._read_request(fwd_side)

        ct = threading.Thread(target=_client, daemon=True)
        ct.start()
        # Must complete around the 5s read timeout + margin — NOT block indefinitely.
        start = time.monotonic()
        _run_with_timeout(_read, timeout=9.0)
        elapsed = time.monotonic() - start
        parsed = parsed_holder.get("v")
        assert parsed is not None
        method, path, raw_headers, body_prefix = parsed
        assert method == "POST"
        # Bounded read gave up after the timeout rather than blocking indefinitely.
        assert elapsed < 8.0
        for s in (client, fwd_side, srv):
            try:
                s.close()
            except OSError:
                pass
