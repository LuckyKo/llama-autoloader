"""Raw TCP forwarder for llama-autoloader.

Eliminates SSE streaming buffering by removing the ASGI/uvicorn/httpx relay from the
LLM data path. It listens on the main client-facing port and, for OpenAI-style LLM
requests (chat/completions, completions, embeddings) whose model is already loaded,
transparently pipes bytes between the client and the target llama-server using raw
sockets + ``select`` (in the spirit of the ``oproxy`` package). No ASGI, no httpx, no
buffering — bytes flow straight through. Piping is identical for streaming and
non-streaming; there is no stream-specific branch.

Management/status/state requests, WebSocket upgrades, and LLM requests whose model is
not yet loaded fall through to the internal FastAPI app (which runs on a separate port)
so all existing endpoints keep working unchanged.

Design notes / constraints (see the implementation plan):
  * ``resolve_request_model`` (inference routing) and ``self.loaded`` are guarded by an
    ``asyncio.Lock`` and are async-only. We therefore NEVER open a fresh event loop from
    a worker thread. Instead we schedule the resolution coroutine onto the MAIN running
    loop with ``asyncio.run_coroutine_threadsafe(...).result(timeout)``. The forwarder
    must be started only after the main loop is running (it is, from FastAPI's startup
    hook). Management endpoints use ``resolve_model_id_strict``, but inference must use
    the request resolver so an unknown model name falls back to the resident model
    rather than failing the request.
  * We forward the ORIGINAL raw header bytes verbatim (preserving casing/order and
    ``Content-Length`` / ``Transfer-Encoding: chunked`` / ``Expect: 100-continue``)
    rather than re-encoding headers from a dict.
  * Both pipe sockets are kept BLOCKING. Windows ``select`` cannot be mixed with
    ``settimeout``/non-blocking mode, so we never call ``settimeout`` on the data
    sockets (the server socket uses a timeout only for clean shutdown).
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import re
import select
import socket
import threading
from typing import Dict, Optional, Tuple

log = logging.getLogger("autoloader")

# How much of the request body we read up-front to extract `model`. AC-style chat bodies carry
# a long system prompt + history BEFORE the model field, so 4 KiB was not enough and extraction
# failed -> the request silently fell through to FastAPI (404). 16 KiB covers realistic small
# bodies; larger ones are handled by _read_more_for_model.
_BODY_PREFIX_LIMIT = 16384

# Safety ceiling for _read_more_for_model: only a guard against a pathological client that streams
# an unbounded body and never terminates. It is deliberately very high (1 GiB) because we pipe the
# full request body to llama-server anyway, so any legitimate request finishes far below this. Do NOT
# lower it toward realistic body sizes — doing so truncated chunked request bodies >256KB mid-body,
# hiding the "model" field and wrongly rejecting valid requests with 400 (see _read_more_for_model).
_MODEL_SCAN_HARD_CAP = 1 << 30

_MODEL_RE = re.compile(rb'"model"\s*:\s*"([^"]*)"')

# CORS headers restored to the piped LLM response. Before the raw-TCP passthrough commit the
# LLM path was the FastAPI/uvicorn proxy, which added exactly these three headers to every LLM
# response. llama-server emits NO CORS headers of its own, so a cross-origin browser client
# (CWrite: Vite on :3136 -> fetch() to 127.0.0.1:1234) had the response blocked by the browser
# and saw nothing. We re-add them verbatim into the first backend response (see _pipe_to_backend).
_CORS_HEADERS = (
    b"Access-Control-Allow-Origin: *\r\n"
    b"Access-Control-Allow-Methods: *\r\n"
    b"Access-Control-Allow-Headers: *\r\n"
)

# Bound for a READ-phase socket timeout used ONLY when a request has no Content-Length (the
# client may be waiting for our response before sending more body bytes, so an unbounded read
# could hang). Cleared before the pipe phase; the AC fast path always carries a Content-Length.
_READ_PHASE_TIMEOUT = 5.0

# Every connection now gets a read deadline. Previously the header loop had NO timeout and the
# body loop only timed out when Content-Length was absent, so a client that connected and stalled
# (or declared a body and never sent it) pinned a worker thread forever.
_HEADER_READ_TIMEOUT = max(0.1, float(os.environ.get("AUTOLOADER_HEADER_TIMEOUT", "30")))
_BODY_READ_TIMEOUT = max(0.1, float(os.environ.get("AUTOLOADER_BODY_TIMEOUT", "30")))

# Cap on simultaneously-handled connections. Each one is a thread; without a cap a few stalled
# clients (or a local misbehaving tool) could exhaust the process's threads/descriptors.
_MAX_CONNECTIONS = max(1, int(os.environ.get("AUTOLOADER_MAX_CONNECTIONS", "256")))

# How long an LLM request waits for its model's FIFO ticket before returning 503.
_QUEUE_WAIT_TIMEOUT = max(0.1, float(os.environ.get("AUTOLOADER_QUEUE_TIMEOUT", "300")))

# Sentinel label passed to _pipe_to_backend to mark the LLM->llama-server fast path (vs
# "fastapi" for management traffic). The CORS-injection gate keys off this value.
_LLAMA_SERVER_LABEL = "llama-server"


def _inject_cors_headers(response_head: bytes, extra_headers: bytes = b"") -> bytes:
    """Insert CORS headers (and optional ``extra_headers``) into an HTTP response head.

    The edit is header-only: the status line is left untouched, the backend's own headers keep
    their original casing/order, and nothing after the blank line (the body) is ever touched — so
    chunked / Content-Length framing is preserved. Both injections land at the same point, right
    after the status line, with CORS first and ``extra_headers`` immediately after it.

    The CORS block is skipped if the backend already sent an ``Access-Control-Allow-Origin``
    header (checked case-insensitively) — but ``extra_headers``, if any, are injected regardless,
    at the same insertion point. Returns the head unchanged when there is no end-of-headers marker
    or no status-line terminator.
    """
    if b"\r\n\r\n" not in response_head:
        return response_head
    cors_already = any(
        re.match(rb"^access-control-allow-origin\s*:", line, re.IGNORECASE)
        for line in response_head.split(b"\r\n")
    )
    status_nl = response_head.find(b"\r\n")
    if status_nl < 0:
        return response_head  # malformed head (no status line terminator); leave as-is
    insert_at = status_nl + 2  # right after the status line, before the rest of the headers
    cors = b"" if cors_already else _CORS_HEADERS  # never double-inject CORS
    return response_head[:insert_at] + cors + extra_headers + response_head[insert_at:]


# How long a worker thread blocks waiting for a JIT model load to become ready. Some models
# take minutes; override with AUTOLOADER_JIT_LOAD_TIMEOUT (seconds).
try:
    _JIT_LOAD_TIMEOUT = float(os.environ.get("AUTOLOADER_JIT_LOAD_TIMEOUT", "300"))
except ValueError:
    _JIT_LOAD_TIMEOUT = 300.0

# Extra wait granted to a dead-backend recovery beyond the model's normal load timeout.
# The recovery first has to drain slot actions and verify the old process is gone, then
# reload; without this margin a slow-but-successful recovery would be reported as a 503.
_RECOVERY_TIMEOUT_EXTRA = 60.0


class _BackendUnreachable(Exception):
    """The LLM backend port actively refused a connection.

    Raised only for the llama-server fast path, and only for refusal-class errors: the
    request has not been sent anywhere yet, so it is safe to evict/reload and retry it.
    """

    def __init__(self, target_port: int, cause: BaseException):
        super().__init__(f"backend on port {target_port} refused the connection: {cause}")
        self.target_port = target_port
        self.cause = cause


def _is_connection_refused(exc: BaseException) -> bool:
    """True for the ``WinError 10061`` / ``ECONNREFUSED`` family of connect failures."""
    if isinstance(exc, ConnectionRefusedError):
        return True
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) == 10061:
            return True
        if getattr(exc, "errno", None) == errno.ECONNREFUSED:
            return True
    return False


class RawTCPForwarder:
    """Transparent TCP forwarder with model-based routing.

    Listens on the main client-facing port. LLM requests whose model is already loaded are
    piped directly to that llama-server; everything else (management, not-yet-loaded models)
    is forwarded to the internal FastAPI port. Piping is identical for streaming and
    non-streaming — bytes flow through either way, so there is no stream-specific branch.
    """

    def __init__(self, listen_port: int, internal_api_port: int, model_manager):
        self.listen_port = listen_port
        self.internal_api_port = internal_api_port
        self.model_manager = model_manager  # ModelManager instance (port lookup)
        self._server_sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Thread-safe registry of in-flight llama-server upstream pipes, keyed by target port.
        # Used by cancel_active() to perform a soft (connection-close) interrupt without unloading.
        self._active_conns: Dict[int, set] = {}
        self._active_lock = threading.Lock()
        self._conn_sem = threading.BoundedSemaphore(_MAX_CONNECTIONS)

    def cancel_active(self, port: int) -> int:
        """Best-effort cancel of in-flight requests bound to ``port``.

        Closes the underlying upstream (and client) sockets. For a recent llama-server build this
        causes generation to stop ("srv stop: cancel task"), while leaving the resident model loaded.
        Returns the number of connections cancelled.
        """
        with self._active_lock:
            pairs = list(self._active_conns.pop(port, ()))
        for client_sock, target_sock in pairs:
            for s in (target_sock, client_sock):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass
        return len(pairs)

    def _register_active(self, port: int, client_sock: socket.socket, target_sock: socket.socket) -> None:
        with self._active_lock:
            self._active_conns.setdefault(port, set()).add((client_sock, target_sock))

    def _unregister_active(self, port: int, client_sock: socket.socket, target_sock: socket.socket) -> None:
        with self._active_lock:
            conns = self._active_conns.get(port)
            if conns:
                conns.discard((client_sock, target_sock))
                if not conns:
                    self._active_conns.pop(port, None)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the forwarder in a background thread. Must be called with the main
        event loop already running (it is, from FastAPI's startup hook)."""
        if self._running:
            return
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to the loopback interface — the proxy is local-only (matches manager.host).
        self._server_sock.bind(("127.0.0.1", self.listen_port))
        self._server_sock.listen(128)
        # Timeout only on the LISTENING socket so the accept loop can observe shutdown.
        self._server_sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        log.info(f"Raw TCP forwarder listening on 127.0.0.1:{self.listen_port} "
                 f"(internal API -> {self.internal_api_port})")

    def stop(self) -> None:
        """Stop the forwarder. In-flight handler threads are daemon and finish on close."""
        self._running = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass

    # ------------------------------------------------------------------ accept loop
    def _accept_loop(self) -> None:
        while self._running:
            try:
                client_sock, _addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed during shutdown
            if not self._conn_sem.acquire(blocking=False):
                log.warning("Forwarder connection limit reached (%d); rejecting client",
                            _MAX_CONNECTIONS)
                try:
                    self._send_error_response(client_sock, 503, "Too many concurrent connections")
                finally:
                    try:
                        client_sock.close()
                    except OSError:
                        pass
                continue
            t = threading.Thread(target=self._run_connection, args=(client_sock,), daemon=True)
            try:
                t.start()
            except Exception as e:  # noqa: BLE001
                self._conn_sem.release()
                try:
                    client_sock.close()
                except OSError:
                    pass
                log.error("Failed to start connection handler: %s", e)
                continue

    def _run_connection(self, client_sock: socket.socket) -> None:
        try:
            self._handle_connection(client_sock)
        finally:
            self._conn_sem.release()

    # ------------------------------------------------------------------ connection
    def _handle_connection(self, client_sock: socket.socket) -> None:
        try:
            parsed = self._read_request(client_sock)
            if parsed is None:
                return  # already closed inside _read_request
            method, path, raw_headers, body_prefix = parsed
            # One request per client connection. Without this, a keep-alive client's
            # SECOND request travelled through the still-open pipe to the first
            # request's backend, so a request for model B could be answered by model A.
            raw_headers = self._force_connection_close(raw_headers)

            # LLM data path: only OpenAI-style generation endpoints carry a `model` field and
            # are piped straight to llama-server. All other requests (management/status/state,
            # /ws, unknown paths) go to FastAPI.
            if self._is_llm_endpoint(path):
                # OPTIONS preflight on an LLM endpoint: answer directly with a CORS-allowing
                # response WITHOUT piping to llama-server. CWrite's current fetch is "simple" so
                # no preflight fires today, but a future Authorization/custom-header change could
                # trigger one; this keeps the loader as robust as the old uvicorn path.
                if method == "OPTIONS":
                    self._send_cors_preflight_response(client_sock)
                    return
                # Only POST (carries a `model` field in the body) is piped to llama-server. Any other
                # method on an LLM path keeps its original fallthrough-to-FastAPI behaviour (e.g. a
                # WebSocket upgrade or GET), which was never LLM generation traffic anyway.
                if method != "POST":
                    self._pipe_to_backend(client_sock, raw_headers,
                                          body_prefix, self.internal_api_port, "internal API")
                    return
                model_id = self._extract_model(body_prefix)
                # The `model` field may sit beyond the bytes we read up-front (large AC bodies, or a
                # body split across TCP segments). Read more — bounded by content-length and a sane
                # cap — until we can locate it. We must NEVER fall through an LLM POST to FastAPI:
                # the LLM proxy routes were removed there, so that path only ever yields a 404.
                if not model_id:
                    body_prefix = self._read_more_for_model(client_sock, raw_headers, body_prefix)
                    model_id = self._extract_model(body_prefix)
                    log.info(f"LLM {method} {path}: 'model' extracted after extra read ({len(body_prefix)}B)")
                if not model_id:
                    # Truly unresolvable (no content-length and body has no "model", or cap hit).
                    log.warning(f"LLM {method} {path}: no 'model' in body ({len(body_prefix)}B); "
                                f"refusing FastAPI fallthrough -> 400")
                    self._send_error_response(client_sock, 400,
                                              "Could not determine 'model' from request body for LLM endpoint")
                    return
                target_port = self._resolve_target_port(model_id)
                if target_port:
                    # Model already loaded -> pipe directly, but only after acquiring this
                    # model's FIFO ticket. The ticket is granted in strict submission order,
                    # so a state restore enqueued before this request runs first (fixes the
                    # full-reprocess race). No per-request log: this is the hot path.
                    queue = self.model_manager.queue_registry.get_or_create(target_port, model_id)
                    self._pipe_with_ticket(client_sock, raw_headers, body_prefix,
                                           target_port, model_id, queue)
                    return
                # Model not loaded -> JIT load it, wait for readiness, then pipe.
                log.info(f"Model '{model_id}' not loaded; triggering JIT load...")
                target_port = self._jit_load_and_wait(model_id)
                if target_port:
                    # Acquire the ticket only AFTER the model is loaded and ready, so the
                    # load itself is never gated by the queue.
                    queue = self.model_manager.queue_registry.get_or_create(target_port, model_id)
                    self._pipe_with_ticket(client_sock, raw_headers, body_prefix,
                                           target_port, model_id, queue)
                    return
                # JIT load failed (unknown model / timeout / spawn error).
                log.error(f"JIT load failed for '{model_id}' on {method} {path}")
                self._send_error_response(client_sock, 503, f"Failed to load model '{model_id}'")
                return
            # Non-LLM endpoint -> forward to FastAPI.
            self._pipe_to_backend(client_sock, raw_headers,
                                  body_prefix, self.internal_api_port, "internal API")
        except Exception as e:  # noqa: BLE001 — never let one connection kill the loop
            log.error(f"Forwarder error: {e}")
        finally:
            try:
                client_sock.close()
            except OSError:
                pass

    # ------------------------------------------------------------------ request parse
    def _read_request(self, sock: socket.socket) -> Optional[Tuple[str, str, bytes, bytes]]:
        """Read the HTTP request line + headers + a body prefix.

        Returns (method, path, raw_header_bytes, body_prefix_bytes) or None on failure
        (socket closed / malformed). ``raw_header_bytes`` includes the terminating
        ``\\r\\n\\r\\n`` and is forwarded verbatim to preserve original header casing/order
        and framing headers.
        """
        try:
            sock.settimeout(_HEADER_READ_TIMEOUT)
        except (OSError, AttributeError):
            pass
        header_data = b""
        while b"\r\n\r\n" not in header_data:
            try:
                chunk = sock.recv(4096)
            except (socket.timeout, TimeoutError):
                log.warning("Timed out reading request headers (%.0fs)", _HEADER_READ_TIMEOUT)
                return None
            except OSError:
                return None
            if not chunk:
                return None
            header_data += chunk
            # Guard against a pathological/absent end-of-headers.
            if len(header_data) > 1_048_576:
                return None

        header_end = header_data.index(b"\r\n\r\n") + 4
        raw_headers = header_data[:header_end]          # request line + headers + \r\n\r\n
        body_prefix = header_data[header_end:]           # any body bytes already received

        lines = raw_headers.decode("utf-8", errors="replace").split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) < 3:
            return None
        method, path = parts[0], parts[1]

        # Compatibility alias: clients whose base URL ends in /api request
        # /api/v1/models, /api/v1/chat/completions, ... . Our FastAPI app and
        # llama-server only serve /v1/..., so rewrite the request target BEFORE
        # routing. This also makes inference requests pipe to the right backend:
        # without it, /api/v1/chat/completions would fall through to FastAPI and 404.
        normalized_path = self._normalize_api_prefix(path)
        if normalized_path != path:
            path = normalized_path
            first_line = f"{method} {path} {parts[2]}".encode("utf-8")
            # Preserve every header byte after the request line.
            raw_headers = first_line + b"\r\n" + raw_headers.split(b"\r\n", 1)[1]

        # Read enough of the body to be confident about the model field: up to
        # min(content_length, _BODY_PREFIX_LIMIT) bytes. _BODY_PREFIX_LIMIT is raised from 4 KiB
        # because AC-style chat bodies carry a long system prompt + history BEFORE the model field;
        # if it still isn't in this prefix, LLM endpoints get one more bounded read (see
        # _read_more_for_model). The remainder of the body always flows through the pipe later.
        raw_cl = self._header_value(raw_headers, "content-length")
        has_content_length = raw_cl is not None
        try:
            content_length = int(raw_cl) if has_content_length else 0
        except (ValueError, TypeError):
            content_length = 0
            has_content_length = False

        # Expect: 100-continue — tell the client to go ahead BEFORE reading its body, or it will
        # wait for our interim response and we would deadlock. Only sent when actually requested.
        expect = self._header_value(raw_headers, "expect")
        if expect is not None and "100-continue" in expect.lower():
            try:
                sock.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
            except OSError:
                return None

        # Bound the READ phase for EVERY request, not just chunked/no-length ones.
        # A client that declares Content-Length and then stalls used to pin this
        # worker thread forever. The timeout is cleared before the pipe phase
        # (Windows select cannot mix with settimeout). No-length/chunked clients
        # keep the shorter historical bound because they may wait for our response.
        read_timeout = _BODY_READ_TIMEOUT if content_length else _READ_PHASE_TIMEOUT
        try:
            sock.settimeout(read_timeout)
        except (OSError, AttributeError):
            pass

        if self._is_llm_endpoint(path) and not has_content_length:
            # Chunked / no Content-Length: scan up to the prefix cap for "model".
            target = _BODY_PREFIX_LIMIT
        else:
            # An explicit Content-Length (including 0) bounds the body exactly.
            target = min(content_length, _BODY_PREFIX_LIMIT)
        timed_out = False
        while len(body_prefix) < target:
            try:
                chunk = sock.recv(4096)
            except (socket.timeout, TimeoutError):
                timed_out = True
                break
            except OSError:
                break
            if not chunk:
                break
            body_prefix += chunk

        # A declared body that did not fully arrive is a broken request: do not
        # forward a truncated body and then wait forever in the pipe.
        if timed_out and content_length and len(body_prefix) < target:
            log.warning("Timed out reading request body (declared %d bytes, got %d)",
                        content_length, len(body_prefix))
            try:
                sock.settimeout(None)
            except (OSError, AttributeError):
                pass
            return None

        # Restore blocking mode before the pipe phase (Windows select can't mix with settimeout).
        try:
            sock.settimeout(None)
        except (OSError, AttributeError):
            pass

        # If the first recv already swallowed a pipelined second request, discard
        # it. Connection: close forces the client to retry it on a new connection;
        # forwarding it here would send it to THIS request's model.
        if has_content_length and len(body_prefix) > content_length:
            log.warning("Discarding %d pipelined byte(s) after request body",
                        len(body_prefix) - content_length)
            body_prefix = body_prefix[:content_length]

        return method, path, raw_headers, body_prefix

    @staticmethod
    def _force_connection_close(raw_headers: bytes) -> bytes:
        """Return request headers with ``Connection: close`` and no keep-alive tokens.

        The forwarder's pipe is per-connection, not per-request. Advertising close
        makes the backend (and FastAPI) end the connection after the response, so a
        second request from a keep-alive client can never travel to the first
        request's model.
        """
        head, sep, rest = raw_headers.partition(b"\r\n\r\n")
        if not sep:
            return raw_headers
        # Never rewrite a protocol upgrade (WebSocket): it needs the original
        # Connection/Upgrade headers, and it is a single long-lived stream rather
        # than a keep-alive HTTP connection.
        if re.search(rb"(?im)^upgrade\s*:", head):
            return raw_headers
        lines = head.split(b"\r\n")
        kept = [lines[0]]
        for line in lines[1:]:
            if re.match(rb"^(connection|keep-alive|proxy-connection)\s*:", line, re.IGNORECASE):
                continue
            kept.append(line)
        kept.append(b"Connection: close")
        return b"\r\n".join(kept) + b"\r\n\r\n" + rest

    @staticmethod
    def _header_value(raw_headers: bytes, name: str) -> Optional[str]:
        """Return the value of a header from raw header bytes (case-insensitive)."""
        needle = f"{name.lower()}:".encode("ascii")
        for line in raw_headers.split(b"\r\n"):
            if line.lower().startswith(needle):
                return line[len(needle):].strip().decode("utf-8", errors="replace")
        return None

    # ------------------------------------------------------------------ routing helpers
    def _extract_model(self, body_prefix: bytes) -> Optional[str]:
        """Extract the 'model' field from the JSON request body prefix."""
        if not body_prefix:
            return None
        try:
            data = json.loads(body_prefix.decode("utf-8", errors="replace"))
            if isinstance(data, dict):
                model = data.get("model")
                if isinstance(model, str) and model:
                    return model
        except Exception:  # noqa: BLE001 — incomplete JSON; fall through to regex
            pass
        m = _MODEL_RE.search(body_prefix)
        if m:
            return m.group(1).decode("utf-8", errors="replace")
        return None

    def _read_more_for_model(self, sock: socket.socket, raw_headers: bytes,
                              body_prefix: bytes) -> bytes:
        """Read additional request-body bytes until the `model` field is locatable or EOF.

        Used only for LLM endpoints when `_extract_model` failed on the initial prefix (the model
        field sits beyond what we read up-front — common with large AC bodies, and ALWAYS the case
        for chunked-transfer requests whose body exceeds any fixed cap).

        IMPORTANT: there is deliberately NO realistic byte cap here. We pipe this exact request's full
        body to llama-server anyway, so the body is bounded by either its declared Content-Length or
        (for chunked transfer) the client sending the complete body before waiting for our response.
        On a keepalive connection the client will NOT send further *body* bytes on this socket until we
        respond, so reading to EOF of THIS request's body is well-defined and cannot hang. Capping at a
        fixed size (the old 256 KB _MODEL_SCAN_MAX_BYTES) was a bug: it stopped the scan mid-body for
        chunked requests >256 KB, so a model field past that point was never seen and the request was
        wrongly rejected with 400. The remainder of the body always flows through the pipe later, so
        reading more here never loses data — it only moves bytes from "piped" to "forwarded verbatim".

        A very high safety ceiling (_MODEL_SCAN_HARD_CAP) still guards against a pathological client
        that streams an unbounded body and never terminates; if we hit it without finding the model we
        return what we have (caller will 400), but normal requests finish far below it.
        """
        prefix = bytearray(body_prefix)
        try:
            content_length = int(self._header_value(raw_headers, "content-length"))
        except (ValueError, TypeError):
            content_length = 0
        try:
            sock.settimeout(_BODY_READ_TIMEOUT if content_length else _READ_PHASE_TIMEOUT)
        except (OSError, AttributeError):
            pass
        try:
            while True:
                model_id = self._extract_model(bytes(prefix))
                if model_id:
                    break
                if len(prefix) >= _MODEL_SCAN_HARD_CAP:
                    log.warning(f"_read_more_for_model hit hard cap ({len(prefix)}B) without finding "
                                f"'model'; returning partial body for caller to reject")
                    break
                if content_length and len(prefix) >= content_length:
                    break  # complete declared body; no model field anywhere
                try:
                    want = 65536
                    if content_length:
                        want = min(want, content_length - len(prefix))
                    chunk = sock.recv(want)
                except (socket.timeout, TimeoutError):
                    log.warning("Timed out reading request body while locating 'model'")
                    break
                except OSError:
                    break
                if not chunk:
                    break  # EOF / peer closed (full body received or client aborted)
                prefix.extend(chunk)
                if content_length:
                    # Never swallow bytes belonging to a pipelined second request.
                    del prefix[content_length:]
        finally:
            try:
                sock.settimeout(None)
            except (OSError, AttributeError):
                pass
        return bytes(prefix)

    @staticmethod
    def _normalize_api_prefix(path: str) -> str:
        """Map the ``/api/v1`` compatibility prefix onto the canonical ``/v1``.

        Some OpenAI clients are configured with a base URL ending in ``/api`` and
        therefore request ``/api/v1/models``, ``/api/v1/chat/completions``, etc.
        Neither FastAPI nor llama-server exposes that extra segment, so without
        this alias every such request 404s. Only ``/api/v1`` is rewritten; other
        ``/api/...`` paths (and ``/api/v2/...``) are left untouched.
        """
        if path == "/api/v1" or path.startswith("/api/v1/") or path.startswith("/api/v1?"):
            return path[len("/api"):]
        return path

    @staticmethod
    def _is_llm_endpoint(path: str) -> bool:
        """True for OpenAI-style generation endpoints that carry a `model` field.

        These are the requests we pipe straight to llama-server when the model is loaded.
        Accepts both bare (/chat/completions) and /v1-prefixed forms, plus a trailing path.
        """
        p = RawTCPForwarder._normalize_api_prefix(path.split("?", 1)[0].rstrip("/"))
        for prefix in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings",
                       "/chat/completions", "/completions", "/embeddings"):
            if p == prefix or p.startswith(prefix + "/"):
                return True
        return False

    def _resolve_request_model(self, model_id: str) -> Optional[str]:
        """Resolve a request's model on the MAIN event loop (async-only resolver).

        Uses the manager's ``resolve_request_model`` so an unrecognised inference
        ``model`` value falls back to the resident model. Returns None when there is
        no running loop, the resolver fails, or no model can be resolved.
        """
        try:
            main_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in this thread; fall back to the manager's loop if it has one.
            main_loop = getattr(self.model_manager, "_main_loop", None)
        if main_loop is None or not main_loop.is_running():
            return None

        # Schedule the async-only resolver onto the MAIN loop and block this worker thread
        # until it completes (or times out). This is the safe cross-thread pattern — never
        # a fresh event loop.
        resolver = getattr(self.model_manager, "resolve_request_model", None)
        if resolver is None:
            # Backwards-compatible with lightweight fakes and older managers: the
            # non-strict resolver is the one that keeps the loaded-model fallback.
            resolver = (getattr(self.model_manager, "resolve_model_id", None)
                        or self.model_manager.resolve_model_id_strict)
        coro = resolver(model_id)
        try:
            return asyncio.run_coroutine_threadsafe(coro, main_loop).result(timeout=5.0)
        except Exception as e:  # noqa: BLE001
            log.error(f"Model resolution failed for {model_id!r}: {e}")
            return None

    def _resolve_target_port(self, model_id: str) -> Optional[int]:
        """Look up the llama-server port for a loaded model.

        Resolves via :meth:`_resolve_request_model` on the MAIN event loop (never a fresh
        one) and returns the port only if that model is already loaded AND ready. Returns
        None otherwise so the caller can JIT-load the requested model.

        ``resolve_request_model`` (not the strict variant) is deliberate: an inference
        request naming a model we do not recognise must be served by the already-loaded
        model instead of failing. Management endpoints stay strict because redirecting
        them could act on an unrelated resident model.
        """
        resolved_mid = self._resolve_request_model(model_id)
        if not resolved_mid:
            return None

        # Snapshot the loaded model. NOTE: manager._lock is an asyncio.Lock bound to the
        # main loop — it CANNOT be acquired from this worker thread. Dict reads are atomic
        # under the GIL, so a plain read is safe here (matches how list_models snapshots).
        lm = self.model_manager.loaded.get(resolved_mid)
        if not (lm and lm.ready):
            return None

        # A piped request IS active usage of this model. Update last_used so the idle
        # reaper does not unload a model that is still being served through the fast
        # raw-TCP path. touch() only writes a float (atomic under the GIL), so it is
        # safe to call from this worker thread without the manager's asyncio lock.
        lm.touch()
        return lm.port

    def _recover_dead_backend(self, model_id: str, dead_port: int) -> Optional[int]:
        """Ask the manager to evict/reload the engine on ``dead_port``; return its new port.

        Runs the async recovery on the MAIN event loop (never a fresh loop) and blocks
        this handler thread until it finishes. Concurrent requests for the same dead
        port share one recovery task inside the manager, so this is safe under a
        thundering herd. Returns None when no running loop/recovery is available or the
        replacement did not become ready.
        """
        manager = self.model_manager
        if manager is None:
            log.error(f"Cannot recover dead backend on port {dead_port}: no model manager")
            return None
        recover = getattr(manager, "recover_dead_model", None)
        if recover is None:
            log.error(f"Cannot recover dead backend on port {dead_port}: "
                      f"manager has no recover_dead_model()")
            return None

        try:
            main_loop = asyncio.get_running_loop()
        except RuntimeError:
            main_loop = getattr(manager, "_main_loop", None)
        if main_loop is None or not main_loop.is_running():
            log.error(f"Cannot recover dead backend on port {dead_port}: "
                      f"no running main event loop")
            return None

        # Bridge cards legitimately need minutes to load; use the manager's per-engine
        # timeout (same source as the JIT path) plus time for drain/kill/teardown.
        timeout = _JIT_LOAD_TIMEOUT
        resolved = self._resolve_request_model(model_id)
        if resolved:
            try:
                mc = manager.models.get(resolved)
                if mc and getattr(mc, "is_bridge", False):
                    timeout = manager.ready_timeout_for(resolved)
            except Exception:  # noqa: BLE001 - timeout refinement is best-effort
                pass
        timeout = max(1.0, float(timeout)) + _RECOVERY_TIMEOUT_EXTRA

        try:
            future = asyncio.run_coroutine_threadsafe(
                recover(model_id, dead_port), main_loop)
        except Exception as e:  # noqa: BLE001 - loop closed between check and submit
            log.error(f"Could not submit dead-backend recovery for '{model_id}' "
                      f"on port {dead_port}: {e}")
            return None
        try:
            lm = future.result(timeout=timeout)
        except Exception as e:  # noqa: BLE001 — includes TimeoutError / spawn errors
            log.error(f"Dead-backend recovery for '{model_id}' on port {dead_port} "
                      f"failed: {e}")
            return None
        if lm is not None and getattr(lm, "ready", False):
            log.info(f"Recovered model '{getattr(lm, 'model_id', model_id)}' on "
                     f"port {lm.port} after backend death on port {dead_port}")
            return lm.port
        log.error(f"Dead-backend recovery for '{model_id}' on port {dead_port} "
                  f"returned no ready model")
        return None

    def _jit_load_and_wait(self, model_id: str) -> Optional[int]:
        """Trigger a JIT load of ``model_id`` and block until it is ready. Returns the
        llama-server port on success, or None on failure (unknown model / timeout / error).

        The request's model is resolved FIRST, so an unknown name that already has a
        resident fallback joins that model's load instead of trying to load a model
        that does not exist. Runs the async ``load_model`` on the MAIN event loop
        (never a fresh one) and blocks this worker thread for up to _JIT_LOAD_TIMEOUT
        seconds. load_model() is idempotent and concurrency-safe: if another request
        is already loading the same model it joins that in-flight task rather than
        double-loading.
        """
        resolved_mid = self._resolve_request_model(model_id)
        if not resolved_mid:
            log.error(f"JIT load for {model_id!r} skipped: no model resolved "
                      f"(unknown name and no loaded model to fall back to)")
            return None
        model_id = resolved_mid

        try:
            main_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in this thread; use the manager's captured main loop.
            main_loop = getattr(self.model_manager, "_main_loop", None)
        if main_loop is None or not main_loop.is_running():
            log.error(f"JIT load for {model_id!r} skipped: no running main event loop")
            return None

        # Bridge cards can take minutes to load; ready_timeout_for() returns the engine's
        # value for those and the llama default otherwise (single source of truth with
        # the load path in server.py, so the two cannot drift apart).
        mc = self.model_manager.models.get(model_id)
        timeout = (self.model_manager.ready_timeout_for(model_id)
                   if (mc and mc.is_bridge) else _JIT_LOAD_TIMEOUT)
        future = asyncio.run_coroutine_threadsafe(
            self.model_manager.load_model(model_id), main_loop
        )
        try:
            lm = future.result(timeout=timeout)
        except Exception as e:  # noqa: BLE001 — includes TimeoutError, KeyError, spawn errors
            log.error(f"JIT load failed for {model_id!r}: {e}")
            return None
        if lm and getattr(lm, "ready", False):
            log.info(f"Model '{model_id}' loaded on port {lm.port}")
            return lm.port
        log.error(f"Model '{model_id}' load completed but not ready")
        return None

    # ------------------------------------------------------------------ piping
    def _connect_backend(self, target_port: int) -> socket.socket:
        """Open the connection to a backend on 127.0.0.1.

        Factored out of _pipe_to_backend so tests can substitute a real connected socket pair
        (a fake llama-server / FastAPI peer) without needing a live port.
        """
        return socket.create_connection(("127.0.0.1", target_port), timeout=10)

    def _pipe_with_ticket(self, client_sock: socket.socket, raw_headers: bytes,
                          body_prefix: bytes, target_port: int, model_id: str,
                          queue, _recovery_attempts: int = 0) -> None:
        """Acquire the model's FIFO ticket, then pipe; surface busy timeouts as 503.

        A connection-refused failure on the llama-server fast path means the engine died
        silently between the registry check and the connect. Nothing was sent, so the
        ticket is released (by the context manager), the manager evicts/reloads the model,
        and the SAME request is retried once on the recovered port. This is what turns a
        permanent 502 loop into a one-request recovery.
        """
        try:
            with queue.ticket(timeout=_QUEUE_WAIT_TIMEOUT):
                self._pipe_to_backend(client_sock, raw_headers,
                                      body_prefix, target_port, _LLAMA_SERVER_LABEL)
        except _BackendUnreachable:
            if _recovery_attempts >= 1:
                log.error(f"Model '{model_id}' engine on port {target_port} still "
                          f"refused after one recovery attempt")
                self._send_error_response(
                    client_sock, 503,
                    f"Model '{model_id}' engine is unavailable after a reload attempt")
                return
            log.warning(f"Model '{model_id}' engine on port {target_port} refused the "
                        f"connection; evicting and reloading")
            new_port = self._recover_dead_backend(model_id, target_port)
            if not new_port:
                self._send_error_response(
                    client_sock, 503,
                    f"Model '{model_id}' engine died and could not be restarted")
                return
            log.info(f"Retrying request for model '{model_id}' on recovered port {new_port}")
            retry_queue = self.model_manager.queue_registry.get_or_create(new_port, model_id)
            self._pipe_with_ticket(client_sock, raw_headers, body_prefix,
                                   new_port, model_id, retry_queue,
                                   _recovery_attempts=_recovery_attempts + 1)
        except TimeoutError:
            log.warning(f"Queue wait timed out for model '{model_id}' on port {target_port}")
            self._send_error_response(
                client_sock, 503,
                f"Model '{model_id}' is busy (queue wait exceeded {_QUEUE_WAIT_TIMEOUT:.0f}s)")
        except RuntimeError as e:
            # The recovery path drops the dead port's FIFO queue, which wakes any
            # requests still waiting on it. They are not doomed: join the recovery and
            # retry once on the replacement port instead of surfacing a spurious 503.
            active = getattr(self.model_manager, "recovery_in_progress", None)
            if (_recovery_attempts < 1 and active is not None
                    and active(target_port)):
                new_port = self._recover_dead_backend(model_id, target_port)
                if new_port:
                    retry_queue = self.model_manager.queue_registry.get_or_create(
                        new_port, model_id)
                    self._pipe_with_ticket(client_sock, raw_headers, body_prefix,
                                           new_port, model_id, retry_queue,
                                           _recovery_attempts=_recovery_attempts + 1)
                    return
            log.error(f"Queue unavailable for model '{model_id}': {e}")
            self._send_error_response(
                client_sock, 503, f"Model '{model_id}' queue is unavailable")

    def _pipe_to_backend(self, client_sock: socket.socket, raw_headers: bytes,
                          body_prefix: bytes, target_port: int, label: str) -> None:
        """Track the request as active, then run the raw pipe.

        The active counter is what stops the idle reaper / LRU eviction from
        unloading a model in the middle of a long generation.
        """
        tracking = label == _LLAMA_SERVER_LABEL
        if tracking and self.model_manager is not None:
            self.model_manager.note_request_start(target_port)
        try:
            self._pipe_to_backend_inner(
                client_sock, raw_headers, body_prefix, target_port, label)
        finally:
            if tracking and self.model_manager is not None:
                self.model_manager.note_request_end(target_port)

    def _pipe_to_backend_inner(self, client_sock: socket.socket, raw_headers: bytes,
                          body_prefix: bytes, target_port: int, label: str) -> None:
        """Forward the request (reconstructed verbatim) to a backend and pipe bidirectionally."""
        try:
            target_sock = self._connect_backend(target_port)
        except Exception as e:  # noqa: BLE001
            log.error(f"Failed to connect to {label} on port {target_port}: {e}")
            if label == _LLAMA_SERVER_LABEL and _is_connection_refused(e):
                # Nothing has been sent to the backend yet, so the request is safe to
                # retry after a reload. The caller owns sending the error response.
                raise _BackendUnreachable(target_port, e) from e
            self._send_error_response(client_sock, 502, f"{label} unreachable")
            return

        # Forward the request VERBATIM: ``raw_headers`` already contains the original
        # request line (method path HTTP/1.1\r\n) plus all headers and the terminating
        # \r\n\r\n, so we must NOT prepend another request line — doing so produced a
        # duplicate first line that parsers reject as an invalid header ("Invalid HTTP
        # request received"). The remainder of the body (beyond what we read up-front)
        # flows through the bidirectional pipe below.
        try:
            target_sock.sendall(raw_headers + body_prefix)
        except OSError as e:
            log.error(f"Failed to send request to {label}: {e}")
            target_sock.close()
            self._send_error_response(client_sock, 502, f"{label} unreachable")
            return

        # Disable Nagle on both data sockets so SSE frames are not coalesced into delayed-ACK
        # batches (the exact class of buffering this forwarder exists to avoid). Best-effort.
        for s in (client_sock, target_sock):
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        # CORS injection (LLM->llama-server fast path ONLY). llama-server sends no CORS headers,
        # so a cross-origin browser client (CWrite) is blocked. We inject the three headers into
        # the FIRST response head using a non-blocking peek that never delays the first byte:
        # if the head isn't fully in the kernel buffer yet, we skip injection entirely and let
        # the pipe deliver everything as-is. The trade-off (CORS occasionally missing on very
        # slow first responses) is acceptable because llama-server's response head arrives in a
        # single segment under normal conditions. Management traffic to FastAPI is left
        # untouched (label != _LLAMA_SERVER_LABEL).
        if label == _LLAMA_SERVER_LABEL:
            self._inject_cors_nonblocking(client_sock, target_sock)

        # Both sockets stay blocking (Windows select can't mix with settimeout/non-blocking).
        tracking = label == _LLAMA_SERVER_LABEL
        if tracking:
            self._register_active(target_port, client_sock, target_sock)
        try:
            self._bidirectional_pipe(client_sock, target_sock)
        finally:
            if tracking:
                self._unregister_active(target_port, client_sock, target_sock)

    def _inject_cors_nonblocking(self, client_sock: socket.socket,
                                 target_sock: socket.socket) -> None:
        """Inject CORS headers into the first response head using a zero-wait peek.

        Strategy: ZERO-WAIT non-blocking peek. ``select`` is called with a timeout of 0.0, i.e. a
        pure poll that returns IMMEDIATELY with whatever the backend has already buffered — it never
        waits for bytes that are not there yet. If the complete response head is already in the
        kernel buffer (a warm-cache hit), we recv it once, inject CORS, and forward; the added
        latency is at most a single recv/sendall of the head. If nothing is buffered yet (a cold
        prompt miss — full reprocess taking seconds to minutes), select returns empty and we skip
        injection entirely: ``_bidirectional_pipe`` then delivers the first byte the instant it
        arrives from the backend.

        The socket stays in BLOCKING mode throughout (no setblocking toggle). A blocking recv after
        select reports ready does not block — it returns what is already buffered — so we never need
        to drop into non-blocking mode. This also keeps the pipe's Windows select working (it cannot
        be mixed with settimeout/non-blocking sockets; see _pipe_to_backend).

        No TTFB measurement and no Cache-Status header are emitted here: prompt-cache hit/miss is
        now classified authoritatively by AC from llama.cpp's ``cached_tokens`` in the usage object.
        """
        # Pure poll: timeout 0.0 means "return immediately" — no waiting for the backend's first
        # bytes. Warm-cache hits have their head already in the kernel buffer and are caught here;
        # cold misses return empty instantly and skip injection, so the data path is never delayed.
        try:
            ready, _, _ = select.select([target_sock], [], [], 0.0)
        except (OSError, ValueError):
            return  # socket error or already closed; pipe loop will handle it

        if not ready:
            # No data buffered yet — cold prompt eval in progress (full reprocess of a large
            # context). Skip injection entirely; the pipe delivers the first byte the instant
            # it arrives. Zero latency added.
            return

        try:
            chunk = target_sock.recv(65536)
        except OSError:
            return  # socket error; the pipe loop will handle it

        if not chunk:
            return  # backend closed before sending anything; pipe handles EOF

        # We have at least some bytes. Check if the head is complete in THIS chunk.
        if b"\r\n\r\n" in chunk:
            head, _, rest = chunk.partition(b"\r\n\r\n")
            payload = _inject_cors_headers(head + b"\r\n\r\n") + rest
        else:
            # Partial head (rare — would mean the status line arrived split across segments).
            # Forward verbatim without injection; the pipe will deliver the remaining bytes.
            # CORS is lost for this response, but zero latency is added.
            payload = chunk

        try:
            client_sock.sendall(payload)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass  # client gone; pipe loop will notice and shut down cleanly

    def _bidirectional_pipe(self, sock_a: socket.socket, sock_b: socket.socket) -> None:
        """Pipe bytes between two blocking sockets until both directions are closed."""
        open_socks = [sock_a, sock_b]
        try:
            while self._running and open_socks:
                try:
                    # Poll every 5s (not 60s) so a shutdown request is noticed promptly on an
                    # idle connection; 5s still far exceeds typical HTTP keepalive idle gaps.
                    readable, _writable, exceptional = select.select(open_socks, [], open_socks, 5)
                except OSError:
                    break
                if exceptional:
                    break
                for sock in readable:
                    other = sock_b if sock is sock_a else sock_a
                    try:
                        data = sock.recv(65536)
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        # Peer reset / socket error: half-close our write side so the peer
                        # gets an orderly EOF instead of a hard RST. finally closes both.
                        try:
                            other.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        return
                    if not data:
                        # This side closed -> half-close the other direction.
                        try:
                            other.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        open_socks.remove(sock)
                        # Close the EOF side explicitly. Leaving it in no list meant
                        # the finally block never closed it; CPython's refcounting
                        # usually did, but that is not a contract.
                        try:
                            sock.close()
                        except OSError:
                            pass
                    else:
                        try:
                            other.sendall(data)
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            try:
                                other.shutdown(socket.SHUT_WR)
                            except OSError:
                                pass
                            return
        finally:
            for sock in open_socks:
                try:
                    sock.close()
                except OSError:
                    pass

    @staticmethod
    def _send_cors_preflight_response(sock: socket.socket) -> None:
        """Answer an OPTIONS preflight on an LLM endpoint with a CORS-allowing 204.

        Sent directly (never piped to llama-server). Mirrors the old uvicorn path so a browser can
        complete a preflight if CWrite ever adds an Authorization/custom header. Connection: close
        since we do not keep this socket in the pipe loop.
        """
        response = (
            b"HTTP/1.1 204 No Content\r\n"
            + _CORS_HEADERS
            + b"Access-Control-Max-Age: 86400\r\n"
            + b"Content-Length: 0\r\n"
            + b"Connection: close\r\n"
            + b"\r\n"
        )
        try:
            sock.sendall(response)
        except OSError:
            pass

    @staticmethod
    def _send_error_response(sock: socket.socket, status_code: int, message: str) -> None:
        """Send a minimal HTTP error response and let the caller close the socket."""
        status_text = {400: "Bad Request", 404: "Not Found", 502: "Bad Gateway",
                       503: "Service Unavailable"}.get(status_code, "Error")
        body = json.dumps({"error": message}).encode("utf-8")
        response = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("latin-1") + body
        try:
            sock.sendall(response)
        except OSError:
            pass
