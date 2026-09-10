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
  * ``resolve_model_id`` and ``self.loaded`` are guarded by an ``asyncio.Lock`` and
    are async-only. We therefore NEVER open a fresh event loop from a worker thread.
    Instead we schedule the resolution coroutine onto the MAIN running loop with
    ``asyncio.run_coroutine_threadsafe(...).result(timeout)``. The forwarder must be
    started only after the main loop is running (it is, from FastAPI's startup hook).
  * We forward the ORIGINAL raw header bytes verbatim (preserving casing/order and
    ``Content-Length`` / ``Transfer-Encoding: chunked`` / ``Expect: 100-continue``)
    rather than re-encoding headers from a dict.
  * Both pipe sockets are kept BLOCKING. Windows ``select`` cannot be mixed with
    ``settimeout``/non-blocking mode, so we never call ``settimeout`` on the data
    sockets (the server socket uses a timeout only for clean shutdown).
"""

from __future__ import annotations

import asyncio
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

# How long a worker thread blocks waiting for a JIT model load to become ready. Some models
# take minutes; override with AUTOLOADER_JIT_LOAD_TIMEOUT (seconds).
try:
    _JIT_LOAD_TIMEOUT = float(os.environ.get("AUTOLOADER_JIT_LOAD_TIMEOUT", "300"))
except ValueError:
    _JIT_LOAD_TIMEOUT = 300.0


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
            t = threading.Thread(target=self._handle_connection, args=(client_sock,), daemon=True)
            t.start()

    # ------------------------------------------------------------------ connection
    def _handle_connection(self, client_sock: socket.socket) -> None:
        try:
            parsed = self._read_request(client_sock)
            if parsed is None:
                return  # already closed inside _read_request
            method, path, raw_headers, body_prefix = parsed

            # LLM data path: only OpenAI-style generation endpoints carry a `model` field and
            # are piped straight to llama-server. All other requests (management/status/state,
            # /ws, unknown paths) go to FastAPI.
            if self._is_llm_endpoint(path):
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
                    # Model already loaded -> pipe directly. (No per-request log: this is the hot path.)
                    self._pipe_to_backend(client_sock, raw_headers,
                                          body_prefix, target_port, "llama-server")
                    return
                # Model not loaded -> JIT load it, wait for readiness, then pipe.
                log.info(f"Model '{model_id}' not loaded; triggering JIT load...")
                target_port = self._jit_load_and_wait(model_id)
                if target_port:
                    self._pipe_to_backend(client_sock, raw_headers,
                                          body_prefix, target_port, "llama-server")
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
        header_data = b""
        while b"\r\n\r\n" not in header_data:
            try:
                chunk = sock.recv(4096)
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

        # Read enough of the body to be confident about the model field: up to
        # min(content_length, _BODY_PREFIX_LIMIT) bytes. _BODY_PREFIX_LIMIT is raised from 4 KiB
        # because AC-style chat bodies carry a long system prompt + history BEFORE the model field;
        # if it still isn't in this prefix, LLM endpoints get one more bounded read (see
        # _read_more_for_model). The remainder of the body always flows through the pipe later.
        try:
            content_length = int(self._header_value(raw_headers, "content-length"))
        except (ValueError, TypeError):
            content_length = 0
        if self._is_llm_endpoint(path):
            target = min(content_length, _BODY_PREFIX_LIMIT) if content_length else _BODY_PREFIX_LIMIT
        else:
            target = min(content_length, _BODY_PREFIX_LIMIT)
        while len(body_prefix) < target:
            try:
                chunk = sock.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            body_prefix += chunk

        return method, path, raw_headers, body_prefix

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
        while True:
            model_id = self._extract_model(bytes(prefix))
            if model_id:
                break
            if len(prefix) >= _MODEL_SCAN_HARD_CAP:
                log.warning(f"_read_more_for_model hit hard cap ({len(prefix)}B) without finding "
                            f"'model'; returning partial body for caller to reject")
                break
            try:
                chunk = sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break  # EOF / peer closed (full body received or client aborted)
            prefix.extend(chunk)
        return bytes(prefix)

    @staticmethod
    def _is_llm_endpoint(path: str) -> bool:
        """True for OpenAI-style generation endpoints that carry a `model` field.

        These are the requests we pipe straight to llama-server when the model is loaded.
        Accepts both bare (/chat/completions) and /v1-prefixed forms, plus a trailing path.
        """
        p = path.split("?", 1)[0].rstrip("/")
        for prefix in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings",
                       "/chat/completions", "/completions", "/embeddings"):
            if p == prefix or p.startswith(prefix + "/"):
                return True
        return False

    def _resolve_target_port(self, model_id: str) -> Optional[int]:
        """Look up the llama-server port for a loaded model.

        Runs the async ``resolve_model_id`` on the MAIN event loop (never a fresh one)
        and returns the port only if the model is already loaded AND ready. Returns None
        otherwise so the caller falls through to FastAPI (which JIT-loads).
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
        coro = self.model_manager.resolve_model_id(model_id)
        try:
            resolved_mid = asyncio.run_coroutine_threadsafe(coro, main_loop).result(timeout=5.0)
        except Exception as e:  # noqa: BLE001
            log.error(f"Port resolution failed for {model_id!r}: {e}")
            return None
        if not resolved_mid:
            return None

        # Snapshot the loaded model. NOTE: manager._lock is an asyncio.Lock bound to the
        # main loop — it CANNOT be acquired from this worker thread. Dict reads are atomic
        # under the GIL, so a plain read is safe here (matches how list_models snapshots).
        lm = self.model_manager.loaded.get(resolved_mid)
        port = lm.port if (lm and lm.ready) else None
        return port

    def _jit_load_and_wait(self, model_id: str) -> Optional[int]:
        """Trigger a JIT load of ``model_id`` and block until it is ready. Returns the
        llama-server port on success, or None on failure (unknown model / timeout / error).

        Runs the async ``load_model`` on the MAIN event loop (never a fresh one) and blocks
        this worker thread for up to _JIT_LOAD_TIMEOUT seconds. load_model() is idempotent
        and concurrency-safe: if another request is already loading the same model it joins
        that in-flight task rather than double-loading.
        """
        try:
            main_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in this thread; use the manager's captured main loop.
            main_loop = getattr(self.model_manager, "_main_loop", None)
        if main_loop is None or not main_loop.is_running():
            log.error(f"JIT load for {model_id!r} skipped: no running main event loop")
            return None

        future = asyncio.run_coroutine_threadsafe(
            self.model_manager.load_model(model_id), main_loop
        )
        try:
            lm = future.result(timeout=_JIT_LOAD_TIMEOUT)
        except Exception as e:  # noqa: BLE001 — includes TimeoutError, KeyError, spawn errors
            log.error(f"JIT load failed for {model_id!r}: {e}")
            return None
        if lm and getattr(lm, "ready", False):
            log.info(f"Model '{model_id}' loaded on port {lm.port}")
            return lm.port
        log.error(f"Model '{model_id}' load completed but not ready")
        return None

    # ------------------------------------------------------------------ piping
    def _pipe_to_backend(self, client_sock: socket.socket, raw_headers: bytes,
                          body_prefix: bytes, target_port: int, label: str) -> None:
        """Forward the request (reconstructed verbatim) to a backend and pipe bidirectionally."""
        try:
            target_sock = socket.create_connection(("127.0.0.1", target_port), timeout=10)
        except Exception as e:  # noqa: BLE001
            log.error(f"Failed to connect to {label} on port {target_port}: {e}")
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

        # Both sockets stay blocking (Windows select can't mix with settimeout/non-blocking).
        self._bidirectional_pipe(client_sock, target_sock)

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
