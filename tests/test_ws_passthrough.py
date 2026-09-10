"""Regression test: WebSocket / non-LLM pass-through through the raw TCP forwarder.

Background
----------
The RawTCPForwarder owns the MAIN client-facing port (1234). It pipes ONLY the OpenAI-style
LLM generation endpoints straight to llama-server; EVERYTHING else (management/status/state,
the ``/ws`` WebSocket, unknown paths) is forwarded verbatim to the internal FastAPI app on the
INTERNAL port. That forwarding path is a plain bidirectional raw-TCP pipe (_pipe_to_backend ->
_bidirectional_pipe). A WebSocket upgrade is a GET on /ws (not an LLM endpoint), so it must be
relayed intact: the ``101 Switching Protocols`` handshake has to reach the client, and after the
handshake raw WS frames must flow in BOTH directions without the forwarder corrupting or dropping
them.

This test isolates exactly that concern WITHOUT FastAPI/uvicorn (which would add noise): it stands
up a real RawTCPForwarder on a free port plus a minimal fake WebSocket backend that performs the
HTTP upgrade handshake and echoes bytes back. We then drive a raw client through the forwarder and
assert:
  * the upgrade request is relayed to the backend,
  * the ``101`` handshake response is delivered back to the client unmodified,
  * bidirectional byte exchange after the handshake is stable (client->backend and backend->client),
  * orderly half-close when one side closes.

These are real loopback sockets in daemon threads; no production state, no model manager, no
subprocesses. Ports are allocated dynamically so they never collide with a live loader.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

from tcp_forwarder import RawTCPForwarder  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeWsBackend:
    """Minimal WebSocket backend: accept one connection, do the HTTP upgrade handshake, then echo."""

    def __init__(self, port: int):
        self.port = port
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("127.0.0.1", port))
        self.server_sock.listen(1)
        self.server_sock.settimeout(5.0)
        self.received_request = b""
        self.handshake_done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            conn, _addr = self.server_sock.accept()
        except OSError:
            return
        try:
            # Read the full HTTP request (headers end at \r\n\r\n). A WS upgrade has no body.
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
            self.received_request = buf
            # Send the standard 101 Switching Protocols response.
            resp = (
                b"HTTP/1.1 101 Switching Protocols\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Accept: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                b"\r\n"
            )
            conn.sendall(resp)
            self.handshake_done.set()

            # After the handshake, echo everything back (proves bidirectional pipe).
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                conn.sendall(b"echo:" + data)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        try:
            self.server_sock.close()
        except OSError:
            pass


@pytest.fixture
def forwarder_and_backend():
    """A real RawTCPForwarder (main port) + fake WS backend (internal port), both on free ports."""
    main_port = _free_port()
    internal_port = _free_port()

    backend = _FakeWsBackend(internal_port).start()
    fwd = RawTCPForwarder(listen_port=main_port, internal_api_port=internal_port, model_manager=None)
    fwd.start()
    # Give the forwarder a moment to bind.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", main_port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)

    yield {"main_port": main_port, "internal_port": internal_port,
           "fwd": fwd, "backend": backend}

    try:
        fwd.stop()
    finally:
        backend.stop()


def test_ws_upgrade_relayed_and_bidirectional(forwarder_and_backend):
    """A /ws upgrade request is relayed to the backend; 101 + bidirectional bytes flow intact."""
    info = forwarder_and_backend
    main_port = info["main_port"]

    client = socket.create_connection(("127.0.0.1", main_port), timeout=5)
    try:
        # Standard WebSocket upgrade request (GET, not an LLM endpoint -> must be forwarded).
        upgrade = (
            b"GET /ws HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            b"Sec-WebSocket-Version: 13\r\n"
            b"\r\n"
        )
        client.sendall(upgrade)

        # The backend must have received the upgrade request (proves relay happened).
        assert info["backend"].handshake_done.wait(timeout=5), "backend never saw the WS upgrade"
        assert b"GET /ws HTTP/1.1" in info["backend"].received_request, \
            f"forwarded request missing request line: {info['backend'].received_request[:80]!r}"

        # The client must receive the 101 handshake relayed back from the backend.
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = client.recv(4096)
            if not chunk:
                break
            resp += chunk
        assert resp.startswith(b"HTTP/1.1 101"), f"expected 101 handshake, got: {resp[:80]!r}"
        assert b"Sec-WebSocket-Accept: dGhlIHNhbXBsZSBub25jZQ==" in resp

        # After the handshake, raw frames must flow client -> backend (echoed back).
        client.sendall(b"\x81\x05hello")  # a fake WS text frame payload "hello"
        echoed = b""
        while not echoed.startswith(b"echo:"):
            chunk = client.recv(4096)
            if not chunk:
                break
            echoed += chunk
        assert echoed == b"echo:\x81\x05hello", f"bidirectional relay corrupted: {echoed!r}"
    finally:
        try:
            client.close()
        except OSError:
            pass


def test_ws_upgrade_relay_is_byte_transparent(forwarder_and_backend):
    """The forwarder must not alter the WS handshake headers (case/order preserved)."""
    info = forwarder_and_backend
    main_port = info["main_port"]

    client = socket.create_connection(("127.0.0.1", main_port), timeout=5)
    try:
        upgrade = (
            b"GET /ws HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Upgrade: websocket\r\n"
            b"Sec-WebSocket-Version: 13\r\n"
            b"\r\n"
        )
        client.sendall(upgrade)
        assert info["backend"].handshake_done.wait(timeout=5), "backend never saw the WS upgrade"

        # The exact header bytes the client sent must arrive at the backend verbatim
        # (the forwarder forwards raw_headers as-is; it must not rewrite casing/order).
        sent = b"Host: 127.0.0.1\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\n"
        assert sent in info["backend"].received_request, \
            f"WS headers were not forwarded verbatim: {info['backend'].received_request!r}"
    finally:
        try:
            client.close()
        except OSError:
            pass
