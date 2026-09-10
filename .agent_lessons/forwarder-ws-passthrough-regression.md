---
tags: [tcp-forwarder, websocket, passthrough, regression-test, streaming]
aliases: [ws-upgrade-relay, non-llm-fallthrough]
related: [[raw-tcp-forwarder-architecture]], [[forwarder-model-extraction-fallthrough-fix]]
confidence: verified
---

# Forwarder WebSocket / non-LLM pass-through — regression coverage

## What it protects
The `RawTCPForwarder` (main port 1234) pipes ONLY OpenAI-style LLM generation endpoints
(`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` + bare forms) straight to llama-server.
**Everything else** — management/status/state routes, the `/ws` WebSocket, unknown paths — is relayed
verbatim to the internal FastAPI app on the INTERNAL port (default 1235) via `_pipe_to_backend` →
`_bidirectional_pipe`. A WebSocket upgrade is a `GET /ws` (not an LLM endpoint), so it MUST be relayed
intact: the `101 Switching Protocols` handshake reaches the client, and after the handshake raw WS frames
flow in BOTH directions without the forwarder corrupting/dropping bytes.

## Why this matters
The whole point of the forwarder was to remove ASGI/uvicorn from the *streaming* path. But the
*control plane* (incl. `/ws` for live UI status) still rides through the forwarder's raw pipe to FastAPI.
If someone "optimizes" `_pipe_to_backend` or `_bidirectional_pipe` (e.g. assumes only LLM traffic, adds a
body cap, rewrites headers, or breaks half-close handling), the WebSocket and management routes silently
break while LLM streaming still works — easy to miss because AC only exercises LLM paths on 1234.

## The regression test
`tests/test_ws_passthrough.py` (real loopback sockets, daemon threads, dynamic free ports — never collides
with a live loader; NO FastAPI/uvicorn, so it isolates the forwarder's pipe behavior):
- `test_ws_upgrade_relayed_and_bidirectional`: upgrade request reaches backend → 101 relayed to client →
  bidirectional byte exchange after handshake is stable (echo intact).
- `test_ws_upgrade_relay_is_byte_transparent`: WS headers forwarded verbatim (case/order preserved — the
  forwarder must NOT rewrite `raw_headers`).

Fake backend = minimal `_FakeWsBackend` that does the HTTP upgrade handshake then echoes bytes. No model
manager, no subprocesses.

## Invariants to preserve (do not regress)
1. Non-LLM requests (incl. `/ws`) are forwarded to `internal_api_port`, never rejected or piped to llama-server.
2. `raw_headers` (which already contains the request line) is sent verbatim — do NOT prepend a second
   request line (that caused uvicorn "Invalid HTTP request received"). See [[forwarder-model-extraction-fallthrough-fix]].
3. `_bidirectional_pipe` must handle half-close: when one side closes, half-close the other direction and
   drain until the peer also closes (orderly EOF, not a hard RST).
4. The 5s `select` timeout keeps idle connections responsive to shutdown without breaking long-lived WS/SSE.

## Run it
`python -m pytest tests/test_ws_passthrough.py -v`
