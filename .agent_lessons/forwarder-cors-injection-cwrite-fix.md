---
tags: [tcp-forwarder, cors, cwrite, browser-client, windows-testing, latency]
aliases: [cwrite-cors-broken, forwarder-cors-restore, cors-blocking-first-byte-delay]
related: [[raw-tcp-forwarder-architecture]], [[forwarder-ws-passthrough-regression]]
confidence: verified
updated: 2026-09-14
---

# Raw TCP forwarder — CORS injection restores CWrite (browser) after raw-passthrough commit

Commit `e4cbe26` moved the LLM path from the FastAPI/uvicorn proxy to a raw-TCP pipe. The old
proxy added CORS headers to every LLM response; llama-server emits **none**, so a cross-origin
browser client (CWrite: Vite on :3136 → `fetch()` to 127.0.0.1:1234) had its response blocked and
saw nothing. AgentCascade (Python client) is not subject to CORS, so it kept working — the bug was
invisible to non-browser clients.

## ⚠ CRITICAL: The original implementation caused multi-second first-byte delays
The first version (`e73a3db`) used a **blocking recv() loop** in `_inject_cors_into_first_response`
that buffered until `\r\n\r\n` was found. Under real AC load (concurrent sessions, ~70k context,
multi-second prompt eval), this held the entire first response burst hostage in Python while the
client waited for its first byte. User verified: rolling back to pre-CORS code eliminated ALL delays.

**Fixed in `340de52`**: replaced with `_inject_cors_nonblocking` — one non-blocking recv(), if head
is complete inject and forward immediately; if partial/empty, forward verbatim with no injection.
Zero bytes held hostage. See [[autoloader-turn-delay-is-llamacpp-prompt-cache-thrash]] for full evidence.

## ⚠⚠ STILL-LATENT REGRESSION: the 2 s select wait (fixed 2026-09-14)
The `340de52` version of `_inject_cors_nonblocking` used `select.select([target_sock], [], [], 2.0)`
BEFORE its single recv. On a **cold prompt-cache miss** (TTFB > 2 s — full reprocess of a large context)
the head is NOT yet in the kernel buffer, so select burned the FULL 2 s before returning empty and letting
`_bidirectional_pipe` start. User felt this +2 s live on every cold call and had to bypass the loader entirely.

**Zero-latency fix (this session)**: change the timeout `2.0 → 0.0`. A blocking `recv` after select reports
ready does NOT block (it returns what's already buffered), so a pure poll (`select(..., 0.0)`) is the true
zero-wait peek — NO `setblocking(False)` needed, NO restore to manage, and the socket stays BLOCKING
(which the pipe's Windows select requires; see comment at `_pipe_to_backend`). Warm hit → head already buffered
→ one recv/sendall of the head (injection as before). Cold miss → empty poll → skip injection, pipe delivers
first byte instantly. `rest` after `\r\n\r\n` is re-forwarded via `payload = injected_head + rest` (lossless —
the pipe only forwards bytes arriving AFTER this point). Do NOT touch `_cache_status_header`,
`_inject_cors_headers`, or the TTFB threshold constant.

## The fix (`tcp_forwarder.py`, as of 2026-09-14 zero-wait)
- `_CORS_HEADERS` = `Access-Control-Allow-Origin/Methods-Headers: *` (the exact trio the old proxy sent).
- Module fn `_inject_cors_headers(head)`: header-only edit — inserts the trio right after the status
  line, before the blank line. Never touches the body (chunked/Content-Length framing preserved).
  No-op if `access-control-allow-origin` already present (case-insensitive) → no double-inject.
- `_pipe_to_backend`: when `label == _LLAMA_SERVER_LABEL` only, calls `_inject_cors_nonblocking`
  before `_bidirectional_pipe`. Zero-wait: `select(..., 0.0)` pure poll (socket stays BLOCKING), then ONE
  recv(65536): if it contains `\r\n\r\n`, partitions head/rest, injects into head, sends `injected + rest`;
  if partial or empty, forwards verbatim (no injection) and the pipe handles everything. Management traffic
  to FastAPI (`label != _LLAMA_SERVER_LABEL`) is NOT modified.
- `_send_cors_preflight_response`: OPTIONS on an LLM endpoint → 204 with the trio + `Access-Control-Max-Age`
  sent directly (no piping). Branch added in `_handle_connection` before the POST/LLM routing.
- Tertiary no-hang hardening in `_read_request`: send `HTTP/1.1 100 Continue\r\n\r\n` when the request has
  `Expect: 100-continue`; bound the READ phase with a socket timeout (`_READ_PHASE_TIMEOUT = 5.0`) ONLY when
  there is no Content-Length, then clear it before the pipe phase. AC fast path always has Content-Length → no-op.

## Constraints honored
`_bidirectional_pipe` core loop unchanged (TCP_NODELAY + blocking+select); port wiring / routing /
`_resolve_target_port` / `_jit_load_and_wait` / `last_used` touch untouched; 256 KB scan cap NOT reintroduced.

## ⚠ Windows testing gotcha (cost real time)
**`socket.socketpair()` on Windows IGNORES `settimeout`/`setblocking`** (overlapped I/O) — a blocking `recv`
with no data never times out and hangs forever, and you cannot observe a "would block" state via timing. So the
bounded-read-phase / 100-continue tests MUST use a **real loopback server + client** (`_loopback_client()` helper:
bind 127.0.0.1:0, listen, connect, accept — returns `(client, server_sock, listen_sock)`, THREE values), NOT
socketpairs. Real connected sockets honour `settimeout` (verified: recv timed out at the set interval).

For the zero-wait cold-path regression test (`test_cold_path_no_blocking_wait_and_socket_stays_blocking`): use a
real loopback pair with nothing sent, assert elapsed < 1.0 s (the old `2.0`-timeout code would take ~2 s here —
robust discriminator) and that `getblocking()` is still True afterwards. Note `_loopback_client()` returns 3 values
(a first draft unpacked only 2 → `ValueError: too many values to unpack`).

## Testing the pipe without a live backend
`_pipe_to_backend` opens the backend via `_connect_backend(port)` (factored out specifically so tests can
patch it to return a pre-connected socket pair — a fake llama-server). In `_client_side()` the test client
keeps its socket open while reading the response, then `shutdown(SHUT_RDWR)` after capture so the forwarder's
pipe sees client-side EOF and exits (a follow-up request instead blocks in `sendall` to the already-closed
backend — no RST is raised on a closed-but-not-reset socket).

Run: `pytest tests/test_tcp_forwarder.py -v` → 58 passed (as of 2026-09-14; includes the zero-wait cold/warm-path
regression tests in `TestCacheStatusOnLlmPath`).
