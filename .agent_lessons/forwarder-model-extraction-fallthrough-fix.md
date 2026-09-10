---
tags: [tcp-forwarder, routing, model-extraction, bugfix, streaming]
aliases: [llm-endpoint-404-fallthrough, extract-model-none]
related: [[raw-tcp-forwarder-architecture]]
confidence: verified
---

# Forwarder LLM endpoint 404 fallthrough — root cause & fix (Round 14)

## The bug
On the live instance, `POST /v1/chat/completions` hit the **internal FastAPI** and returned
**404 Not Found** (LLM proxy routes were removed there in Round 12) instead of being piped to
llama-server. Log signature: repeated uvicorn `"POST /v1/chat/completions HTTP/1.1" 404` lines,
with NO forwarder `Model 'X' not loaded; triggering JIT load...` and no pipe log line.

## Root cause (verified)
In `_handle_connection`, routing was:
```
if self._is_llm_endpoint(path):
    model_id = self._extract_model(body_prefix)   # can be None
    if model_id: ...pipe/jit...; return
# falls through to FastAPI when model_id is None/empty  <-- THE BUG
self._pipe_to_backend(..., "internal API")
```
`_extract_model` returned `None` when the `"model"` field was **not yet in `body_prefix`**. Two triggers:
1. AC-style bodies put a long system prompt + history BEFORE `model`; `_BODY_PREFIX_LIMIT` was only
   **4096**, and `_read_request` capped its up-front body read at `min(content_length, 4096)`.
2. Even within the cap, if the body arrived split across TCP segments and `model` sat in a later segment.

When extraction failed, execution silently fell through to FastAPI → guaranteed 404 (the LLM route is gone).
Falling an LLM endpoint through to the management API is *always* wrong — it can only ever 404.

## The fix (`_tcp_forwarder.py`)
- Non-POST guard: inside the LLM-endpoint branch, `if method != "POST"` → fall through to FastAPI as before.
  Only POST carries a `model` body field; this preserves original behavior for GET/WS-upgrade on an LLM path
  (the only real WS route is `/ws`, which is not an LLM endpoint) and avoids any WS regression.
- `_BODY_PREFIX_LIMIT`: 4096 → **16384**; new `_MODEL_SCAN_MAX_BYTES = 262144` (hard cap for the model hunt).
- New `_read_more_for_model(sock, raw_headers, body_prefix)`: if `model_id` is None on an LLM endpoint,
  keep reading body (bounded by content-length & scan cap) until `_extract_model` succeeds or EOF/cap.
- `_handle_connection` **never falls through** an LLM endpoint to FastAPI: extra read → still None → return
  a forwarder-generated **400** (not a FastAPI 404). Added `400: "Bad Request"` to `_send_error_response`.
- Reading more body never loses data — the remainder always flows through the bidirectional pipe.

## Evidence (debug instance, ports 9126/1296)
Isolated repro (`_repro_isolated.py`, segmented FakeSock, model field at byte ~6072):
- **original @4096 → routed to "internal API" (BUG reproduced)**
- **fixed → routed to "llama-server"**
Live probe (`_repro_probe.py`) on 9126: small/big/verybig bodies all HTTP 200 from llama-server, incl.
model at byte offset **20148** (> both prefix caps) → extra-read path proven.
Streaming via `_live_stream_test.py` (Qwen2.5-0.5B): **83 SSE frames, max inter-frame gap 0.01s → SMOOTH**.

## Notes for future work
- The in-memory repro could NOT reproduce the bug with a plain `recv(4096)` FakeSock: Python loopback grabs
  all buffered bytes at once, so extraction succeeded even in "original" code. Must simulate segmentation
  (small recv chunks) to trigger it. See [[raw-tcp-forwarder-architecture]] for cross-thread gotchas.

## REGRESSION (Round 15): chunked bodies >256KB wrongly rejected with 400
The `_MODEL_SCAN_MAX_BYTES = 262144` "hard cap" introduced above was **wrong**. It capped the model-hunt at
256 KB, but AC requests use `Transfer-Encoding: chunked` (no Content-Length) and their bodies grow past
~280 KB as session history accumulates. Once a body exceeded 256 KB, `_read_more_for_model` stopped reading
mid-body — BEFORE the `"model"` field — so extraction failed and the request was rejected with **400**.

Log signature (intermittent "came out of the blue" mid-session):
```
'model' extracted after extra read (278528B)   # _read_more_for_model RETURNED 278KB buffer
no 'model' in body (278528B); refusing -> 400  # but extraction on that SAME buffer still None
```
The INFO/WARNING contradiction is the tell: the extra read returned bytes, yet `"model"` isn't in them → the
field was past the cap. **Intermittent** because whether `model` lands before/after byte 256 KB depends on how
much history AC has in that session (short context = works; long context = 400).

Fix: removed the realistic cap entirely. We pipe this exact request's full body to llama-server anyway, so it
is always terminated (Content-Length or chunked terminator / EOF); a keepalive client sends no further *body*
bytes until we respond, so reading to EOF is well-defined and cannot hang. Replaced `_MODEL_SCAN_MAX_BYTES`
with `_MODEL_SCAN_HARD_CAP = 1 GiB` — purely a guard against a pathological unbounded stream, never hit by real
requests. **Lesson: never cap a read at "realistic body size" when you're going to consume the whole body anyway.**

Regression test: `tests/test_tcp_forwarder.py::TestReadMoreForModel::test_chunked_body_model_past_256kb_now_found`.
