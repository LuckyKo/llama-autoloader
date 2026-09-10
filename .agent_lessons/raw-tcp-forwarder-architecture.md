---
tags: [tcp-forwarder, streaming, sse-buffering, architecture, windows]
aliases: [sse-burst-final-fix, raw-socket-pipe]
related: [[uvicorn-logging-architecture]], [[forwarder-model-extraction-fallthrough-fix]]
confidence: verified
---

# Raw TCP Forwarder — final fix for SSE streaming burst (Round 11)

> **Round 14 addendum:** an LLM endpoint could silently fall through to FastAPI (→404) when the `model`
> field wasn't in the up-front body prefix. Root cause + fix: [[forwarder-model-extraction-fallthrough-fix]].

## The decision
After 10+ rounds of app-level fixes on the FastAPI/uvicorn/httpx relay all failed
(TCP_NODELAY, keepalive pings, anti-buffering headers, event-loop policy, sync/threadpool),
the root cause was identified as **buffering in the ASGI/uvicorn layer itself** — Windows
socket coalescing during ~47s prefill delivered the whole SSE response in one end-of-stream
burst. The fix removes the ASGI layer from the streaming data path entirely.

## Architecture (NEW — simplified in Round 12)
- `RawTCPForwarder` (`_tcp_forwarder.py`) owns the **MAIN** client-facing port
  (`manager.port`, from `CFG['launcher']['port']`; config.yaml currently 9124).
- FastAPI/uvicorn moved to an **INTERNAL** port: env `AUTOLOADER_INTERNAL_PORT` (default 1235),
  set in the `__main__` block (`uvicorn.run(app, ..., port=INTERNAL_API_PORT)`).
- `_startup` captures `manager._main_loop = asyncio.get_running_loop()` then starts the forwarder;
  `_shutdown` stops it.
- Routing is now **endpoint-based, not stream-based** (Round 12 simplification): if the path is an
  LLM endpoint (`_is_llm_endpoint`: /v1|bare chat/completions, completions, embeddings) AND a model
  was extracted AND that model is loaded/ready → transparent raw socket pipe (blocking sockets +
  `select`, oproxy-style) straight to the llama-server port. TCP_NODELAY set on both data sockets.
  Piping is identical for streaming and non-streaming — there is NO `_is_streaming` branch anymore.
- Everything else (management/status/state, `/ws` WebSocket upgrade, LLM path with a not-yet-loaded
  model) is forwarded **verbatim** to the internal FastAPI port (raw header bytes preserved).

### Round 12: old FastAPI proxy route REMOVED (dead code)
- `ModelManager.proxy()`, `_make_proxy_route`, `proxy_raw`, `proxy_catchall`, and all `/v1/chat/
  completions` + `/completions` + `/embeddings` + `/v1/{path}` route registrations were DELETED from
  server.py — the LLM data path is now served ONLY by the forwarder. FastAPI keeps management/status/
  state routes + `/ws` only (22 routes). Unused imports `StreamingResponse`, `AsyncIterator`, `Iterator`
  removed. NOTE: a live E2E test (`tests/test_real_model.py::test_real_completion_roundtrip`) previously
  POSTed /v1/chat/completions to FastAPI directly; it now targets the llama-server child port directly.

### Round 13: forwarder handles JIT-load (forwarder owns the ENTIRE LLM data path)
- `_handle_connection` no longer falls through to FastAPI for a not-loaded LLM model. New method
  `_jit_load_and_wait(model_id)` schedules `manager.load_model(mid)` on the MAIN loop via
  `asyncio.run_coroutine_threadsafe(...).result(timeout=_JIT_LOAD_TIMEOUT)` and blocks the worker thread
  until ready, returning `lm.port` (or None → forwarder sends 503 to the client).
- `_JIT_LOAD_TIMEOUT` = env `AUTOLOADER_JIT_LOAD_TIMEOUT` (default 300s; some models load for minutes).
- Safe because `load_model` releases its asyncio.Lock BEFORE awaiting `_wait_until_ready`, so the main loop
  stays free for other requests. Same-model concurrent requests dedup via `_loading_tasks` (both threads join
  one task → same port, no double-load). On timeout the load coroutine is NOT cancelled — it keeps loading in
  the background; a later request joins it. This makes FastAPI's LLM route truly unnecessary.

## CRITICAL cross-thread gotchas (do NOT regress these)
1. `resolve_model_id` / `self.loaded` are guarded by an `asyncio.Lock` and are async-only.
   A worker thread MUST resolve ports via `asyncio.run_coroutine_threadsafe(coro, main_loop)`
   against the MAIN running loop — NEVER `asyncio.new_event_loop().run_until_complete(...)`
   (unsafe: fresh loop can't share state / locks). See `_resolve_target_port`.
2. `manager._lock` is an `asyncio.Lock` bound to the main loop and **cannot** be acquired from a
   worker thread. The forwarder reads `self.model_manager.loaded.get(mid)` as a plain GIL-atomic
   dict read (matches how `list_models` snapshots). Do NOT add `with self._lock:` in a worker thread.
3. Both pipe DATA sockets stay BLOCKING — Windows `select` cannot mix with `settimeout`/non-blocking.
   Only the LISTENING socket uses `settimeout(1.0)` (for clean shutdown in the accept loop).

## Request parsing
- `_read_request` reads until `\r\n\r\n`, then up to `min(content_length, 4KB)` of body.
- Forwarding re-sends `request_line + raw_headers + body_prefix`; the rest of the body flows through
  the bidirectional pipe (no double-send). Chunked bodies: `content-length` absent → prefix only, piped.
- `_extract_model` / `_is_streaming`: try JSON parse first, fall back to regex on truncated bodies.

## Tests
`tests/test_tcp_forwarder.py` — 19 parsing-only unit tests (no live sockets), all pass. Full suite: 166 passed.

### LIVE E2E verified (Round 14, 2026-09-10) — SSE burst IS fixed
Ran a real end-to-end streaming test through the forwarder on isolated debug ports
(main 9126 / internal 1296 / llama-server base_port 9141) with `Qwen2.5-0.5B-Instruct-Q8_0.gguf`
(506 MB, smallest chat model). Result: **83 SSE frames over a ~0.28 s span, max inter-frame gap 0.01 s** —
incremental, one frame per backend token, NO end-of-stream burst. llama-server itself generated
81 tokens in 283.8 ms (~3.55 ms/tok); the forwarder delivered each frame as it was emitted with no
batching delay added. Confirms the raw-socket pipe removes the ASGI/uvicorn buffering that caused the
old burst. Test harness: `_live_stream_test.py` (points at a tiny model, measures inter-frame gaps;
verdict = "incremental" iff ≥5 frames and no single gap > 0.5 s). See [[sidecar-args-invalid-flag-jit-load-fail]]
for the only failure hit during this test (a per-model sidecar arg that crashed llama-server — NOT a
forwarder bug; the forwarder correctly surfaced it as HTTP 503).

## Open assumptions / edge cases
- Forwarder binds `127.0.0.1` only (proxy is local-only; matches `manager.host`). If a non-loopback
  main host is ever configured, update the bind address in `RawTCPForwarder.start`.
- Internal port must not collide with llama-server ports (`base_port`+n) or the main port. Default 1235
  is safe for current config (base_port 9001); override via `AUTOLOADER_INTERNAL_PORT` if needed.
- Streaming request for a model that is NOT yet loaded falls through to FastAPI → JIT load (correct).
