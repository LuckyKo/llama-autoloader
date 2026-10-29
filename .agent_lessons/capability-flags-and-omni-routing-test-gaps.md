---
tags: [capabilities, multimodal, audio, tcp-forwarder, test-gaps]
aliases: [capability-flags-test-gap, ws-method-guard-gap, query-url-decoding-gap]
related: [[lfm3-omni-smoke-test-tts-realtime-pass-multimodal-flag-gap]], [[model-type-derived-vs-override-split]], [[llama-autoloader-pytest-canonical-invocation]], [[forwarder-ws-passthrough-regression]]
confidence: verified
author: researcher_omni-loader-research_20261029_094206.jsonl
contributors:
  - reviewer_verify-omni-anchors_20261029_094206.jsonl
---

# Capability flags + omni routing: all plan anchors verified exact; three test/behaviour gaps

**Tree pin:** llama-autoloader @ HEAD `ee02b40c82e45233a7384a2b5033f91b5bde810c`, **dirty** working tree
(`git status`: `config.yaml`, `server.py`, `tcp_forwarder.py`, `test_server.py`,
`tests/test_tcp_forwarder.py` modified; 4 `.agent_lessons/*.md` + 2 `.bug_tracker/BUG_0015/0016` untracked).
Line numbers below were re-anchored against the **live dirty tree** on 2026-10-29, not against the pin.

## Anchor verification (no drift found)

| Plan anchor | Live location | Status |
|---|---|---|
| `server.py:4811-4812` multimodal/audio | exact | **exact** |
| `tcp_forwarder.py:1307-1320` `_extract_model_from_query` | def at `:1307`, body to `:1320` | **exact** |
| `tcp_forwarder.py:826` WS branch | `if self._is_ws_model_endpoint(path):` | **exact** |
| `tcp_forwarder.py:829-833` 400 path | `if not model_id:` → `_send_error_response(400)` → `return` | **exact** |
| `test_server.py:279` model_type assert | `test_model_type_present_for_capability_flags` | **exact** |

## Gap 1 — no test asserts the `multimodal` / `audio` booleans anywhere

Whole-tree grep for `multimodal|"audio"` over `*.py` (excluding `temp/`) returns **only** `server.py`
(definitions + the payload builder) and one docstring mention at `test_server.py:280`. The only payload
test is `test_model_type_present_for_capability_flags` (`test_server.py:278-285`), which asserts
`model_type` presence/validity — **not** the derived flags. Classifier behaviour is pinned at
`tests/test_regression.py` (e.g. `test_tts_arch`, `test_generic_arch_rescued_by_filename`), i.e. the
*input* to the derivation is tested but the *output* flags are not.

Corollary: `static/app.js` consumes **`model_type` only** (`TYPE_META`, `typeBadgeHTML`, group/filter
logic) and never reads `multimodal` or `audio`. The flags are currently unconsumed by the WebUI, so a
flag change cannot be caught by any UI-side assertion either.

## Gap 2 — `_extract_model_from_query` does no URL decoding

`_extract_model_from_query` (`tcp_forwarder.py:1307-1320`) splits on `?`/`&` and `partition("=")`, then
returns the **raw** value. No percent-decoding and no `+`→space handling. `grep 'urllib|unquote|parse_qs'`
over `tcp_forwarder.py` returns **zero** hits — `urllib` is not imported in the file (imports at
`:32-44`: asyncio, errno, json, logging, os, re, select, socket, threading, time, typing). So a model id
containing a percent-encoded character (space, `/`, non-ASCII) resolves to the wrong string and lands on
the JIT-load / unknown-model path instead of the loaded model.

Existing tests at `tests/test_tcp_forwarder.py:1266-1287` (`TestExtractModelFromQuery`) cover only
already-decoded ASCII ids (`lfm3-omni`, `llama-8b`) — no percent-encoded case, no `+`, no duplicate
`model=` params.

## Gap 3 — the WS branch has no method guard

`_is_ws_model_endpoint` (`:1296-1304`) is **method-agnostic**: it matches the path only. In
`_handle_connection` the only method checks before it are `method == "OPTIONS"` → preflight (`:823`) and,
in the *other* branch, `method != "POST"` → FastAPI (`:838`). So a `POST`/`PUT`/`DELETE` on
`/v1/realtime` enters the WS branch and is piped to llama-server as if it were an upgrade.
House style for method comparison is **case-sensitive against uppercase literals** — `_read_request`
assigns `method, path = parts[0], parts[1]` verbatim (`:953`) with no `.upper()`, and every existing
comparison uses `"OPTIONS"` / `"POST"`. A guard consistent with the file is `method != "GET"`.

No test exercises a non-GET method on `/v1/realtime`, and **no test drives the 400 path at `:829-833`**
through `_handle_connection` — `TestOmiRoutingDecision.test_realtime_ws_without_model`
(`tests/test_tcp_forwarder.py:1304-1307`) only asserts the helper returns `None`.

**Scope of the gap (reviewer correction, 2026-10-29):** the WS branch validates **neither the HTTP method
nor an `Upgrade:` header**. Every non-OPTIONS method (GET/POST/PUT/DELETE/HEAD) on `/v1/realtime` is
piped as a WS upgrade. The `method == "OPTIONS"` preflight at `:823` is the *only* method-specific guard
and it runs **before** the WS branch, so `OPTIONS /v1/realtime` never reaches the 400. The gap also applies
to `/api/v1/realtime`, because `_is_ws_model_endpoint` calls `_normalize_api_prefix` at `:1303`
(the alias is applied a second time in `_read_request` at `:960`).

**Name collision to avoid when citing:** the route handler `async def list_models()` is at
`server.py:4769` (inside `@app.get("/v1/models")` at `:4768`); a *different* `ModelManager.list_models`
method is at `server.py:1969`. Cite the path + the decorator, not just the name.


## Reusable harness for a `_handle_connection` routing test

`test_handle_connection_routes_options_to_preflight` (`tests/test_tcp_forwarder.py:840-894`; next class
`TestTertiaryNoHang` at `:895`) is the template: real `_loopback_client()` sockets (`:441-459`), patch `f._pipe_to_backend` and
`f._send_cors_preflight_response` into counters (`staticmethod(...)` needed for the latter), run
`f._handle_connection(fwd_side)` on the main thread (`:880`) while a client thread sends the request, restore the
originals in `finally`, assert on the counters. `_loopback_client` (not `_socket_pair`) is required
because Windows overlapped sockets ignore `settimeout` — see the docstring at `:444-446`.

Omni/WS test cluster (reviewer-confirmed anchors): `TestOmiPostEndpoints` `:1215`,
`TestIsWsModelEndpoint` `:1245`, `TestExtractModelFromQuery` `:1266`, `TestOmiRoutingDecision` `:1290-1315`.

## Scratch-copy sweep (item 6)

**No** `server_base.py` / `_base_server.py` / `tcp_forwarder_base.py` or any other copy of the two
production modules exists. Root `dir /b *server*.py *forwarder*.py` returns only `server.py`,
`test_server.py`, `tcp_forwarder.py`, `test_kv_cache_reuse.py`.

Two live-tree traps that look like impostors but are **not**:
- **root `test_server.py` is the live server test file** — the canonical invocation is
  `python -m pytest tests/ test_server.py -q` and `conftest.py:6` says so explicitly. There is no
  `tests/test_server.py`. Editing `tests/test_server.py` would create a *second* file, not edit the live one.
- `test_kv_cache_reuse.py` (root) is an ad-hoc live-HTTP script, `collect_ignore`d at `conftest.py:16` —
  it is **not** part of the suite.

`temp/` holds ad-hoc scripts (`omni_smoke_test.py`, `test_kv_fix.py`, `test_direct.py`,
`test_slot_restore.py`, `probe_*.py`) and is excluded via `pytest.ini` `norecursedirs`.

## Current counts (re-measured 2026-10-29)

`python -m pytest tests/test_tcp_forwarder.py test_server.py --collect-only -q` → **194 tests collected,
0 errors**. Full-suite counts from [[llama-autoloader-pytest-canonical-invocation]] (397 @ d977c82) are
historical — re-collect before quoting a "N passed" baseline.
