---
tags: [autoloader, slot-save-restore, error-handling, http-probe-loop]
aliases: [slot-action-probe-loop]
related: [[async-cutoff-restore-model-swap]]
confidence: verified
author: orchestrator_Maine_20261027_002908.jsonl
contributors:
  - coder_loader-slot-error-fix_20261027_025826.jsonl
---

# Autoloader: slot save/restore probe loop turns a clean engine 4xx into a 3-call spam + misleading 500

## Symptom (reproduced 2026-10-27 @ ninfer-serve build `...gf7db07b6-dirty`, port 1234)
When the engine returns a **genuine structured 400** on a slot `state/save` or `state/restore`, the loader issues 3 HTTP calls and returns **500**, discarding the engine's real error:
```
Saving slot 0 state to file: ...DFlash2.zz-repro-save.bin
[WARNING] Slot save attempt failed [URL param: ?action=save]: HTTP 400 — {"error":{"code":"slot_save_failed","message":"slot holds no retained session",...}}
[WARNING] Slot save attempt failed [URL param: ?action=save]: HTTP 404 —
[WARNING] Slot save attempt failed [JSON body params]: HTTP 404 —
POST .../state/save HTTP/1.1" 500 Internal Server Error
```
Client body: `{"detail":"save failed: save failed across all slot endpoints. Last error: [JSON body params] HTTP 404:"}` — the real `slot_save_failed` code/message is **lost**; only the last 404 survives.

## Root cause (file:line, autoloader @ current tree)
`_perform_slot_action` (`server.py:3691`): builds 3 request shapes (`/slots/0?action=`, `/slots?action=`, `/slots` JSON body) and loops them. The **only terminal HTTP break is `if r.status_code >= 500: break` (`server.py:3716`)**; the other break (`:3724`) is for an unreachable socket. A genuine engine **400** is NOT terminal, so the loop continues to the other two shapes (which return 404 because that endpoint shape doesn't exist), then `raise RuntimeError(... last_err ...)` (`:3734`) where `last_err` is the **last** probe (`:3712`), not the real first error. Callers `save_state` (`:3777`) / `load_state` (`:3855`) wrap it as `HTTPException(500, ...)`.

The probe loop exists to handle different engine API shapes — a **404 legitimately means "this endpoint shape is unsupported, try the next."** The bug is conflating that with a **4xx that carries a structured error body** (= "the engine understood the request and genuinely refused the operation").

## Fix (loader-side; engine already replies correctly)
In `_perform_slot_action`, treat a response that carries a structured engine error body (JSON with `error.code`/`error.message`) as **terminal**: break immediately and surface the engine's code/message to the caller as a proper **4xx** (e.g. 400/409), NOT a 500. Continue probing ONLY on a bare 404 (endpoint-shape mismatch). The final error must carry the FIRST real engine error, not the last 404.
- "Preferably no issue in the first place" for the restore case is already handled by the engine's make-room loop (restore into a fresh slot now returns 200; the "does not fit" 400 no longer fires). The save "no retained session" 400 is a legitimate refusal and must be surfaced cleanly, not spammed.

## Reproduction harness
`bench/quick_aba.py` (AgentWorkspace/bench) drives the loader's `state/save`/`state/load` endpoints directly and captures the probe sequence + final status. Deterministic trigger: call `state/save` on a freshly-loaded slot that holds no retained session.

## Implemented (2026-10-27, autoloader working tree — NOT committed)
Loader-side fix landed; engine untouched.

- **New module-level `EngineSlotError(RuntimeError)`** (defined just after `_UNREACHABLE_ERRORS`): carries `code`, `message`, `status_code`. Subclasses `RuntimeError` so existing `except Exception` handlers still catch it (same convention as `RecoveryCooldownError`).
- **Single source of truth for classification:** `_parse_engine_error(text)` returns `(code, message)` only when the body is JSON with an `error` object carrying `code`/`message`; otherwise `None` (bare 404 → keep probing). No loader-side status inference: `EngineSlotError.status_code` is the engine's OWN returned 4xx, forwarded verbatim (guaranteed <500 by the caller's `status_code < 500` gate). An earlier `_engine_error_status` keyword→409 heuristic was tried and REMOVED as brittle — the engine already chose the status deliberately, so guessing is both redundant and wrong.
- **`_perform_slot_action`:** after the 200 success check, `if r.status_code < 500:` probe for a structured error and `raise EngineSlotError(...)` immediately (terminal — no further shapes). A bare 404 falls through to the next shape (multi-shape probing preserved). The `>=500` break and the `_UNREACHABLE_ERRORS` break are unchanged. Added `except EngineSlotError: raise` **before** the generic `except Exception` — required, since `EngineSlotError` is an `Exception` and would otherwise be swallowed and the loop would keep probing. Renamed `last_err`→`first_err` (keep the FIRST real error, not the last 404); final message now says "First error:".
- **Callers `save_state` / `load_state`:** `except EngineSlotError as e: raise HTTPException(e.status_code, e.message)` placed **before** the generic `except Exception → HTTPException(500, ...)`. So "slot holds no retained session" (engine 400) → clean 400 with the engine message; a refusal the engine returned as 409 → 409 (forwarded, not inferred).
- **Internal callers** (auto-save-on-unload, auto-restore) still wrap in `except Exception → log.warning`, so the now-terminal `EngineSlotError` degrades gracefully there — and benefits from single-probe behavior.

**Tests** (`tests/test_regression.py::TestState`, driven directly with a `_FakeSlotClient` in parity with the real `httpx.AsyncClient` — `.post` + `.aclose`): structured 400 → 1 call + `EngineSlotError` with code/message + caller 400; bare-404→200 still probes and succeeds (2 calls); all-404 → generic `RuntimeError`/500 (3 calls, and asserts the FIRST probe's body is carried while later probes' are NOT); a structured **5xx** body → still generic `RuntimeError`/500 (1 call — the `status_code < 500` gate wins over the structured check); a structured refusal returned as **409** → surfaced as 409 (verbatim forward). Suite: 447 passed, 3 errors (all `test_real_model.py::TestRealModelLoad` fixture-setup, missing CUDA `llama-server.exe` binary — environmental, pre-existing). NOTE: `test_two_instances_do_not_collide_on_engine_ports` intermittently fails under full-suite load (`_port_in_use` loopback-detection race on Windows) and passes in isolation — a known flake unrelated to this change.
