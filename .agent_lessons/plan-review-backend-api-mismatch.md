---
tags: [plan-review, backend-api, frontend-data]
aliases: [verify-backend-exposes-frontend-fields]
confidence: verified
author: ui_correctness_20261006_110235.jsonl
contributors:
  - coder_ui_builder_20261006_112342.jsonl
---

# Plan review gotcha: backend API must expose all fields the frontend expects

When reviewing a WebUI feature plan, always verify that the backend API actually returns the fields the frontend code intends to use. **Never assume** that because a field exists in `ModelConfig` or is documented in the brief, it is included in the API response.

## The gotcha

In the Strata bridge-card plan (v1), F1 said: "Add to the per-model map: `engine`, `strata_config`, `resolved_binary`, `min_free_vram_mib`, `extra_args`." This assumes the `/v1/models` endpoint returns these fields. However, `server.py:1862-1888` only returns a subset of fields and **excludes** exactly those five. A coder implementing F1 literally would get `undefined` values and break the UI.

## How to catch it

1. **Read the actual API handler.** Don't just grep for endpoint names—read the full function body and note every key in every return statement (including fallback/empty branches).
2. **Cross-check with the frontend mapping.** Compare the keys the frontend accesses against the keys the API returns. Any mismatch is a red flag.
3. **Check `list_models()` vs `/v1/models`.** Sometimes the manager method returns more fields than the public endpoint exposes. The plan may conflate the two.

## Prevention

- Include a step in plan review: "Verify backend exposes all frontend-required fields."
- If fields are missing, the plan must explicitly include backend changes to extend the response schema and any DTOs.
- For settings or config fields, remember that persistence alone is not enough—there must be a read path from UI to backend to UI.

## Resolution (build phase, 2026-10-06)

The v2 plan repeated the same false premise: B1b claimed "engine, strata_config, resolved_binary are ALREADY there (749-751)". That is true of **`ModelManager.list_models()`** (`server.py` ~749) but **NOT** of the HTTP handler `list_models()` at ~1862, which re-projects every field key-by-key. The WebUI consumes the **HTTP** one via `refreshAll()` → `GET /v1/models`, so F1 would have mapped `undefined`.

Fix applied: added `engine`, `strata_config`, `resolved_binary`, `min_free_vram_mib`, `extra_args` as pass-through keys to the `/v1/models` response dict. This does not conflict with the plan's "don't refactor the /v1/models serializer" — that line was about the *POST* response body (return `{"id": mid}`), not about the GET projection.

**Generalized rule: a new WebUI-visible field needs FOUR edits, not one** — dataclass field → manager `list_models` dict → HTTP handler projection → app.js `refreshAll` whitelist. Check all four; missing any one silently yields `undefined` in the UI.

## Related

- [[fullstack-data-feature-plumbing]] — emphasizes the six-seam data path for settings.
- [[plan-anchor-verification]] — verifies line numbers and structural assumptions.
- [[llama-autoloader-pytest-canonical-invocation]] — the run command for this repo.
