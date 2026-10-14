---
tags: [strata, bridge-card, autoloader, lifecycle, health]
aliases: [strata-bridge-lifecycle-facts]
related: [[raw-tcp-forwarder-architecture]], [[model-id-gguf-name-collision]], [[sidecar-args-invalid-flag-jit-load-fail]], [autoloader-registry-is-gguf-shaped]]
confidence: verified
author: researcher_strata_researcher_20261006_071922.jsonl
contributors:
  - researcher_strata-save-restore-scope_20261014_110135.jsonl   # marked the /slots claim SUPERSEDED at Strata 296b59d2; flagged all line numbers as stale
---

# Strata serve-layer facts needed to bridge it into llama-autoloader

Verified against `N:\work\WD\Strata\serve\server.py` on 2026-10-06. Non-obvious items only.

- **`/health` is 200 whenever the socket is listening, never gated on readiness.**
  `serve/server.py:2172-2175` returns `{"loaded": svc.loaded(), "service": "strata", ...}` with no
  branch. `loaded()` (`:996-997`) is `not hasattr(engine,"alive") or engine.alive()`;
  `alive()` is `:407-408`.
- **Non-lazy spawn makes 200 a true readiness signal** — `StrataEngine.__init__` blocks on the
  `READY <ctx>` line (`:342-351`) and `serve()` binds only at `:3008`, i.e. after the constructor
  returned. Lazy mode short-circuits at `:325-326`, so 200 then means "listening, not loaded".
- **`--lazy` is fatal with a `vision` config**: `ap.error(...)` at `:2904-2905`.
  `strata-coder-iq1_m.json` has a `vision` block ⇒ bridge cards must spawn non-lazy.
- **`aliases` has NO CLI flag** (`:999-1012`, `:2942`) — config-file only. So a bridge that must
  inject its own `model_id` **cannot** do it with argv and must write a derived config.
- **The request `model` field is never validated**: `model_for()` (`:1017-1021`) silently
  substitutes `self.model` for any unknown name. Alias injection is still needed, but for
  `/v1/models` reporting (`:2193-2202`) and because `/props?model=` 404s (`:2313-2317`).
- **`"port"` in a Strata config JSON is dead config** — only `--port` is read (`:2841`, `:2874`).
  `strata-coder-iq1_m.json:48` sets `"port": 8080` and it is ignored.
- **`--api-key` defaults from `$STRATA_API_KEY`** (`:2850`), and an *empty* key is a hard exit 2
  (`:2947-2952`). A bridge must **delete** the var from the child env, not blank it.
- **CORS is a real trap for the raw forwarder**: the forwarder forwards `Origin` verbatim, and
  `_foreign_page()` (`:2000-2015`) 403s any `/v1/*` POST whose Origin is not in
  `trusted_origins + cors_origins` when there is no api key. A bridged card needs
  `cors_origins: ["*"]`.
- **No orphan risk on unload**: `strata.exe` / `strata-vision.exe` are plain children
  (`:338-339`, `:673-675`, no new session/detach) *and* are put in a
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` job by `contain()` (`:340`, `:675`;
  `serve/winjob.py:53,61-74`). psutil recursive-children kill (autoloader `server.py:960-976`) is
  sufficient.
- **`--before-load` runs with `shell=True`** (`:1066`) — do not use it to call back into the
  autoloader; evict in-process before `Popen` instead.
- `StrataEngine` has no `/slots` save/restore equivalent, so the autoloader's
  `auto_save_state` must be forced off for strata cards.
  > **SUPERSEDED (2026-10-14).** This was true when written. At Strata HEAD `296b59d2` it is **false**:
  > `StrataEngine.session_file()` exists (`serve/server.py:2029-2126`) and the llama-server-compatible
  > `POST /slots/0?action=save|restore` endpoint exists (`serve/server.py:3111-3177`). The autoloader's
  > `is_llama` gate and forced `auto_save_state = False` are now guarding a limitation upstream has
  > removed. See [[strata-save-restore-blocked-by-405-gate]] for the three current blocking points and
  > [[strata-session-save-restore-is-fully-implemented]] for the feature's actual scope.
  >
  > **All line numbers in this file are stale** — e.g. `/health` cited here at `:2172-2175` is at
  > `serve/server.py:5337` today. The *facts* about `aliases`, dead `"port"` config keys and
  > `STRATA_API_KEY` scrubbing remain directionally useful; the citations must not be reused.

Plan: `N:\work\WD\AgentCascade\plans\strata-bridge-plan.md`.
See also [[raw-tcp-forwarder-architecture]].