---
tags: [webui, app.js, render-drift, in-place-update, button-row]
aliases: [appjs-inplace-refresh, appjs-structure-compare]
confidence: verified
---

# WebUI (static/app.js) Render-Path Drift & Button-Row Structure — 2026-08-17

`renderModels()` has TWO code paths that must stay in sync: **build-new-card** (template string) and **in-place-update** for existing cards. They DRIFTED, causing stale UI after Edit-modal save or rescan. Fixed by a refinement pass (surgical, single file). See [[webui-appjs-robustness-fixes]] for the earlier defensive-coding pass.

## Root causes fixed
1. **In-place path was patching only meta/status/stats/buttons** — never name/tags/args. After an edit or rescan changed a model's `name`/`tags`/`args`/`mmproj`, the card kept stale values. Fix: extract shared helpers so BOTH paths compute identical output:
   - `tagsRowHTML(m)` → user tags + default/pinned/autosave/vision spans (used by both template and in-place `.tags-row`).
   - `argsDisplay` computed ONCE at top of the `list.forEach(m)` body (conditional `parts.push` for ctx_size/ngl only when `!= null`, plus `args`) and reused by both paths.
   - In-place path now also updates `.model-name` (textContent), `.tags-row` (innerHTML), `.args-box` (textContent).
2. **`updateButtonRow()` could never add/remove the Vision button** — it only updated an existing one, and had a dead `if (!loadBtn)` fallback. Fix: **structure-compare** approach. A row's structure is determined by `m.loaded` (Load vs Unload) and `m.has_mmproj` (Vision present). Read current from DOM (`currentLoaded`, `currentHasVision`); if either differs → `btnRow.innerHTML = buildButtonRowHTML(m); return;`. Otherwise fine-grained text/disabled updates only. One branch covers load/unload swaps AND vision create/remove.

## Patterns to keep
- **Shared helpers over duplicated template logic** — any field shown on a card must be produced by the same helper in both render paths, or they will drift again.
- **`data-stat="vram|ram|ctx|uptime"` attributes** on `.model-stats` child divs; in-place path uses `querySelector('[data-stat=...]')` with null-checks (NO positional `.children[N]`).
- **`getInFlightState(id)`** helper returns `{act, isLoadInFlight, isUnloadInFlight, isVisionInFlight}` — used by both `buildButtonRowHTML` and `updateButtonRow`.
- **Modal state globals cleared on cancel**: `modal-cancel` → `editingId=null`; `sm-cancel` → `stateManagingId=null`.
- **Toast timing constants** `TOAST_MS=3500`, `TOAST_FADE_MS=300` at top of file.

## Verification
`node --check static/app.js` clean; full pytest suite green (E2E over real HTTP); `test_server.py` 59 passed; production files (server.py/config.yaml/index.html/styles.css) untouched. Note: on a machine without the fake-backend subprocess, one E2E test (`TestRealSubprocessLoad`) is env-skipped via `skipif`, so total may read "130 passed, 1 skipped" instead of "131 passed" — not a regression.
