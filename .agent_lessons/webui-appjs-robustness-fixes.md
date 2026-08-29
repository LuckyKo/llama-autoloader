---
tags: [webui, app.js, robustness, null-guards, error-handling]
aliases: [appjs-null-guards, webui-toast-errors]
confidence: verified
---

# WebUI (static/app.js) Robustness Fixes — 2026-08-17

Fixed 8 defensive-coding issues in `static/app.js` (surgical, single-file). Verified by full test suite (131 passed incl. E2E over real HTTP) and `node --check`.

Key patterns now in place:
- **Null guards on render fns**: `renderRAM(ram)` early-returns with `—` placeholders if `ram` is falsy; `argsDisplay` built via conditional `parts.push()` so missing `ctx_size`/`n_gpu_layers` are omitted (not "undefined"); stats-row `ctx_size` shows `—` when null.
- **No silent error swallowing**: `refreshAll()` catch now shows a toast (was console-only); `btn-rescan` and `btn-unload-all` handlers wrapped in try/catch with error toasts (were unhandled promise rejections).
- **Settings inputs** use `?? ''` nullish coalescing so null API values don't render as "null" in the form.
- **Empty-state message** distinguishes "No models found." (no filter) vs "No models match your filter." (`filter || statusFilter !== 'all'`). Note: `status-filter` select default value is `"all"` (see index.html).
- **Toasts**: `showToast` uses `textContent` (XSS-safe); removed decorative single quotes around interpolated ids/labels for readability.
- **`renderBackendSelectors`** only rebuilds the edit-modal `<select>` (`fSel`) when `$('modal-overlay')` has class `active`, avoiding wasteful DOM churn on every WS tick (~1 Hz).
