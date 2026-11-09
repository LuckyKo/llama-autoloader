# BUG_0021 — WebUI model-card renders reset page scroll and drop button focus

**Status:** FIXED
**Filed:** 2026-11-09
**Fixed:** 2026-11-09
**Commit:** `f3d39a6e29eaa1d0e50d53afa5f3cc76ffa3b715`
**Found in:** Manual UI report — clicking a button on a model card below the fold appeared to "reset the page scroll". Reproduced in headless Chrome via CDP against the real `static/index.html` + `static/app.js` with stubbed `/v1/*` responses and `/ws` pushes.
**Severity:** low-medium — no data loss, but the viewport jumps upward under the cursor on every card-button click and again on the 1 Hz status tick while that button has focus, making the WebUI awkward to use on a long model list.
**Component:** `static/app.js` (`renderModels`, `renderCard`, new `placeAfter` helper) — build, style and tests untouched.

## Symptom

On a long model list, scrolling down and clicking any button on a model card
(Load/Unload, Edit, Vision, cURL, Copy ID, Chat, States) made the page jump back up to
the top of the model grid. Cards near the top looked unaffected because their viewport
position and the jump target nearly coincided.

The jump did not require the clicked action to touch the server: `Copy ID` and `cURL`
reproduced it just as reliably as `Load`. It also reproduced without a second click — a
click focuses the button, and the next 1 Hz `/ws` render moved the page on its own.

## Root cause

`renderModels()` unconditionally re-appended every card and every group section on each
render:

```js
g.grid.appendChild(renderCard(m, ...));   // every card, every pass
container.appendChild(g.section);         // every group, every pass
```

`appendChild` on an already-connected node is a **remove + insert**, even when the node is
already the last child. It therefore:

1. blurs the focused element inside the moved subtree (a mouse click leaves the card
   button focused), and
2. makes Chromium re-run scroll anchoring over the moved cards.

Isolation on the real page (`scrollY = 1895` before each trial):

| operation | scroll after |
| --- | --- |
| `focus()` only | 1895 |
| `renderModels()` only | 1895 |
| `focus()` + `renderModels()` | **432** |
| `focus()` + re-append all cards | **432** |
| `focus()` + re-append only sections | 1895 |
| `focus()`, blur, `renderModels()` | 1895 |

So the trigger is the focused card button *plus* the mass card re-append; neither alone
moves the page. The landing position (432 in the repro) is the top of the model grid, not
`scrollY = 0`.

## Fix (implemented)

Added a position-aware placement helper and used it for both cards and sections:

```js
// Insert `node` after `prev` in `parent` ONLY when it is not already in that position.
function placeAfter(parent, node, prev) {
  const next = prev ? prev.nextElementSibling : parent.firstElementChild;
  if (next !== node) parent.insertBefore(node, next);
}
```

`renderModels()` now tracks `prevCard` per group and `prevSection` (starting at the
`#models-grid` placeholder) and calls `placeAfter`. A render whose order has not changed
performs no DOM moves, so the focused button survives and the viewport is untouched. New
cards and genuine reorders still work, because `insertBefore` covers both insertion and
moves.

## Verification

- Headless Chrome (CDP, real `static/index.html` + `static/app.js`, stubbed `/v1/*` + WS):
  49 real mouse clicks across 8 cards (load, unload, edit, toggle-vision, cURL, copy-id,
  chat, states) with one `/ws` tick after each click.
  Before: **36/49** changed scroll (e.g. `4417 → 432`, delta −3985; `1895 → 432`, delta
  −1463), focus fell back to `BODY`. After: **0/49** changed scroll, focus retained on the
  clicked button.
- Ordering invariants over the same harness: baseline, `modelsCache.reverse()`,
  deterministic shuffle, type filter on/off, a model changing `model_type`, and text
  filter on/off all keep sections in `TYPE_ORDER`, cards in `modelsCache` order, with no
  duplicates, omissions, or stale cards.
- `node --check static/app.js` clean.
- `python -m pytest tests/ test_server.py -q` → **517 passed, 3 errors**; all three are
  `tests/test_real_model.py` setup errors for a missing
  `backends/llama.cpp-win-x86_64-nvidia-cuda12-avx2-local/llama-server.exe`, unrelated to
  this change.

## Related

- [[webui-card-render-scroll-jump]] — lesson documenting the Chromium focus /
  scroll-anchoring behavior and the `placeAfter` pattern.
- [[webui-appjs-render-drift]] — earlier in-place render refactor; the removed
  "appendChild MOVES an existing node" comment endorsed the unconditional re-append that
  caused this bug.
