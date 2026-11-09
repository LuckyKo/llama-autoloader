---
tags: [webui, app.js, scroll, focus, dom-reorder, chromium, renderModels]
aliases: [webui-scroll-jump, card-button-scroll-reset, appendchild-scroll-anchoring]
confidence: verified
---

# WebUI: re-appending already-placed cards reset the page scroll — 2026-11-09

Clicking any button on a model card could make the page jump back up to the top of the
models section. The jump was NOT caused by the click handler itself: it needed a focused
card button + a `renderModels()` run, so the 1 Hz `/ws` tick alone triggered it within a
second of the click. Cards near the top looked fine; cards further down appeared to
"reset the page scroll".

## Root cause

`renderModels()` unconditionally did:

```js
g.grid.appendChild(renderCard(m, ...));   // every card, every render
container.appendChild(g.section);         // every group, every render
```

`appendChild` on an already-connected node is a **remove + insert**, even when the node is
already the last child. It therefore:

1. blurs the focused element inside the moved subtree (a click leaves the card button
   focused), and
2. makes Chromium re-run scroll anchoring over the moved cards.

The result is a scroll jump (in the repro: `scrollY 1895 → 432`, i.e. the top of the model
grid), plus lost button focus. A synthetic page only reproduced it when *both* a focused
card button and the mass re-append happened; `renderModels()` alone or `focus()` alone did
not.

## Fix (implemented)

`static/app.js` `renderModels()` now uses `placeAfter(parent, node, prev)`, which only
calls `insertBefore` when the node is not already immediately after `prev`:

```js
function placeAfter(parent, node, prev) {
  const next = prev ? prev.nextElementSibling : parent.firstElementChild;
  if (next !== node) parent.insertBefore(node, next);
}
```

Sections are synced after the `#models-grid` placeholder (`prevSection = emptySlot`) and
cards after `prevCard`; a render with unchanged order now performs zero DOM moves. New
cards and genuine reorders still work because `insertBefore` handles both.

Do **not** revert to unconditional `appendChild` for "cheap" in-place updates — the DOM
insertion is cheap, but it is not side-effect free while a child has focus.

## Verification

- Headless Chrome (CDP, real `static/index.html` + `static/app.js`, stubbed `/v1/*` + WS):
  49 real mouse clicks across 8 cards (load/unload, edit, toggle-vision, cURL, copy-id,
  chat, states) with a WS tick after each: pre-fix 36/49 reset scroll (e.g. −3985 px),
  post-fix **0/49**, focus retained on the clicked button.
- Ordering invariants: baseline, `modelsCache.reverse()`, shuffle, type filter on/off, a
  model changing `model_type`, and text filter on/off all keep sections in `TYPE_ORDER`
  and cards in `modelsCache` order with no duplicates/omissions.
- `node --check static/app.js` clean; `python -m pytest tests/ test_server.py -q` →
  517 passed, 3 env errors (missing real `llama-server.exe` in `backends/...`, unrelated).

## Related

- [[BUG_0021]] — the filed bug entry (`f3d39a6`).
- [[webui-appjs-render-drift]] — earlier in-place-render refactor; the removed
  "appendChild MOVES an existing node" comment endorsed the pattern that caused this.
