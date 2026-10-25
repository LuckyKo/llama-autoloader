---
tags: [bridge-card, auto-save-state, engine-config, review-trap, dirty-tree]
aliases: [card-cannot-flip-auto-save-state, make-engine-config-force]
related: [[perform-slot-save-model-id-wip-signature-drift]], [[bridge-card-model-id-vs-overlay-key-invariant]]
confidence: verified
author: coder_bridge_greenup_20261025_132007.jsonl
---

# Bridge-card `_make_engine_config` must force `auto_save_state = False`

In `_make_engine_config` (`server.py`), the "Force AFTER the overlay" block that sets
`base.engine` / `base.engine_config` **must also set `base.auto_save_state = False`**.
A card file must never be able to opt a bridge card into per-card slot-save. The global
`default_auto_save_state` still governs at the use sites, which all apply the OR pattern
`cfg.auto_save_state or self.default_auto_save_state` (and gate on `(cfg.is_llama or cfg.engine == "ninfer")`).
Per-card opt-in is intentionally **not** supported.

`tests/test_bridge_engines.py::test_card_overlay_cannot_flip_engine` pins this: it builds a
card with `"auto_save_state": true` and asserts the resulting `ModelConfig.auto_save_state is False`.

## Why this broke once (WIP regression, not a HEAD bug)
At `b97491c7` (HEAD), `_make_engine_config` **did** contain `base.auto_save_state = False`
(`server.py:1507` @ that SHA). A parallel uncommitted refactor in the working tree **removed**
that force line and replaced the comment with "NInfer card overlays opt in with `auto_save_state`: true" —
which is exactly what broke the test. Re-adding the force line (with a corrected comment) fixes it.

## Review trap: diff against the working-tree baseline, not HEAD
A reviewer that runs `git diff` (vs HEAD) in this dirty tree saw HEAD *also* had a force line,
so it misread the re-added line as "comment-only, no production leak" and declared the fix
mislabeled. That verdict is **wrong for the handed-off working tree**, where the line was absent
and the test was failing. When the tree is dirty with other agents' WIP, judge a fix against the
**working-tree baseline / the failing test**, not against HEAD. See [[refute-agent-claimed-root-cause]]
and [[reviewer-git-diff-dirty-tree-false-scope-creep]].
