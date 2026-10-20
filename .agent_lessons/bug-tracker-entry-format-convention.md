---
tags: [bug-tracker, documentation, markdown-format, llama-autoloader]
aliases: [bug-entry-template, bug-tracker-header-convention]
related: [[unload-teardown-exception-orphans-registry]] [[lru-eviction-result-check-and-mid-load-exclusion]]
confidence: verified
author: coder_polish-coder-d_20261020_174404.jsonl
---

# Canonical `.bug_tracker/fixed/` entry format (llama-autoloader)

All entries in `.bug_tracker/fixed/` follow one template. Follow it when filing a new
BUG_XXXX entry; the format was normalized across all entries on 2026-10-20.

## Front-matter fields (in this exact order, bold labels, no blank lines between)

```
# BUG_XXXX — <one-line symptom title>

**Status:** FIXED
**Filed:** YYYY-MM-DD
**Fixed:** YYYY-MM-DD
**Commit:** `<full-or-7-char SHA>`
**Found in:** <how the bug was discovered — production log w/ timestamp + todo.md line range, or "Post-fix audit of BUG_NNNN (`<sha>`)">
**Severity:** <low|low-medium|medium|medium-high|high>
**Component:** `file.py` (`symbol`, `symbol`) — symbol-anchored, not line numbers
```

- `**Fixed:**` carries **only** the date. Commit hash goes in a separate `**Commit:**` field. (BUG_0009 previously conflated them as "date (uncommitted, on top of `sha`)" — corrected.)

## Section headers (H2) — use these exact strings

- `## Symptom`
- `## Root cause`
- `## Fix (implemented)`  ← **not** `## Fix`, **not** `## Fix implemented`
- `## Verification`  ← fold all test names / test-file refs **here**; there is **no separate `## Tests` section**
- `## Related` — backlinks to sibling bugs and `[[memory-name]]` lesson links

Optional design sections that pre-date the fix are allowed and are NOT renamed: `## Fix direction` (proposed approach), `## Evidence`, `## Residuals (deliberately out of scope here)`. Only the *implemented-fix* header is normalized to `## Fix (implemented)`.

## Cross-reference style

`## Related` lists sibling bug files by filename (`BUG_00NN_<slug>.md`) with a one-line note, plus `[[obsidian-backlink]]` to the matching `.agent_lessons/` memory.

## Note on the task that normalized this

Polish pass 2026-10-20: normalized `## Fix implemented`→`## Fix (implemented)` in BUG_0002/0006/0007/0008, `## Fix`→`## Fix (implemented)` in BUG_0009, and folded BUG_0008's `## Tests` section into `## Verification` (header-only, content preserved). Added BUG_0010 (recovery teardown orphan, `28b95ae`) and BUG_0011 (LRU eviction result dropped + mid-load cascade, `d977c82`).
