---
tags: [documentation, comments, dangling-citations, repo-layout]
aliases: [no-reports-dir, reports-md-citations]
related: [[strata-restore-fifo-ordering-gap]], [[httpx-dead-socket-exception-signals]]
confidence: verified
author: safe-mode-autoloader-fixes
---

# There is no `reports/` directory in llama-autoloader — `reports/*.md` citations are unresolvable

**Fact:** code comments in this repo cite `reports/<name>.md §N`, but no `reports/` directory exists anywhere in the tree. Such citations point at a document that never shipped with the code, so a reader cannot follow them.

**Proof method (re-checkable):** recursive `list_dir` of `*.md` over the whole repo (only `README.md`, `DOCUMENTATION.md`, `tests/README.md`, `backends/README.md`, `temp/*_PLAN.md`, `.agent_lessons/*.md` exist) plus a whole-repo grep for `reports/`.

**Instances (working tree at HEAD f3d7f586b27d7100c18ccf1c0bd7528c3763e584; symbol-anchored, re-grep before trusting line numbers):**
- `model_queue.py` module docstring: "Design reference: reports/strata-restore-fifo-design.md §3.5 (ticket-based v2)." — still dangling, NOT fixed (different subsystem).
- `tests/test_model_queue.py` module docstring: "The original design (reports/strata-restore-fifo-design.md §3.5) appended a …" — same dangling doc, NOT fixed.
- `server.py` slot-action constants-block header: cited `reports/SHELL_CMD_SAFE_MODE_EARLY_RELOAD_INVESTIGATION.md §8 F4/F5` — removed 2026-10-19, header now reads "Slot-action sequencing bounds and state-temp cleanup (safe-mode ordering contract)".

**Verification grep that works:** `\bF4b?\b|\bF5\b|reports/` over `*.py`. The narrower `^\s*# ?F[0-9]` gives a **false clean** — it only catches line-leading labels and misses embedded `(F4):` / `(F4b)` / `symptom F4 exists` forms.

**Consequence for F-labels:** the identifiers `F4`, `F4b`, `F5` were defined *only* in that missing report, so every one of them is meaningless to a reader of the code alone. On 2026-10-19 all were folded into descriptive text: `server.py` (constants-block header, `_inflight_slot_actions` field comment, `_recent_unload` field comment, load-in-flight cancellation comment, unload release-recording comment, unload-drain comment, `load_state` 409-refusal comment) and `tests/test_safe_mode_slot_ordering.py` (module docstring, 4 section banners, one test docstring). After that change the only remaining hits for that grep are the two `model_queue` docstrings above.

**Where the real design docs live:** `.agent_lessons/` — e.g. the FIFO ticket design behind `model_queue.py` is [[strata-restore-fifo-ordering-gap]], not `reports/strata-restore-fifo-design.md`.
