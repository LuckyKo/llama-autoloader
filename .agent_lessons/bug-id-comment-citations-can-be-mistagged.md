---
tags: [bug-tracker, comments, code-review, llama-autoloader]
aliases: [bug-id-mistagging, comment-cites-wrong-bug]
related: [[unload-teardown-exception-orphans-registry]], [[lru-eviction-result-check-and-mid-load-exclusion]]
confidence: verified
author: reviewer_polish-reviewer-llama_20261020_162834.jsonl
---

# Bug-ID citations in code comments can name the wrong bug

`server.py` comments justify a guard by citing a `.bug_tracker` ID. At least one such citation
names a **different** defect than the guard actually protects against. Trust the tracker entry,
not the comment.

At `llama-autoloader` @ `d977c82bd3202cdc8ed7523190be478c211e6165`:

- The `except BaseException` handler inside `_recover_dead_model_impl` (`server.py:2392`) opens
  with "**BUG_0005:** the teardown above runs OUTSIDE `self._lock`, so the pop made by
  `_begin_unload_locked` must be undone if it aborts". BUG_0005 is the **lock-scoping** defect
  (why the slow drain/kill must not hold the global lock). The defect being guarded — an aborted
  teardown orphaning a live process from `self.loaded` — is **BUG_0009**.
- Proof it is a known confusion, not my inference: `.bug_tracker/fixed/BUG_0009_unload_unlocked_exception_orphans_registry.md:90`
  states verbatim "unrelated defect, and the source of the tag confusion (this is BUG_0009, not
  BUG_0005)". The sibling handler in `_unload_unlocked` (`server.py:2682`) tags the same mechanism
  correctly as BUG_0009.
- The memory [[unload-teardown-exception-orphans-registry]] already attributes BUG_0005 correctly
  (lock scoping) — so the code comment is the outlier in this repo, not the docs.

## Why it matters

A mis-tagged comment converts an invariant into an apparently-scoped one: a reader who believes
the guard belongs to the lock-scoping fix will delete it when that refactor lands. Two handlers
for the same defect class then silently diverge again.

## Review heuristic

Before accepting a `BUG_NNNN` citation in a comment as evidence, open
`.bug_tracker/fixed/BUG_NNNN_*.md` and compare its **Component** and **Root cause** sections
against what the comment claims the guard does. Mismatch → report a comment defect and cite the
tracker line that proves it.

## Related hygiene gap (same tree, same SHA)

Two of the three fixes committed that day (`28b95ae` recovery restore, `d977c82` LRU result check
— see [[lru-eviction-result-check-and-mid-load-exclusion]]) have **no** tracker entry, while their
code comments reference defect classes. `BUG_0009`'s entry is the only one of nine missing its
`**Commit:**` field ("Fixed: … (uncommitted, on top of `03614f1`)", written ~13 min before the
commit landed). Absence of an entry is not evidence a change was reviewed; grep `git log --oneline`
for the commit and cross-check the entry's mtime against the commit timestamp.
