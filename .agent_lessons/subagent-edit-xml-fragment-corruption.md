# Sub-agent edits can leave tool-XML fragments hidden in comments (tests still pass)

## What happened
During the Strata bridge-card WebUI build, a sub-agent `edit_file` left literal tool-XML
fragments in `tests/test_strata_adapter.py` — a line read:

```python
assert "engine" not in data   # engine is derived at scan time, never written</new_string><justification>Update exact-dict assertion for B2</justification>
```

Because the junk was appended **after a `#`**, it was a comment, so the suite passed
(223/223). The corruption was invisible to every test and to a plain read of the assertion.

## How it was caught
A POLISH-review nit flagged a stale "plan B2" comment reference. Reading that region revealed
the `</new_string>…` junk on the adjacent line. A repo-wide grep for `</new_string>|<justification>`
confirmed it was the only occurrence.

## Prevention (do this after any sub-agent code edit, before committing)
Grep the changed files for stray tool-XML / edit-tool markers:

```
grep -rE '</new_string>|<justification>|</old_content>|<new_string>' <changed-files>
```

Corruption inside comments or string literals **passes the test suite** — only a text grep
catches it. Don't rely on "tests green" as proof the edit was clean.

## Related
- [[plan-review-backend-api-mismatch]] — the B1b deviation that motivated re-reading the actual
  endpoint projection (where this corruption also lived, in the test asserting the card shape).
