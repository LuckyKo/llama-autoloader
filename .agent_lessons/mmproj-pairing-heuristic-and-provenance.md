---
name: mmproj-pairing-heuristic-and-provenance
description: Projector pairing used a positional character heuristic that scored unrelated files ABOVE correct ones, and an inferred pairing was indistinguishable from a user-set one because scan() writes the inferred name back into the sidecar field.
metadata:
  type: project
  verified: 2026-10-18
---

Adding vision provenance (which projector gives a model vision, and did we infer that or
did you?) surfaced two defects in the pairing rules that had been invisible because the
UI only ever showed a bare `vision` tag.

**1. The positional heuristic actively picks the wrong file.**

`_best_mmproj_by_name` compared model and projector stems **character by index** to find
a "common prefix". That is dominated by the `mmproj-` prefix, which shifts every
position. For `alpha.gguf` in a directory with `mmproj-alpha.gguf` and
`orphan-mmproj.gguf`:

| candidate | compared against | score |
| --- | --- | --- |
| `mmproj-alpha` | `alpha` vs `mmpro` | 1 |
| `orphan-mmproj` | `alpha` vs `orpha` | 3 |

The **orphan** won. Index 2 of `mmproj-*` is always `p`, so every projector scores at
least 1 against every model.

**Fix:** prefer `_mmproj_affinity` (strip the `mmproj-` prefix, then score exact /
containment) whenever **any** candidate has a real name relationship. Fall back to the
positional heuristic only when none does — which preserves today's answer for real
libraries whose projectors carry no model name (`mmproj-F16` / `mmproj-F32` beside one
model), where there is no better information and changing the result would alter a
working setup for no gain.

**2. Provenance is destroyed by the code that sets it.**

`scan()` writes an inferred pairing back into `cfg.mmproj_file`:

```python
if not cfg.mmproj_file:
    cfg.mmproj_file = paired_mmproj.name
```

So after one scan, "we guessed this" and "you set this" are the same string, forever.
`mmproj_source` must therefore be read **before** that line. There is a regression test
(`test_provenance_is_read_before_the_inferred_name_is_written_back`) that asserts
`cfg.mmproj_file` is populated *and* source is still `auto`.

**3. A trailing space in a hand-edited sidecar poisons the resolved Path.**

`mmproj_file: "mmproj-F16.gguf "` (trailing space) produced
`WindowsPath('.../mmproj-F16.gguf ')`, which pathlib **preserves** and which therefore
never `==` the canonical path from `rglob`. It put a stray space inside the `--mmproj`
launch arg. It only "worked" because Windows file APIs tolerate trailing spaces; on Linux
it would fail. Fixed with `.strip()` at the single point where a string becomes a Path.

**Lesson:** a heuristic that scores every candidate above zero is not a heuristic, it is
a coin flip with extra steps — and a value written back into user config as a
"convenience" erases the provenance you need to audit it later.

Symptom that exposed #1 and #3 together: a verification script reported a projector as
`UNPAIRED` when it was actually paired to a model. Both bugs were in the same pairing
path; the report was correct and the code was wrong.

Related: [[model-type-derived-vs-override-split]]