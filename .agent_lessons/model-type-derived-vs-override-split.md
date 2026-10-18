---
name: model-type-derived-vs-override-split
description: Keep a card's DETECTED model type and the user's type PIN in separate fields, or a mere auto-detection silently becomes a permanent override on the first sidecar save.
metadata:
  type: project
  verified: 2026-10-18
---

`ModelConfig` carries two separate fields for a card's type, and conflating them was a
real bug found by a test while building model-type grouping:

- `model_type` — DERIVED. The effective type; recomputed by `_apply_gguf_info()` on every
  scan and on every config save.
- `type_override` — USER DATA. Persisted in the sidecar / bridge card. Empty = follow
  auto-detection.

**Why the split matters.** With one field, `scan()` writes the auto-detected value onto the
config; the next unrelated sidecar save (`save()` → `asdict()`) persisted it to disk; the
next `load()` read it back as if the user had chosen it. From then on the UI showed a
"pinned ✎" badge on a type nobody pinned, and changing the detection table could never
move the card again. Two tests lock this down:
`test_auto_detected_type_is_not_persisted_as_an_override` and
`test_sidecar_override_survives_rescan`.

**Rule:** any field that is recomputed from the file must be excluded from persistence.
`ModelConfig.DERIVED_FIELDS = {"arch", "model_type"}` is the single list that does this —
add to it, don't sprinkle `save()` special cases.

**API asymmetry (intentional):** the WebUI / API field is `model_type` (what the user is
"setting"), but it writes `type_override`. The handler therefore `setattr`s
`type_override` and `continue`s — never `setattr(cfg, "model_type", ...)`. The effective
type is re-derived in `ModelManager.update_config`.

**Reset semantics:** `""` means "go back to auto", and it takes effect on the very next
render because `update_config` re-runs classification instead of waiting for a rescan.

Related: [[autoloader-registry-is-gguf-shaped]], [[webui-appjs-render-drift]]