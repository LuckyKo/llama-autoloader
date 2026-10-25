---
tags: [unload, save-state, wip, test-failures, out-of-scope]
aliases: [perform-slot-save-model-id, safe-mode-test-reds]
related: [[bridge-card-model-id-vs-overlay-key-invariant]]
confidence: verified
author: coder_bug0014_fix_20261025_124244.jsonl
---

# 4 test_safe_mode_slot_ordering.py reds are pre-existing WIP signature drift, not a regression

At `llama-autoloader` working tree on top of `b97491c74f16c12c14f571cb378bcc624d789d58` (uncommitted bridge/ninfer WIP):

`save_state` calls `self._perform_slot_save(lm.port, state_path, timeout=..., model_id=model_id)`, but the test
helpers in `tests/test_safe_mode_slot_ordering.py` (`slow_slot_save._save`, `very_slow_save`, `save`) do NOT accept a
`model_id` kwarg → `TypeError: ... got an unexpected keyword argument 'model_id'`. This reddens exactly 4 tests:

- `TestUnloadDrain::test_unload_waits_for_granted_slot_action`
- `TestUnloadDrain::test_drain_cap_proceeds_with_kill`
- `TestUnloadDrain::test_drain_covers_queue_handoff`
- `TestUnloadInProgressRefusal::test_flag_lifetime_is_exactly_the_teardown`

**Proof they are not caused by an unrelated edit:** swapping in the pre-edit `server.py` (the tool's earliest
per-edit backup, which lacks the new code under review) reproduces the identical 4 failures →
`4 failed, 9 passed` in that file. See [[prove-failure-preexisting-via-tool-file-backup-baseline]].

Do NOT chase these when your task is unload/recovery/safe-mode-logic: they are the WIP's `_perform_slot_save`
signature mismatch, fixed by the bridge/ninfer work, not by any unload fix. A full `test_safe_mode_slot_ordering.py`
run in this tree is expected to show these 4 reds until the WIP lands.
