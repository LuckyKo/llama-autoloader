---
name: update-config-two-lock-name-race
description: Pre-existing non-atomic name-uniqueness window in ModelManager.update_config (server.py)
tags: [llama-autoloader, server, concurrency, mmproj]
---

## Finding (pre-existing, NOT introduced by the mmproj save-before-pairing fix)

`ModelManager.update_config()` in `server.py` uses **two separate** `async with self._lock` blocks:

1. First lock: validates `mid in self.models`, reads `path`, runs the name-uniqueness guard against `self.models`.
2. Outside lock: `await asyncio.to_thread(new_cfg.save, path)` (and now the mmproj pairing/fill).
3. Second lock: commits `self.models[mid] = new_cfg` and syncs `self.mmproj_paths`.

Because the uniqueness check and the commit are in different critical sections, two concurrent
`update_config` calls assigning the **same name to two different models** can both pass the guard
and both commit → duplicate retrieval-key names. The model-existence assumption is likewise not
re-verified under the second lock.

This window existed **before** the mmproj fix (the save() call already split the two locks).
The mmproj reordering only moved pure reads (`_list_mmproj_files` / `_pair_mmproj`) into the gap;
they mutate no shared state, so they add no new race.

## If it ever needs fixing
- Re-check uniqueness under the second lock immediately before committing (two-phase), or
- Move pairing+save inside a single critical section (accepts I/O blocking on the lock).

Related: [[mmproj-pairing-determinism]]
