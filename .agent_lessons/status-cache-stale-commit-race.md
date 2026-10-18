---
name: status-cache-stale-commit-race
description: The status cache is built OUTSIDE the lock across slow nvidia-smi awaits, so a load/unload landing mid-build lets the builder commit its pre-mutation snapshot with a fresh timestamp.
metadata:
  type: project
  verified: 2026-10-18
---

`ModelManager._build_status()` snapshots `self.loaded`, then
`await asyncio.gather(to_thread(self.gpus), to_thread(self.per_model_ram_vram, ...))`.
Those two `nvidia-smi` subprocess calls take 100-500ms. So a build is **in flight across
the await**, and a load/unload can complete in that window.

**The race (pre-existing, found 2026-10-18):**

1. `_status_cache_updater` starts a build; it has already snapshotted `loaded = {model}`.
2. An unload runs: pops the model and sets `self._status_cache = None`.
3. The updater finishes and writes its **pre-mutation** snapshot into `_status_cache`
   with a fresh `_status_cache_time`.
4. The next `/v1/status` sees a fresh cache and serves the stale payload —
   `models_loaded: 1` for a model that was just unloaded, until the next poll.

**Fix:** a generation counter. `_status_gen` is bumped by `_invalidate_status()` (called
under `self._lock` from the load and unload paths). Builders stamp the generation they read
before building and commit only `if gen == self._status_gen`. A discarded build leaves the
cache invalidated, so the next reader rebuilds.

**Symptom that exposed it:** `tests/test_real_model.py::test_real_unload_cleans_up`
intermittently failed with `expected 0 loaded after unload, got 1` while the log showed
`Unloading <model> (port ...)` having run correctly. It passed 3/3 in isolation, so it is
timing, not logic — do NOT "fix" that test.

**Lesson:** any cache populated by a slow await must be generation-guarded, or clearing it
from another task is not enough. Clearing (`= None`) is not atomic w.r.t. an in-flight
builder.

Related: [[asyncio-lock-blocking-io-starvation]]