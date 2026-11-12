# BUG_0022 — admission refuses a load that misses the free-memory test by 9 MB

**Status:** FIXED
**Filed:** 2026-11-12
**Fixed:** 2026-11-12
**Commit:** `efdd6861618ab26655b2be216a583245d2e3a5b5`
**Found in:** Live incident 2026-11-12 11:51:36 (`logs/console.log`) — JIT load of
`qwen3.8-flash-next-unsloth-ud-iq1_m` returned 503 after evicting the resident model.
**Severity:** medium — no data loss, but every model switch could fail and cascade into a
caller-side breaker trip; the resident model had already been evicted, so the instance sat
with no model loaded for ~54s.
**Component:** `server.py` (`ModelManager._make_room_for` + shortfall helpers),
`models.vram.marginal_shortfall_mb` config key.

## Symptom

```
11:51:36,257 Evicting LRU model qwen38-27b-swift15-nvfp4full-dflash2 (last used 0.3s ago)
11:51:38,466 JIT load failed for 'qwen3.8-flash-next-unsloth-ud-iq1_m':
             Cannot load ...: measured need VRAM {GPU0: 13907MB, GPU1: 14222MB} / RAM 40351MB
             exceeds available VRAM {GPU0: 15539MB, GPU1: 14213MB} / RAM 49642MB
             and no evictable model is available
```

GPU1 was short by **9 MB** of 16311 MB. The endpoint returned 503 for a model that loads
fine; the same model had already been admitted five times that day with the identical need
and available GPU1 values of 14245–14470 MB. The caller (AgentCascade) treated the 503 as
server-wide load pressure: two attempts 1.1s apart, breaker OPEN 20s, the other 1234
endpoints skipped, `ServerBusyError` retried, recovery only at 11:52:31.

## Root cause

Admission was exact-to-the-megabyte with no floor:

* `gpus_mb` in an occupancy profile is a coarse **system-used delta** captured around the
  load, so it overcounts by whatever else moved on the desktop during the window.
* `models.vram.headroom_mb` is `0` in the live config (persisted from the WebUI Global
  Settings), so no slack absorbs that drift.
* `reclaim_wait_s` is one sleep + one re-sample; WDDM may still be returning the evicted
  model's pages when the comparison runs.

Once `_pick_lru_eviction_candidate()` returned `None` (the only resident had already been
evicted for this load), the code raised unconditionally, however small the gap.

## Fix

New `models.vram.marginal_shortfall_mb` (default `64`, `0` = strict). When no evictable
candidate remains **and** `len(loaded) < max_loaded_models`, a per-card or RAM shortfall
within the tolerance is admitted with a `WARNING` that names the gap and the eviction
history; the full measured need is still reserved and the engine's own fit check remains
the final authority. The count-cap condition is checked at the call site, so the tolerance
can never raise the resident count or act as a soft cap.

The refusal message is also more accurate: it names the per-card/RAM shortfall and *why*
no resident could be evicted (`pinned.gguf (pinned)`, `busy: N active request(s)`,
`mid-load`, or `no other residents`) instead of the generic "pinned, busy, or mid-load".

## Verification

* `tests/test_memory_admission.py` — six new cases: 9 MB gap admitted after eviction (the
  incident's exact shape), gap beyond tolerance refused with the gap named, `0` strict,
  a tolerated gap still blocked by a full count cap with a pinned resident, marginal RAM
  shortfall, default value matches the module constant, invalid config rejected.
* `python -m pytest tests/test_memory_admission.py -q` → **30 passed**.
* `python -m pytest -q` → **571 passed, 4 errors**: the three
  `tests/test_real_model.py` environment gates (missing pinned CUDA backend) plus
  `test_server.py::TestCleanupOldStates::test_max_keep_one`, a Windows file-lock flake that
  passes in isolation and is unrelated to admission.

## Related

* [[autoloader-marginal-vram-admission]] — lesson with the incident numbers, the
  coarse-delta explanation, and the `PUT /v1/settings` in-memory-rewrite gotcha.
* BUG_0017 — the earlier soft-cap bug; this fix deliberately does **not** reopen it: the
  count cap stays hard.
