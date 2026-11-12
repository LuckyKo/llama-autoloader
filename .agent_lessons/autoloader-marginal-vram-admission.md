---
tags: [llama-autoloader, memory-admission, vram, eviction, occupancy-profile, 503, breaker-cascade]
aliases: [marginal-shortfall-refuses-load, admission-9mb-gap, post-eviction-refusal]
related: ["[[autoloader-lru-eviction-result-ignored]]", "[[autoloader-aba-switch-facts-2026-10-31]]"]
confidence: verified
scope: [llama-autoloader]
---

# llama-autoloader: an MB-scale post-eviction gap refused a load the engine could run

Observed live on 2026-11-12 11:51:36 (repo HEAD `033e0ac`, live PID started 10:19).

## The facts (numbers are from `logs/console.log`)

A JIT load of `qwen3.8-flash-next-unsloth-ud-iq1_m` (a Strata bridge card) evicted the
resident `qwen38-27b-swift15-nvfp4full-dflash2` (the agents' own brain), then refused:

```
measured need VRAM {GPU0: 13907MB, GPU1: 14222MB} / RAM 40351MB
exceeds available VRAM {GPU0: 15539MB, GPU1: 14213MB} / RAM 49642MB
```

GPU1 was short by **9 MB** out of 16311 MB. Nothing else was evictable (the model had
already been evicted), so `_make_room_for` raised, the JIT load returned 503, and the
caller's breaker/retry cascade began (two 1.1s-spaced attempts → breaker OPEN 20s → every
1234 endpoint skipped → `ServerBusyError` retried → recovered only 54s later).

All the *successful* admissions of the same model that day were equally marginal: available
GPU1 was 14245, 14262, 14335, 14464, 14470 MB against a need of 14222 MB. The engine itself
loads fine; only the admission arithmetic was tight.

## Why the arithmetic is tight (not a bug in the profile itself)

`gpus_mb` is a coarse **system-used delta** captured around the load: it overcounts by
whatever else moved on the desktop during the window. The residual after eviction is
therefore normally a few MB, not zero. Two things made it fatal here:

1. `models.vram.headroom_mb` is `0` in the live config (the WebUI Global Settings value is
   persisted back into `config.yaml`). With zero headroom every MB of desktop drift lands
   inside the need-vs-free comparison.
2. `reclaim_wait_s` is a single sleep + one re-sample, so the comparison happens while
   WDDM may still be handing pages back.

The refusal was absolute: once `_pick_lru_eviction_candidate` returns `None`, the old code
raised regardless of how small the gap was.

## Fix

`models.vram.marginal_shortfall_mb` (default `64`, `0` = strict). Applied **only** when

* no evictable candidate remains, **and**
* `len(self.loaded) < self.max_loaded_models` (checked at the call site, not inside the
  helper) — so it can never raise the resident count or act as a soft cap.

Admission then logs a WARNING naming the per-card gap and reserves the full measured need;
the engine's own fit check remains the final authority (it is the thing that would fail if
the tolerance were ever too generous).

## Gotcha worth remembering

* Refusal text now names the shortfall per card and *who* blocked eviction
  (`pinned.gguf (pinned)`, `busy: 1 active request(s)`, `mid-load`), instead of a generic
  "pinned, busy, or mid-load". A bare "exceeds available" made a 9 MB miss look like a
  4 GB miss in the logs.
* `PUT /v1/settings` dumps the **in-memory** `cfg` back to `config.yaml`. A key added only
  to the file is dropped by the next WebUI settings save until the process restarts; add
  new keys to the file *and* restart (or accept the code default meanwhile).
* A 503 from a per-model load failure is not "the server is out of memory": see the AC-side
  breaker cascade in the 2026-11-12 incident notes.
