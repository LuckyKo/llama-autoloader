# BUG_0020 — `slot-drain-insufficient` cannot pass: the KV pool auto-expands to fit the graft's pages

**Status:** DONE (2026-11-04) — fixed via fix option 1 (graft-in-pool) + graft-aliasing test design; scenario verified passing on GPU.
**Filed:** 2026-11-03
**Closed:** 2026-11-04 — `ninfer-all` commit `756427f6` (dev/prefill-microchunk) implements fix option 1:
`explicit_page_groups` (kv_capacity.cpp) and `startup.cpp` no longer add `resident_main_pages`
on top of the explicit token budget, so the graft's pages live *inside* the pool. Combined with the
test's graft-aliasing design (`input.options.graft = "g"` so the long session aliases the graft's 13
pinned pages and its snapshot carries them inline), the restore now throws `CapacityShortfall` (400)
after the full drain. Verified: `ninfer_qwen3_5_prefix_real_test` scenario `slot-drain-insufficient`
passes on the RTX 5060 Ti with `NINFER_TEST_GRAFT=graft_fixture\container.bin`
(n_slots=800 → G=13 pages) and the abliterated-swift15-mtp artifact; drain reclaimed 1 parked owner.
The `run_drain_insufficient.txt` failure in `graft_fixture/` is stale (11/03, pre-fix).
**Found in:** Tier-1 polish of the 9-file `CapacityShortfall`/full-drain change. We added `tools/make_graft_fixture.cpp` to produce the `direct_kv` graft the scenario needs, but the scenario still fails.
**Component:** `src/runtime/engine/kv_capacity.cpp` `explicit_page_groups` (44–55). **Code lives in `ninfer-all`** (no local `.bug_tracker`); filed here.

## Why the scenario can't pass

Premise: a pinned graft keeps its KV in the pool even after the full drain, so a snapshot needing more free KV than the drain can release still 400s.

But `explicit_page_groups` sizes the pool as `ceil(explicit_tokens / main_page_tokens) + resident_main_pages` page groups, where `resident_main_pages = ceil(n_slots / main_page_tokens)` is the graft's pages. The graft's pages are **added to** the pool (per `resources.h:404` "an explicit capacity asks for them on top of the requested tokens"), not injected into it. So free KV after the drain is always `>= explicit_tokens` (since `resident_main_pages * main_page_tokens >= n_slots`). With `explicit_tokens = 512` and a 400-token snapshot, free KV is always >= 512 > 400 → the snapshot always fits → no `CapacityShortfall`.

**Empirical:** `n_slots` ∈ {64, 128, 256, 384, 448, 480, 500, 504, 508, 511} all fail with "the graft may not have filled enough KV"; `n_slots` ∈ {512, 1024} fail at engine construction ("graft 'g' leaves no room for a prompt within max_context").

## Fix options
1. **Engine:** stop expanding the pool for graft pages (inject the graft's KV into the existing pool). Lets the scenario pass, but shrinks the pool for all graft configs — needs a re-baseline of the other three scenarios.
2. **Test:** smaller `kv_capacity`/`max_context` doesn't help, because `kv_capacity >= max_context >= snapshot_size` ⇒ free KV always >= snapshot_size.

The fixture tool (`tools/make_graft_fixture.cpp`, target `ninfer-make-graft-fixture`) is correct and produces a valid `direct_kv` graft (`N:\work\WD\AgentWorkspace\graft_fixture\container.bin` + `.json`); the blocker is purely the pool-sizing mechanism.

## Related
- The other three slot scenarios pass without a graft.
- `ninfer-all/.agent_lessons/slot-drain-retest-config-and-gate-ordering.md`.
