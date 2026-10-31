---
tags: [llama-autoloader, aba-switch, lru-eviction, slot-save-restore, ninfer-cards, lesson-corrections]
aliases: [autoloader-aba-switch-facts, slot-action-engine-error-now-terminal, ninfer-dflash2-persistence-supported]
related: ["[[lru-eviction-result-check-and-mid-load-exclusion]]", "[[bridge-card-auto-save-state-must-be-forced-off]]", "[[ninfer-bridge-engine-facts]]", "[[autoloader-slot-action-error-observability-gap]]", "[[autoloader-unload-drain-inflight-slot-action]]", "[[ninfer-launch-cards-configured-26600be6]]"]
confidence: verified
author: researcher_autoloader-reader_20261031_005348.jsonl
---

# A→B→A model switching on the autoloader side: cold reload always; three earlier lessons are now stale

All line numbers below are the CURRENT working tree at `llama-autoloader` @ HEAD `e900a1506a20fc83c2774bf04b5fab346f5b7336`
(tree dirty only in `.agent_lessons/memory_stats.json`, so `server.py` on disk == HEAD — line numbers are valid for both).
**Later on 2026-10-31** the working tree also gained the BUG_0017 hard-cap fix (`server.py` +11 lines after `:2956`);
line numbers below are HEAD-relative and shift by +11 from that point onward.

## 1. There is no park/sleep/keep-alive state — unload always kills the process
- No `park` / `keep_alive` / `Sleeping` concept exists anywhere in `server.py` (grep: only `asyncio.sleep` timers and
  `_LEGACY_STRATA_OVERLAY`'s `idle_unload_s` at `server.py:740`).
- `_unload_finish` (`server.py:3244`) → `_kill_proc` (`server.py:3292`) kills the whole process tree and *verifies* it.
- Therefore a request for a previously-unloaded model is **always a full cold reload**: the forwarder's
  `_resolve_target` misses (`tcp_forwarder.py:859`), it logs "Model '…' not loaded; triggering JIT load…"
  (`tcp_forwarder.py:888-890`) and `_jit_load_and_wait_target` runs `load_model` on the main loop
  (`tcp_forwarder.py:1512-1554`). The only "cache hit" is a still-resident `LoadedModel`
  (`_load_model_serialized` double-check, `server.py:2533-2535`) or joining an in-flight load task (`:2542-2551`).
- The closest thing to a warm restart is auto-restore of `<model>.auto.bin` at `server.py:3055-3073`, but it is gated on
  `(cfg.auto_save_state or self.default_auto_save_state)` AND `(cfg.is_llama or cfg.engine == "ninfer")`, and
  `default_auto_save_state` comes from `models.auto_save_state` (`server.py:1043`) = **false** in `config.yaml:73`.
  Empirical confirmation: `states/` contains **no `*.auto.bin` file at all** (only explicit `<model>.<label>.bin`
  saves made by agents), so in this production setup A→B→A is a cold reload with no state restore.

## 2. LRU eviction: result IS checked now (the old "ignored result" lesson is closed)
- `_evict_lru_if_needed` (`server.py:2342`) returns `Optional[UnloadResult]`; candidate filter excludes target,
  `pinned`, `active_requests != 0` and mid-load (`not self._is_loading(mid)`, `:2363-2368`); sorted by `last_used`
  (`:2375`); returns `await self._unload_unlocked(lru_id)` (`:2378`).
- The call site enforces it: `evict_result = await self._evict_lru_if_needed(model_id)` then
  `self._require_engine_gone(...)` (`server.py:2953-2957`); helper at `server.py:2924-2936` raises
  `RuntimeError("…: process {pid} is still running ({reason})")`. The force-reload branch uses the same helper
  (`:2949-2952`). Order inside `async with self._lock`: unload-existing → evict-LRU → allocate port → spawn →
  install `LoadedModel` → **then** `_wait_until_ready` outside the lock (`:3019-3024`).
- `max_loaded_models` is a **hard cap** as of 2026-10-31: checked once (`:2361`); when every candidate is
  pinned/busy/mid-load it warns and returns `None` (`:2369-2374`), and `_do_load_model` now refuses the load with
  `Cannot load …: max_loaded_models (n) reached and no evictable model is available` instead of proceeding over the
  cap. (Before this, the same `None` launched a second engine — see the 2026-10-31 04:33 double-strata OOM in
  [[lru-eviction-result-check-and-mid-load-exclusion]] addendum.) Only other auto-unload is `idle_reaper` on
  `idle_timeout_seconds` (`server.py:4275-4291`; 3600 s in `config.yaml:74`).

## 3. Slot actions: the engine's own error is no longer discarded (corrects the observability-gap lesson)
- `_perform_slot_action` (`server.py:3734`) probes 3 shapes: `POST /slots/0?action=<save|restore>&model=<id>` with
  `{"filename": <basename>}`, then `/slots?id_slot=0&…`, then `/slots` with `{"action","id_slot","filename"}`
  (`:3740-3750`). `&model=` is appended only when `model_id` is non-empty (`:3739`) — NInfer needs it.
- **New behaviour:** any `<500` response whose body parses as `{error:{code|message}}` (`_parse_engine_error`,
  `server.py:818-840`) is TERMINAL — logged at WARNING with code+message and raised as `EngineSlotError`
  (`server.py:3762-3772`, class at `:800-815`). `save_state`/`load_state` convert it to
  `HTTPException(e.status_code, e.message)` (`:3846-3849`, `:3928-3931`) — the engine's 4xx is forwarded verbatim,
  not degraded to 500. Bare 404 bodies are logged truncated (`r.text[:500]`, `:3773`) and the **first** error is kept
  in the RuntimeError (`:3776-3777`, `:3803` "First error:", not "Last error:"). 5xx still breaks the loop (`:3781`);
  `_UNREACHABLE_ERRORS` breaks it too (`:3793-3801`).
- What IS still discarded: the **successful** 200 body (`return True`, `:3751-3753`) — the loader never reads the
  engine's slot metadata. `--slot-save-path` for llama comes from `to_launch_args` (`server.py:616-617`); for NInfer it
  is a literal in the engine argv template (`config.yaml:45-46`), pointing at `models.save_state_dir` (`config.yaml:77`).

## 4. NInfer cards: what is actually passed (and what is NOT)
- Engine registry: `config.yaml:35-52` — argv template `ninfer-serve.exe {artifact} --port {port} --host 127.0.0.1
  --model-id {model_id} --slot-save-path <states dir>`, `configs_dir: N:/work/WD/ninfer-all/ninfer-cards`,
  `config_key: artifact`, `derive_config: false`, `ready_timeout_seconds: 600`. No `min_free_vram_flag` for ninfer.
- Engine configs (3): `ninfer-cards/qwen38-27b-unc-dflash2.json`, `qwen38-27b-swift15abl-nvfp4full-dflash2.json`,
  `qwen38-27b-ablit-iq3xxs-mtp.json` — each is ONLY `artifact` + `model_name` (serve_args would be dead metadata).
- Overlay cards (3) in `llama-autoloader/engine_cards/` of the same stems: `extra_args` carry
  `--vision --max-context 120000|90112 --devices 0,1|0 --gdn-state-fp16 --kv-dtype rk8v4 --spec dflash2|mtp
  --draft-tokens 7 [--lm-head-draft] --wddm-evictable-budget --device-state-slots 4|8 --kv-capacity <same as max-context>
  --value-aware-demote --recency-eviction --host-state-slots 25 [--branch-anchors] [--prefill-cublas --prefill-chunk 4096
  --prefill-microchunk-tokens 1024] --stage-layers 30,34`.
  `--max-concurrency 1` appears ONLY on the swift15abl nvfp4 card (`engine_cards/qwen38-27b-swift15abl-nvfp4full-dflash2.json:21-22`).
- **NOT present in any card or config:** `--disk-kv-path`, `--disk-kv-gib`, `--disk-kv-restore`, `--disk-kv-directstorage`,
  `--slot-save-path` (whole-tree grep: only hits are the engine's own `--help` text echoed into rotated logs, e.g.
  `logs/console.log.4:2685-2691`). NInfer's disk tier is therefore OFF: `disk_kv_path` defaults empty and the other
  disk-kv flags without a path are a hard `invalid_argument` (`ninfer-all/src/serve/serve_options.cpp:939-957, 1764-1767`).
- `auto_save_state` is **forced False** for every bridge card (`server.py:1622`), and cannot be flipped by a card file
  (see [[bridge-card-auto-save-state-must-be-forced-off]]); the global `models.auto_save_state: false` (`config.yaml:73`)
  keeps auto-save off for llama cards too. Verified live: no `*.auto.bin` in `states/`.
- Live argv proof (matches the cards exactly): `logs/console.log:177-178` and `:1468-1469` —
  `Launching ninfer for qwen38-27b-swift15abl-nvfp4full-dflash2 on port 9025` + full argv.

## 5. Lesson corrections
- **[[ninfer-bridge-engine-facts]] addendum is stale**: "session persistence does not support the DFlash/DFlash2 backends"
  is gone from the engine tree — grep for that string in `ninfer-all/src` returns 0 hits; `dflash_context_frontier` is now
  persisted (e.g. `src/models/qwen3_5/program/transactions/materialization.cpp:636-639`,
  `src/models/qwen3_5/program/prefix/hybrid_program.cpp:908-911`). Empirically, DFlash2 state files exist and are large:
  `states/qwen38-27b-swift15abl-nvfp4full-dflash2.*.bin` (942 MB / 617 MB / 2.0 GB, 2026-10-31). So DFlash2 save/restore
  works now; do not keep telling future agents to set `auto_save_state` off "because DFlash2 can't persist".
- **[[autoloader-slot-action-error-observability-gap]] 2026-10-27 addendum is superseded**: its "a 400 is NOT terminal,
  only the last 404 survives, body discarded" no longer holds — structured engine errors are terminal and surfaced
  (`server.py:3762-3772`), and the aggregate message says "First error" (`:3803`).
- **[[autoloader-lru-eviction-result-ignored]] is closed** — see §2 and [[lru-eviction-result-check-and-mid-load-exclusion]].
