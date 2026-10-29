---
tags: [slot-state, retention, gc, states-dir, kv-snapshot]
aliases: [state-file-retention, states-bin-orphaning, save-state-max-keep]
related: [[bridge-card-auto-save-state-must-be-forced-off]]
confidence: verified
author: researcher_ninfer-state-retention-research_20261025_144538.jsonl
---

# llama-autoloader slot-state save/restore lifecycle + retention gap

Retention **already exists** but is per-model, count-based, and skipped on the auto-save path. The
16 GB / 17-file accumulation is NOT "no cleanup" — it is per-model `keep-5` × large per-file KV
sizes, plus an uncleaned auto-save path and no global size budget.

Pins: `llama-autoloader` @ `997e562`; `ninfer-all` @ `7c88d180`.

## 1. Where state files are written + filename derivation
- Dir: `ModelManager.__init__` → `self.save_state_dir = Path(cfg["models"].get("save_state_dir","./states"))` (`server.py:980`).
- Filename template: `f"{mid_clean}.{label_clean}.bin"` — written at `save_state` (`server.py:3736`),
  `load_state` (`:3808`), delete (`:3846`). The auto-restore (`:2992`) and auto-save-on-unload
  (`:3195`) paths use a DISTINCT hardcoded form `f"{mid_clean}.auto.bin"` (fixed `auto` label), not
  the `{label}` template.
- `<model-id>` = `self._sanitize_model_id(model_id)` (model_id includes the `.gguf` extension, so
  the real prefix is e.g. `qwen3.8-...gguf`; the observed names drop it for display).
- `<client-identity>` = `self._sanitize_label(label)`. Explicit `save_state(model_id, label)` passes the
  caller's label (e.g. `Maine`, `ninfer-prefill-design`); the **auto-save-on-unload** path always uses
  the fixed label `auto` → one `<model>.auto.bin` per model (overwrites, never accumulates).
- Engine side: llama-server gets `--slot-save-path <dir>` (`to_launch_args`, `server.py:616-617`) but
  **NOT** `--auto-save-evicted`, so the engine only writes when the autoloader POSTs
  `.../slots/0?action=save&filename=...`. The autoloader is the sole authority over `.bin` naming.

## 2. How restore matches a file to the current client
- `load_state(model_id_input, label)` rebuilds the SAME path `f"{mid_clean}.{label_clean}.bin"`
  (`server.py:3806-3808`) and calls `_perform_slot_restore` (`:3719`, raises FileNotFoundError if absent).
- Auto-restore-on-load (`:2990-3008`) only ever restores the fixed `<model>.auto.bin` if it exists.
- **Restore-safety invariant:** a file is needed later iff a client will re-attach with the SAME
  `(model_id, label)`. There is NO session/client registry (see §5), so the only safe proxy is recency
  (mtime) — never delete a recent file.

## 3. Existing retention / cleanup (there IS some)
- `_cleanup_old_states(model_id_input)` (`server.py:3857-3898`): per-model, keeps newest
  `save_state_max_keep` (default **5**, `server.py:984`, validated `>=1` at `:1204-1206`) by
  `st_mtime`, deletes the rest via `unlink()`.
- **Only ONE call site:** `save_state` (`:3758`). NOT called on the auto-save-on-unload path
  (`_unload_finish`, `:3192-3199`), NOT at startup, NOT periodically. Confirmed by whole-file grep
  (`_cleanup_old_states` appears only at `:3758` + `:3857` + tests).
- `_cleanup_state_temps` (`:3631`, called from `_perform_slot_save` `:3715`) only reclaims interrupted
  `*.bin.<hash>.tmp` scratch files (`STATE_TEMP_MAX_AGE_S=3600`, `:771`); comment at `:770` "Nothing
  else ever reclaims them" refers to `.tmp`, NOT `.bin`.
- No total-size cap, no TTL, no age check on `.bin`.

## 4. Per-file SIZE driver (engine session_snapshot format)
`ninfer-all` @ `7c88d180`: `src/models/qwen3_5/program/storage/session_snapshot.cpp` (header comment
`:25-38`) + `include/ninfer/engine.h:219-230` (`save_slot`/`restore_slot`).
- The `.bin` is the **retained session** = resident prefix ledger + digests + checkpoint directory +
  a deduplicated table of **StateImages** (the source's term; these are qwen3_5's recurrent/linear-attention
  state images — GDN is a plausible but uncited label, do not assert it) + **paged Text and backend KV payload**.
- **Weights are NOT included** (re-uploaded on `resume()`; `engine.h:205-207`).
- KV payload is bounded by the **committed/executed frontier** (`:31` "covers every checkpoint frontier,
  because every checkpoint lies at or below the committed frontier"; `text_committed_frontier`/
  `backend_committed_frontier` `:249-250`). So size ∝ **used/retained context length**, NOT full
  `--ctx-size` capacity. That is why large-context sessions produce 5 GB+ files and short ones ~672 MB.
- Consequence: size is only reducible by shortening the retained prefix (undesirable); the practical
  lever is **count/total-size management**, not per-file shrink.

## 5. Active vs orphaned client
- **No active-client / session-id set exists.** Whole-file grep for `self.clients|client_id|session_id|
  ws_clients|active_client` returns **zero matches repo-wide**. `tcp_forwarder.py` tracks only raw
  `client_sock` socket handles (e.g. `:660`, `:769`) — connection plumbing, not a client/session registry.
  The `/ws` is a backend passthrough (see [[forwarder-ws-passthrough-regression]]). The autoloader cannot
  tell which `(model,label)` a live client will restore.
- Therefore option (c) "keep only active-client states" is **not implementable** without first adding a
  client registry. The only safe signals are mtime (recency) and file size.

## Recommended policy
Layer a **global total-size budget with a per-model recency floor** on top of the existing keep-N:
1. Keep the existing per-model `keep-N` (newest N per model) as an untouchable floor — it is what
   protects restore-safety for recent sessions.
2. Add `models.save_state_max_bytes` (default e.g. 8 GiB). After any save, if total `.bin` size over
   budget, evict **oldest-first by mtime across all models**, but never below each model's keep-N floor
   and never the file just written. This is option (d) made safe by the §2 invariant.
3. Run it on **write-time** (add a `_enforce_state_budget()` next to `_cleanup_old_states` at
   `:3899`, call from `save_state` after `:3758` AND from `_unload_finish` after `:3199`) — the
   auto-save path currently skips ALL cleanup, so write-time-only on save would never touch `.auto.bin`
   churn.
4. Add a one-time **startup sweep** (call `_enforce_state_budget()` in `ModelManager.__init__` after
   `:985`) to reclaim the already-orphaned 16 GB, since no future save may touch those models.
Lowering `save_state_max_keep` alone is the cheapest interim knob (already validated) but is
per-model and still unbounded globally; the size budget is the robust fix.
