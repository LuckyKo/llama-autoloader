---
tags: [state-restore, seq_rm, kv-cache, reprocess, llama.cpp, image_gen]
aliases: [seqrm-reprocess, restored-slot-reprocess]
related: [[model-id-gguf-name-collision]]
confidence: verified
---

# Restored-slot `seq_rm` failure → full reprocess (GENERIC, not image/caption specific)

**Fact:** After a successful `state/load` (KV restore), the *next* chat request triggers
`restored slot seq_rm(<p0>, -1) returned false; ... clearing sequence and resetting cached
tokens to 0 to reprocess prompt fresh` and reprocesses the **entire** prompt (~54s for ~37k tokens).
This is **not** caused by captioning polluting slot 0, by image token mismatch, or by slot routing.
It is a **generic property** of the llama.cpp restored-slot reconciliation path.

**Evidence:**
- `server-context.cpp:3486-3504`: for a restored slot (`slot.restore_kv_valid==true`, set at :2677),
  the code calls `llama_memory_seq_rm(mem, slot.id, p0, -1)` to trim the stale tail. On failure it
  takes the designed safe fallback: full `seq_rm(0,-1)` + `slot.prompt_clear()` + `restore_kv_valid=false`
  → `n_past=0` → full reprocess.
- The line appears **~40 times** across `logs/console.log{,.1,.2,.5}` with wildly varying `p0` (3, 211,
  33199, 54054, 77854, 106542) and task ids, **including task 2 (p0=211) with NO caption between
  restore and request** (console.log.1:50842). Two message variants exist across builds:
  `reprocess prompt fresh` (current) and `clearing sequence to satisfy M-RoPE invariants` (older).
- LCP selection matches the restored slot correctly: `f_sim_best=0.959, f_keep=1.000` (console.log:6817).
  The prefix (incl. image chunks, which `server_tokens` round-trips via `server-common.cpp:298-308`)
  lines up — so it is NOT a token-position conflict. `p0` is a KV *position* count (image chunks expand
  to multiple positions via `pos_next`, `server-common.cpp:372-413`), which is why `p0=33199` < 37208 logical tokens.

**Mechanism (why the partial seq_rm fails):**
- Pure-attention `llama_kv_cache::seq_rm` returns `true` unconditionally (`llama-kv-cache.cpp:383-450`)
  → a plain attention cache would NOT produce this failure.
- The failure requires a memory backend that can fail a **partial tail-rollback**:
  `llama_memory_recurrent::seq_rm` (`llama-memory-recurrent.cpp:161-216`) only succeeds if the rollback
  distance fits the per-token snapshot buffer `n_rs_seq` (lines 194-203). A large restored prefix → large
  rollback distance → returns false. Hybrid/ISWA/DSA wrappers propagate a sub-cache `false`
  (`llama-memory-hybrid.cpp:143-149`, `llama-kv-cache-dsa.cpp:61-65`).
- Model launched with `--spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 3 --kv-unified`
  (console.log:6772). The MTP draft-spec / unified memory is the *likely* source of the recurrent sub-cache.

**Open item:** exact backend that returns false needs a runtime check (log memory type / `n_rs_seq` at :3494).

**Implication for image_gen tool (AgentCascade `tools/image_gen.py`):** the save→unload→caption→restore
ordering is CORRECT (restore-after-caption is fine; images round-trip). Do NOT chase the caption/slot
hypothesis. The only way to eliminate the reprocess across all restore flows is a llama.cpp fix at
:3494-3504 (don't full-clear when the LCP prefix actually matches). Short-term mitigation: compress the
agent context before image_gen, and route captioning to a separate small vision endpoint.

**Full report:** `N:\work\WD\AgentWorkspace\reports\image_gen_reprocess_investigation.md`
