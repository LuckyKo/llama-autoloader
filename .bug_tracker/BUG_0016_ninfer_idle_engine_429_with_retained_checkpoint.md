# BUG_0016 — NInfer rejects a request with 429 `context cache exhausted: an idle Engine found no placement` while an idle retained checkpoint still owns KV pages

**Status:** FIXED (upstream) — fix landed in `ninfer-all` commit `ef6d8d3e` (branch `dev/prefill-microchunk`, 2026-10-29). The `candidates.empty()` liveness regression was removed (root candidate always appended with its real `root_candidate_index`), a `BlockReason` now distinguishes a genuine capacity floor (`NoReclaimableOwner`) from a reclaimable no-placement block (`NoFeasiblePlacement`), and `try_admit_one` performs at most one bounded thorough re-search only when an evictable owner exists. Verified: `ctest -R "resource_manager|admission_policy|context_store|host_kv|pressure|materialization|ladder|arena"` 8/8 pass; GPU integration test skips without CUDA.
**Still open (unchanged):** the residual production symptom may partly be a capacity/ownership floor (the device-only baseline page `b7208203` names, occupancy floors at 1). For that case the fixed path now yields a *clear* refusal (`NoReclaimableOwner`) instead of the misleading "idle Engine found no placement" 429 — but it does not make the request fit. Confirming whether the 2026-10-29 04:42 failure was that floor vs. a search-budget bail needs a traced GPU reproduction (`--checkpoint-trace`, `NINFER_MAT_DEBUG=1`, A1→A2→B→A2+ on the `qwen38-27b-swift15abl-nvfp4full-dflash2` card). If it is a true floor, the remaining fix is client-side `max_tokens` sizing (see the now-documented reservation semantics). See `N:\work\WD\ninfer-all\.agent_lessons\bug16_fix_plan.md` for the full verified analysis.
**Filed:** 2026-10-29
**Found in:** Production log 2026-10-29 04:29–04:42 (`N:\work\stuff\Beta\llama-autoloader\logs\console.log`); the same failure is present on 2026-10-26 14:53 and 2026-10-27 07:33 / 11:11 / 11:35 (`logs/console.log.1`, `logs/console.log.2`). Engine build `2026-10-26.2+2026-10-26.2-45-g26600be6`.
**Severity:** high — a follow-up agent turn is rejected in ~4 ms and every retry fails the same way; the conversation is only unstuck when another model is loaded and the whole engine is torn down.
**Component:** `ninfer-all` — `src/runtime/engine/engine_core.h` (`admit_pending`, `inspect_admission`), `src/runtime/engine/context_cache/resource_manager.h` (`inspect`, `plan_materialization`), `docs/serving.md` (Default output limit)

## Symptom

A request is accepted by HTTP and then fails during generation:

```
04:42:25,403 LLM POST /v1/chat/completions: 'model' extracted after extra read (81917B)
04:42:25,469 [qwen38-27b-swift15abl-nvfp4full-dflash2] req#3 started
             | openai-chat stream | 25 messages | max output 99,155 | thinking medium | tools 27
04:42:25,473 [qwen38-27b-swift15abl-nvfp4full-dflash2] req#3 failed during generation
             | openai-chat | HTTP 429 | server overloaded
             | context cache exhausted: an idle Engine found no placement
04:42:26,733 [qwen38-27b-swift15abl-nvfp4full-dflash2] req#4 started (identical retry)
04:42:26,737 [qwen38-27b-swift15abl-nvfp4full-dflash2] req#4 failed during generation
             | HTTP 429 | context cache exhausted: an idle Engine found no placement
04:42:26,855 Auto-evicting LRU model qwen38-27b-swift15abl-nvfp4full-dflash2 (last used 0.1s ago)
```

The engine is healthy (`/health` passes, no crash, prompt preparation succeeds); only admission fails. Because the 429 arrives after the prompt is parsed, the autoloader logs it as a generation failure rather than a transport error, which makes it look like a "model crash".

## Reproduction pattern (deterministic in the logs)

Every occurrence is: a short "side" request whose output budget is the *entire remaining context* finishes; the next, larger request arrives immediately after and is rejected.

| Time | Previous request | Follow-up request | Result |
|---|---|---|---|
| 04:42:17 → :25 | 2 msgs, prompt 6,222, **max output 113,779** (`6,222 + 113,778 = 120,000` = full KV pool) | 25 msgs, **max output 99,155** (prompt ≈ 20.5K, as req#1's 23-msg prompt was 20,459) | 429, retry 429 |
| 04:31:38 → :46 | 2 msgs, prompt 6,193, **max output 113,808** | 19 msgs, **max output 102,044** | 429, retry 429 |
| 04:29:30 → :46 | 2 msgs, prompt 6,023, **max output 113,978** | 6 msgs, **max output 107,751** | 429, retry 429 |
| 10-26 14:53:40 → :41 | 5,829-token side request finishes | 97 msgs, **max output 42,064** | 429 ×10 (req#9–#18) |

The engine config for the affected card (`engine_cards/qwen38-27b-swift15abl-nvfp4full-dflash2.json`) is `--max-context 120000 --kv-capacity 120000 --device-state-slots 4 --host-state-slots 25 --spec dflash2 --branch-anchors --value-aware-demote --recency-eviction`.

Arithmetic for the first row: the follow-up's own entitlement is ~119.7K (prompt + output) and fits the 120K pool, but the previous request's 6,222-token retained session overlaps it; ~125.9K > 120,000. Retrying reaches the same retained state, so it fails identically.

## Root cause

NInfer reserves KV for the prompt **plus the full output budget** (plus speculative/draft KV). `docs/serving.md` §"Default output limit" (:2694) says an omitted limit receives "the largest output whose admission entitlement … fits one lane's share … clamped to the remaining context". Clients that send `max_tokens = context_window - prompt_tokens` therefore reserve essentially the *whole* KV pool on every turn. After the request completes, its checkpoint/continuation remains cached, still owning those pages.

Admission then hits this branch (`engine_core.h:2777-2790`):

```cpp
const ActiveAdmissionSet active = scheduler_.active_admission_set(slots_, max_concurrency_);
if (active.size == 0) {
    // No active lane will release anything, so the head cannot wait its way in.
    ++cumulative_stats_.context_cache_exhausted_requests;
    consecutive_context_cache_exhaustions_.fetch_add(1, std::memory_order_relaxed);
    (void)remove_pending_error(head, std::make_exception_ptr(RequestError(
              RequestErrorKind::Overloaded,
              "context cache exhausted: an idle Engine found no placement")));
    ...
}
```

Problems with that reasoning:

1. **"No active lane" does not mean "nothing can be reclaimed".** The pages are held by *idle retained checkpoints/continuations*, which the engine is allowed to evict or demote (`--recency-eviction --value-aware-demote`). The debug trace on 2026-10-26 14:53 shows pressure planning running (`EVICT`/`DEMOTE` decisions) and still failing, then this branch firing.
2. **An in-flight context transaction also lands here.** `ResourceManager::inspect` returns `TemporarilyBlocked` immediately while a transaction is open (`resource_manager.h:370-373`), and `plan_materialization` likewise returns `TemporarilyBlocked` when it cannot select a placement (`:626-646`). Right after a request finishes, its capture/publish transaction and state transfer can still be running; the next request is then classified as "idle Engine found no placement" even though the transaction would complete shortly and release or expose the state.
3. **No root/cold fallback is attempted** when reuse candidates exist but none can be placed. `inspect` only builds a root candidate when `candidates.empty()` (`resource_manager.h:626`); a matching-but-unplaceable continuation prevents the root (re-prefill) plan from being considered. The request should be able to run by dropping the reuse and prefilling, especially when output is the binding constraint.
4. **429 is the wrong class of error for this state.** The condition is transient and reclaimable, not "server overloaded"; clients retry immediately and deterministically fail again.

## Expected behaviour

When no request is active and admission is blocked only by idle retained checkpoints or an in-progress context transaction, the engine should make progress rather than reject:

- evict/demote idle retained checkpoints and re-inspect; and/or
- fall back to a root placement (drop reuse, full prefill) for this request; and/or
- wait for the in-flight context transaction to finish (bounded by the existing planning allowance / `--pending-timeout-ms`).

A 429 `Overloaded` should be reserved for genuinely unreclaimable pressure (e.g. active lanes that must release work).

## Fix options (upstream, in `ninfer-all`)

1. **`engine_core.h` admission (`admit_pending`)**: before the `active.size == 0` failure, attempt a forced reclaim of all idle catalog owners (or re-run inspection with a "must make progress" flag) and retry. Only fail if the second inspection is still `TemporarilyBlocked` with no reclaimable owner.
2. **`resource_manager.h` (`inspect` / `plan_materialization`)**: distinguish `TemporarilyBlocked` due to an in-progress transaction from blocked-by-retained-cache; in the latter case force the maximal fallback (root candidate / drop the unplaceable continuation) instead of returning blocked. Allow a root plan even when non-root candidates existed but failed placement (`:626`).
3. **`engine_core.h` error mapping**: map a transaction-in-progress block to a short retryable wait (queue) rather than `RequestErrorKind::Overloaded`; the engine already has `--max-pending-requests` / `--pending-timeout-ms` machinery.
4. **Docs/contract**: state explicitly that `max_tokens` is a KV *reservation*, not just a stop condition, so clients can size it sanely. (A client-side `max_tokens` cap is the immediate workaround; it is not an engine fix.)

## Verification (to be added with the upstream fix)

- Integration test: two sequential requests on one lane where the first requests `max_tokens = context - prompt` and completes; submit the second immediately. Expect 200 (via eviction/root fallback), not 429.
- Unit test around `admit_pending`: `active.size == 0` with one idle catalog owner large enough to matter → the head is admitted after reclaim; with no reclaimable owner → the existing 429 is still returned.
- Regression coverage for the transaction case: open a materialization/capture transaction, submit a request, assert it is queued/waited for, not failed with `context cache exhausted`.

## Cross-references

- `BUG_0015_state_load_not_serialized_with_inflight_requests.md` — same KV-occupancy family: an operation needing KV capacity is not serialized against the owner of those pages. There the failing op is `state/load`; here it is a generation request.
- Logs: `logs/console.log` (2026-10-29 04:29:46, 04:31:46, 04:42:25), `logs/console.log.1` (2026-10-27), `logs/console.log.2` (2026-10-26 14:53, includes `[pressure]`/`[escape-hatch]` debug decisions showing eviction attempts before the 429).
- Engine card: `engine_cards/qwen38-27b-swift15abl-nvfp4full-dflash2.json`; startup argv/limits in `logs/console.log` at 04:42:03 (`capacity | KV 120,000 tokens … pages 1,875/1,875 | runtime 3.76 GiB | free 801.0 MiB`).
- NInfer docs: `docs/serving.md` §"Default output limit" (:2694) and metrics table (`ninfer:context_cache_exhausted_requests_total`, :791).
