# BUG_0018 — `state/load` succeeds but the next request re-prefills (load-window ticket gap → catalog race)

**Status:** FIXED (2026-11-03) — load-window ticket gap closed; pre-load reservation in `load_state` + model_id-keyed queue. Verified by load-window repro (see Verification).
**Filed:** 2026-11-03
**Found in:** Production log 2026-11-03 10:28:03–10:30:49 (`N:\work\stuff\Beta\llama-autoloader\logs\console.log` lines 43944–43983). A safe-mode return restored `Maine.bin` (109,455 tok, session `9a9cdf47`) successfully (200 OK), but the very next request (req#2, 122 msgs, 110,782 tok) did a **full cold re-prefill (cache 0%)** instead of the expected ~98.8% hit.
**Severity:** high — session continuity is silently degraded: the turn after a safe-mode return is a full cold re-prefill (~1m11s for 110K tokens) instead of a near-instant cache hit. No error is surfaced, so it's easy to miss.
**Component:** `server.py` `load_state` (4084–4089), `tcp_forwarder.py` `_pipe_with_ticket` / post-load path (888–899, 1575–1595), `model_queue.py` (FIFO ticket queue).

## Symptom

A `POST /v1/models/{id}/state/load` returns **200 OK** and logs the restored token count, but the *next* chat request re-prefills the entire prompt (`cache 0 (0.0%)`) rather than reusing the just-restored KV. Reproducible on the ninfer/dflash2 model after a safe-mode return where a concurrent request lands during the model's JIT-load window.

## Evidence

Verbatim from `console.log` (dflash2, port 9024):

```
43944  10:28:03,953  Restoring slot 0 state from file: ...Maine.bin
43945  10:28:05,503  throughput | 5.0s | prefill 61.0 tok/s (305 tok) | decode 0.8 tok/s (4 tok) | running 0   ← 305-tok request INSIDE the restore window
43946  10:28:08,173  slot restore | slot 0 | ...maine.bin | 109455 tokens | 3.11 GiB | session 9a9cdf472645c9bc | 4.01 s
43947  10:28:08,174  State restored for model=...dflash2 label=Maine
43948  10:28:08,176  "POST /v1/models/...dflash2/state/load HTTP/1.1" 200 OK
43949  10:28:08,521  LLM POST /v1/chat/completions: 'model' extracted after extra read (344064B)
43950  10:28:08,739  req#2 started | openai-chat stream | 122 messages | max output 32,768
43983  10:30:49,619  req#2 done | prompt 110,782 | output 6,058 | cache 0 (0.0%) | TTFT 1m 11.3s | total 2m 41.1s
```

- Restored state: **109,455 tok**, session `9a9cdf47`, digest stable across save/restore/save.
- req#2 prompt: **110,782 tok** = that session + ~1,327 new tokens → prefix *should* match → expected ~98.8% cache.
- Actual: **cache 0 (0.0%)** → full re-prefill.
- The **305-tok request (line 43945)** ran at `10:28:05`, inside the `Maine` restore window (`10:28:03.953 → 10:28:08.173`).

Control case (same binary, no concurrent traffic in the load window): the A/B/C test `strata_aba_test.py` restore → next request hits **99.1%** consistently. The miss is *intermittent* and only occurs when a concurrent request slips into the load window.

## Root cause

The per-model FIFO ticket queue (`model_queue.py` / `ModelQueueRegistry`) is designed to order LLM requests against slot save/restore — `tcp_forwarder.py:870-872`: *"The ticket is granted in strict submission order, so a state restore enqueued before this request runs first (fixes the full-reprocess race)."*

**But the ticket is acquired only *after* the model is loaded and ready:**
- `tcp_forwarder.py:893-894` — *"Acquire the ticket only AFTER the model is loaded and ready, so the load itself is never gated by the queue."*
- `server.py:load_state` (4085) — the manual restore acquires its ticket **after** `load_model` returns.

So the entire **JIT-load window is unsynchronized**. A chat request that arrives while the model is (re)loading grabs its ticket *before* the restore does, runs to completion, and catalogs its own (shorter, different) continuation into the engine's context-cache catalog. The restore then lands the target session into a catalog cell, but the just-cataloged concurrent continuation now competes with it — and the next request's prefix-match misses the restored session → full re-prefill.

This is the same load-window gap as [[BUG_0015]], but a *different* symptom: BUG_0015 is the restore **failing** (HTTP 400, KV occupied by an in-flight request); this one is the restore **succeeding** while the next request **misses** the restored KV (catalog race). Both share the root cause: the ticket is taken after the load, leaving the load window open to concurrent requests.

## Fix

Close the load window so a restore enqueued during a safe-mode return always wins the queue over any request that lands while the model is coming back up:

1. **`server.py` `load_state`** — acquire the model's FIFO ticket **before** triggering `load_model` (reserve the ticket at the top of `load_state`, then load, then run `_perform_slot_restore` under that same ticket, releasing it in `finally`). Keep the current post-load ticket behavior for the *auto*-restore path in `load_model` (3131) only if it doesn't already inherit this.
2. **`tcp_forwarder.py` post-load path (888–899)** — keep acquiring the ticket after readiness for *pure* chat requests (so the load itself is never gated), but ensure a queued restore that was reserved *before* the load is granted first (strict submission order already gives this once the restore holds the ticket pre-load).

Net effect: the restore's ticket is reserved at safe-mode-return time, so any chat request arriving during the load window queues *behind* it; the concurrent continuation no longer gets cataloged ahead of the restored session.

## Verification

Verified 2026-11-03 (post-fix, restarted autoloader, q2_0 + dflash2, safe mode):

- **Clean A/B/C control** (`bench/strata_aba_test.py qwen3.8-flash-next-q2_0`): A cold 34984/0%/60.9s → save 200/0.8s → B (dflash2, evicts A) 1647/0%/21.7s → restore 200/37.9s → **A+ 35293 cached 34979 (99.1%)**/2.7s. PASS — no regression on the clean case.
- **Load-window repro** (`bench/strata_aba_loadwindow_test.py qwen3.8-flash-next-q2_0`): same A/save/B/restore, but a **probe request fired 0.8s into the restore's JIT-load window**. Post-fix the probe is ordered *behind* the restore (pre-load ticket reservation), so it hits the restored KV: **probe 34984 cached 34979 (100.0%)**, then **A+ 35293 cached 34979 (99.1%)**. PASS — the concurrent request no longer shadows the restored catalog cell.

Before the fix, the probe would grab its ticket *before* the restore (restore reserved post-load), run inside the load window, and catalog its own shorter continuation → A+ re-prefilled at 0%. The pre-load reservation closes that window.

## Related

- [[BUG_0015]] — same root cause (load-window ticket gap); that one is the restore *failing* (400), this one is the next request *missing* the restored KV.
- `model_queue.py` — the FIFO ticket queue; `queue.ticket()` context manager holds the ticket for the request duration.
- `tcp_forwarder.py:1575` `_pipe_with_ticket` — where chat requests take the ticket.
