# BUG_0015 — `state/load` not serialized with in-flight generation requests (restore fails when KV is occupied)

**Status:** OPEN
**Filed:** 2026-10-26
**Found in:** Production log 2026-10-26 07:26:01–07:26:03 (`N:\work\stuff\Beta\llama-autoloader\logs\console.log`). Autoloader fires `POST /state/load` to restore `Maine.bin` immediately after model load, but the bench (or AgentCascade) sends a generation request that occupies KV pages before the restore completes. The restore then fails with HTTP 400.
**Severity:** high — state restore is a critical operation for session continuity; if it silently fails (500 to the caller), the agent's entire conversation context is lost and the next turn is a full cold re-prefill.

## Symptom

```
07:26:01,972 [INFO]    Restoring slot 0 state from file: ...Maine.bin
07:26:03,355 [INFO]    throughput | 5.0s | prefill 907.6 tok/s (4,529 tok) | decode 0.8 tok/s (4 tok)
07:26:03,930 [WARNING] Slot restore attempt failed [URL param: ?action=restore]:
                     HTTP 400 — {"error":{"code":"slot_restore_failed",
                     "message":"session snapshot does not fit the free KV capacity;
                     evict other sessions first",...}}
07:26:03,931 [WARNING] Slot restore attempt failed [URL param: ?action=restore]: HTTP 404 —
07:26:03,933 [WARNING] Slot restore attempt failed [JSON body params]: HTTP 404 —
07:26:03,933 [INFO]    POST /v1/models/.../state/load HTTP/1.1 500 Internal Server Error
```

The generation request (req#9, 4,529 tokens) is running and holding KV pages when the state restore is attempted. The restore needs 54K tokens of KV but can't fit because the in-flight request's session is occupying pages.

## Root Cause

The autoloader fires the state restore (HTTP `POST /state/load`) to NInfer without serializing it against concurrent generation requests. The NInfer server accepts the restore call while a generation request is already in flight, and NInfer's `state/load` handler returns 400 when the snapshot doesn't fit the remaining free KV capacity — it does not evict or wait for the in-flight session.

The autoloader's load sequence is:
1. `POST /v1/models/{id}/load` → 200 (model is now accepting requests)
2. Client immediately sends generation requests (bench, AgentCascade, etc.)
3. `POST /v1/models/{id}/state/load` → 400 (KV occupied by step 2's requests)

Steps 2 and 3 race. The autoloader should ensure no generation requests are in-flight (or are queued) during the state restore window.

## Expected Behavior

When the autoloader initiates a state restore after model load:
- In-flight generation requests should complete (or be evicted) before the restore is attempted, OR
- New generation requests should be queued/paused until the restore completes, OR
- The restore should retry (with backoff) until the KV capacity is available

The state restore should not fail with 500 simply because another client's request happened to be in-flight at that instant.

## Fix Options

1. **Autoloader-side queue**: The autoloader should track in-flight requests per model. Before issuing `state/load`, wait for all in-flight requests to complete (with a timeout). New requests arriving during the restore window should be queued.
2. **NInfer-side auto-evict**: `POST /state/load` should evict other sessions to make room (the error message already says "evict other sessions first" — the engine could do this itself instead of refusing).
3. **Combined**: Autoloader waits for in-flight requests to drain (option 1), and NInfer's `state/load` auto-evicts any remaining cached (non-active) sessions to guarantee capacity (option 2).

## Cross-references

- NInfer engine: the `state/load` 400 response is correct engine behavior (it can't silently evict an active generation). The bug is in the autoloader's sequencing.
- AgentCascade: the agent's next turn after a failed restore will be a full cold re-prefill (~45s for 38K tokens) because the state is lost.
