# Asyncio Lock Holding Across Blocking I/O

**Date:** 2026-10-14  
**Repo:** llama-autoloader  
**Trigger:** lock, asyncio, blocking, starvation, performance

## Problem

Holding an `asyncio.Lock` while awaiting a blocking operation (e.g., `asyncio.to_thread()`, network I/O, subprocess wait) can starve other async tasks that need the lock, because the lock remains held throughout the entire await.

## Example in llama-autoloader

In `interrupt_model` (server.py), the lock is held across `await manager._unload_unlocked(mid)`, which internally calls `await asyncio.to_thread(_kill_proc)`. The `_kill_proc` function can take several seconds if the process does not terminate quickly.

```python
async with manager._lock:          # Lock acquired
    # ... short operations ...
    await manager._unload_unlocked(mid)  # Holds lock during thread call
```

## Impact

- Other requests that need `manager._lock` are blocked for the entire duration of the blocking call.
- Can lead to increased latency and potential timeouts under load.
- Not a correctness bug, but a concurrency limitation.

## Recommended Patterns

1. **Release lock before long-running operations** – If the operation does not depend on shared state protected by the lock, move it outside the `async with` block.
2. **Use a separate mutex for independent resources** – Isolate the critical section to only what truly needs exclusive access.
3. **Document lock-holding expectations** – Make it explicit in comments if a lock must be held across an I/O operation.

## Related Memory

- See `concurrency-code-review-checklist` for lock-ordering verification.
- See `code-refinement-review` for edge-case analysis of blocking calls.
