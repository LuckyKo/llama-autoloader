# BUG_0004 — Active-request accounting is keyed by port and can decrement a different engine

**Status:** FIXED
**Filed:** 2026-10-20
**Fixed:** 2026-10-20
**Commit:** `6ba95c4`
**Found in:** Post-fix audit of BUG_0001 (`c31e08e`)
**Severity:** medium
**Component:** `server.py` (`note_request_start` / `note_request_end`), `tcp_forwarder.py` (`_active_conns`, `cancel_active`, `_pipe_to_backend`)

## Symptom

`active_requests` is tracked by walking `self.loaded` for a matching **port**, not by an engine
instance. Ports are stable per model, so after a recovery reload the same numerical port can
represent a different `LoadedModel`.

Sequence:

1. Request A starts a long generation against engine instance E1 on port P.
   `note_request_start(P)` increments `E1.active_requests`.
2. E1 dies and BUG_0001 recovery replaces it with E2 on the same port P.
3. Request A eventually finishes/aborts and `note_request_end(P)` finds whichever instance is
   currently on P — now E2 — and decrements `E2.active_requests`.
4. If E2 has its own active generation (counter 1), it is decremented to 0 while still running.
   The idle reaper and LRU eviction both use `active_requests == 0` as “safe to unload”, so E2 can
   be killed mid-generation.

`cancel_active(port)` has the same generation-mixing problem: its `_active_conns` registry is
keyed by port, so old and new engine connections can share a cancellation bucket.

## Evidence

`server.py:1919-1930`:

```python
for lm in list(self.loaded.values()):
    if lm.port == port:
        lm.active_requests += 1
        ...
```

and the matching decrement uses only `port`. `tcp_forwarder.py:183` likewise stores
`_active_conns` as `Dict[int, set]` keyed by target port.

## Impact

A stale end event can make a genuinely busy replacement engine eligible for idle/LRU eviction.
The window is small but real for long generations, state restores, and recovery cycles — exactly
the workloads this tracking exists to protect.

## Fix direction

Return an opaque token from `note_request_start` (engine generation or the `LoadedModel` instance)
and require it on `note_request_end(token)`. Decrement only if the token still identifies the
current instance; otherwise ignore the stale event.

Apply the same identity to `_active_conns` and `cancel_active`, so cancellation targets one
engine generation rather than every connection that ever used that port. This should share the
generation primitive proposed in BUG_0003.

## Fix (implemented)

- `note_request_start(port, generation)` returns the exact `LoadedModel` instance it
  incremented (or `None` if that generation is already gone).
- `note_request_end(engine)` takes that instance token and decrements that object
  directly; a late completion from E1 can never touch E2's counter. The old
  port-only end form was removed so an ambiguous stale event cannot decrement a
  replacement.
- The forwarder captures the generation from `_resolve_target()` and passes it
  through `_pipe_with_ticket` / `_pipe_to_backend`.
- `_active_conns` is keyed by `(port, generation)`, and
  `cancel_active(port, generation)` closes only that instance's connections.
  Omitting `generation` keeps the legacy port-wide behavior for explicit callers.

## Verification

- `tests/test_dead_backend_recovery.py::test_note_request_end_releases_original_generation_only`
  — stale E1 end leaves E2's counter at 1.
- `tests/test_dead_backend_recovery.py::test_note_request_start_rejects_stale_generation`
  — a request routed to E1 cannot attach to E2 after port reuse.
- `tests/test_dead_backend_recovery.py::test_cancel_active_targets_one_engine_generation`
  — cancelling E1 leaves E2's sockets open.
- `tests/test_dead_backend_recovery.py::test_pipe_with_ticket_carries_generation_through_recovery`
  — generation is propagated through the retry path.
- Full suite: 370 passed after the BUG_0003/0004/0005 change; one Windows
  `WinError 10055` event-loop setup flake passed when rerun.

## Related

- `BUG_0003_recovery_evicts_healthy_replacement_on_reused_port.md` — same identity/generation gap
- `BUG_0001_tcp_forwarder_no_dead_backend_recovery.md` — increases the frequency of port reuse
