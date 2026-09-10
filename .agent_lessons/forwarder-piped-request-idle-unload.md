---
tags: [tcp-forwarder, idle-timeout, last_used, touch, bugfix]
aliases: [piped-request-idle-unload, forwarder-last-used-stale]
related: [[raw-tcp-forwarder-architecture]], [[forwarder-model-extraction-fallthrough-fix]]
confidence: verified
---

# Piped (already-loaded) requests never refreshed `last_used` → premature idle unload

## Symptom
A model that was *recently used* still got unloaded by the background `idle_reaper`.

## Root cause
After the raw-TCP forwarder refactor ([[raw-tcp-forwarder-architecture]]), LLM requests for an
**already-loaded** model bypass FastAPI entirely and are piped directly by
`RawTCPForwarder._pipe_to_backend()` (tcp_forwarder.py). The only place that path consults the
manager is `_resolve_target_port()`, which calls `resolve_model_id()` and reads
`self.model_manager.loaded.get(mid)` to grab the port — but it **never called `lm.touch()`**.

So `last_used` was only updated at:
- model load time (server.py `_do_load_model` → `lm.touch()`), and
- requests that go through the JIT/fallback path (`load_model`, server.py).

Every subsequent *piped* request left `last_used` stale, so after `idle_timeout_seconds` the
`idle_reaper` (server.py, checks `now - lm.last_used > self.idle_timeout`) unloaded a model that
was still being served.

## Fix
In `tcp_forwarder.py::_resolve_target_port`, when the resolved model is loaded AND ready, call
`lm.touch()` before returning its port. `touch()` only writes a float (`self.last_used = time.time()`),
which is atomic under the GIL — safe to call from the forwarder's worker thread without the
manager's asyncio lock (the same "plain dict read is safe" rationale already used there).

## Testing note (gotcha)
`_resolve_target_port` blocks on `asyncio.run_coroutine_threadsafe(...).result(timeout=5)`.
Calling it **from within the same event loop** it schedules onto deadlocks (test hangs → returns None
on timeout). In production it runs on a worker thread. The regression test
(`tests/test_tcp_forwarder.py::TestResolveTargetPort`) therefore spins up a dedicated event loop in a
background thread, sets `mgr._main_loop`, and calls the method from the *test* thread to mirror real
cross-thread usage.

## Files
- Fix: `tcp_forwarder.py` `_resolve_target_port`
- Tests: `tests/test_tcp_forwarder.py` (class `TestResolveTargetPort`)
