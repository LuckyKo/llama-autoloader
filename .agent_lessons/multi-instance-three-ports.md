---
name: multi-instance-three-ports
description: The autoloader binds THREE ports (forwarder / internal FastAPI / engine base) and --port only moved one, so a second instance died with a bare WinError 10048.
metadata:
  type: project
  verified: 2026-10-18
---

`python server.py --port 9126` alongside an already-running instance used to fail like this:

```
Raw TCP forwarder listening on 127.0.0.1:2234 (internal API -> 1235)
ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 1235)
Application shutdown complete.
```

**The trap:** the app binds *three* things, and `--port` moved only the first.

| What | Default | Control |
|------|---------|---------|
| Main client port (raw TCP forwarder) | `launcher.port` | `--port` |
| Internal FastAPI / uvicorn | `$AUTOLOADER_INTERNAL_PORT` or **1235** | env var only (pre-2026-10-18) |
| First engine port | `launcher.base_port` | config only |

The forwarder bound the requested port fine, then uvicorn still targeted the hardcoded
1235 — owned by the running instance — and the process died with a raw `WinError 10048`.
Nothing in `--help`, the README or the error hinted a second port existed.

**Fixed by:** `--internal-port` and `--base-port` flags, plus `_resolve_internal_port()`:

- Busy + **implicit** (env/default) → advance to the next free port and log a warning
  naming the port taken. A second instance must be able to start.
- Busy + **explicit** (`--internal-port N`) → `SystemExit` with a readable message.
  Never silently move a port the user named — that hides the collision being diagnosed.
- Busy **main** port → `SystemExit` too, before uvicorn ever binds.

**Use `_port_in_use()`, not a bind probe, to test "is this taken".** On Windows
`SO_REUSEADDR` (which `ModelManager._is_port_available` uses for *its own* allocation
logic) does not cleanly distinguish "free" from "another instance owns it". A connect
probe is unambiguous.

**Cross-instance engine-port safety comes from `_is_port_available`, not `used_ports`.**
`_allocate_port` only knows about *its own* manager's `loaded` dict, so two instances
sharing a `base_port` are kept apart purely by the real-socket probe. Don't add a test
that asserts they self-exclude via `loaded` — that is not the mechanism.

**Known shared state (not instance-scoped):** `models.save_state_dir`, `engine_cards/`
and `temp/`. `temp/<engine>/<model>.json` derived configs are written on load and
unlinked on unload, so one instance unloading a bridge card deletes the config another
instance is using. The startup banner warns about this; fixing it means per-instance
temp dirs, which would break the derived-config cleanup compat documented in
[[strata-bridge-lifecycle-facts]].

Related: [[windows-batch-port-kill-gotchas]], [[start-bat-port-wait-relaunch]],
[[raw-tcp-forwarder-architecture]]