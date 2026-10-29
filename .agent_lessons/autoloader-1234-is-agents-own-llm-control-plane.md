---
tags: [autoloader, control-plane, agentcascade, restart, vram]
aliases: [autoloader-is-own-brain, killing-autoloader-kills-llm]
related: [[start-bat-port-wait-relaunch]], [[lfm3-omni-smoke-test-tts-realtime-pass-multimodal-flag-gap]]
confidence: verified
author: coder_omni-live-smoke_20261029_092439.jsonl
---

# The live autoloader on 127.0.0.1:1234 IS AgentCascade's own LLM control plane

**Fact:** the autoloader instance serving this machine (port 1234, internal 1235) is the
control plane for the LLM that AgentCascade agents — including the one doing the work —
are running on. There is **only one** such instance. Killing it drops the resident model
(the agent's "brain"); the supervisor/harness restarts the autoloader, and the LLM
JIT-reloads on the next request, but every agent mid-turn loses its model for the gap.

**Verified 2026-10-29:** a smoke-test agent killed the running autoloader (PID 26596,
started 07:55) to restart it with freshly edited code; the user had to restart it
manually (new PID 29044 @ 09:25:57). The agent's own launch lost a port race to the
user's restart ("Main port 1234 is already in use" — see [[start-bat-port-wait-relaunch]]:
the forwarder binds 1234 only inside FastAPI's `_startup`, so "port free" checks race).

## Rules
1. **Before killing/restarting the autoloader, compare the live process start time with
   the code mtimes** (`Get-Process -Id <pid> | Select StartTime` vs. `server.py`/
   `tcp_forwarder.py` mtimes). If the process is already newer than the code, it runs the
   new code — no restart needed.
2. Prefer the supervisor/user to restart it; if you must, expect the LLM to be down
   until the next request JIT-loads it, and do NOT launch a competing instance while a
   restart is in flight.
3. `max_loaded_models: 1` (config.yaml) means loading a smoke-test model **evicts the
   resident LLM** — plan for the brain being out during the test and back on the next turn.
