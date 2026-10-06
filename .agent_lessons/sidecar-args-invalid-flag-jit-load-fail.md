---
tags: [jit-load, sidecar-config, llama-server, debug]
aliases: [restore-reeval-tokens-invalid, model-sidecar-crash]
related: [[raw-tcp-forwarder-architecture]]
confidence: verified
---

# Sidecar `args` with an unsupported flag makes JIT load fail (llama-server exit 1) → forwarder returns HTTP 503

## The fact
Each `.gguf` can have a sidecar `<name>.gguf.json`. Its `"args"` field is appended **verbatim** to the
llama-server argv at launch (`server.py` `ModelConfig.build_argv`, ~line 246: `if self.args: argv +=
self.args.split()`). If any flag in that string is not recognized by the *currently selected* llama-server
build, the child exits immediately with code 1 and JIT load fails.

## Concrete repro (Round 14)
- Model `Qwen2.5-0.5B-Instruct-Q8_0.gguf` sidecar had `"args": "--restore-reeval-tokens 16"`.
- Selected build `llama.cpp-win-x86_64-nvidia-cuda12-avx2-0.3.0-custom-2` does NOT support that flag →
  child printed `error: invalid argument: --restore-reeval-tokens` and exited code 1 within ~2 s.
- Forwarder correctly detected the failure and returned **HTTP 503** to the client (NOT a forwarder bug —
  it did exactly its job). Log lines in `logs/console_debug.log`:
  `[Qwen...] error: invalid argument: --restore-reeval-tokens` →
  `llama-server process exited prematurely with code 1` →
  `JIT load failed ... : Model ... failed to start`.

## Diagnosis tip
When a JIT load fails fast (~2 s) with a forwarder HTTP 503, read `logs/console_debug.log` for the
llama-server child's stderr — the "invalid argument" / crash line is there. It is almost always a
**config/sidecar problem**, not a `_tcp_forwarder.py` bug.

## Fix / workaround (verified)
Remove or empty the offending `"args"` entry in the model's sidecar JSON, then make the running instance
re-read it: `POST http://127.0.0.1:<main>/v1/scan` (`server.py` `/v1/scan` → `manager.scan()`, which calls
`ModelConfig.load(p)` fresh for every model and rebuilds `self.models`). No restart needed — the in-memory
config is rebuilt from disk on rescan. After that, the same request JIT-loads cleanly.

## Caveat
This was a **per-model sidecar** issue (one specific `.gguf.json`), not a global one. The user fixed it by
deleting the `args` line entirely. Do NOT assume all models' sidecars are clean — different llama-server
builds support different flags, so a sidecar valid for one build can crash another.
