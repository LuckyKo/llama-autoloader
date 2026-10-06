---
tags: [autoloader, registry, scan, model-config]
aliases: [scan-gguf-shaped-registry]
related: [[model-id-gguf-name-collision]], [[raw-tcp-forwarder-architecture]], [strata-bridge-lifecycle-facts]]
confidence: verified
author: researcher_strata_researcher_20261006_071922.jsonl
---

# llama-autoloader's registry is GGUF-shaped, not engine-agnostic

The load/unload/forward/kill core IS engine-agnostic, but the **registry** is not. Anyone adding a
second backend must know this.

- `ModelManager.scan()` (server.py:521-582) rebuilds `self.gguf_paths = {}` (`:533`) and
  `self.models = found` (`:554`) wholesale from `root_dir.rglob("*.gguf")` (`:523`). Any non-gguf
  model registered in `self.models` is **destroyed by a rescan** (the `/api/rescan` endpoint at
  `:1800` and `start()` at `:1335` both call it).
- `self.gguf_paths` is gguf-specific in meaning. The GGUF metadata loop (`:560-580`) iterates it,
  so putting a `.json` path in it would make `_read_gguf_metadata` run on a JSON file.
  Safer: keep a separate `strata_cards: Dict[str, Path]` and resolve
  `self.gguf_paths.get(mid) or self.strata_cards.get(mid)` at spawn time.
- `_do_load_model` (`:842`) reads **both** `self.models[model_id]` (`:850`) and
  `self.gguf_paths[model_id]` (`:851`) — an unknown id KeyErrors.
- `ModelConfig.backend` (`:181`) already means the **llama build directory**
  (`resolve_binary`, `:407-432`), persisted in `config.yaml` as `selected_backend`. It is NOT free
  for a `llama|strata` engine discriminator — add a new `engine` field.
- `list_models` calls `self.resolve_binary(cfg)` unconditionally (`:634`) — a non-llama card must
  be guarded there.
- The GGUF metadata loop must stay gguf-only for the same reason.

Bridge design and the full line-anchored change list: `N:\work\WD\AgentCascade\plans\strata-bridge-plan.md`.