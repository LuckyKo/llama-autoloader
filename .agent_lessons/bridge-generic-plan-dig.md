---
tags: [llama-autoloader, bridge-card, engines, config, refactor, dig-phase]
aliases: [bridge-generic-plan-dig]
related: [[strata-bridge-lifecycle-facts]], [[strata-dedup-deliberate-delta]], [[plan-review-cycle-before-implementation]], [autoloader-registry-is-gguf-shaped]
confidence: verified
author: researcher_bridge_generic_planner_20261006_123928.jsonl
---

# DIG findings for the llama-autoloader generic multi-engine bridge

Plan: `N:\work\WD\AgentCascade\plans\bridge-generic-plan.md` (reviewed 2026-10-06).

## The bug, confirmed at source

`strata_adapter.py:70-78` subscripts `mgr.strata_cfg["venv_python"]` /
`["serve_script"]` unguarded. The shipped `config.yaml` has **no `strata:` section at all**
(25 lines, verified in full), so `mgr.strata_cfg == {}` (`server.py:357`) and JIT-load dies with
`KeyError`. `validate_config` (`server.py:435-444`) only *warns* — deliberately, so a bad bridge
config can never stop llama from booting. **Keep that property**: validate into a per-engine
`error` string at boot, surface it via a `GET /v1/engines` endpoint, and turn it into a 400 in
`_do_load_model`. Fail-fast-at-boot would violate the project's own stated constraint.

## Every Strata-specific behaviour is already DATA, not code

Verified by reading `strata_adapter._launch_strata`: argv tokens
(`--engine strata --config … --port … --host`), the derived-config overlay
(`aliases`/`cors_origins`/`lazy_load`/`idle_unload_s`), and the env scrub list
(`STRATA_API_KEY`, `STRATA_ALLOWED_HOSTS`). All three fit a config-driven argv template.
⇒ A `{engine_name: launch_function}` registry would be **over-abstraction**: it still needs a
code change per new engine (defeating the stated "config-only" goal) *and* re-introduces
`if engine == …` dispatch. Recommend the data-driven template. This is the load-bearing
architectural argument of the plan.

## Invariants that are easy to break in this refactor

- `_launch_llama` (`strata_adapter.py:50-60`) must stay **byte-identical**, returning
  `(argv, None, None)`. Do NOT unify its signature with the bridge launcher (it takes `path`).
  Llama stays out of the engine registry entirely — that is the guarantee.
- `temp/strata/<mid>.json` derived-config path must not change, or pre-refactor derived files
  stop being cleaned up on unload (`server.py:1179-1186`).
- `env_scrub` must **DELETE** env vars, not blank them. See [[strata-bridge-lifecycle-facts]]:
  an empty `STRATA_API_KEY` is a hard exit(2) in `serve/server.py`.
- `strata_adapter.py:81-83` (`--min-free-vram-mib` only when `> 0`, then `extra_args`) must be
  preserved exactly; only the flag *name* becomes config (`min_free_vram_flag`).

## Non-obvious UI bugs found while surveying

- `index.html:210-213` — the `f-strata-config` input **exists but app.js never reads it** ⇒
  users cannot edit the engine-config path. Root cause of the reported complaint.
- `index.html:214-225` — duplicate tags inputs `f-tags` and `f-tags-strata`; the second is never
  read either ⇒ **tags are silently dropped** on every Strata card save.
- `styles.css:114-116` — `.modal { width: 560px }` is the "cramped" editor.

## Deliberate invariant reversals (must appear in the commit message)

The "card files never record `engine`" rule (`server.py:250-268`, asserted by
`tests/test_strata_adapter.py:326,363`) was an artefact of a single-engine world. With a shared
card directory the engine *must* be stored in the file. Reversing it breaks 2 tests by design.

## Review discipline that paid off

An independent reviewer confirmed all ~30 cited line numbers as accurate but found 6
enumeration gaps (comments at `server.py:1083`/`:1132`; test lines 348/349/361/471). Grep-driven
enumeration catches sites; hand-written line ranges do not. For DIG plans, enumerate from a grep
of every spelling variant, then have a reviewer re-grep independently.

Plan: `N:\work\WD\AgentCascade\plans\bridge-generic-plan.md`.
See also [[plan-review-cycle-before-implementation]].
