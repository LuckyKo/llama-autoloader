---
tags: [generation-token, dead-backend-recovery, predicate-refactor, reachability-proof]
aliases: [engine-generation-always-positive, gen-matches-unreachable-divergence]
related: [[generation-match-predicate-single-source]]
confidence: verified
author: reviewer_gen_refactor_reviewer_20261020_181319.jsonl
---

# `LoadedModel.generation` is always >= 1 in production — sole writer is `_do_load_model`

The engine identity token (`LoadedModel.generation`, and `RecoveryFailure.generation` seeded from it)
can only ever be **0 (the legacy sentinel) or a positive integer**. Negative values are impossible in
production, which matters because `gen_matches` treats **exactly 0** as the wildcard — an older inline
guard of the form `expected_generation > 0 and recorded != expected_generation` treated *anything* <= 0
as legacy, so a swap between the two shapes is a sentinel-domain narrowing.

## Proof (re-checkable)

At `llama-autoloader` @ `d977c82` (note: the working tree carries concurrent uncommitted WIP):

- `grep "LoadedModel("` across the repo → exactly one production constructor, in
  `ModelManager._do_load_model` (`server.py:2556`), which runs
  `self._engine_generation += 1` (`server.py:2555`) **before** constructing the record with
  `generation=self._engine_generation`. The counter starts at 0 in `ModelManager.__init__`
  (`self._engine_generation = 0`, `server.py:846`) and is never reset.
- `LoadedModel.generation: int = 0` (`server.py:642`) is the only path to 0, and no production
  constructor omits the kwarg.
- The *caller's* token is derived from the same attribute of the same object:
  `RawTCPForwarder._resolve_target` returns `lm.port, int(getattr(lm, "generation", 0) or 0)`
  (`tcp_forwarder.py:1226`), so the recorded and expected sides cannot disagree for one instance.

## The one divergence this leaves (unreachable, both directions)

Truth table over `recorded ∈ {-1,0,1,2,3}` × `expected ∈ {-1,0,1,2,3}` (old inline guards vs
`gen_matches`) yields exactly two families:

1. **`recorded == 0, expected > 0`** — `_find_engine` / `note_request_start` become *more* permissive
   (they now match a generation-0 resident); `ModelManager._recover_dead_model_impl` becomes *less*
   permissive — its skip-branch (`if not self._gen_matches(lm.generation, expected_generation)`) now
   **suppresses** recovery for a generation-0 resident and returns it as a "healthy replacement".
   Suppression is the dangerous direction: a legacy/fake engine would never be recovered, the request
   retries the same dead engine and hits the `_recovery_attempts >= 1` cap → 503.
   `recovery_retry_after` has **no** divergence at all here — its old guard
   `expected_generation > 0 and rec.generation not in (0, expected_generation)` was already symmetric.
2. **`expected < 0`** — all match predicates flip from wildcard to strict-no-match, and
   `_record_recovery_failure`'s seed check changed from `if generation <= 0:` to
   `if generation == LEGACY_GENERATION:`, so a negative token is now stored verbatim instead of
   resolved from the registry.

Both families are unreachable by the invariant above. **Caveat: they are reachable in tests**, because
the fixtures default the token to the sentinel — `_add_resident(..., generation: int = 0)` and
`_new_resident(..., generation: int = 0)` in `tests/test_dead_backend_recovery.py`. Tests that pass a
concrete `expected_generation` first create the resident with a positive one, so the current suite does
not hit them either — but any new test using the default helper plus a concrete expectation lands on
family 1.

See [[generation-match-predicate-single-source]] for why the matcher is deliberately symmetric and
where the shared definition lives.
