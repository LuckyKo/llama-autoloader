# Strata DRY Refactor Deliberate Delta

**Date:** 2026-10-06  
**Review by:** polish_review  

## Context
The DRY refactor of the Strata‑bridge feature extracted inline guards into helper methods. One change was **deliberately not byte‑identical**:

### Original gate (before refactor)
```python
if cfg and (cfg.engine or "llama") != "llama":
    raise HTTPException(405, "Operation not supported for this engine")
```

### New helper (`ModelManager._require_llama_engine`)
```python
def _require_llama_engine(self, cfg) -> None:
    if cfg is None or not cfg.is_llama:
        raise HTTPException(405, "Operation not supported for this engine")
```

## Behavior change
- **Old:** `cfg is None` → condition false → execution falls through to inner try.
- **New:** `cfg is None` → raises 405 immediately.

## Reachability analysis
- `toggle_vision_ep`: `mid` is non‑empty (404 raised if missing). `cfg = await manager.get_config(mid)` can be `None` only if the model is not in `self.models` (edge case).
- `save_state_ep` / `load_state_ep`: `cfg = await manager.get_config(mid) if mid else None`. If `mid` is `None` (model resolution fails), old code fell through; new code raises 405.

## Verdict
The change was **explicitly specified by the task author** as an intentional delta to provide a clearer error when configuration is missing. It is acceptable and does not constitute a regression.

## Why this matters
Future reviewers should **not assume pure DRY = zero deltas**. When a helper signature differs from the original inline guard, document the delta and its acceptability before declaring the refactor complete.
