---
tags: [launch-cards, prefill-config, vram-fit, quant-specific]
aliases: [autoloader-engine-cards-configured]
related: [[model-type-derived-vs-override-split]]
confidence: verified
author: orchestrator_Maine_20261028_055403.jsonl
contributors:
  - coder_ninfer-iq3-cards_*.jsonl
---

# NInfer launch cards configured (engine `g26600be6`)

The three NInfer Qwen3.8-27B launch cards in `llama-autoloader/engine_cards/` + their `ninfer-cards/` engine configs were set to **measured** values. The optimal config is **quant- and spec-dependent** — the DFlash2 champion does NOT transfer to IQ3 quants.

## Key finding: cuBLAS is quant-specific
`--prefill-cublas` targets Q4/Q5 linear shapes. IQ3 quants have none → it is a **no-op / slightly negative** on IQ3_S (measured T vs F ≈ −0.5%). `--prefill-microchunk-tokens` is likewise neutral-to-negative on IQ3. Only `--prefill-chunk 4096` gives a marginal (+2%, within noise) help. So on DFlash2/rk8v4 cuBLAS is a +35% win, but it must NOT be carried onto IQ3 cards.

## Key finding: omit `--stage-layers` → engine auto-splits
The engine's `default_stage_layers` (`qwen3_5/load.cpp:188-283`, `solve_stage_plan`) computes a VRAM+KV-aware **byte-balanced** split. A hand-pinned even split (`--stage-layers 32,32`) is wrong. Cards should OMIT `--stage-layers` and let the engine balance. (DFlash2 card keeps a hand-pinned `26,38` — that is the correct unbalanced split for that model, per the user; not auto-derived.)

## The three cards (as authored)
- `qwen38-27b-unc-dflash2.json` (DFlash2, 19 GB artifact, 2-GPU): champion `--prefill-cublas --prefill-chunk 4096 --prefill-microchunk-tokens 1024`, `--stage-layers 26,38`, `--spec dflash2`. Measured 1186 tok/s (+10%). Set by the user. ⚠️ Caveats: `--lm-head-draft`/`--branch-anchors` added by user are legal but untested for speed; the cuBLAS+micro+2-GPU path is NOT covered by the byte-identity oracle (`test_engine_stages_real.cpp` has no cuBLAS field, runs `.devices={0,0}`).
- `qwen38-27b.json` (IQ3_S, 14 GB artifact, 2-GPU, `--spec dflash2`): FIXED — removed the wrong `--stage-layers 32,32` (now auto-split), removed `--prefill-cublas` (no-op on IQ3), kept `--prefill-chunk 4096`. Measured ~969 tok/s.
- `qwen38-27b-ablit-iq3xxs-mtp.json` (Huihui IQ3_XXS, 10.3 GB artifact, `--spec mtp`): CREATED — **single-card** on the non-monitor card (`--devices 0`, index 0), `--max-context 90112 --kv-capacity 90112` (the max that fits with ≥1.5 GB headroom; 120K needs 2-GPU). Measured ~875 tok/s prefill, ~49 tok/s gen. Single-device removes all cross-card sync.

## Binding constraint
Monitor card = nvidia-smi index 1 (uuid `GPU-c4d40dba`), keep ≥1.5 GB free (drives the monitors). The other card (index 0, `GPU-4780c954`) is the tight one in 2-GPU runs.

## Open
- `--draft-tokens 10` measured better for gen than 7, but only at ~100-token context — re-measure at realistic context before adopting.
- 90K context cap on IQ3_XXS single-card; a 120K 2-GPU variant is available if needed.
