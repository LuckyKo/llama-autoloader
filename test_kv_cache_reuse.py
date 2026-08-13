"""Test KV cache reuse after slot restore with our patched llama-server.

This tests the specific fix in server-context.cpp lines 3358-3365:
After restoring slot state, resending the SAME prompt should use the restored
KV cache instead of forcing full reprocess (the old buggy behavior).

Flow:
  1. Load model
  2. Send large prompt (~8k tokens) — baseline timing
  3. Save slot state  
  4. Unload model
  5. Reload model (new process)
  6. Restore slot state
  7. Send EXACT SAME prompt again — should be fully cached now with our fix
     Before fix: ~8k tokens reprocessed in ~100s+
     After fix: near-zero processing time, all tokens cached
"""

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:9124"  # autoloader port from config.yaml
MODEL = "Qwen3-4B-Instruct-2507-Q6_K.gguf"  # Smaller Qwen3 model, no GPU-specific config
PROMPT_REPEATS = 200  # ~8k tokens
HTTP_TIMEOUT = 600


def api(path: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read())


def direct(port: int, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(url, data=data, method="POST")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read())


def chat(port: int, messages: list, max_tokens: int = 30) -> dict:
    return direct(port, "/v1/chat/completions", {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    })


def build_large_prompt(n_repeats: int = PROMPT_REPEATS) -> str:
    paragraph = (
        "The quick brown fox jumps over the lazy dog. "
        "Artificial intelligence has transformed how we interact with technology. "
        "Machine learning models continue to improve in both speed and accuracy. "
        "Natural language processing enables computers to understand human speech. "
        "Deep learning architectures like transformers have revolutionized the field. "
        "Large language models can generate coherent and contextually relevant text. "
        "Each new generation of models builds upon the insights of the previous one. "
        "Research in AI safety and alignment remains an active area of study. "
    )
    return (paragraph * n_repeats).strip()


def main():
    large_text = build_large_prompt()
    messages = [{"role": "user", "content": large_text}]

    # Step 0: Unload model to ensure clean state
    print("=" * 60)
    print("STEP 0: Ensuring clean state...")
    print("=" * 60)
    try:
        api(f"/v1/models/{MODEL}/unload", "POST")
        print("  Model unloaded")
    except Exception as e:
        print(f"  Note: Model not loaded or already unloaded ({e})")

    # Step 1: Load model
    print("\n" + "=" * 60)
    print("STEP 1: Loading model...")
    print("=" * 60)
    res = api(f"/v1/models/{MODEL}/load", "POST")
    if not res.get("ready"):
        print(f"  ERROR: Model loaded but not ready (port={res.get('port')})")
        sys.exit(1)
    port = res['port']
    print(f"  Loaded on port {port}, pid {res['pid']}, ready={res['ready']}")

    # Reset slot to clear any auto-restored cache
    try:
        direct(port, "/slots/0?action=reset")
        print("  Slot reset")
    except Exception as e:
        print(f"  WARNING: Failed to reset slot ({e})")

    # Step 2: Send large prompt — baseline (first time, no cache)
    print("\n" + "=" * 60)
    print("STEP 2: Sending large prompt (~8k tokens) — BASELINE...")
    print("=" * 60)
    t0 = time.time()
    res_step2 = chat(port, messages, max_tokens=30)
    t1 = time.time()
    
    usage = res_step2["usage"]
    timings = res_step2.get("timings", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    
    print(f"  Prompt tokens: {usage['prompt_tokens']}")
    print(f"  Cached tokens: {cached}, Processed: {prompt_n}")
    print(f"  Prompt eval time: {prompt_ms:.1f} ms ({prompt_ms/max(prompt_n,1):.2f} ms/tok)")
    print(f"  Wall time: {t1-t0:.2f}s")
    
    baseline_prompt_tokens = usage['prompt_tokens']
    baseline_processed = prompt_n
    baseline_ms = prompt_ms

    # Step 3: Save slot state
    print("\n" + "=" * 60)
    print("STEP 3: Saving slot state...")
    print("=" * 60)
    save_res = direct(port, "/slots/0?action=save", {"filename": f"{MODEL}.test_kv.bin"})
    n_saved = save_res.get('n_saved', 0)
    n_written = save_res.get('n_written', 0)
    print(f"  Saved {n_saved} tokens, {n_written:,} bytes")
    if n_saved <= 0:
        print(f"  ERROR: Save reported {n_saved} tokens (expected > 0)")
        sys.exit(1)

    # Step 4: Unload model
    print("\n" + "=" * 60)
    print("STEP 4: Unloading model...")
    print("=" * 60)
    api(f"/v1/models/{MODEL}/unload", "POST")
    print("  Model unloaded")

    # Step 5: Reload model (new process on new port)
    print("\n" + "=" * 60)
    print("STEP 5: Reloading model...")
    print("=" * 60)
    t0 = time.time()
    load_res = api(f"/v1/models/{MODEL}/load", "POST")
    t1 = time.time()
    if not load_res.get("ready"):
        print(f"  ERROR: Model loaded but not ready (port={load_res.get('port')})")
        sys.exit(1)
    port2 = load_res['port']
    print(f"  Loaded on port {port2}, pid {load_res['pid']}, ready={load_res['ready']}")
    print(f"  Load time: {t1 - t0:.1f}s")

    # Step 6: Restore slot state
    print("\n" + "=" * 60)
    print("STEP 6: Restoring slot state...")
    print("=" * 60)
    restore_res = direct(port2, "/slots/0?action=restore", {"filename": f"{MODEL}.test_kv.bin"})
    n_restored = restore_res.get('n_restored', 0)
    print(f"  Restored {n_restored} tokens")
    if n_restored <= 0:
        print(f"  ERROR: Restore reported {n_restored} tokens (expected > 0)")
        sys.exit(1)

    # Step 7: CRITICAL TEST — Send EXACT SAME prompt again
    # With our fix: should be fully cached (prompt_n ~= 0 or very small delta)
    # Without fix: full reprocess (~8k tokens, ~100s+)
    print("\n" + "=" * 60)
    print("STEP 7: CRITICAL TEST — Resending SAME prompt after restore...")
    print("=" * 60)
    t0 = time.time()
    res_step7 = chat(port2, messages, max_tokens=30)
    t1 = time.time()
    
    usage = res_step7["usage"]
    timings = res_step7.get("timings", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    
    print(f"  Prompt tokens: {usage['prompt_tokens']}")
    print(f"  Cached tokens: {cached}, Processed: {prompt_n}")
    print(f"  Prompt eval time: {prompt_ms:.1f} ms ({prompt_ms/max(prompt_n,1):.2f} ms/tok if processed > 0)")
    print(f"  Wall time: {t1-t0:.2f}s")

    # Step 8: Verify the fix worked
    print("\n" + "=" * 60)
    print("STEP 8: RESULTS")
    print("=" * 60)
    print(f"\n  BASELINE (first send, no cache):")
    print(f"    Processed: {baseline_processed} tokens in {baseline_ms:.1f} ms")
    print(f"\n  AFTER RESTORE (same prompt, should use cached KV):")
    print(f"    Processed: {prompt_n} tokens in {prompt_ms:.1f} ms")
    print(f"    Cached: {cached} tokens")

    # With our fix: processed should be very small (< 5% of baseline), time similarly small
    # Without fix: processed ~= baseline_processed, time ~= baseline_ms
    if prompt_n < baseline_processed * 0.1:
        print(f"\n  ✅ FIX WORKING! Only {prompt_n} tokens reprocessed vs {baseline_processed} baseline")
        print(f"     Speedup: {baseline_ms/max(prompt_ms,1):.1f}x faster")
    elif cached >= baseline_prompt_tokens * 0.9:
        print(f"\n  ✅ FIX WORKING! {cached}/{baseline_prompt_tokens} tokens cached")
    else:
        print(f"\n  ❌ FIX NOT WORKING — Full reprocess detected!")
        print(f"     Expected <{int(baseline_processed*0.1)} tokens processed, got {prompt_n}")
        sys.exit(1)

    # Cleanup
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)
    try:
        api(f"/v1/models/{MODEL}/unload", "POST")
        print("  Model unloaded")
    except Exception as e:
        print(f"  Note: Unload during cleanup failed ({e})")

    import os
    state_file = f"./states/{MODEL}.test_kv.bin"
    if os.path.exists(state_file):
        try:
            os.remove(state_file)
            print(f"  Removed test artifact: {state_file}")
        except Exception as e:
            print(f"  Note: Failed to remove {state_file} ({e})")

    print("\n" + "=" * 60)
    print("TEST PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()