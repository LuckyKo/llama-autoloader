"""Test slot state save/restore with large prompt (~8k+ tokens).

Flow:
  0. Unload model via autoloader to ensure clean state
  1. Load model, reset slot cache
  2. Send large prompt (8k+ tokens) — baseline timing
  3. Save slot state
  4. Unload model (kills process)
  5. Reload model (new process on new port)
  6. Restore slot state from saved file
  7. Send continuation — verify cached_tokens shows restored KV cache is working
  8. Send second continuation — verify ongoing cache usage

Note: llama.cpp's slot save/restore preserves the raw KV cache across process
restarts, but does NOT populate the separate RAM-based prompt cache (cache_prompt).
So resending the exact same prompt after restore will reprocess tokens, but
continuations that build on restored state WILL show cached_tokens. This is
by design — slot state is for conversation persistence, not prompt deduplication.
"""

import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:1234"
MODEL = "Qwen3.6-27B-Fable-Fus-MTP-Q4_K_M.gguf"
PROMPT_REPEATS = 200
HTTP_TIMEOUT = 120  # seconds for long-running operations (model load/unload)


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


def chat(port: int, messages: list, max_tokens: int = 50) -> dict:
    return direct(port, "/v1/chat/completions", {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    })


def check_response(res: dict, step_name: str) -> None:
    """Validate chat completion response structure."""
    if "error" in res:
        print(f"  ERROR [{step_name}]: Server error: {res['error']}")
        sys.exit(1)
    if "choices" not in res or "usage" not in res:
        print(f"  ERROR [{step_name}]: Unexpected response structure: {json.dumps(res, indent=2)[:300]}")
        sys.exit(1)
    if not res["choices"]:
        print(f"  ERROR [{step_name}]: Empty choices in response")
        sys.exit(1)


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

    # Step 0: Unload model via autoloader to ensure clean state
    print("=" * 60)
    print("STEP 0: Ensuring clean state (unloading model if loaded)...")
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

    # Reset slot to clear auto-restored cache
    try:
        direct(port, "/slots/0?action=reset")
        print("  Slot reset")
    except Exception as e:
        print(f"  WARNING: Failed to reset slot ({e})")

    # Step 2: Send large prompt — baseline
    print("\n" + "=" * 60)
    print("STEP 2: Sending large prompt (~8k+ tokens) — baseline...")
    print("=" * 60)
    t0 = time.time()
    res_step2 = chat(port, messages, max_tokens=30)
    check_response(res_step2, "Step 2")
    t1 = time.time()
    usage = res_step2["usage"]
    timings = res_step2.get("timings", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    print(f"  Prompt tokens: {usage['prompt_tokens']}")
    print(f"  Cached tokens: {cached}, Processed: {prompt_n}")
    print(f"  Prompt time: {prompt_ms:.1f} ms ({prompt_ms/max(prompt_n,1):.1f} ms/tok)")
    print(f"  Wall time: {t1-t0:.2f}s")
    print(f"  Completion: {res_step2['choices'][0]['message']['content'][:80]}...")

    # Step 3: Save slot state
    print("\n" + "=" * 60)
    print("STEP 3: Saving slot state...")
    print("=" * 60)
    save_res = direct(port, "/slots/0?action=save", {"filename": f"{MODEL}.slot_test.bin"})
    n_saved = save_res.get('n_saved', 0)
    n_written = save_res.get('n_written', 0)
    print(f"  Saved {n_saved} tokens, {n_written:,} bytes")
    if n_saved <= 0:
        print(f"  ERROR: Save reported {n_saved} tokens saved (expected > 0)")
        sys.exit(1)

    # Step 4: Unload model
    print("\n" + "=" * 60)
    print("STEP 4: Unloading model...")
    print("=" * 60)
    unload_res = api(f"/v1/models/{MODEL}/unload", "POST")
    print(f"  Unloaded: {unload_res['unloaded']}")

    # Step 5: Reload model
    print("\n" + "=" * 60)
    print("STEP 5: Reloading model...")
    print("=" * 60)
    t0 = time.time()
    load_res = api(f"/v1/models/{MODEL}/load", "POST")
    t1 = time.time()
    if not load_res.get("ready"):
        print(f"  ERROR: Model loaded but not ready (port={load_res.get('port')})")
        sys.exit(1)
    port = load_res['port']
    print(f"  Loaded on port {port}, pid {load_res['pid']}, ready={load_res['ready']}")
    print(f"  Load time: {t1 - t0:.1f}s")

    # Step 6: Restore slot state
    print("\n" + "=" * 60)
    print("STEP 6: Restoring slot state...")
    print("=" * 60)
    restore_res = direct(port, "/slots/0?action=restore", {"filename": f"{MODEL}.slot_test.bin"})
    n_restored = restore_res.get('n_restored', 0)
    print(f"  Restored {n_restored} tokens")
    if n_restored <= 0:
        print(f"  ERROR: Restore reported {n_restored} tokens restored (expected > 0)")
        sys.exit(1)
    if n_restored != n_saved:
        print(f"  WARNING: Restored {n_restored} tokens but saved {n_saved} — possible partial restore")

    # Step 7: Send continuation — should use restored KV cache for history
    # After restore, the slot already has the full conversation in its KV cache.
    # Sending the same messages again lets llama.cpp match cached tokens and skip reprocessing.
    print("\n" + "=" * 60)
    print("STEP 7: Sending continuation via /chat/completions...")
    print("=" * 60)
    step2_response = res_step2["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": step2_response})
    messages.append({"role": "user", "content": "Now count from 1 to 10."})
    t0 = time.time()
    try:
        res_step7 = chat(port, messages, max_tokens=30)
    except Exception as e:
        print(f"  ERROR during chat request: {e}")
        sys.exit(1)
    check_response(res_step7, "Step 7")
    t1 = time.time()
    usage = res_step7["usage"]
    timings = res_step7.get("timings", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    total_prompt = usage['prompt_tokens']
    print(f"  Prompt tokens: {total_prompt}")
    print(f"  Cached tokens: {cached}, Processed: {prompt_n}")
    print(f"  Prompt time: {prompt_ms:.1f} ms ({prompt_ms/max(prompt_n,1):.1f} ms/tok)")
    print(f"  Wall time: {t1-t0:.2f}s")
    print(f"  Completion: {res_step7['choices'][0]['message']['content'][:80]}...")

    # Verify cache hit: majority of prompt tokens should be cached (history from restored state)
    if cached >= n_restored * 0.9:
        print(f"  ✅ KV cache restore confirmed: {cached}/{n_restored} tokens cached")
    elif cached >= total_prompt * 0.8:
        print(f"  ✅ History caching confirmed: {cached}/{total_prompt} prompt tokens cached")
    elif prompt_n < 50:
        print(f"  ✅ Low processed tokens suggests cache hit: {prompt_n}")
    else:
        print(f"  ❌ No clear cache hit signal (cached={cached}, restored={n_restored})")
        sys.exit(1)

    # Step 8: Send another continuation to verify ongoing cache usage
    print("\n" + "=" * 60)
    print("STEP 8: Sending second continuation (should use cache for all history)...")
    print("=" * 60)
    step7_response = res_step7["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": step7_response})
    messages.append({"role": "user", "content": "What is 2+2?"})
    t0 = time.time()
    try:
        res_step8 = chat(port, messages, max_tokens=30)
    except Exception as e:
        print(f"  ERROR during chat request: {e}")
        sys.exit(1)
    check_response(res_step8, "Step 8")
    t1 = time.time()
    usage = res_step8["usage"]
    timings = res_step8.get("timings", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    total_prompt = usage['prompt_tokens']
    print(f"  Prompt tokens: {total_prompt}")
    print(f"  Cached tokens: {cached}, Processed: {prompt_n}")
    print(f"  Prompt time: {prompt_ms:.1f} ms")
    print(f"  Wall time: {t1-t0:.2f}s")
    print(f"  Completion: {res_step8['choices'][0]['message']['content'][:80]}...")

    # Verify ongoing cache usage
    if cached >= total_prompt * 0.8 or prompt_n < 20:
        print(f"  ✅ Ongoing cache usage confirmed: {cached} cached, {prompt_n} processed")
    else:
        print(f"  ❌ Expected cache hit but got cached={cached}, processed={prompt_n}")
        sys.exit(1)

    # Step 9: Cleanup
    print("\n" + "=" * 60)
    print("STEP 9: Cleanup — unloading loader model...")
    print("=" * 60)
    try:
        api(f"/v1/models/{MODEL}/unload", "POST")
        print("  Model unloaded")
    except Exception as e:
        print(f"  Note: Unload during cleanup failed ({e})")

    # Clean up test artifact
    state_file = f"./states/{MODEL}.slot_test.bin"
    if os.path.exists(state_file):
        try:
            os.remove(state_file)
            print(f"  Removed test artifact: {state_file}")
        except Exception as e:
            print(f"  Note: Failed to remove {state_file} ({e})")

    print("  Done!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()