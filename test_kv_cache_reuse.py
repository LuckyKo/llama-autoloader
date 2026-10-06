# WARNING: Run this test in SYNCHRONOUS mode (not async/background).
# The test takes several minutes (model load + 3 turns + save/restore + step 7).
# Running it async causes the agent running on AC to interfere with the test
# If you are an AI agent: use shell_cmd with timeout=600 and do set async_mode=false.

"""Test KV cache reuse after slot restore with our patched llama-server.

Simulates real AC usage: model is unloaded (LRU eviction), reloaded, and
slot state is restored. The next prompt (with appended content) should use
the restored KV cache via LCP matching, with only the suffix + re-eval
tokens processed instead of a full reprocess.

Flow:
  1. Load model
  2. Send large prompt (~8k tokens) — baseline timing
  3. Save slot state
  4. Unload model (simulates LRU eviction / model switch)
  5. Reload model (new process)
  6. Restore slot state (loads KV + tokens from disk)
  7. Send same prompt + appended content (simulates AC's next message)
     Expect: most tokens cached via LCP, only suffix + re-eval processed
"""

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:1234"  # autoloader port from config.yaml
MODEL = "Qwen3.8-27B-Ablit"  # Smaller Qwen3 model, no GPU-specific config
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

    # Build a multi-turn conversation history (mirrors real AC usage where
    # multiple turns accumulate before a model switch / LRU eviction happens).
    # We do 3 turns to build up realistic context.
    history: list[dict] = []
    turn_texts = [
        large_text,
        "Summarize the key points from the text above in one sentence.",
        "Now explain what 'transformers' means in the context of that summary.",
    ]

    messages = [{"role": "user", "content": large_text}]

    # Step 0: Unload ALL models to ensure clean state
    # (The target model may not be the one currently loaded; with max_loaded_models=1,
    #  any other loaded model will block our load and cause a 503.)
    print("=" * 60)
    print("STEP 0: Ensuring clean state...")
    print("=" * 60)
    try:
        models = api("/v1/models", "GET")
        loaded = [m for m in models.get("data", []) if m.get("loaded")]
        if not loaded:
            print("  No models loaded, clean state confirmed")
        for m in loaded:
            mid = m["id"]
            print(f"  Unloading: {mid}")
            try:
                api(f"/v1/models/{mid}/unload", "POST")
            except Exception as e:
                print(f"    Note: unload failed ({e})")
        # Brief pause to let the autoloader fully release the slot
        import time as _t
        _t.sleep(2)
    except Exception as e:
        print(f"  Warning: Could not query model list ({e}), proceeding anyway")

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

    # Step 2: Build multi-turn history (mirrors AC's accumulated conversation)
    print("\n" + "=" * 60)
    print("STEP 2: Building multi-turn history (3 turns)...")
    print("=" * 60)
    last_res = None
    for i, turn_text in enumerate(turn_texts):
        # Build messages with full history up to this turn
        if i == 0:
            msgs = [{"role": "user", "content": turn_text}]
        else:
            msgs = history + [{"role": "user", "content": turn_text}]

        t0 = time.time()
        last_res = chat(port, msgs, max_tokens=30)
        t1 = time.time()

        usage = last_res["usage"]
        timings = last_res.get("timings", {})
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        prompt_n = timings.get("prompt_n", 0)
        prompt_ms = timings.get("prompt_ms", 0)
        msg_obj = last_res["choices"][0]["message"]
        assistant_msg = msg_obj.get("content", "") or ""
        reasoning_msg = msg_obj.get("reasoning_content", "") or ""

        asst_obj = {"role": "assistant", "content": assistant_msg}
        if reasoning_msg:
            asst_obj["reasoning_content"] = reasoning_msg

        print(f"  Turn {i+1}: prompt={usage['prompt_tokens']} tok, "
              f"cached={cached}, processed={prompt_n}, "
              f"eval={prompt_ms:.0f}ms, wall={t1-t0:.1f}s")

        # Append to history: user turn + assistant response (with reasoning_content field)
        history.append({"role": "user", "content": turn_text})
        history.append(asst_obj)

    baseline_prompt_tokens = last_res["usage"]["prompt_tokens"]
    baseline_processed = timings.get("prompt_n", 0)
    baseline_ms = prompt_ms
    print(f"\n  History built: {len(history)} messages, "
          f"last prompt={baseline_prompt_tokens} tokens")

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

    # Step 7: CRITICAL TEST — Send next turn with full history (mirrors AC)
    # In real AC, after a state restore the next request includes the full
    # conversation history plus new user input. The tokenized prompt matches
    # slot.prompt.tokens (which includes all prior turns + generated tokens),
    # so LCP should give us a high n_past and only the new turn is processed.
    print("\n" + "=" * 60)
    print("STEP 7: CRITICAL TEST — Next turn after restore (full history)...")
    print("=" * 60)
    followup = "What is the capital of France? Answer in one word."
    # Full history + new user message (exactly what AC sends)
    messages_step7 = history + [{"role": "user", "content": followup}]
    t0 = time.time()
    res_step7 = chat(port2, messages_step7, max_tokens=30)
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

    # With our fix: most of the original prompt is cached (LCP match), only the
    # appended suffix + re-eval tokens are processed. Expect < 20% of baseline.
    # Without fix: full reprocess ~= baseline_processed or more.
    if prompt_n < baseline_processed * 0.2:
        print(f"\n  [SUCCESS] FIX WORKING! Only {prompt_n} tokens reprocessed vs {baseline_processed} baseline")
        print(f"     Speedup: {baseline_ms/max(prompt_ms,1):.1f}x faster")
        print(f"     (Expected: ~suffix tokens + up to 16 re-eval tokens)")
    elif cached >= baseline_prompt_tokens * 0.8:
        print(f"\n  [SUCCESS] FIX WORKING! {cached}/{baseline_prompt_tokens} original tokens cached")
    else:
        print(f"\n  [FAILED] FIX NOT WORKING -- Full reprocess detected!")
        print(f"     Expected <{int(baseline_processed*0.2)} tokens processed, got {prompt_n}")
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
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()