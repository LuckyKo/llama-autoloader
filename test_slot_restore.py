"""Test slot state save/restore with large prompt (~8k+ tokens).

Flow:
  1. Load model, reset slot cache
  2. Send large prompt (8k+ tokens) — baseline timing
  3. Save slot state
  4. Unload model
  5. Reload model, reset slot cache (clear auto-restore)
  6. Restore slot state
  7. Send same prompt — verify cache hit (cached_tokens ≈ prompt tokens)
  8. Send continuation — verify only new tokens processed
"""

import json
import time
import urllib.request

BASE = "http://127.0.0.1:1235"
MODEL = "Qwen2.5-0.5B-Instruct-Q8_0.gguf"


def api(path: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def direct(port: int, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(url, data=data, method="POST")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def slot_info(port: int) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/slots", method="GET")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())[0]


def chat(port: int, messages: list, max_tokens: int = 50) -> dict:
    return direct(port, "/v1/chat/completions", {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    })


def build_large_prompt(n_repeats: int = 200) -> str:
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
    large_text = build_large_prompt(200)
    messages = [{"role": "user", "content": large_text}]

    # Clean state
    try:
        api(f"/v1/models/{MODEL}/unload", "POST")
    except Exception:
        pass

    # Step 1: Load model
    print("=" * 60)
    print("STEP 1: Loading model...")
    print("=" * 60)
    res = api(f"/v1/models/{MODEL}/load", "POST")
    port = res['port']
    print(f"  Loaded on port {port}, pid {res['pid']}, ready={res['ready']}")

    # Reset slot to clear auto-restored cache
    try:
        direct(port, "/slots/0?action=reset")
    except Exception:
        pass
    info = slot_info(port)
    print(f"  Slot n_prompt_tokens: {info.get('n_prompt_tokens', info.get('n_past', '?'))}")

    # Step 2: Send large prompt — baseline
    print("\n" + "=" * 60)
    print("STEP 2: Sending large prompt (~8k+ tokens) — baseline...")
    print("=" * 60)
    res = chat(port, messages, max_tokens=30)
    usage = res["usage"]
    timings = res.get("timings", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    print(f"  Prompt tokens: {usage['prompt_tokens']}")
    print(f"  Cached tokens: {cached}, Processed: {prompt_n}")
    print(f"  Prompt time: {prompt_ms:.1f} ms ({prompt_ms/max(prompt_n,1):.1f} ms/tok)")
    print(f"  Completion: {res['choices'][0]['message']['content'][:80]}...")
    baseline_prompt_ms = prompt_ms
    baseline_prompt_n = prompt_n

    # Step 3: Save slot state
    print("\n" + "=" * 60)
    print("STEP 3: Saving slot state...")
    print("=" * 60)
    save_res = direct(port, "/slots/0?action=save", {"filename": f"{MODEL}.slot_test.bin"})
    print(f"  Saved {save_res.get('n_saved', '?')} tokens, {save_res.get('n_written', '?')} bytes")

    # Step 4: Unload model
    print("\n" + "=" * 60)
    print("STEP 4: Unloading model...")
    print("=" * 60)
    res = api(f"/v1/models/{MODEL}/unload", "POST")
    print(f"  Unloaded: {res['unloaded']}")

    # Step 5: Reload model
    print("\n" + "=" * 60)
    print("STEP 5: Reloading model...")
    print("=" * 60)
    t0 = time.time()
    res = api(f"/v1/models/{MODEL}/load", "POST")
    t1 = time.time()
    port = res['port']
    print(f"  Loaded on port {port}, pid {res['pid']}, ready={res['ready']}")
    print(f"  Load time: {t1 - t0:.1f}s")

    # Reset slot to clear auto-restored cache
    try:
        direct(port, "/slots/0?action=reset")
    except Exception:
        pass
    info = slot_info(port)
    print(f"  Slot n_prompt_tokens after reset: {info.get('n_prompt_tokens', info.get('n_past', '?'))}")

    # Step 6: Restore slot state
    print("\n" + "=" * 60)
    print("STEP 6: Restoring slot state...")
    print("=" * 60)
    restore_res = direct(port, "/slots/0?action=restore", {"filename": f"{MODEL}.slot_test.bin"})
    print(f"  Restored {restore_res.get('n_restored', '?')} tokens")
    info = slot_info(port)
    print(f"  Slot n_prompt_tokens after restore: {info.get('n_prompt_tokens', info.get('n_past', '?'))}")

    # Step 7: Send SAME large prompt — should hit cache
    print("\n" + "=" * 60)
    print("STEP 7: Sending same large prompt (should hit cache)...")
    print("=" * 60)
    res = chat(port, messages, max_tokens=30)
    usage = res["usage"]
    timings = res.get("timings", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    print(f"  Prompt tokens: {usage['prompt_tokens']}")
    print(f"  Cached tokens: {cached}, Processed: {prompt_n}")
    print(f"  Prompt time: {prompt_ms:.1f} ms")
    print(f"  Completion: {res['choices'][0]['message']['content'][:80]}...")
    speedup = baseline_prompt_ms / max(prompt_ms, 1)
    print(f"  Speedup vs baseline: {speedup:.1f}x")
    assert cached > 5000, f"Expected >5000 cached tokens, got {cached}"

    # Step 8: Send continuation
    print("\n" + "=" * 60)
    print("STEP 8: Sending continuation (should use cache for history)...")
    print("=" * 60)
    first_response = res["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": first_response})
    messages.append({"role": "user", "content": "Now count from 1 to 10."})
    res = chat(port, messages, max_tokens=30)
    usage = res["usage"]
    timings = res.get("timings", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    print(f"  Prompt tokens: {usage['prompt_tokens']}")
    print(f"  Cached tokens: {cached}, Processed: {prompt_n}")
    print(f"  Prompt time: {prompt_ms:.1f} ms")
    print(f"  Completion: {res['choices'][0]['message']['content'][:80]}...")
    assert cached > 5000, f"Expected >5000 cached tokens, got {cached}"

    # Cleanup
    print("\n" + "=" * 60)
    print("STEP 9: Cleanup — unloading model...")
    print("=" * 60)
    api(f"/v1/models/{MODEL}/unload", "POST")
    print("  Done!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()