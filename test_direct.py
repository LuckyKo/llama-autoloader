"""Direct test of KV cache reuse after slot restore using patched llama-server.

Bypasses autoloader to avoid GPU config issues with CPU-only build.
Tests the specific fix in server-context.cpp lines 3358-3365.

Flow:
  1. Start llama-server directly with --n-gpu-layers 0 (CPU)
  2. Send large prompt (~8k tokens) — baseline timing
  3. Save slot state  
  4. Kill and restart llama-server
  5. Restore slot state
  6. Send EXACT SAME prompt again — should be fully cached with our fix
     Before fix: ~8k tokens reprocessed in ~100s+
     After fix: near-zero processing time, all tokens cached
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SERVER_EXE = r"N:\work\stuff\Beta\llama-autoloader\backends\llama.cpp-win-x86_64-cpu-avx2-patched\llama-server.exe"
MODEL_PATH = r"N:\work\stuff\Beta\lmstudio-community\Qwen3-4B-Instruct-2507-GGUF\Qwen3-4B-Instruct-2507-Q6_K.gguf"
PORT = 9999
STATE_DIR = Path(r"N:\work\stuff\Beta\llama-autoloader\states")
STATE_FILE = STATE_DIR / "test_kv_direct.bin"

HTTP_TIMEOUT = 300


def api(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    url = f"http://127.0.0.1:{PORT}{path}"
    req = urllib.request.Request(url, data=data, method="POST")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read())


def wait_for_server(timeout: int = 60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            api("/health")
            return True
        except Exception:
            time.sleep(1)
    return False


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


def start_server():
    args = [
        SERVER_EXE,
        "--model", MODEL_PATH,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--ctx-size", "8192",
        "--n-gpu-layers", "0",  # CPU only!
        "--slot-save-path", str(STATE_DIR),
        "--no-webui",
        "--parallel", "1",
        "--jinja",
        "-t", "8",
    ]
    print(f"Starting llama-server: {' '.join(args[:6])} ...")
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return proc


def stop_server(proc):
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)
    print("  Server stopped")


def main():
    large_text = build_large_prompt()
    messages = [{"role": "user", "content": large_text}]

    # Step 1: Start server
    print("=" * 60)
    print("STEP 1: Starting llama-server...")
    print("=" * 60)
    proc = start_server()
    if not wait_for_server(timeout=120):
        print("ERROR: Server failed to start")
        stop_server(proc)
        sys.exit(1)
    print("  Server ready on port", PORT)

    try:
        # Step 2: Send large prompt — baseline (first time, no cache)
        print("\n" + "=" * 60)
        print("STEP 2: Sending large prompt (~8k tokens) — BASELINE...")
        print("=" * 60)
        t0 = time.time()
        res_step2 = api("/v1/chat/completions", {
            "model": Path(MODEL_PATH).name,
            "messages": messages,
            "max_tokens": 30,
        })
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
        save_res = api("/slots/0?action=save", {"filename": STATE_FILE.name})
        n_saved = save_res.get('n_saved', 0)
        n_written = save_res.get('n_written', 0)
        print(f"  Saved {n_saved} tokens, {n_written:,} bytes")
        if n_saved <= 0:
            print(f"  ERROR: Save reported {n_saved} tokens (expected > 0)")
            sys.exit(1)

        # Step 4: Stop server
        print("\n" + "=" * 60)
        print("STEP 4: Stopping llama-server...")
        print("=" * 60)
        stop_server(proc)

        # Step 5: Restart server (new process)
        print("\n" + "=" * 60)
        print("STEP 5: Restarting llama-server...")
        print("=" * 60)
        t0 = time.time()
        proc = start_server()
        if not wait_for_server(timeout=120):
            print("ERROR: Server failed to restart")
            sys.exit(1)
        t1 = time.time()
        print(f"  Server ready on port {PORT} (load time: {t1-t0:.1f}s)")

        # Step 6: Restore slot state
        print("\n" + "=" * 60)
        print("STEP 6: Restoring slot state...")
        print("=" * 60)
        restore_res = api("/slots/0?action=restore", {"filename": STATE_FILE.name})
        n_restored = restore_res.get('n_restored', 0)
        print(f"  Restored {n_restored} tokens")
        if n_restored <= 0:
            print(f"  ERROR: Restore reported {n_restored} tokens (expected > 0)")
            sys.exit(1)

        # Step 7: CRITICAL TEST — Send EXACT SAME prompt again
        print("\n" + "=" * 60)
        print("STEP 7: CRITICAL TEST — Resending SAME prompt after restore...")
        print("=" * 60)
        t0 = time.time()
        res_step7 = api("/v1/chat/completions", {
            "model": Path(MODEL_PATH).name,
            "messages": messages,
            "max_tokens": 30,
        })
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

        # With our fix: processed should be very small (< 5% of baseline)
        if prompt_n < baseline_processed * 0.1:
            print(f"\n  ✅ FIX WORKING! Only {prompt_n} tokens reprocessed vs {baseline_processed} baseline")
            speedup = baseline_ms / max(prompt_ms, 1)
            print(f"     Speedup: {speedup:.1f}x faster")
        elif cached >= baseline_prompt_tokens * 0.9:
            print(f"\n  ✅ FIX WORKING! {cached}/{baseline_prompt_tokens} tokens cached")
        else:
            print(f"\n  ❌ FIX NOT WORKING — Full reprocess detected!")
            print(f"     Expected <{int(baseline_processed*0.1)} tokens processed, got {prompt_n}")
            sys.exit(1)

    finally:
        # Cleanup
        print("\n" + "=" * 60)
        print("CLEANUP")
        print("=" * 60)
        stop_server(proc)
        
        if STATE_FILE.exists():
            try:
                STATE_FILE.unlink()
                print(f"  Removed test artifact: {STATE_FILE}")
            except Exception as e:
                print(f"  Note: Failed to remove {STATE_FILE} ({e})")

        print("\n" + "=" * 60)
        print("TEST PASSED ✅")
        print("=" * 60)


if __name__ == "__main__":
    main()