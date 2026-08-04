#!/usr/bin/env python3
"""
Benchmark script to compare inference speed with and without --swa-full flag.

Target: Qwen3.6-27B-Fable-Fus-MTP-Q4_K_M.gguf via llama-autoloader proxy at http://127.0.0.1:1234
"""

import json
import time
import statistics
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import requests

BASE_URL = "http://127.0.0.1:1234"
MODEL_ID = "qwen3.6-27b-fable-fus-mtp"

NUM_ITERATIONS = 10
WARMUP_ITERATIONS = 3
MAX_CONSECUTIVE_FAILURES = 2

# Fixed prompt (~60 tokens) - moderate length for stable measurements
BENCHMARK_PROMPT = """You are a helpful assistant. Explain how merge sort works, including its time complexity and why it's useful for sorting large datasets."""

# Request config to get consistent output length
REQUEST_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 150,
    "stream": False,
}


@dataclass
class IterationResult:
    ttft_ms: float          # Time to first token (ms)
    tokens_per_sec: float   # Tokens generated per second
    total_time_ms: float    # Total completion time (ms)
    tokens_generated: int   # Number of output tokens


@dataclass
class BenchmarkRun:
    name: str
    iterations: List[IterationResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.iterations:
            return {}

        ttfts = [r.ttft_ms for r in self.iterations]
        tps = [r.tokens_per_sec for r in self.iterations]
        totals = [r.total_time_ms for r in self.iterations]
        tokens = [r.tokens_generated for r in self.iterations]

        def stats(vals):
            return {
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }

        return {
            "ttft_ms": stats(ttfts),
            "tokens_per_sec": stats(tps),
            "total_time_ms": stats(totals),
            "tokens_generated": stats(tokens),
        }


def print_separator(char="=", length=100):
    print(char * length)


def run_single_iteration(session: requests.Session, config: dict) -> IterationResult:
    """Run a single chat completion (non-streaming) and extract timing metrics.

    Uses non-streaming endpoint so we get accurate token counts from usage stats.
    TTFT is approximated as the time until response headers arrive.
    """

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": BENCHMARK_PROMPT}],
        **config,
    }
    payload["stream"] = False  # non-streaming for accurate token counts

    start_time = time.perf_counter()
    resp_start_time = None

    with session.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        stream=True,  # stream only to capture header arrival time (TTFT proxy)
        timeout=300,
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"Request failed: {resp.status_code} - {resp.text[:500]}")

        # Time until first byte = TTFT approximation (headers received)
        resp_start_time = time.perf_counter()

        # Read full response body
        body = b""
        for chunk in resp.iter_content(chunk_size=8192):
            body += chunk

    end_time = time.perf_counter()

    data = json.loads(body.decode("utf-8"))

    # Extract token counts from usage field (OpenAI-compatible)
    usage = data.get("usage", {})
    tokens_generated = usage.get("completion_tokens", 0) or usage.get("tokens_generated", 0)

    if tokens_generated == 0:
        # Fallback: estimate from response text word count if usage missing
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens_generated = max(len(content.split()), 1)

    total_time_ms = (end_time - start_time) * 1000
    ttft_ms = (resp_start_time - start_time) * 1000 if resp_start_time else total_time_ms

    # Tokens/sec based on total time (non-streaming: all tokens at once)
    total_time_sec = end_time - start_time
    tokens_per_sec = tokens_generated / total_time_sec if total_time_sec > 0 else 0.0

    return IterationResult(
        ttft_ms=ttft_ms,
        tokens_per_sec=tokens_per_sec,
        total_time_ms=total_time_ms,
        tokens_generated=tokens_generated,
    )


def check_model_ready(session: requests.Session) -> bool:
    """Check if the target model is currently ready."""
    try:
        resp = session.get(f"{BASE_URL}/v1/models/{MODEL_ID}", timeout=5)
        if resp.status_code != 200:
            return False
        data = resp.json()
        return data.get("ready", False)
    except Exception:
        return False


def run_benchmark(config: dict, label: str) -> BenchmarkRun:
    """Run a full benchmark with warmup + measurement iterations."""
    print(f"\n{'='*60}")
    print(f"Running benchmark: {label}")
    print(f"{'='*60}")

    run = BenchmarkRun(name=label)
    session = requests.Session()
    consecutive_failures = 0

    # Warmup iterations (discard results)
    print(f"Warming up ({WARMUP_ITERATIONS} iteration(s))...", flush=True)
    for i in range(WARMUP_ITERATIONS):
        try:
            run_single_iteration(session, config)
            print("  done", flush=True)
            consecutive_failures = 0
        except Exception as e:
            print(f"  Warning during warmup: {e}", flush=True)

    # Measurement iterations
    print(f"\nMeasurement iterations ({NUM_ITERATIONS}x):", flush=True)
    for i in range(NUM_ITERATIONS):
        print(f"  Iteration {i+1}/{NUM_ITERATIONS}...", flush=True)

        # Verify model is ready before each measurement
        if not check_model_ready(session):
            print(f"    Model not ready, attempting to reload...", flush=True)
            try:
                load_resp = session.post(f"{BASE_URL}/v1/models/{MODEL_ID}/load", timeout=60)
                load_resp.raise_for_status()
                time.sleep(3)
            except Exception as e:
                print(f"    Reload failed: {e}", flush=True)

        if not check_model_ready(session):
            consecutive_failures += 1
            print(f"    SKIPPED (model not ready, failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})", flush=True)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\nABORTING benchmark '{label}': too many consecutive failures.", flush=True)
                return run

        try:
            result = run_single_iteration(session, config)
            run.iterations.append(result)
            consecutive_failures = 0
            print(
                f"    TTFT={result.ttft_ms:.0f}ms "
                f"TPS={result.tokens_per_sec:.1f} "
                f"Total={result.total_time_ms:.0f}ms "
                f"Tokens={result.tokens_generated}",
                flush=True,
            )
        except Exception as e:
            consecutive_failures += 1
            print(f"    FAILED ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}", flush=True)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\nABORTING benchmark '{label}': too many consecutive failures.", flush=True)
                return run

    return run


def ensure_model_loaded(session: requests.Session):
    """Ensure the target model is loaded and ready."""
    resp = session.get(f"{BASE_URL}/v1/models")
    resp.raise_for_status()
    models = resp.json().get("data", [])

    for m in models:
        if m["id"] == MODEL_ID or m.get("name") == MODEL_ID:
            if m.get("ready"):
                print(f"Model {MODEL_ID} is already loaded and ready (port={m.get('port')})")
                return True
            elif m.get("loaded"):
                print(f"Model {MODEL_ID} is loading, waiting for ready...")
                time.sleep(2)
                return ensure_model_loaded(session)
            else:
                print(f"Model {MODEL_ID} found but not loaded. Loading now...")
                load_resp = session.post(f"{BASE_URL}/v1/models/{MODEL_ID}/load")
                load_resp.raise_for_status()
                print("Waiting for model to become ready...")
                time.sleep(5)
                return ensure_model_loaded(session)

    print(f"ERROR: Model {MODEL_ID} not found in autoloader!")
    return False


def reload_model_with_config(session: requests.Session, new_args: str):
    """Unload, update args via sidecar/config API, and reload the model."""
    model_id = MODEL_ID

    # Unload current instance
    print(f"Unloading {model_id}...")
    resp = session.post(f"{BASE_URL}/v1/models/{model_id}/unload")
    resp.raise_for_status()
    time.sleep(5)  # wait for process to fully terminate and VRAM to free

    # Update config args via API
    print(f"Updating model config with args: {new_args}")
    resp = session.put(
        f"{BASE_URL}/v1/models/{model_id}/config",
        json={"args": new_args},
    )
    resp.raise_for_status()

    # Reload with new config
    print(f"Reloading {model_id}...")
    resp = session.post(f"{BASE_URL}/v1/models/{model_id}/load")
    resp.raise_for_status()

    # Wait for ready
    print("Waiting for model to become ready...")
    time.sleep(3)
    ensure_model_loaded(session)


def get_current_args(session: requests.Session) -> str:
    """Get current args from the model config."""
    resp = session.get(f"{BASE_URL}/v1/models/{MODEL_ID}")
    resp.raise_for_status()
    data = resp.json()
    return data.get("config", {}).get("args", "")


def print_comparison_table(run_without: BenchmarkRun, run_with: BenchmarkRun):
    """Print a clear comparison table."""
    s1 = run_without.summary()
    s2 = run_with.summary()

    if not s1 or not s2:
        print("ERROR: One or both benchmarks have no data to compare.")
        return

    def pct_diff(base, test):
        if base == 0:
            return float("inf") if test != 0 else 0.0
        return ((test - base) / base) * 100

    print_separator()
    print("SWA-FULL BENCHMARK COMPARISON RESULTS")
    print(f"Model: {MODEL_ID}")
    print(f"Iterations per config: {NUM_ITERATIONS}")
    print_separator()

    # Helper to format stats compactly
    def fmt(v):
        return f"{v:.2f}"

    def fmt_ms(v):
        return f"{v:.0f}ms"

    def pct_str(base, test):
        d = pct_diff(base, test)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}%"

    # Header
    print(f"\n{'Metric':<25} {'Without --swa-full':>22} {'With --swa-full':>20} {'Difference':>14}")
    print_separator("-")

    # TTFT (ms) - lower is better (approximated as time to first byte in non-streaming mode)
    ttft_base = s1["ttft_ms"]["mean"]
    ttft_test = s2["ttft_ms"]["mean"]
    print(
        f"{'TTFT* (mean, ms)':<25} "
        f"{fmt_ms(ttft_base):>22} "
        f"{fmt_ms(ttft_test):>20} "
        f"{pct_str(ttft_base, ttft_test):>14}"
    )

    # Tokens/sec - higher is better
    tps_base = s1["tokens_per_sec"]["mean"]
    tps_test = s2["tokens_per_sec"]["mean"]
    print(
        f"{'Tokens/sec (mean)':<25} "
        f"{fmt(tps_base):>22} "
        f"{fmt(tps_test):>20} "
        f"{pct_str(tps_base, tps_test):>14}"
    )

    # Total time (ms) - lower is better
    total_base = s1["total_time_ms"]["mean"]
    total_test = s2["total_time_ms"]["mean"]
    print(
        f"{'Total time (mean, ms)':<25} "
        f"{fmt_ms(total_base):>22} "
        f"{fmt_ms(total_test):>20} "
        f"{pct_str(total_base, total_test):>14}"
    )

    # Tokens generated (should be similar)
    tok_base = s1["tokens_generated"]["mean"]
    tok_test = s2["tokens_generated"]["mean"]
    print(
        f"{'Tokens generated (mean)':<25} "
        f"{fmt(tok_base):>22} "
        f"{fmt(tok_test):>20} "
        f"{pct_str(tok_base, tok_test):>14}"
    )

    print_separator("-")

    # Detailed stats
    print("\nDETAILED STATISTICS:\n")

    for metric in ["ttft_ms", "tokens_per_sec", "total_time_ms"]:
        label = metric.replace("_", " ").title()
        base_stats = s1[metric]
        test_stats = s2[metric]

        print(f"  {label}:")
        print(f"    Without --swa-full: mean={base_stats['mean']:.2f} median={base_stats['median']:.2f} "
              f"stdev={base_stats['stdev']:.2f} [{base_stats['min']:.2f}, {base_stats['max']:.2f}]")
        print(f"    With --swa-full:    mean={test_stats['mean']:.2f} median={test_stats['median']:.2f} "
              f"stdev={test_stats['stdev']:.2f} [{test_stats['min']:.2f}, {test_stats['max']:.2f}]")

    print("\nNotes:")
    print("  * TTFT = time to first byte (non-streaming mode, proxy for actual TTFT)")
    print("  * Tokens/sec = completion_tokens / total_time (includes prompt processing)")
    print("  * Token counts from API usage field (completion_tokens)")

    print_separator()


def main():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    # Verify autoloader is reachable
    try:
        resp = session.get(f"{BASE_URL}/v1/models", timeout=5)
        resp.raise_for_status()
        print(f"Connected to llama-autoloader at {BASE_URL}")
    except Exception as e:
        print(f"ERROR: Cannot reach autoloader at {BASE_URL}: {e}")
        sys.exit(1)

    # Get original args to restore later
    original_args = get_current_args(session)
    print(f"\nOriginal model args:\n  {original_args}\n")

    # Ensure model is loaded first (for baseline without --swa-full)
    if not ensure_model_loaded(session):
        sys.exit(1)

    try:
        # Phase 1: Baseline WITHOUT --swa-full
        print("\n" + "=" * 60)
        print("PHASE 1: Baseline (WITHOUT --swa-full)")
        print("=" * 60)

        # Make sure --swa-full is NOT in args for baseline
        baseline_args = original_args
        if "--swa-full" in baseline_args:
            baseline_args = baseline_args.replace("--swa-full", "").strip()
            reload_model_with_config(session, baseline_args)

        run_without = run_benchmark(REQUEST_PARAMS, "Without --swa-full")

        # Phase 2: Test WITH --swa-full
        print("\n" + "=" * 60)
        print("PHASE 2: Testing (WITH --swa-full)")
        print("=" * 60)

        swa_args = baseline_args + " --swa-full" if baseline_args else "--swa-full"
        reload_model_with_config(session, swa_args)

        run_with = run_benchmark(REQUEST_PARAMS, "With --swa-full")

        # Print comparison
        print_comparison_table(run_without, run_with)

    finally:
        # Restore original config
        print("\nRestoring original model configuration...")
        reload_model_with_config(session, original_args)
        print("Done. Model restored to original args.")


if __name__ == "__main__":
    main()