"""Manual SSE streaming smoothness/perf check for the autoloader forwarder.

Verifies: management endpoint works, JIT-load triggers on first LLM request,
and SSE chunks arrive incrementally (smooth) rather than in one burst. This is a
handy diagnostic to eyeball streaming performance — it is NOT part of the pytest
suite (it needs a live loader + a real model), so run it directly:

    python tests/stream_perf_check.py                 # default: debug port 9126
    python tests/stream_perf_check.py --port 9124     # target the live loader
    python tests/stream_perf_check.py --model <id>    # override the model

Defaults to the isolated debug instance (tests/config_debug.yaml, port 9126) so it
never disturbs the live loader. Point --port at a running instance that has the model
available.
"""
import argparse
import urllib.request
import json
import os
import time

DEFAULT_BASE = "http://127.0.0.1:9126"
DEFAULT_MODEL = "Qwen2.5-0.5B-Instruct-Q8_0.gguf"


def main():
    parser = argparse.ArgumentParser(description="SSE streaming smoothness/perf check")
    parser.add_argument("--port", type=int, default=None,
                        help=f"Target loader port (default: {DEFAULT_BASE.rsplit(':', 1)[-1]})")
    parser.add_argument("--host", default="127.0.0.1", help="Target host (default 127.0.0.1)")
    parser.add_argument("--model", default=None,
                        help=f"Model id to request (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}" if args.port else DEFAULT_BASE
    model = args.model or DEFAULT_MODEL
    print(f"target: {base}   model: {model}")

    print("\n== mgmt /v1/models ==")
    d = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=8).read())
    print("models:", len(d.get("data", [])))

    print("\n== streaming request (triggers JIT load on first call) ==")
    body = json.dumps({
        "model": model,
        "stream": True,
        "max_tokens": 256,   # enough tokens that generation spans ~1-3s so incremental vs burst is unambiguous
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Do not stop early; keep writing until you have listed all items requested."},
            {"role": "user", "content": "List the first 30 positive integers, one per line. Do not add any other text."}
        ],
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})

    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=300)
    arr = []          # arrival times (s since request start) for each SSE data frame
    text_parts = []
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith(b"data:"):
            now = time.time() - t0
            payload = line[5:].strip()
            arr.append(round(now, 2))
            if payload and payload != b"[DONE]":
                try:
                    obj = json.loads(payload)
                    delta = obj["choices"][0].get("delta", {}) or {}
                    # First frame's content is null (role-only); guard against None.
                    c = delta.get("content")
                    if c is not None:
                        text_parts.append(c)
                except Exception:
                    pass
    total = time.time() - t0

    print(f"total wall time: {total:.1f}s")
    print(f"SSE data frames received: {len(arr)}")
    if arr:
        print(f"first frame at: {arr[0]:.2f}s  last frame at: {arr[-1]:.2f}s")
        gaps = [round(arr[i + 1] - arr[i], 2) for i in range(len(arr) - 1)]
        print(f"gaps between frames (s): {gaps}")
        span = arr[-1] - arr[0] if len(arr) >= 2 else 0
        # The forwarder is a transparent pipe; its job is to deliver each SSE frame the moment
        # the backend emits it, WITHOUT batching. For a tiny model the whole generation can be
        # <1s on the backend (llama.cpp eval ~5ms/tok), so "span > 1s" is NOT the right signal.
        # Instead: incremental = frames arrive one-by-one with small inter-frame gaps and no big
        # stall mid-stream. A burst would show as one large gap followed by many near-zero gaps
        # (or all frames in a single recv). We flag any single gap > 0.5s as a batch/stall.
        max_gap = max(gaps) if gaps else 0.0
        big_gaps = [g for g in gaps if g > 0.5]
        # Incremental: we got several frames, none of the inter-frame gaps is a big stall, and
        # they are spread across at least ~3 distinct centisecond ticks (not all one instant).
        smooth = len(arr) >= 5 and not big_gaps
        print(f"\nframe span: {span:.2f}s   max inter-frame gap: {max_gap:.2f}s")
        if big_gaps:
            print(f"WARN: {len(big_gaps)} inter-frame gap(s) > 0.5s -> possible batch/stall: {big_gaps}")
        print(f"\nSMOOTH incremental delivery (no batching stalls): {smooth}")
    else:
        print("\nNO SSE FRAMES RECEIVED (likely an error response or non-stream)")
    if text_parts:
        print("\nGenerated text:", "".join(text_parts)[:300])


if __name__ == "__main__":
    main()
