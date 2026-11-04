"""Graceful autoloader restart: save state → unload all → JIT reload → restore.

Usage:
    python restart.py [--port 1234] [--label restart]

Sequence:
  1. GET /v1/status → list loaded models
  2. POST /v1/models/{id}/state/save  (for each loaded model)
  3. POST /v1/unload_all
  4. Wait for backend to die
  5. JIT loader restarts the backend on next model request
  6. POST /v1/models/{id}/state/load  (for each previously loaded model)
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:1234"
LABEL = "restart"


def api(method, path, body=None, timeout=120):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else b"{}"
    req = urllib.request.Request(url, data=data, method=method,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read().decode("utf-8", "replace")
        elapsed = time.time() - t0
        try:
            return json.loads(raw), elapsed
        except json.JSONDecodeError:
            return raw, elapsed
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        return f"HTTP {e.code}: {raw[:200]}", time.time() - t0


def main():
    global BASE, LABEL
    if "--port" in sys.argv:
        port = sys.argv[sys.argv.index("--port") + 1]
        BASE = f"http://127.0.0.1:{port}"
    if "--label" in sys.argv:
        LABEL = sys.argv[sys.argv.index("--label") + 1]

    print(f"Autoloader: {BASE}")
    print(f"Label: {LABEL}")
    print()

    # 1. Get loaded models
    print("[1] Checking loaded models...")
    status, _ = api("GET", "/v1/status")
    if isinstance(status, str):
        print(f"    FAIL: {status}")
        return 1
    loaded = status.get("uptime_models", [])
    if not loaded:
        print("    No models loaded. Nothing to restart.")
        print("    NOTE: if you just ran a safe-mode command, the models were")
        print("    already evacuated. The next model request will spawn the")
        print("    current binary automatically - no restart needed.")
        return 0
    model_ids = [m["id"] for m in loaded]
    print(f"    Loaded: {model_ids}")

    # 2. Save state for each loaded model
    print("[2] Saving state...")
    for mid in model_ids:
        result, elapsed = api("POST", f"/v1/models/{mid}/state/save",
                              {"label": LABEL})
        if isinstance(result, str) and result.startswith("HTTP"):
            print(f"    {mid}: {result} ({elapsed:.1f}s)")
        else:
            print(f"    {mid}: saved ({elapsed:.1f}s)")

    # 3. Unload all
    print("[3] Unloading all...")
    result, elapsed = api("POST", "/v1/unload_all")
    if isinstance(result, str) and result.startswith("HTTP"):
        print(f"    FAIL: {result} ({elapsed:.1f}s)")
        return 1
    failed = result.get("failed", [])
    print(f"    Unloaded {result.get('unloaded', 0)} model(s), "
          f"{len(failed)} failed ({elapsed:.1f}s)")

    # 4. Wait for backend processes to die
    print("[4] Waiting for backends to exit...")
    time.sleep(3)

    # 5. Restore state for each model (triggers JIT reload)
    print("[5] Restoring state (triggers JIT reload)...")
    for mid in model_ids:
        result, elapsed = api("POST", f"/v1/models/{mid}/state/load",
                              {"label": LABEL}, timeout=300)
        if isinstance(result, str) and result.startswith("HTTP"):
            print(f"    {mid}: {result} ({elapsed:.1f}s)")
        else:
            print(f"    {mid}: restored ({elapsed:.1f}s)")

    print()
    print("Restart complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
