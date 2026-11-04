#!/usr/bin/env python3
"""In-place autoloader SERVER restart with KV state preservation.

Unlike ``restart.py`` (model-level: save -> unload -> reload -> restore inside
the SAME process), this restarts the whole server process in place via
``POST /v1/restart`` while preserving each loaded model's KV state across the
restart:

  1. save state for every loaded model (KV -> disk)
  2. POST /v1/restart   (the endpoint unloads models, then re-execs in place)
  3. wait for the server to come back (poll /health)
  4. restore state for every previously-loaded model (JIT-loads + KV <- disk)

The terminal that started the server stays the running server (the re-exec is
in-place, not a detached child). Usage:

    python restart_server.py [--port 1234] [--label restart] [--timeout 120]
"""

import argparse
import sys
import time

import requests


def base(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def get_loaded_models(port: int) -> list[str]:
    """Return the ids of models with a live engine (loaded=True)."""
    try:
        r = requests.get(f"{base(port)}/v1/models", timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        models = data.get("data", data) if isinstance(data, dict) else data
        return [m["id"] for m in models if m.get("loaded")]
    except requests.exceptions.RequestException as e:
        print(f"  ! cannot list models: {e}")
        return []


def save_state(port: int, model: str, label: str) -> bool:
    # StateLabel is a JSON body (label: str = "default") - NOT a query param.
    r = requests.post(f"{base(port)}/v1/models/{model}/state/save",
                      json={"label": label}, timeout=120)
    return r.status_code == 200


def restore_state(port: int, model: str, label: str) -> bool:
    # JIT-loads the model if needed, then restores the KV state from disk.
    r = requests.post(f"{base(port)}/v1/models/{model}/state/load",
                      json={"label": label}, timeout=180)
    return r.status_code == 200


def wait_up(port: int, timeout: float) -> bool:
    """Poll /health until the re-exec'd server answers (or the timeout elapses)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base(port)}/health", timeout=5).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="In-place autoloader server restart with KV state preservation.")
    ap.add_argument("--port", type=int, default=1234,
                    help="main client-facing port (default 1234)")
    ap.add_argument("--label", default="restart",
                    help="state label used for save/restore (default 'restart', "
                         "so the 'default' state is not clobbered)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="max seconds to wait for the server to come back up "
                         "(default 120)")
    args = ap.parse_args()

    models = get_loaded_models(args.port)
    if models:
        print(f"Saving state for {len(models)} loaded model(s) [label={args.label}]...")
        for m in models:
            print(f"  saved {m}" if save_state(args.port, m, args.label)
                  else f"  ! save failed for {m} (restart proceeds)")
    else:
        print("No loaded models; skipping state save.")

    print("Restarting server (POST /v1/restart)...")
    try:
        requests.post(f"{base(args.port)}/v1/restart", timeout=10)
    except requests.exceptions.RequestException:
        # The connection closes when the old process exits mid re-exec - expected.
        pass

    print(f"Waiting for the server to come back (up to {args.timeout:.0f}s)...")
    if not wait_up(args.port, args.timeout):
        print("Server did not come back within the timeout.")
        sys.exit(1)
    print("Server is back up.")

    if models:
        print(f"Restoring state for {len(models)} model(s) [label={args.label}]...")
        ok = 0
        for m in models:
            if restore_state(args.port, m, args.label):
                print(f"  restored {m}")
                ok += 1
            else:
                print(f"  ! restore failed for {m}")
        print(f"Restored {ok}/{len(models)} model(s).")
    print("Done.")


if __name__ == "__main__":
    main()
