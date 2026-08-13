#!/usr/bin/env python3
"""Test KV cache reuse after state restore with patched llama-server."""
import requests
import json
import time

AUTOLOADER = "http://127.0.0.1:9124"  # autoloader proxy for state save/load
SERVER = "http://127.0.0.1:9001"      # direct llama-server for completions
MODEL = "Qwen3-4B-Instruct-2507-Q6_K.gguf"

# Generate a substantial prompt (~500+ tokens) to establish context
prompt = """You are an expert software engineer specializing in Python and system architecture. 
Please provide a detailed explanation of the following topics:

1. Design Patterns:
   - Singleton pattern and its use cases
   - Factory pattern variations
   - Observer pattern implementation details
   
2. System Architecture:
   - Microservices vs monolithic architecture trade-offs
   - Event-driven architecture principles
   - Caching strategies (Redis, Memcached)
   
3. Database Design:
   - Normalization vs denormalization
   - Indexing strategies for performance
   - Transaction isolation levels
   
4. API Design:
   - RESTful best practices
   - GraphQL advantages and disadvantages
   - Rate limiting implementations

For each topic, provide concrete examples and explain when to use each approach."""

def chat_completion(messages, stream=False):
    """Send a chat completion request via autoloader proxy."""
    url = f"{AUTOLOADER}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": stream,
    }
    resp = requests.post(url, json=payload)
    return resp.json()

def get_timings(completion):
    """Extract timing info from completion response."""
    if "usage" in completion:
        usage = completion["usage"]
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    return {}

print("=" * 60)
print("KV Cache Reuse Test")
print("=" * 60)

# Step 1: Initial completion with full prompt processing
print("\n[Step 1] Initial completion (full prompt processing)...")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

start = time.time()
completion1 = chat_completion(messages)
elapsed1 = time.time() - start

timings1 = get_timings(completion1)
print(f"  Prompt tokens: {timings1.get('prompt_tokens', 'N/A')}")
print(f"  Completion tokens: {timings1.get('completion_tokens', 'N/A')}")
print(f"  Total time: {elapsed1:.2f}s")

# Step 2: Save state via autoloader proxy
print("\n[Step 2] Saving slot state...")
save_resp = requests.post(f"{AUTOLOADER}/v1/models/{MODEL}/state/save", json={"slot_id": 0})
print(f"  Save response: {save_resp.status_code} - {save_resp.text[:200]}")

# Step 3: Restore state via autoloader proxy  
print("\n[Step 3] Restoring slot state...")
restore_resp = requests.post(f"{AUTOLOADER}/v1/models/{MODEL}/state/load", json={"slot_id": 0})
print(f"  Restore response: {restore_resp.status_code} - {restore_resp.text[:200]}")

# Step 4: Send same prompt again - should use KV cache
print("\n[Step 4] Same prompt after restore (should use KV cache)...")
start = time.time()
completion2 = chat_completion(messages)
elapsed2 = time.time() - start

timings2 = get_timings(completion2)
print(f"  Prompt tokens: {timings2.get('prompt_tokens', 'N/A')}")
print(f"  Completion tokens: {timings2.get('completion_tokens', 'N/A')}")
print(f"  Total time: {elapsed2:.2f}s")

# Summary
print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"  Initial prompt processing:  {elapsed1:.2f}s ({timings1.get('prompt_tokens', '?')} tokens)")
print(f"  After restore (cached):     {elapsed2:.2f}s ({timings2.get('prompt_tokens', '?')} tokens)")

if elapsed1 > 0 and elapsed2 > 0:
    speedup = elapsed1 / elapsed2
    print(f"  Speedup factor:           {speedup:.1f}x")
    
    if speedup > 5:
        print("\n  ✓ KV cache reuse WORKING - prompt was cached after restore!")
    elif speedup > 1.5:
        print("\n  ~ Partial caching observed")
    else:
        print("\n  ✗ KV cache reuse NOT working - full reprocess detected")