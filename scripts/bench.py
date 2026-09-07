"""TTFT and tokens/s, with and without the Chromium/Java relay hop."""
import argparse
import json
import time

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--host-base", default="http://127.0.0.1:8787")
parser.add_argument("--host-token", default="devtoken")
parser.add_argument("--relay-base", default="http://127.0.0.1:18080")
parser.add_argument("--relay-token", default="javadevtoken")
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--prompt", default="Count from 1 to 30, one number per line.")
args = parser.parse_args()


def bench_once(base, token, prompt):
    t0 = time.time()
    ttft = None
    n_chunks = 0
    n_chars = 0
    with httpx.stream(
        "POST", f"{base}/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={"model": "composer-2.5", "messages": [{"role": "user", "content": prompt}], "stream": True},
        timeout=180,
    ) as r:
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            if ttft is None:
                ttft = time.time() - t0
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or ""
            n_chars += len(content)
            n_chunks += 1
    total = time.time() - t0
    tps = (n_chars / 4) / total if total > 0 else 0
    return {"ttft_s": round(ttft or 0, 3), "total_s": round(total, 3), "chunks": n_chunks, "approx_tokens_per_s": round(tps, 2)}


def bench_series(label, base, token):
    print(f"--- {label} ---")
    for i in range(args.runs):
        result = bench_once(base, token, args.prompt)
        print(f"run {i}: {result}")


if __name__ == "__main__":
    bench_series("direct host (no browser hop)", args.host_base, args.host_token)
    bench_series("through Java relay (Chromium hop)", args.relay_base, args.relay_token)
