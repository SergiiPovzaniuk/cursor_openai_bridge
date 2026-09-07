"""Continue emulator driven straight against the host's /v1 (no browser)."""
import argparse
import json
import threading
import time

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:8787")
parser.add_argument("--token", default="devtoken")
args = parser.parse_args()

BASE = args.base
H = {"Authorization": f"Bearer {args.token}"}
WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
    },
}


def post(payload, stream=False, timeout=180):
    if stream:
        return httpx.stream("POST", f"{BASE}/v1/chat/completions", headers=H, json=payload, timeout=timeout)
    return httpx.post(f"{BASE}/v1/chat/completions", headers=H, json=payload, timeout=timeout)


def test_tool_loop():
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Immediately call the write_file tool once with path='hello.txt' content='hi'. Do not call any other tool, do not explore, do not read files. Just call write_file now."},
    ]
    r = post({"model": "composer-2.5", "messages": messages, "tools": [WRITE_TOOL], "stream": False}, timeout=300)
    body = r.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls", body
    calls = choice["message"]["tool_calls"]
    assert calls, body
    call = calls[0]
    assert call["function"]["name"] == "write_file"

    messages.append(choice["message"])
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": "wrote hello.txt"})
    r2 = post({"model": "composer-2.5", "messages": messages, "tools": [WRITE_TOOL], "stream": False}, timeout=300)
    body2 = r2.json()
    assert body2["choices"][0]["finish_reason"] == "stop", body2
    print("tool_loop OK:", body2["choices"][0]["message"]["content"][:120])


def test_concurrent_chats(n=4):
    results = {}

    def run(i):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Remember the secret word BANANA and reply OK."},
        ]
        r = post({"model": "composer-2.5", "messages": messages, "stream": False})
        results[i] = r.json()["choices"][0]["message"]["content"]

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"concurrent_chats OK: {n} chats in {time.time() - t0:.1f}s", results)


def test_cancellation():
    messages = [{"role": "user", "content": "Count slowly from 1 to 50, one number per line."}]
    with post({"model": "composer-2.5", "messages": messages, "stream": True}, stream=True) as r:
        it = r.iter_lines()
        next(it)
        r.close()
    print("cancellation OK: closed mid-stream without hanging")


def test_overflow():
    big = "x" * 2_000_000
    messages = [{"role": "user", "content": big}]
    r = post({"model": "composer-2.5", "messages": messages, "stream": False}, timeout=30)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["code"] == "context_length_exceeded", err
    print("overflow OK:", err["message"])


if __name__ == "__main__":
    test_tool_loop()
    test_concurrent_chats()
    test_cancellation()
    test_overflow()
    print("ALL SIMULATE_CONTINUE TESTS PASSED")
