import argparse
import threading
import time

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:18080")
parser.add_argument("--token", default="devtoken")
parser.add_argument("--duration", type=int, default=90)
parser.add_argument("--concurrency", type=int, default=6)
args = parser.parse_args()

H = {"Authorization": f"Bearer {args.token}"}
stats = {"ok": 0, "err": 0}
lock = threading.Lock()
stop_at = time.time() + args.duration


def worker(i):
    # Simulates one real user: a handful of long-lived chats, each with several
    # follow-up turns (session reuse), rather than a new agent per message.
    while time.time() < stop_at:
        messages = [{"role": "user", "content": f"Reply with just the word: start{i}-{time.time()}"}]
        for turn in range(5):
            if time.time() >= stop_at:
                break
            try:
                r = httpx.post(
                    f"{args.base}/v1/chat/completions",
                    headers=H,
                    json={"model": "composer-2.5", "messages": messages, "stream": False},
                    timeout=60,
                )
                ok = r.status_code == 200
                with lock:
                    stats["ok" if ok else "err"] += 1
                if not ok:
                    print(f"worker {i} status {r.status_code}: {r.text[:200]}")
                    break
                content = r.json()["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Now reply with just the number {turn}."})
            except Exception as e:
                with lock:
                    stats["err"] += 1
                print(f"worker {i} error: {e}")
                break


threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.time() - t0
print(f"soak done: {stats['ok']} ok, {stats['err']} err in {elapsed:.1f}s across {args.concurrency} workers")
