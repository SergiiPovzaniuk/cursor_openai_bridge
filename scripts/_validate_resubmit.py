"""Reproduce: Continue resends the ORIGINAL first prompt on the SAME conversation id
(simulating a dropped-stream retry / "Resubmit last message"). Before the fix this silently
orphaned the live session and created a duplicate, causing the model to redo the read+edit
from scratch. Assert only one session is ever created for this conversation id."""
import os
import uuid
import httpx

BASE = "http://127.0.0.1:18080"
TOKEN = "vNjo6ZDNQamYb2w98solftTvzoyM1vdZM5dQLIyhi4"
CONV_ID = f"conv-{uuid.uuid4().hex}"
H = {"Authorization": f"Bearer {TOKEN}", "X-Conversation-Id": CONV_ID}
WORKDIR = os.path.join(os.environ["TEMP"], "relay_tool_validate")
os.makedirs(WORKDIR, exist_ok=True)
TARGET = os.path.join(WORKDIR, "call_me3.txt")
with open(TARGET, "w", encoding="utf-8") as f:
    f.write("hello world\n")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Reply with exactly the word: pong"},
]


def post():
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=H, json={"model": "claude-sonnet-4-5", "messages": messages, "stream": False}, timeout=60)
    r.raise_for_status()
    return r.json()


HOST_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "host_stdout.log")


def session_created_count():
    with open(HOST_LOG, encoding="utf-8", errors="ignore") as f:
        return sum(1 for line in f if '"session created"' in line or "session created" in line)


before = session_created_count()
r1 = post()
print("turn 1:", r1["choices"][0]["message"]["content"][:80])
mid = session_created_count()
print(f"sessions created by turn 1: {mid - before}")
assert mid - before == 1, f"expected exactly 1 new session, got {mid - before}"

r2 = post()  # resubmit of the SAME original messages, same conv id
print("turn 2 (resubmit):", r2["choices"][0]["message"]["content"][:80])
after = session_created_count()
print(f"sessions created by resubmit: {after - mid}")
assert after - mid == 0, f"resubmit created a NEW session instead of reusing existing one ({after - mid} new sessions) -- this is the orphan/duplicate bug!"

print("\nPASS: resubmit on same conversation id reused the existing session, no orphan/duplicate.")
