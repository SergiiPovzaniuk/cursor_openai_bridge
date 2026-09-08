"""Reproduce: ask the agent to create a file with NO path given (like a real Continue user
would), and check whether the filepath argument it picks leaks our internal sandbox cwd."""
import json
import uuid
import httpx

BASE = "http://127.0.0.1:18080"
TOKEN = "vNjo6ZDNQamYb2w98solftTvzoyM1vdZM5dQLIyhi4"
H = {"Authorization": f"Bearer {TOKEN}", "X-Conversation-Id": f"conv-{uuid.uuid4().hex}"}

TOOLS = [{"type": "function", "function": {"name": "create_new_file", "description": "Create a new file within the project", "parameters": {"type": "object", "required": ["filepath", "contents"], "properties": {"filepath": {"type": "string"}, "contents": {"type": "string"}}}}}]

messages = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Create a new file called notes.txt with the content: hello"},
]

r = httpx.post(f"{BASE}/v1/chat/completions", headers=H, json={"model": "claude-sonnet-4-5", "messages": messages, "tools": TOOLS, "stream": False}, timeout=60)
body = r.json()
choice = body["choices"][0]
print("finish_reason:", choice["finish_reason"])
if choice["finish_reason"] == "tool_calls":
    for call in choice["message"]["tool_calls"]:
        args = json.loads(call["function"]["arguments"])
        print("filepath chosen by model:", args.get("filepath"))
        if "sandbox" in args.get("filepath", "").lower():
            print("\n*** LEAK CONFIRMED: model used our internal sandbox path ***")
        else:
            print("\nOK: model used a plain relative/neutral path")
else:
    print(choice["message"].get("content"))
