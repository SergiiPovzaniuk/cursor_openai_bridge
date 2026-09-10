"""Reproduce: ask the agent to create a file with NO path given (like a real Continue user
would), and check whether the filepath argument it picks leaks our internal sandbox cwd."""
import json
import os
import uuid
import httpx

BASE = "http://127.0.0.1:18080"
TOKEN = os.getenv("BEARER_TOKEN", "continue-local")
H = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Conversation-Id": f"conv-{uuid.uuid4().hex}",
    "X-Continue-Workspace": r"C:\remote\workspace",
    "X-Continue-OS": "windows",
    "X-Continue-Shell": "powershell",
}

TOOLS = [{"type": "function", "function": {"name": "create_new_file", "description": "Create a new file within the project", "parameters": {"type": "object", "required": ["filepath", "contents"], "properties": {"filepath": {"type": "string"}, "contents": {"type": "string"}}}}}]

messages = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Create a new file called notes.txt with the content: hello"},
]

r = httpx.post(f"{BASE}/v1/chat/completions", headers=H, json={"model": "claude-sonnet-4-5", "messages": messages, "tools": TOOLS, "stream": False}, timeout=60)
r.raise_for_status()
body = r.json()
choice = body["choices"][0]
assert choice["finish_reason"] == "tool_calls", body
call = next(c for c in choice["message"]["tool_calls"] if c["function"]["name"] == "create_new_file")
filepath = json.loads(call["function"]["arguments"])["filepath"]
assert "sandbox" not in filepath.lower(), filepath
assert filepath.replace("\\", "/").lstrip("./") == "notes.txt", filepath
print("PASS:", filepath)
