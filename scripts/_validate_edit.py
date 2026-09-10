"""End-to-end validation: agent MODIFIES an existing real file via a search/replace edit tool."""
import json
import os
import httpx

BASE = "http://127.0.0.1:18080"
TOKEN = os.getenv("BEARER_TOKEN", "continue-local")
WORKDIR = os.path.join(os.environ["TEMP"], "relay_tool_validate")
H = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Continue-Workspace": WORKDIR,
    "X-Continue-OS": "windows",
    "X-Continue-Shell": "powershell",
}
os.makedirs(WORKDIR, exist_ok=True)
TARGET = os.path.join(WORKDIR, "config.txt")

ORIGINAL = "app_name = demo\nport = 8000\ndebug = false\n"
with open(TARGET, "w", encoding="utf-8") as f:
    f.write(ORIGINAL)

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Edit an existing file by replacing an exact substring with a new one",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["filepath", "old_string", "new_string"],
        },
    },
}


def post(payload):
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=H, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


print("=== edit_file (modify existing file) ===")
print("original content:", repr(ORIGINAL))

messages = [
    {"role": "system", "content": "You are a coding agent with file editing tools."},
    {"role": "user", "content": f"In the file {TARGET}, change the port from 8000 to 9090. Use edit_file with the exact old_string 'port = 8000' and new_string 'port = 9090'. Call it exactly once."},
]
body = post({"model": "claude-sonnet-4-5", "messages": messages, "tools": [EDIT_TOOL], "stream": False})
choice = body["choices"][0]
assert choice["finish_reason"] == "tool_calls", body
call = choice["message"]["tool_calls"][0]
assert call["function"]["name"] == "edit_file", call
args = json.loads(call["function"]["arguments"])
print("model called edit_file with:", args)
assert args["filepath"] == TARGET, args

# Actually perform the edit on disk, exactly as Continue's local tool executor would.
with open(args["filepath"], encoding="utf-8") as f:
    current = f.read()
assert args["old_string"] in current, f"old_string not found in file: {args}"
updated = current.replace(args["old_string"], args["new_string"], 1)
with open(args["filepath"], "w", encoding="utf-8") as f:
    f.write(updated)

with open(TARGET, encoding="utf-8") as f:
    on_disk = f.read()
print("file on disk now contains:")
print(on_disk)
assert "port = 9090" in on_disk, on_disk
assert "port = 8000" not in on_disk, on_disk
assert "app_name = demo" in on_disk, on_disk  # untouched lines preserved
assert "debug = false" in on_disk, on_disk

messages.append(choice["message"])
messages.append({"role": "tool", "tool_call_id": call["id"], "content": "Edit applied successfully."})
body2 = post({"model": "claude-sonnet-4-5", "messages": messages, "tools": [EDIT_TOOL], "stream": False})
choice2 = body2["choices"][0]
assert choice2["finish_reason"] == "stop", body2
print("follow-up after edit:", choice2["message"]["content"][:200])

print("\nPASS: edit_file correctly modified only the targeted line, preserved the rest of the file, and the agent continued correctly.")
