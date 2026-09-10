"""Reproduce Continue's real agent loop: read_file -> edit_file across separate HTTP turns,
each carrying the full growing transcript with a stable conversation id header, exactly like
Continue does. Checks that this stays ONE underlying session (no session churn / re-reads loop)."""
import json
import os
import uuid
import httpx

BASE = "http://127.0.0.1:18080"
TOKEN = os.getenv("BEARER_TOKEN", "continue-local")
WORKDIR = os.path.join(os.environ["TEMP"], "relay_tool_validate")
H = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Conversation-Id": f"conv-{uuid.uuid4().hex}",
    "X-Continue-Workspace": WORKDIR,
    "X-Continue-OS": "windows",
    "X-Continue-Shell": "powershell",
}
os.makedirs(WORKDIR, exist_ok=True)
TARGET = os.path.join(WORKDIR, "call_me.txt")
with open(TARGET, "w", encoding="utf-8") as f:
    f.write("name = old_name\nvalue = 1\n")

READ_TOOL = {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
EDIT_TOOL = {"type": "function", "function": {"name": "edit_file", "description": "Replace an exact substring in a file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["filepath", "old_string", "new_string"]}}}
TOOLS = [READ_TOOL, EDIT_TOOL]


def post(messages):
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=H, json={"model": "claude-sonnet-4-5", "messages": messages, "tools": TOOLS, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json()


def apply_tool(name, args):
    if name == "read_file":
        with open(args["path"], encoding="utf-8") as f:
            return f.read()
    if name == "edit_file":
        with open(args["filepath"], encoding="utf-8") as f:
            cur = f.read()
        assert args["old_string"] in cur, f"old_string not in file! model may be looping on stale state: {args}"
        new = cur.replace(args["old_string"], args["new_string"], 1)
        with open(args["filepath"], "w", encoding="utf-8") as f:
            f.write(new)
        return "edit applied"
    raise ValueError(name)


messages = [
    {"role": "system", "content": "You are a coding agent with file tools."},
    {"role": "user", "content": f"First read {TARGET}, then edit it to change 'old_name' to 'new_name'. Do this step by step, one tool call at a time."},
]

seen_tool_names = []
for turn in range(6):
    body = post(messages)
    choice = body["choices"][0]
    finish = choice["finish_reason"]
    print(f"--- turn {turn}: finish_reason={finish} ---")
    if finish == "stop":
        print("final message:", choice["message"]["content"])
        break
    assert finish == "tool_calls", body
    msg = choice["message"]
    messages.append(msg)
    for call in msg["tool_calls"]:
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])
        seen_tool_names.append(name)
        print(f"  tool call #{len(seen_tool_names)}: {name}({args})")
        result = apply_tool(name, args)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
else:
    raise AssertionError(f"did not converge after 6 turns -- tool call sequence: {seen_tool_names}")

with open(TARGET, encoding="utf-8") as f:
    final = f.read()
print("\nfinal file content:")
print(final)
assert "new_name" in final and "old_name" not in final, final
print("\ntool call sequence:", seen_tool_names)
assert seen_tool_names.count("edit_file") == 1, f"edit_file was called {seen_tool_names.count('edit_file')} times -- this is the repeated-edit loop bug!"
print("\nPASS: converged in", turn + 1, "turns with exactly one edit_file call, no repeated-edit loop.")
