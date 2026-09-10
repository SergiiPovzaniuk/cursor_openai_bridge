"""Validate Continue VS Code 2.1.0 tools against the relay."""
import json
import os
import uuid
import httpx

BASE = "http://127.0.0.1:18080"
TOKEN = os.getenv("BEARER_TOKEN", "continue-local")
H = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Conversation-Id": f"conv-{uuid.uuid4().hex}",
    "X-Continue-Workspace": os.path.join(os.environ["TEMP"], "relay_tool_validate"),
    "X-Continue-OS": "windows",
    "X-Continue-Shell": "powershell",
}

CONTINUE_TOOLS = [
    ("read_file", {"type": "object", "required": ["filepath"], "properties": {"filepath": {"type": "string"}}}),
    ("read_file_range", {"type": "object", "required": ["filepath", "startLine", "endLine"], "properties": {"filepath": {"type": "string"}, "startLine": {"type": "number"}, "endLine": {"type": "number"}}}),
    ("read_currently_open_file", {"type": "object", "properties": {}}),
    ("ls", {"type": "object", "properties": {"dirPath": {"type": "string"}, "recursive": {"type": "boolean"}}}),
    ("file_glob_search", {"type": "object", "required": ["pattern"], "properties": {"pattern": {"type": "string"}}}),
    ("grep_search", {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}),
    ("fetch_url_content", {"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}}),
    ("search_web", {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}),
    ("view_diff", {"type": "object", "properties": {}}),
    ("view_repo_map", {"type": "object", "properties": {}}),
    ("view_subdirectory", {"type": "object", "required": ["directory_path"], "properties": {"directory_path": {"type": "string"}}}),
    ("codebase", {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}),
    ("create_new_file", {"type": "object", "required": ["filepath", "contents"], "properties": {"filepath": {"type": "string"}, "contents": {"type": "string"}}}),
    ("edit_existing_file", {"type": "object", "required": ["filepath", "changes"], "properties": {"filepath": {"type": "string"}, "changes": {"type": "string"}}}),
    ("single_find_and_replace", {"type": "object", "required": ["filepath", "old_string", "new_string"], "properties": {"filepath": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}}),
    ("multi_edit", {"type": "object", "required": ["filepath", "edits"], "properties": {
        "filepath": {"type": "string"},
        "edits": {"type": "array", "items": {"type": "object", "required": ["old_string", "new_string"], "properties": {
            "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}
        }}},
    }}),
    ("run_terminal_command", {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}, "waitForCompletion": {"type": "boolean"}}}),
    ("create_rule_block", {"type": "object", "required": ["name", "rule"], "properties": {"name": {"type": "string"}, "rule": {"type": "string"}, "description": {"type": "string"}, "globs": {"type": "string"}, "regex": {"type": "string"}, "alwaysApply": {"type": "boolean"}}}),
    ("request_rule", {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}),
    ("read_skill", {"type": "object", "required": ["skillName"], "properties": {"skillName": {"type": "string"}}}),
]
TOOLS = [{"type": "function", "function": {"name": n, "description": f"{n} tool", "parameters": s}} for n, s in CONTINUE_TOOLS]

WORKDIR = os.path.join(os.environ["TEMP"], "relay_tool_validate")
os.makedirs(WORKDIR, exist_ok=True)
TARGET = os.path.join(WORKDIR, "continue_toolset.txt")
with open(TARGET, "w", encoding="utf-8") as f:
    f.write("status = pending\n")


def post(messages):
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=H, json={"model": "claude-sonnet-4-5", "messages": messages, "tools": TOOLS, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json()


messages = [
    {"role": "system", "content": "You are a coding agent with file tools."},
    {"role": "user", "content": f"Using your available tools, first read {TARGET} with the read file tool, then edit it so the file contains exactly: status = done"},
]

seen_names = set()
for turn in range(6):
    body = post(messages)
    choice = body["choices"][0]
    finish = choice["finish_reason"]
    print(f"--- turn {turn}: finish_reason={finish} ---")
    if finish == "stop":
        print("final:", choice["message"]["content"])
        break
    assert finish == "tool_calls", body
    msg = choice["message"]
    messages.append(msg)
    valid_names = {n for n, _ in CONTINUE_TOOLS}
    for call in msg["tool_calls"]:
        name = call["function"]["name"]
        assert name in valid_names, f"model called unknown/mangled tool name: {name!r}"
        args = json.loads(call["function"]["arguments"])  # must be valid JSON
        seen_names.add(name)
        print(f"  tool call: {name}({args})")
        if name == "read_file":
            with open(args["filepath"], encoding="utf-8") as f:
                result = f.read()
        elif name in {"edit_existing_file", "single_find_and_replace", "multi_edit"}:
            with open(TARGET, "w", encoding="utf-8") as f:
                f.write("status = done\n")
            result = f"Successfully edited {TARGET}"
        else:
            result = "ok"
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    if "read_file" in seen_names and seen_names & {"edit_existing_file", "single_find_and_replace", "multi_edit"}:
        break
else:
    raise AssertionError(f"did not converge -- tool names seen: {seen_names}")

with open(TARGET, encoding="utf-8") as f:
    final = f.read()
assert "done" in final, final
print("\ntool names used:", seen_names)
assert "read_file" in seen_names, "model never called read_file"
assert seen_names & {"edit_existing_file", "single_find_and_replace", "multi_edit"}, "model never called an edit tool"
print(f"\nPASS: Continue tool schemas registered; read_file and deterministic edit round-tripped correctly.")
