"""Same read_file -> edit_file agent loop as _validate_multiturn.py but over real SSE
streaming (stream=true), which is what Continue actually uses."""
import json
import os
import uuid
import httpx

BASE = "http://127.0.0.1:18080"
TOKEN = os.getenv("BEARER_TOKEN", "continue-local")
H = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Conversation-Id": f"conv-{uuid.uuid4().hex}",
    "X-Continue-Workspace": os.environ["TEMP"],
    "X-Continue-OS": "windows",
    "X-Continue-Shell": "powershell",
}
WORKDIR = os.path.join(os.environ["TEMP"], "relay_tool_validate")
os.makedirs(WORKDIR, exist_ok=True)
TARGET = os.path.join(WORKDIR, "call_me2.txt")
with open(TARGET, "w", encoding="utf-8") as f:
    f.write("name = old_name\nvalue = 1\n")

READ_TOOL = {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]}}}
EDIT_TOOL = {"type": "function", "function": {"name": "single_find_and_replace", "description": "Replace an exact substring in a file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["filepath", "old_string", "new_string"]}}}
TOOLS = [READ_TOOL, EDIT_TOOL]


def post_stream(messages):
    """Mimic a real OpenAI streaming client: accumulate tool_calls by index, concatenating
    argument fragments, exactly like Continue's SSE consumer does."""
    tool_calls: dict[int, dict] = {}
    content = ""
    finish_reason = None
    chunk_count = 0
    with httpx.stream("POST", f"{BASE}/v1/chat/completions", headers=H, json={"model": "claude-sonnet-4-5", "messages": messages, "tools": TOOLS, "stream": True}, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            chunk_count += 1
            choice = chunk["choices"][0]
            delta = choice["delta"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if "content" in delta and delta["content"]:
                content += delta["content"]
            for tc in delta.get("tool_calls") or []:
                idx = tc["index"]
                slot = tool_calls.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
    print(f"  ({chunk_count} SSE chunks)")
    ordered_calls = [tool_calls[i] for i in sorted(tool_calls)]
    return finish_reason, content, ordered_calls


def apply_tool(name, args):
    if name == "read_file":
        with open(args["filepath"], encoding="utf-8") as f:
            return f.read()
    if name == "single_find_and_replace":
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
    finish, content, calls = post_stream(messages)
    print(f"--- turn {turn}: finish_reason={finish} calls={[c['function']['name'] for c in calls]} ---")
    if finish == "stop":
        print("final message:", content)
        break
    assert finish == "tool_calls", (finish, content, calls)
    msg = {"role": "assistant", "content": content or None, "tool_calls": calls}
    messages.append(msg)
    for call in calls:
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])
        seen_tool_names.append(name)
        print(f"  tool call #{len(seen_tool_names)}: {name}({args})")
        result = apply_tool(name, args)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    if "single_find_and_replace" in seen_tool_names:
        break
else:
    raise AssertionError(f"did not converge after 6 turns -- tool call sequence: {seen_tool_names}")

with open(TARGET, encoding="utf-8") as f:
    final = f.read()
print("\nfinal file content:")
print(final)
assert "new_name" in final and "old_name" not in final, final
print("\ntool call sequence:", seen_tool_names)
assert seen_tool_names.count("single_find_and_replace") == 1
print("\nPASS (streaming): Continue 2.1.0 tool calls and results round-tripped.")
