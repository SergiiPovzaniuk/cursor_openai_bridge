"""End-to-end validation: real file write + a second distinct tool, through the Java relay."""
import json
import os
import httpx

BASE = "http://127.0.0.1:18080"
TOKEN = "vNjo6ZDNQamYb2w98solftTvzoyM1vdZM5dQLIyhi4"
H = {"Authorization": f"Bearer {TOKEN}"}
WORKDIR = os.path.join(os.environ["TEMP"], "relay_tool_validate")
os.makedirs(WORKDIR, exist_ok=True)
TARGET = os.path.join(WORKDIR, "hello.txt")
CONTENT = "hello from tool validation test"

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file on disk",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    },
}
READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the full contents of a file on disk",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}


def post(payload):
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=H, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def test_write_file():
    print("=== write_file ===")
    messages = [
        {"role": "system", "content": "You are a coding agent with file tools."},
        {"role": "user", "content": f"Call write_file with path='{TARGET}' and content='{CONTENT}'. Call it exactly once, nothing else."},
    ]
    body = post({"model": "claude-sonnet-4-5", "messages": messages, "tools": [WRITE_TOOL], "stream": False})
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls", body
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "write_file", call
    args = json.loads(call["function"]["arguments"])
    print("model called write_file with:", args)

    # Actually perform the write, exactly as Continue would locally.
    with open(args["path"], "w", encoding="utf-8") as f:
        f.write(args["content"])
    assert os.path.isfile(args["path"])
    with open(args["path"], encoding="utf-8") as f:
        written = f.read()
    print("file on disk now contains:", repr(written))
    assert written == args["content"]

    messages.append(choice["message"])
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": f"Wrote {len(args['content'])} bytes to {args['path']}"})
    body2 = post({"model": "claude-sonnet-4-5", "messages": messages, "tools": [WRITE_TOOL], "stream": False})
    choice2 = body2["choices"][0]
    assert choice2["finish_reason"] == "stop", body2
    print("follow-up after write:", choice2["message"]["content"][:200])
    print("PASS: write_file created a real file on disk and the agent continued correctly.\n")
    return args["path"]


def test_read_file(path):
    print("=== read_file (second, distinct tool) ===")
    messages = [
        {"role": "system", "content": "You are a coding agent with file tools."},
        {"role": "user", "content": f"Call read_file with path='{path}' to see what's in it, then tell me the exact contents."},
    ]
    body = post({"model": "claude-sonnet-4-5", "messages": messages, "tools": [READ_TOOL], "stream": False})
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls", body
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read_file", call
    args = json.loads(call["function"]["arguments"])
    print("model called read_file with:", args)
    assert args["path"] == path, args

    with open(args["path"], encoding="utf-8") as f:
        real_content = f.read()

    messages.append(choice["message"])
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": real_content})
    body2 = post({"model": "claude-sonnet-4-5", "messages": messages, "tools": [READ_TOOL], "stream": False})
    choice2 = body2["choices"][0]
    assert choice2["finish_reason"] == "stop", body2
    final_text = choice2["message"]["content"]
    print("follow-up after read:", final_text[:300])
    assert CONTENT in final_text, f"expected model to relay real file content, got: {final_text}"
    print("PASS: read_file returned the real on-disk content and the agent relayed it correctly.\n")


if __name__ == "__main__":
    p = test_write_file()
    test_read_file(p)
    print("ALL TOOL VALIDATIONS PASSED")
