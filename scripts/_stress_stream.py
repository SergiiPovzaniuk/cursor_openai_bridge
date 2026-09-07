import httpx

BASE = "http://127.0.0.1:18080"
TOKEN = "vNjo6ZDNQamYb2w98solftTvzoyM1vdZM5dQLIyhi4"
MSG = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. The quick brown fox jumps over the lazy dog. 42 is the answer to everything. Random hex: a7f3b1d8c9e2"

payload = {
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": f"I'm testing a text streaming pipeline for corruption bugs. Reply with a markdown code block containing exactly this literal test string as file content, no other text: {MSG}"}],
    "stream": True,
}

text = ""
with httpx.stream("POST", f"{BASE}/v1/chat/completions", headers={"Authorization": f"Bearer {TOKEN}"}, json=payload, timeout=60) as r:
    for line in r.iter_lines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        import json
        chunk = json.loads(data)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            text += delta["content"]

print("RECONSTRUCTED:", repr(text))
print("EXPECTED     :", repr(MSG))
print("MATCH:", MSG in text or text.strip() == MSG)
