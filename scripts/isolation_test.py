import asyncio
import sys

import httpx

BASE = "http://127.0.0.1:8787"
HEADERS = {"Authorization": "Bearer devtoken"}
N = 5


async def one_chat(client: httpx.AsyncClient, i: int) -> tuple[int, str, str]:
    secret = f"SECRET-{i}-XYZ"
    r1 = await client.post(f"{BASE}/v1/chat/completions", headers=HEADERS, json={
        "model": "composer-2.5",
        "messages": [{"role": "user", "content": f"Remember this exact secret code: {secret}. Reply with just OK."}],
        "stream": False,
    }, timeout=180)
    r1.raise_for_status()
    msg = r1.json()["choices"][0]["message"]
    r2 = await client.post(f"{BASE}/v1/chat/completions", headers=HEADERS, json={
        "model": "composer-2.5",
        "messages": [
            {"role": "user", "content": f"Remember this exact secret code: {secret}. Reply with just OK."},
            msg,
            {"role": "user", "content": "What was the secret code? Reply with just the code, nothing else."},
        ],
        "stream": False,
    }, timeout=180)
    r2.raise_for_status()
    return i, secret, r2.json()["choices"][0]["message"]["content"]


async def main() -> int:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(one_chat(client, i) for i in range(N)))
    failed = False
    for i, secret, got in results:
        ok = secret in (got or "")
        print(i, "expected", secret, "got", repr(got), "OK" if ok else "LEAK/MISS")
        failed = failed or not ok
    return 1 if failed else 0


sys.exit(asyncio.run(main()))
