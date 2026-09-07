from __future__ import annotations

from cursor_sdk import TokenUsage

from .config import CONFIG


def usage_to_openai(usage: TokenUsage | None) -> dict | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def budget_messages_for_replay(messages: list[dict], max_chars: int) -> list[dict]:
    if not messages:
        return messages
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    budget = max_chars
    kept: list[dict] = []
    for m in reversed(rest):
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
        content = truncate_middle(str(content), max(2000, max_chars // 4))
        cost = len(content)
        if budget - cost < 0 and kept:
            break
        budget -= cost
        m2 = dict(m)
        m2["content"] = content
        kept.append(m2)
    kept.reverse()
    return system + kept


def total_chars(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
        total += len(str(content))
    return total


def check_overflow(messages: list[dict], model_id: str, models_cache) -> None:
    limit_chars = models_cache.context_length(model_id) * 4
    if total_chars(messages) > limit_chars * 1.5:
        from .errors import context_length_exceeded

        raise context_length_exceeded(
            f"This conversation exceeds the context window for '{model_id}'. Start a new chat or remove earlier messages."
        )
