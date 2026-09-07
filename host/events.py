from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextEvent:
    text: str


@dataclass
class ReasoningEvent:
    text: str


@dataclass
class ToolCallsEvent:
    calls: list[dict]


@dataclass
class UsageEvent:
    usage: dict | None


@dataclass
class DoneEvent:
    finish_reason: str
    error: str | None = None


@dataclass
class ErrorEvent:
    message: str
    status: int = 500
    code: str | None = None


Event = TextEvent | ReasoningEvent | ToolCallsEvent | UsageEvent | DoneEvent | ErrorEvent
