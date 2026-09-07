from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Metrics:
    requests_total: int = 0
    errors_total: int = 0
    tool_round_trips: int = 0
    active_streams: int = 0
    ttft_samples: deque = field(default_factory=lambda: deque(maxlen=200))
    tokens_per_s_samples: deque = field(default_factory=lambda: deque(maxlen=200))
    started_at: float = field(default_factory=time.monotonic)

    def record_ttft(self, seconds: float) -> None:
        self.ttft_samples.append(seconds)

    def record_tps(self, tokens_per_second: float) -> None:
        self.tokens_per_s_samples.append(tokens_per_second)

    def snapshot(self) -> dict:
        def avg(d: deque) -> float | None:
            return round(sum(d) / len(d), 4) if d else None

        return {
            "uptime_s": round(time.monotonic() - self.started_at, 1),
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "tool_round_trips": self.tool_round_trips,
            "active_streams": self.active_streams,
            "ttft_avg_s": avg(self.ttft_samples),
            "tokens_per_s_avg": avg(self.tokens_per_s_samples),
        }


METRICS = Metrics()
