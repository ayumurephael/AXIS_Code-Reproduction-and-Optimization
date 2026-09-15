from __future__ import annotations

import math
import time


def _format_duration(seconds: float) -> str:
    """Format a duration in a compact human-readable form."""

    seconds = max(0, int(round(float(seconds))))
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minute:02d}:{sec:02d}"
    return f"{minute:02d}:{sec:02d}"


class ProgressPrinter:
    """Print coarse-grained progress updates every fixed percentage."""

    def __init__(self, total: int, *, label: str, step_percent: float = 2.5) -> None:
        self.total = max(0, int(total))
        self.label = str(label)
        self.step_percent = max(0.1, float(step_percent))
        self.started = time.perf_counter()
        self._next_threshold = self.step_percent
        self._started_printed = False

    def start(self, *, extra: str = "") -> None:
        """Print the initial 0% progress line once."""

        if self._started_printed:
            return
        self._started_printed = True
        suffix = f" {extra}" if extra else ""
        print(f"[{self.label}] 0/{self.total} (0.0%) elapsed=00:00 eta=--{suffix}", flush=True)

    def update(self, current: int, *, extra: str = "") -> None:
        """Print progress when the next percentage threshold is reached."""

        if not self._started_printed:
            self.start()
        if self.total <= 0:
            return
        current = max(0, min(int(current), self.total))
        percent = (100.0 * current / self.total) if self.total else 100.0
        should_print = current >= self.total or percent + 1e-9 >= self._next_threshold
        if not should_print:
            return
        elapsed = time.perf_counter() - self.started
        rate = (current / elapsed) if elapsed > 0 and current > 0 else 0.0
        remaining = self.total - current
        eta = (remaining / rate) if rate > 0 else math.inf
        eta_text = "--" if not math.isfinite(eta) else _format_duration(eta)
        suffix = f" {extra}" if extra else ""
        print(
            f"[{self.label}] {current}/{self.total} ({percent:.1f}%) "
            f"elapsed={_format_duration(elapsed)} eta={eta_text}{suffix}",
            flush=True,
        )
        while self._next_threshold <= percent + 1e-9:
            self._next_threshold += self.step_percent
