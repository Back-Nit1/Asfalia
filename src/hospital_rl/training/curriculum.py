"""Difficulty curriculum advanced by held-out validation success, not training return."""
from __future__ import annotations

from collections import deque


class Curriculum:
    def __init__(self, levels: list[str], threshold: float = 0.8, window: int = 3):
        self.levels, self.threshold, self.window = levels, threshold, window
        self.index = 0
        self._recent: deque[float] = deque(maxlen=window)

    @property
    def level(self) -> str:
        return self.levels[self.index]

    def report(self, validation_success: float) -> bool:
        """Record one validation result; advance when the last `window` all meet the threshold."""
        self._recent.append(validation_success)
        if (self.index < len(self.levels) - 1 and len(self._recent) == self.window
                and min(self._recent) >= self.threshold):
            self.index += 1
            self._recent.clear()
            return True
        return False

    def state_dict(self) -> dict:
        return {"index": self.index, "recent": list(self._recent)}

    def load_state_dict(self, s: dict) -> None:
        self.index, self._recent = s["index"], deque(s["recent"], maxlen=self.window)
