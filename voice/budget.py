"""The sightseeing budget (T17.8): steer a fooling visitor back.

On 2026-09-05 one visitor spent three minutes on "turn your head, any
other people?" -- twelve ``look`` calls and eight head and body moves,
every one obeyed -- then "have you ever baked a cake?". The family's
debrief: kids will do this at the Faire; the session has to come back
to the lesson without the robot being a bore about it.

A prompt rule alone did not hold (the model obliged every time), so the
tools carry a budget: during a lesson, ``look``, ``move_head`` and
``turn_body`` together get ``limit`` calls per ``window_secs``; past
that they answer "skipped: enough sightseeing, back to the lesson", the
same shape as the T15.9 "a recorded move is playing" refusal, and the
model has to steer. Outside a lesson (nobody enrolled or recognized:
stranger chat, the intake) there is no budget -- "what do you see?" is a
fair first question. Reset per session by the runner.

Pure Python, clock injectable, unit-tested in the light venv.
"""

from __future__ import annotations

import time

SIGHTSEEING_TOOLS = ("look", "move_head", "turn_body")


class SightseeingBudget:
    def __init__(self, *, limit: int = 2, window_secs: float = 120.0,
                 in_lesson=None, clock=time.monotonic):
        self.limit = limit
        self.window_secs = window_secs
        # ``in_lesson()`` -> is somebody being tutored right now; None
        # means always (tests).
        self.in_lesson = in_lesson
        self._clock = clock
        self._calls: list[float] = []
        self.refused = 0

    def reset(self) -> None:
        self._calls = []
        self.refused = 0

    def _prune(self, now: float) -> None:
        self._calls = [t for t in self._calls if now - t < self.window_secs]

    def allow(self, tool: str, now: float | None = None) -> dict | None:
        """None when the call may go ahead (and it is counted), else the
        result dict the tool should answer with instead."""
        if tool not in SIGHTSEEING_TOOLS:
            return None
        if self.in_lesson is not None and not self.in_lesson():
            return None
        now = self._clock() if now is None else now
        self._prune(now)
        if len(self._calls) >= self.limit:
            self.refused += 1
            return {"skipped": True,
                    "reason": "enough sightseeing for now, this is a "
                              "lesson: say so in one light sentence and "
                              "set the next task in the lesson language. "
                              "Do not describe anything and do not try "
                              "another movement."}
        self._calls.append(now)
        return None

    def used(self, now: float | None = None) -> int:
        now = self._clock() if now is None else now
        self._prune(now)
        return len(self._calls)
