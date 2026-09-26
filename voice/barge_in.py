"""Barge-in (2026-09-25): let a visitor talk over the robot.

"Can we enable fluent conversations with break-in when using Gemini
Live." Gemini Live handles interruptions itself -- its server VAD hears
the visitor, sends ``interrupted`` and pipecat stops the speaker -- but
the agent has muted the mic whenever the robot talks
(``AlwaysUserMuteStrategy``): the speaker and the mic share a table and
there is no echo cancellation, so an open mic would let the robot
interrupt itself with its own voice.

The gate in between: while the robot talks, the mic stays muted unless
it hears something clearly louder than the robot's own echo. The echo
reference is a running high percentile of the mic level over the last
few seconds of robot speech (it follows the volume knob, the loudness
drive and where the mic sits); a visitor counts once the level stays
``margin_db`` above it, and above an absolute floor, for ``onset_ms``.
Then the mic opens for the rest of that reply and Gemini takes it from
there. The first ``onset_ms`` of the visitor's words are not sent;
Gemini still hears the rest.

``EchoGate`` is plain numpy (tested in the light venv);
``EchoGatedUserMuteStrategy`` wraps it for pipecat's user aggregator in
place of ``AlwaysUserMuteStrategy``. Every barge-in is logged with its
numbers (``barge-in: ...``) so the margin can be tuned from the log.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger("barge_in")

DEFAULT_MARGIN_DB = 10.0
DEFAULT_FLOOR_DB = -40.0
DEFAULT_ONSET_MS = 200.0
ECHO_WINDOW_SECS = 6.0      # robot speech the reference is taken over
ECHO_PERCENTILE = 90.0
MIN_ECHO_SECS = 0.5         # before this much, the gate stays shut


def level_db(pcm16: bytes) -> float:
    """RMS level of int16 audio in dBFS (-120 for silence)."""
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean((x / 32768.0) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-6))


class EchoGate:
    """Decides, frame by frame, whether the visitor is talking over the
    robot. ``feed(level, seconds)`` while the robot speaks; ``bot_stopped``
    after each reply."""

    def __init__(self, margin_db: float = DEFAULT_MARGIN_DB,
                 floor_db: float = DEFAULT_FLOOR_DB,
                 onset_ms: float = DEFAULT_ONSET_MS):
        self.margin_db = margin_db
        self.floor_db = floor_db
        self.onset_secs = onset_ms / 1000.0
        # (level, seconds) of robot-speech frames, carried across replies
        self._echo: deque = deque()
        self._echo_secs = 0.0
        self._streak = 0.0
        self._peak = -120.0
        self.open = False
        self.last_trigger: dict | None = None

    def echo_reference(self) -> float | None:
        if self._echo_secs < MIN_ECHO_SECS:
            return None
        levels = np.array([lv for lv, _ in self._echo])
        return float(np.percentile(levels, ECHO_PERCENTILE))

    def threshold(self) -> float | None:
        ref = self.echo_reference()
        if ref is None:
            return None
        return max(ref + self.margin_db, self.floor_db)

    def feed(self, level: float, seconds: float) -> bool:
        """One mic frame while the robot speaks. Returns True the moment
        the gate opens (once per reply)."""
        if self.open:
            return False
        threshold = self.threshold()
        if threshold is not None and level >= threshold:
            self._streak += seconds
            self._peak = max(self._peak, level)
            if self._streak >= self.onset_secs:
                self.open = True
                self.last_trigger = {"level": self._peak,
                                     "echo": self.echo_reference(),
                                     "threshold": threshold}
                return True
            return False        # a candidate: kept out of the echo stats
        self._streak = 0.0
        self._peak = -120.0
        self._echo.append((level, seconds))
        self._echo_secs += seconds
        while self._echo and self._echo_secs - self._echo[0][1] >= ECHO_WINDOW_SECS:
            self._echo_secs -= self._echo.popleft()[1]
        return False

    def bot_stopped(self) -> None:
        self.open = False
        self._streak = 0.0
        self._peak = -120.0


def make_strategy(margin_db: float = DEFAULT_MARGIN_DB,
                  floor_db: float = DEFAULT_FLOOR_DB,
                  onset_ms: float = DEFAULT_ONSET_MS):
    """The pipecat user-mute strategy around an ``EchoGate``."""
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame,
    )
    from pipecat.turns.user_mute.base_user_mute_strategy import (
        BaseUserMuteStrategy,
    )

    class EchoGatedUserMuteStrategy(BaseUserMuteStrategy):
        """Muted while the robot speaks, unless the visitor talks over it."""

        def __init__(self):
            super().__init__()
            self.gate = EchoGate(margin_db, floor_db, onset_ms)
            self._bot_speaking = False
            self.barge_ins = 0
            self.replies = 0

        async def process_frame(self, frame) -> bool:
            await super().process_frame(frame)
            if isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
                self.gate.bot_stopped()
                self.replies += 1
                echo = self.gate.echo_reference()
                if echo is not None and self.replies % 10 == 1:
                    logger.info("barge-in: the robot's own echo at the mic "
                                "is %.0f dBFS; a voice needs %.0f to "
                                "interrupt", echo, self.gate.threshold())
            elif self._bot_speaking and isinstance(frame, InputAudioRawFrame):
                rate = frame.sample_rate or 16000
                seconds = len(frame.audio) / 2 / max(1, frame.num_channels) / rate
                if self.gate.feed(level_db(frame.audio), seconds):
                    self.barge_ins += 1
                    t = self.gate.last_trigger
                    logger.info("barge-in: the visitor spoke over the robot "
                                "(mic %.0f dBFS, echo %.0f, threshold %.0f); "
                                "letting Gemini hear it", t["level"],
                                t["echo"], t["threshold"])
            return self._bot_speaking and not self.gate.open

    return EchoGatedUserMuteStrategy()


# 2026-09-25, evening: a bigger speaker on a USB-to-jack adapter. The
# robot's own speaker is echo-cancelled at its mic by the XVF3800 board;
# another speaker is not, and its sound reached the desk mic at -3 dBFS,
# as loud as any visitor. No level gate can tell the two apart, so the
# agent turns barge-in off on such a speaker (the robot opened the gate
# on its own voice half a second into each reply, over and over) and
# keeps the mic shut a moment longer after each reply, while the room
# still rings with the last word.
DEFAULT_TAIL_SECS = 0.25


def make_tail_strategy(tail_secs: float = DEFAULT_TAIL_SECS):
    """A pipecat user-mute strategy that stays muted ``tail_secs`` after
    the robot stops speaking (use with AlwaysUserMuteStrategy)."""
    import time
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
    )
    from pipecat.turns.user_mute.base_user_mute_strategy import (
        BaseUserMuteStrategy,
    )

    class TailUserMuteStrategy(BaseUserMuteStrategy):
        """Muted for a short tail after each reply."""

        def __init__(self):
            super().__init__()
            self.tail_secs = tail_secs
            self._until = 0.0

        async def process_frame(self, frame) -> bool:
            await super().process_frame(frame)
            if isinstance(frame, BotStartedSpeakingFrame):
                self._until = 0.0
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._until = time.monotonic() + self.tail_secs
            return time.monotonic() < self._until

    return TailUserMuteStrategy()
