"""Pace (T17.10): how fast the robot talks, measured and, if needed, slowed.

The mother, 2026-09-05: "you speak too fast, you don't adapt to how
the person speaks" and "the ends of words are slurred". Gemini Live
has no speaking-rate setting, so two things live here:

* ``wpm(text, seconds)`` -- the number. ``CloudTranscriptLogger`` in
  agent.py logs one ``pace: N words in S s (W wpm)`` line per reply, so
  the next session's log says whether the prompt ("speak slowly,
  short sentences") moved anything. Measure first.

* ``Stretcher`` -- a streaming WSOLA time stretcher on the reply audio,
  pitch kept, for when the prompt does not move it. ``factor`` 1.2
  makes every reply twenty percent longer at the same pitch. Off unless
  ``--speech-rate`` is given; ``as_processor()`` wraps it as a pipecat
  stage over ``TTSAudioRawFrame`` chunks (state carried across chunks,
  flushed when the reply ends).

WSOLA in forty milliseconds' worth of frames: each output frame is
taken from the input near its nominal position, shifted by up to a
quarter frame to where it best continues the previous frame, and
overlap-added with a Hann window at half-frame hops. Plain numpy, no
extra dependency; ~30 M multiply-adds per second of audio at 24 kHz.
"""

from __future__ import annotations

import numpy as np


def wpm(text: str, seconds: float) -> int | None:
    words = len(str(text or "").split())
    if seconds <= 0 or not words:
        return None
    return int(round(words / seconds * 60.0))


def pace_line(text: str, seconds: float) -> str | None:
    rate = wpm(text, seconds)
    if rate is None:
        return None
    return "pace: %d words in %.1fs (%d wpm)" % (len(text.split()), seconds, rate)


class Stretcher:
    """Streaming WSOLA. ``factor`` > 1 slows (output longer), < 1 speeds."""

    def __init__(self, factor: float, sample_rate: int = 24000,
                 frame_ms: float = 40.0):
        if factor <= 0:
            raise ValueError("factor must be positive")
        self.factor = float(factor)
        self.sample_rate = sample_rate
        n = int(sample_rate * frame_ms / 1000.0)
        self.N = n - (n % 2)
        self.Hs = self.N // 2                 # synthesis hop
        self.Ha = self.Hs / self.factor       # analysis hop
        self.delta = self.Hs // 2             # search range either way
        self.win = np.hanning(self.N).astype(np.float32)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._base = 0            # absolute input index of _buf[0]
        self._k = 0               # next output frame index
        self._prev_start = None   # absolute input start of the last segment
        self._ola = np.zeros(0, dtype=np.float32)
        self._ola_base = 0        # absolute output index of _ola[0]
        self._emitted = 0

    @property
    def passthrough(self) -> bool:
        return abs(self.factor - 1.0) < 1e-6

    # -- helpers ---------------------------------------------------------------

    def _avail(self) -> int:
        return self._base + len(self._buf)

    def _seg(self, start: int, length: int) -> np.ndarray:
        i = start - self._base
        return self._buf[i:i + length]

    def _add(self, frame: np.ndarray, at: int) -> None:
        end = at + len(frame)
        need = end - self._ola_base
        if need > len(self._ola):
            self._ola = np.concatenate(
                [self._ola, np.zeros(need - len(self._ola), dtype=np.float32)])
        i = at - self._ola_base
        self._ola[i:i + len(frame)] += frame

    # -- the stretch ---------------------------------------------------------

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Feed float32 mono samples; returns the output samples that are
        final so far (possibly empty)."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self.passthrough:
            return samples
        if samples.size:
            self._buf = np.concatenate([self._buf, samples])
        while True:
            nominal = int(round(self._k * self.Ha))
            if nominal + self.delta + self.N > self._avail():
                break
            if self._prev_start is None:
                start = max(nominal, self._base)
            else:
                ref_start = self._prev_start + self.Hs
                if ref_start + self.N > self._avail():
                    break
                ref = self._seg(ref_start, self.N)
                lo = max(nominal - self.delta, self._base)
                hi = nominal + self.delta
                block = self._seg(lo, hi - lo + self.N)
                if len(block) < self.N:
                    break
                cands = np.lib.stride_tricks.sliding_window_view(block, self.N)
                scores = cands @ ref
                start = lo + int(np.argmax(scores))
            seg = self._seg(start, self.N) * self.win
            self._add(seg, self._k * self.Hs)
            self._prev_start = start
            self._k += 1
            # drop input nobody will read again
            keep_from = min(self._prev_start,
                            int(round(self._k * self.Ha)) - self.delta)
            keep_from = max(keep_from, self._base)
            if keep_from > self._base:
                self._buf = self._buf[keep_from - self._base:]
                self._base = keep_from
        final_until = self._k * self.Hs        # samples below this are done
        n = final_until - self._ola_base
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        out = self._ola[:n]
        self._ola = self._ola[n:]
        self._ola_base += n
        self._emitted += n
        return out

    def flush(self) -> np.ndarray:
        """End of a reply: whatever is left (the input tail, unstretched
        beyond the last full frame, then the overlap buffer)."""
        if self.passthrough:
            return np.zeros(0, dtype=np.float32)
        tail_from = max(self._base, (self._prev_start + self.N)
                        if self._prev_start is not None else self._base)
        tail = self._seg(tail_from, self._avail() - tail_from)
        out = np.concatenate([self._ola, tail]) if tail.size else self._ola
        self.reset()
        return out

    # -- pipecat ---------------------------------------------------------------

    def as_processor(self):
        """A FrameProcessor stretching every TTSAudioRawFrame, flushing on
        the end of a reply and dropping state on an interruption."""
        from pipecat.frames.frames import (
            InterruptionFrame, TTSAudioRawFrame, TTSStoppedFrame,
        )
        from pipecat.processors.frame_processor import FrameProcessor

        stretcher = self

        def to_bytes(samples: np.ndarray) -> bytes:
            clipped = np.clip(samples, -1.0, 1.0)
            return (clipped * 32767.0).astype(np.int16).tobytes()

        class _Pace(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, TTSAudioRawFrame) \
                        and frame.num_channels == 1:
                    if stretcher.sample_rate != frame.sample_rate:
                        stretcher.sample_rate = frame.sample_rate
                        stretcher.__init__(stretcher.factor, frame.sample_rate)
                    pcm = np.frombuffer(frame.audio, dtype=np.int16
                                        ).astype(np.float32) / 32768.0
                    out = stretcher.process(pcm)
                    if out.size:
                        await self.push_frame(TTSAudioRawFrame(
                            audio=to_bytes(out), sample_rate=frame.sample_rate,
                            num_channels=1), direction)
                    return
                if isinstance(frame, TTSStoppedFrame):
                    out = stretcher.flush()
                    if out.size:
                        await self.push_frame(TTSAudioRawFrame(
                            audio=to_bytes(out),
                            sample_rate=stretcher.sample_rate,
                            num_channels=1), direction)
                elif isinstance(frame, InterruptionFrame):
                    stretcher.reset()
                await self.push_frame(frame, direction)

        return _Pace(name="Pace")
