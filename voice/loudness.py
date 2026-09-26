"""Loudness (2026-09-25): the robot's voice as loud as its speaker allows.

"Can we increase the volume much more?" -- the ALSA mixer was already at
60/60 (100 %, 0 dB) on both PCM controls, nothing between the agent and
``hw:2,0`` attenuates, and Gemini Live's audio arrives normalised: one
measured reply peaked at 0 dBFS with a 99th percentile of -2 dBFS and a
voiced RMS of -12 dBFS. Plain gain would only hard-clip.

What is left is making the waveform denser. Measured on that reply, in
the 300 Hz - 5 kHz band a small speaker actually plays:

  * a lookahead peak limiter tops out near +3 dB however hard it is
    driven -- voiced speech has only 6-9 dB of crest factor inside each
    pitch period, and a limiter cannot go below the period;
  * a tanh soft clip gets +3.5 dB at 6 dB of drive, +4.9 at 9 and +6.2
    at 12, with no state, no latency and no lookahead to flush.

The booth's model, gemini-3.1-flash-live-preview, speaks about 4 dB
quieter (peak -1.5 dBFS, voiced RMS -15.6), so the same drives give
+4.6, +6.5 and +8.3 dB there; the log line uses those numbers.

So this is the soft clip. ``drive_db`` is how hard the signal is pushed
into the curve; the output is scaled so full scale lands at -0.5 dBFS.
It adds harmonics (that is where the loudness comes from); speech stays
intelligible well past what this does, but if the voice sounds harsh,
turn the drive down before anything else. Off at 0.
"""

from __future__ import annotations

import numpy as np

CEILING_DB = -0.5

# Measured gain in the speaker band for a few drives on Gemini 3.1 Live
# audio (see the docstring), for the startup log line; in between is
# interpolated.
_MEASURED = ((0.0, 0.0), (6.0, 4.6), (9.0, 6.5), (12.0, 8.3))


def louder_by_db(drive_db: float) -> float:
    """Roughly how much louder a drive makes Gemini's speech, in dB."""
    xs, ys = zip(*_MEASURED)
    return float(np.interp(drive_db, xs, ys))


class SoftClip:
    """Stateless tanh soft clip with ``drive_db`` of drive."""

    def __init__(self, drive_db: float):
        if drive_db < 0:
            raise ValueError("drive_db must be >= 0")
        self.drive_db = float(drive_db)
        self._d = 10.0 ** (self.drive_db / 20.0)
        self._scale = 10.0 ** (CEILING_DB / 20.0) / np.tanh(self._d)

    @property
    def passthrough(self) -> bool:
        return self.drive_db < 1e-6

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Float samples in [-1, 1] -> float samples within the ceiling."""
        if self.passthrough:
            return samples
        return (np.tanh(samples * self._d) * self._scale).astype(np.float32)

    def process_pcm16(self, audio: bytes) -> bytes:
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        out = np.clip(self.process(pcm), -1.0, 1.0)
        return (out * 32767.0).astype(np.int16).tobytes()

    # -- pipecat ---------------------------------------------------------------

    def as_processor(self):
        """A FrameProcessor soft-clipping every TTSAudioRawFrame in place."""
        from pipecat.frames.frames import TTSAudioRawFrame
        from pipecat.processors.frame_processor import FrameProcessor

        clip = self

        class _Loudness(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, TTSAudioRawFrame) and frame.audio:
                    frame.audio = clip.process_pcm16(frame.audio)
                await self.push_frame(frame, direction)

        return _Loudness()
