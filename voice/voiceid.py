"""Voice print: the second identity signal (T13.9).

The family, at the debrief: "it should remember not just the face but the
voice print, and notice when a known face has the wrong voice." Three
pieces, in one module so the two identity signals stay side by side:

* **Embedding.** SpeechBrain's ECAPA-TDNN speaker encoder
  (``speechbrain/spkrec-ecapa-voxceleb``), the same model and threshold
  family the user's other project runs. 192-float unit vector per
  utterance, cosine similarity between them. Measured on this box
  (2026-09-02): model load 5 s, one 3 s clip **5 ms on the GPU, 40 ms on
  the CPU** -- so, unlike the face model (T2), this one really does run
  on CUDA here; ``provider_report()`` says which.

* **``VoiceCollector``.** A pipecat processor that accumulates the
  visitor's speech from the raw input audio and hands each finished
  sample's embedding to a callback. Gated on audio energy rather than
  turn frames, because the cloud service (Gemini Live) emits no
  user-speaking frames; the robot's own voice is excluded by watching
  the bot-speaking frames that the output transport pushes upstream.

* **``VoiceIdentity``.** The fusion policy with the face signal, pure
  and unit-tested:

    face sure  + voice matches      -> verified (log only)
    face sure  + voice mismatches   -> one playful challenge, then, if
                                       still wrong, downgrade to "is
                                       that you?" (the ask band)
    face unsure + voice matches     -> confirmed without asking
    face sure  + no print on file   -> learn the print (family profiles
                                       enrolled from photos have none)

Prints are computed on the booth computer and never leave it: in cloud
mode the *audio* streams to Gemini for the conversation, but the
embedding is local, and the print is deleted with the profile ("forget
me", the end-of-day wipe). Sign 1 says so.

Thresholds are calibrated by ``voice/verify_voiceid.py`` against the
fixture voices; the numbers are in the T13 progress log and asserted in
``tests/t13/test_voiceid.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

logger = logging.getLogger("voiceid")

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain",
                         "spkrec-ecapa-voxceleb")
SAMPLE_RATE = 16000
EMBED_DIM = 192

# Cosine-similarity decision band. Calibrated 2026-09-02 on the fixture
# voices (five Kokoro timbres, four sentences each, verify_voiceid.py,
# ECAPA on cuda:0): same-speaker pairs 0.717-0.898 (hold-out print vs
# clip 0.786-0.929), different-speaker pairs -0.028-0.391 (the 0.39 is
# two female English voices, af_heart vs bf_emma). The band sits in the
# gap with margin on both sides. The user's other project runs a single
# 0.65 match threshold on real human voices; we keep an ask band under
# the accept line, like the face module, because a wrong "you are not
# you" is rude and a wrong "it is you" defeats the point. Re-measure on
# the family's real voices (point the gate at their recordings) before
# trusting these on humans: synthetic voices vary less than people do.
ACCEPT_THRESHOLD = 0.60    # at or above: same voice
REJECT_THRESHOLD = 0.45    # below: a different voice
# between: not enough evidence either way -- keep listening

# How much clean speech one sample needs before it is worth embedding,
# and the most we keep (ECAPA is happiest on 2-10 s).
MIN_SAMPLE_SECS = 1.5
MAX_SAMPLE_SECS = 8.0

_encoder = None
_device = None


def _load_encoder():
    """The shared ECAPA encoder (downloads ~80 MB once into MODEL_DIR)."""
    global _encoder, _device
    if _encoder is None:
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:                       # older SpeechBrain
            from speechbrain.pretrained import EncoderClassifier
        _device = "cuda:0" if torch.cuda.is_available() else "cpu"
        t0 = time.monotonic()
        _encoder = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE, savedir=MODEL_DIR,
            run_opts={"device": _device})
        logger.info("voiceid: ECAPA ready on %s (%.1fs)", _device,
                    time.monotonic() - t0)
    return _encoder


def provider_report() -> str:
    """Which device the encoder runs on ('cuda:0' or 'cpu')."""
    _load_encoder()
    return str(_device)


def warm_up() -> None:
    """Load the model and run one throwaway embedding (startup, not the
    visitor's first sentence, pays the 5 s load)."""
    embed_pcm(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)


def to_float_mono(pcm, sample_rate: int, channels: int = 1) -> np.ndarray:
    """int16 bytes/array (any channel count) -> float32 mono at 16 kHz."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        pcm = np.frombuffer(pcm, dtype=np.int16)
    pcm = np.asarray(pcm)
    if pcm.dtype == np.int16:
        pcm = pcm.astype(np.float32) / 32768.0
    else:
        pcm = pcm.astype(np.float32)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, sample_rate)
        pcm = resample_poly(pcm, SAMPLE_RATE // g, sample_rate // g
                            ).astype(np.float32)
    return pcm


def embed_pcm(pcm, sample_rate: int = SAMPLE_RATE,
              channels: int = 1) -> np.ndarray:
    """Unit-length ECAPA embedding of one utterance."""
    import torch
    audio = to_float_mono(pcm, sample_rate, channels)
    encoder = _load_encoder()
    with torch.no_grad():
        wav = torch.from_numpy(audio).unsqueeze(0).to(_device)
        vec = encoder.encode_batch(wav).squeeze().detach().cpu().numpy()
    vec = np.asarray(vec, dtype=np.float32)
    return vec / (np.linalg.norm(vec) or 1.0)


def embed_wav(path) -> np.ndarray:
    from scipy.io import wavfile
    sr, data = wavfile.read(str(path))
    channels = data.shape[1] if data.ndim == 2 else 1
    return embed_pcm(data.reshape(-1), sr, channels)


def similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


def average(vectors) -> np.ndarray:
    mean = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    return mean / (np.linalg.norm(mean) or 1.0)


@dataclass
class VoiceMatch:
    score: float
    sure: bool          # >= ACCEPT
    rejected: bool      # <  REJECT


def compare(vector, stored) -> VoiceMatch:
    score = similarity(vector, stored)
    return VoiceMatch(score=score, sure=score >= ACCEPT_THRESHOLD,
                      rejected=score < REJECT_THRESHOLD)


# ---------------------------------------------------------------------------
# Fusion with the face signal
# ---------------------------------------------------------------------------

class VoiceIdentity:
    """Per-session voice evidence, fused with whatever the face decided.

    ``holder`` is the tutor's ``CurrentLearner`` (learner = face-sure or
    enrolled/confirmed; candidate = face-unsure). Each sample moves the
    state and returns an *action* the caller turns into a spoken cue:

        None          nothing to do
        "learned"     stored a first print for a face-sure learner
        "verified"    voice agrees with the face (once per session)
        "challenge"   voice disagrees; ask them playfully to say more
        "downgrade"   still disagrees; learner moved to candidate -> ask
        "confirmed"   face was unsure, the voice settled it
    """

    def __init__(self, store, holder, *, learn_after: int = 2,
                 accept: float = ACCEPT_THRESHOLD,
                 reject: float = REJECT_THRESHOLD):
        self.store = store
        self.holder = holder
        self.learn_after = learn_after
        self.accept = accept
        self.reject = reject
        self.reset()

    def reset(self) -> None:
        self.samples: list[np.ndarray] = []
        self.verified = False
        self.mismatches = 0
        self.challenged = False
        self.downgraded = False
        self.last_score: float | None = None

    @property
    def print(self) -> np.ndarray | None:
        """The session's running voice print (mean of samples)."""
        return average(self.samples) if self.samples else None

    def print_list(self) -> list[float] | None:
        p = self.print
        return None if p is None else [float(x) for x in p]

    def _score(self, stored) -> float:
        score = similarity(self.print, stored)
        self.last_score = score
        return score

    def _save_print(self, learner) -> None:
        current = self.store.load(learner.id)
        if current is None:
            return
        current.voice_embedding = self.print_list()
        self.store.save(current)
        learner.voice_embedding = current.voice_embedding

    def on_sample(self, vector) -> str | None:
        self.samples.append(np.asarray(vector, dtype=np.float32))
        learner = self.holder.learner
        candidate = self.holder.candidate

        if learner is not None:
            stored = learner.voice_embedding
            if not stored:
                if len(self.samples) >= self.learn_after:
                    self._save_print(learner)
                    logger.info("voice: learned %s's print from %d samples",
                                learner.id, len(self.samples))
                    return "learned"
                return None
            if self.downgraded:
                return None
            score = self._score(stored)
            if score >= self.accept:
                if not self.verified:
                    self.verified = True
                    logger.info("voice: verified %s (score %.3f)",
                                learner.id, score)
                    return "verified"
                return None
            if score < self.reject:
                self.mismatches += 1
                if self.verified:
                    # One good match beats a later bad one (a cough, a
                    # laugh, someone else chiming in).
                    logger.info("voice: %s mismatch after verification "
                                "(score %.3f), ignoring", learner.id, score)
                    return None
                if not self.challenged:
                    self.challenged = True
                    logger.info("voice: %s does not sound like themselves "
                                "(score %.3f), challenging", learner.id, score)
                    return "challenge"
                if self.mismatches >= 2:
                    self.downgraded = True
                    self.holder.candidate = learner
                    self.holder.learner = None
                    logger.info("voice: still no match for %s (score %.3f), "
                                "downgraded to ask", learner.id, score)
                    return "downgrade"
            return None

        if candidate is not None and candidate.voice_embedding \
                and not self.downgraded:
            score = self._score(candidate.voice_embedding)
            if score >= self.accept:
                self.holder.learner = candidate
                self.holder.candidate = None
                self.verified = True
                logger.info("voice: confirmed %s by voice (score %.3f)",
                            candidate.id, score)
                return "confirmed"
        return None


# ---------------------------------------------------------------------------
# The pipecat tap
# ---------------------------------------------------------------------------

class VoiceCollector:
    """Accumulates the visitor's speech from raw input audio and embeds each
    finished sample. Built as a pipecat FrameProcessor lazily (``as_processor``)
    so this module stays importable without pipecat for the tests.

    Energy-gated: frames whose RMS clears ``gate`` (adaptive: a few times
    the running noise floor) count as speech; ``silence_secs`` of quiet
    after at least ``MIN_SAMPLE_SECS`` of speech closes a sample. Audio
    while the robot itself is speaking is dropped (bot-speaking frames
    arrive upstream from the output transport), which also covers the
    self-hearing problem the mute strategy exists for.

    ``on_sample(vector, seconds)`` is called on the event loop.
    """

    def __init__(self, on_sample, *, sample_rate: int = SAMPLE_RATE,
                 silence_secs: float = 0.8, gate_ratio: float = 3.0,
                 min_rms: float = 0.008, max_secs: float = MAX_SAMPLE_SECS,
                 min_secs: float = MIN_SAMPLE_SECS):
        self.on_sample = on_sample
        self.sample_rate = sample_rate
        self.silence_secs = silence_secs
        self.gate_ratio = gate_ratio
        self.min_rms = min_rms
        self.max_secs = max_secs
        self.min_secs = min_secs
        self.enabled = True
        self.bot_speaking = False
        self._noise = 0.003
        self._speech: list[np.ndarray] = []
        self._speech_secs = 0.0
        self._quiet_secs = 0.0
        self._pending: set = set()
        self.samples_taken = 0

    # -- audio in ------------------------------------------------------------

    def feed(self, pcm_bytes: bytes, sample_rate: int, channels: int = 1):
        """One chunk of input audio. Returns a finished sample's float32
        audio when a sample closes, else None (the async wrapper embeds it)."""
        if not self.enabled or self.bot_speaking:
            return None
        audio = to_float_mono(pcm_bytes, sample_rate, channels)
        if audio.size == 0:
            return None
        secs = audio.size / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio * audio)))
        threshold = max(self.min_rms, self._noise * self.gate_ratio)
        if rms >= threshold:
            self._speech.append(audio)
            self._speech_secs += secs
            self._quiet_secs = 0.0
            if self._speech_secs >= self.max_secs:
                return self._close()
            return None
        # quiet: track the floor slowly, and close a long-enough sample
        self._noise = 0.95 * self._noise + 0.05 * rms
        if self._speech:
            self._quiet_secs += secs
            if self._quiet_secs >= self.silence_secs:
                if self._speech_secs >= self.min_secs:
                    return self._close()
                self._speech, self._speech_secs = [], 0.0
        return None

    def _close(self):
        audio = np.concatenate(self._speech)
        self._speech, self._speech_secs, self._quiet_secs = [], 0.0, 0.0
        return audio

    def set_bot_speaking(self, speaking: bool) -> None:
        self.bot_speaking = speaking
        if speaking:
            # A sample cut off by the robot's reply is still a sample.
            if self._speech and self._speech_secs >= self.min_secs:
                audio = self._close()
                self._spawn(audio)
            else:
                self._speech, self._speech_secs = [], 0.0

    # -- embedding on a thread -----------------------------------------------

    def _spawn(self, audio: np.ndarray) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._embed_and_report(audio))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _embed_and_report(self, audio: np.ndarray) -> None:
        try:
            vector = await asyncio.to_thread(embed_pcm, audio, SAMPLE_RATE)
        except Exception as exc:                                # noqa: BLE001
            logger.warning("voiceid: embedding failed: %s", exc)
            return
        self.samples_taken += 1
        secs = audio.size / SAMPLE_RATE
        logger.info("voice: sample %d (%.1fs of speech)",
                    self.samples_taken, secs)
        await self._deliver(vector, secs)

    async def _deliver(self, vector, secs: float) -> None:
        try:
            result = self.on_sample(vector, secs)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:                                # noqa: BLE001
            logger.warning("voiceid: on_sample failed: %s", exc)

    async def inject_wav(self, path) -> None:
        """Scripted runs (``--voice-source``): hear this file as one sample."""
        vector = await asyncio.to_thread(embed_wav, path)
        self.samples_taken += 1
        logger.info("voice: sample %d (injected from %s)", self.samples_taken,
                    os.path.basename(str(path)))
        await self._deliver(vector, 0.0)

    # -- pipecat ---------------------------------------------------------------

    def as_processor(self):
        """A FrameProcessor that feeds this collector and passes every
        frame through untouched."""
        from pipecat.frames.frames import (
            BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
            InputAudioRawFrame,
        )
        from pipecat.processors.frame_processor import FrameProcessor

        collector = self

        class _Tap(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                try:
                    if isinstance(frame, InputAudioRawFrame):
                        audio = collector.feed(frame.audio, frame.sample_rate,
                                               frame.num_channels)
                        if audio is not None:
                            collector._spawn(audio)
                    elif isinstance(frame, BotStartedSpeakingFrame):
                        collector.set_bot_speaking(True)
                    elif isinstance(frame, BotStoppedSpeakingFrame):
                        collector.set_bot_speaking(False)
                except Exception as exc:                        # noqa: BLE001
                    logger.warning("voiceid tap failed: %s", exc)
                await self.push_frame(frame, direction)

        return _Tap(name="VoiceCollector")


# ---------------------------------------------------------------------------
# CLI: embed one wav (JSON), for the light test venv
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import contextlib
    import json
    ap = argparse.ArgumentParser(description="ECAPA voice print of a wav")
    ap.add_argument("wav")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    with contextlib.redirect_stdout(sys.stderr):
        vec = embed_wav(args.wav)
        device = provider_report()
    print(json.dumps({"device": device,
                      "embedding": [round(float(x), 6) for x in vec]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
