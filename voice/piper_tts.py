"""Piper TTS (T5): the local engine for the languages Kokoro cannot speak.

Kokoro has no Russian voice at all and its Mandarin measured 41%
intelligibility (see multilingual.py's BROKEN_LANGUAGES autopsy). Piper —
built for Raspberry-Pi-class hardware, so the aarch64 wheels are
first-class — ships well-regarded voices for both, and its 1.7 release
phonemizes Mandarin properly (g2pW ONNX, not espeak).

``DualEngineTTS`` occupies the pipeline's one TTS slot and dispatches per
utterance: whatever language its settings currently point at (the
LanguageRouter updates them per turn) is synthesized by Piper if it is a
Piper language, by Kokoro otherwise. Same TTSAudioRawFrame contract, same
resampling path as the Kokoro base class.

Voice models are NOT in git (~63 MB each). Fetch once with:

    voice/.venv/bin/python -m piper.download_voices \
        --download-dir voice/piper_voices ru_RU-irina-medium zh_CN-huayan-medium

Measured here (2026-08-31, CPU): load 0.7 s per voice (cached after first
use), synthesis ~25x realtime, and the smoke sentences round-tripped
through Whisper verbatim. The shipping gate is voice/verify_language.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.transcriptions.language import Language

from multilingual import MultilingualKokoro, Voice

log = logging.getLogger("piper_tts")

DEFAULT_VOICES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "piper_voices")

# The languages Piper owns, keyed like multilingual.LANGUAGES. Everything
# else stays with Kokoro.
PIPER_LANGUAGES: dict[str, Voice] = {
    "ru": Voice(Language.RU, "ru_RU-irina-medium", "Russian"),
    "zh": Voice(Language.ZH, "zh_CN-huayan-medium", "Mandarin"),
}


def piper_available(voices_dir: str | Path = DEFAULT_VOICES_DIR
                    ) -> dict[str, Voice]:
    """The subset of PIPER_LANGUAGES whose model files are on disk."""
    return {code: voice for code, voice in PIPER_LANGUAGES.items()
            if Path(voices_dir, voice.voice + ".onnx").exists()}


class DualEngineTTS(MultilingualKokoro):
    """Kokoro for its verified languages, Piper for ru/zh, one service.

    Engine choice keys off the service's *current* language setting, which
    the LanguageRouter updates ahead of each reply — so a Spanish turn and
    a Russian turn in the same conversation each get the right engine.
    """

    def __init__(self, *, voices_dir: str | Path = DEFAULT_VOICES_DIR,
                 **kwargs):
        super().__init__(**kwargs)
        self._piper_dir = Path(voices_dir)
        self._piper_langs = piper_available(voices_dir)
        self._piper_voices: dict[str, object] = {}

    def _piper_code(self) -> str | None:
        lang = self._settings.language
        code = str(getattr(lang, "value", lang) or "").split("-")[0].lower()
        return code if code in self._piper_langs else None

    def _load_piper(self, code: str):
        from piper import PiperVoice
        name = self._piper_langs[code].voice
        if name not in self._piper_voices:
            path = self._piper_dir / f"{name}.onnx"
            log.info("piper: loading %s", path.name)
            self._piper_voices[name] = PiperVoice.load(path)
        return self._piper_voices[name]

    async def run_tts(self, text: str,
                      context_id: str) -> AsyncGenerator[Frame, None]:
        code = self._piper_code()
        if code is None:
            async for frame in super().run_tts(text, context_id):
                yield frame
            return
        try:
            await self.start_tts_usage_metrics(text)
            voice = await asyncio.to_thread(self._load_piper, code)
            log.info("piper: synthesizing %d chars (%s)", len(text), code)
            chunks = await asyncio.to_thread(
                lambda: list(voice.synthesize(text)))
            for chunk in chunks:
                await self.stop_ttfb_metrics()
                audio_int16 = (np.clip(chunk.audio_float_array, -1.0, 1.0)
                               * 32767).astype(np.int16).tobytes()
                audio = await self._resampler.resample(
                    audio_int16, chunk.sample_rate, self.sample_rate)
                yield TTSAudioRawFrame(
                    audio=audio, sample_rate=self.sample_rate,
                    num_channels=1, context_id=context_id)
        except Exception as e:                                  # noqa: BLE001
            yield ErrorFrame(error=f"piper synthesis failed: {e}")
        finally:
            await self.stop_ttfb_metrics()
