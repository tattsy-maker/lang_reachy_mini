#!/usr/bin/env python3
"""multilingual -- let Reachy answer in the language you spoke to it.

Kokoro is already a multilingual model: the voices file that ships with it
carries 54 voices across nine languages. What pinned the robot to English was
configuration, not capability. Two things were missing:

1. **Whisper was told the language instead of asked.** pipecat defaults its
   MLX Whisper service to `language=EN` and passes that straight to
   `mlx_whisper.transcribe`, which disables Whisper's own detection. Passing
   `None` instead makes it detect, at no measurable cost -- detection happens
   inside the transcribe call that was running anyway.

2. **The detected language was thrown away.** pipecat stamps the *configured*
   language onto the TranscriptionFrame, not the one Whisper found, so nothing
   downstream could act on it. `MultilingualWhisperMLX` keeps it.

`LanguageRouter` then switches the Kokoro voice to match. Voice and language
have to move together: Kokoro voices are language-specific (`ff_siwis` is a
French voice), and reading French text with an English voice produces
confident nonsense rather than an error.

**Why the guards matter more than the switching.** Whisper will happily detect
a language from a cough. An unguarded router flips the robot into Japanese
mid-conversation and it cannot be talked back out, because everything it hears
after that is being transcribed under the wrong assumption. So a switch needs
a transcript long enough to mean something, and a language we actually have a
voice for.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, NamedTuple

import numpy as np
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.settings import assert_given
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

log = logging.getLogger("multilingual")


class Voice(NamedTuple):
    """A language Kokoro can speak, and the voice to speak it with."""

    language: Language
    voice: str
    name: str


# The nine languages Kokoro v1.0 ships voices for, keyed by the code Whisper
# reports. The voice is the default pick; every language has more than one
# except French and Italian, which ship exactly one each.
#
# Kokoro's Portuguese is Brazilian, and its Chinese is Mandarin.
LANGUAGES: dict[str, Voice] = {
    "en": Voice(Language.EN, "af_heart", "English"),
    "es": Voice(Language.ES, "ef_dora", "Spanish"),
    "fr": Voice(Language.FR, "ff_siwis", "French"),
    "it": Voice(Language.IT, "if_sara", "Italian"),
    "pt": Voice(Language.PT, "pf_dora", "Portuguese"),
    "hi": Voice(Language.HI, "hf_alpha", "Hindi"),
}

# Japanese and Chinese are deliberately absent, and it is not an oversight.
#
# Kokoro ships voices for both (`jf_alpha`, `zf_xiaobei` and six more), but
# kokoro-onnx phonemizes every language through espeak, and espeak is not good
# enough at either script to feed this model. Kokoro's CJK voices were trained
# on misaki phonemes, which carry Japanese pitch accent and real word
# segmentation; espeak produces neither.
#
# Measured here by speaking a sentence and transcribing it back (the same
# round trip the other six pass at 96-100%):
#
#     ja  19% match, and 12.2s of audio for a 2.5s sentence -- it loops
#     zh  41% match
#
# "Hello, I am a small robot on your desk" comes back in Japanese as "I read
# my information. I read my information. I read my information." espeak
# collapses the word for "small" (chiisana) to a single "s". It sounds fluent
# and means nothing, which is the worst way for this to fail: the robot is
# confidently talking gibberish and only a Japanese speaker would notice.
#
# The fix, if CJK is wanted, is misaki rather than a different TTS engine:
# `create_stream(..., is_phonemes=True)` accepts phonemes directly, so
# misaki[ja] / misaki[zh] can do the g2p that espeak cannot. Both pull in
# heavier dependencies (pyopenjtalk, jieba), which is why this stops here.
BROKEN_LANGUAGES: dict[str, Voice] = {
    "ja": Voice(Language.JA, "jf_alpha", "Japanese"),
    "zh": Voice(Language.ZH, "zf_xiaobei", "Chinese"),
}

# Whisper reports a base code, so "en" covers both Kokoro English variants.
# Ask for a British voice with --voice bm_george and it stays British.
SPOKEN_LANGUAGES = ", ".join(v.name for v in LANGUAGES.values())

# Below this many characters a transcript is too short to trust a detection
# from. "Yes", "hmm" and a cough all detect as something, and the something is
# often not English.
MIN_CHARS_TO_SWITCH = 12


class MultilingualKokoro(KokoroTTSService):
    """Kokoro with the two language codes espeak actually wants.

    Kokoro phonemizes through espeak-ng, and espeak names French `fr-fr` and
    Mandarin `cmn`. pipecat's language map hands it the bare codes `fr` and
    `zh`, which espeak rejects outright:

        RuntimeError: language "fr" is not supported by the espeak backend

    That lands inside `run_tts`, which turns exceptions into an ErrorFrame
    rather than raising, so the failure mode is a robot that goes silent for
    those two languages while everything else keeps working. The other seven
    (`en-us`, `en-gb`, `es`, `it`, `pt`, `hi`, `ja`) pass through unchanged.
    """

    ESPEAK_CODES = {Language.FR: "fr-fr", Language.ZH: "cmn"}

    def language_to_service_language(self, language: Language) -> str:
        """Map a language to the code espeak accepts."""
        for lang, code in self.ESPEAK_CODES.items():
            if str(getattr(language, "value", language)).lower().startswith(lang.value):
                return code
        return super().language_to_service_language(language)


# Bilingual priming (T7). Whisper conditions its decoding on an initial
# prompt; feeding it one sentence in each language keeps both "in mind"
# mid-utterance, which measurably helps embedded foreign phrases survive
# (see tests/reports/verify_codeswitch_*). One entry per tutoring language.
PRIMING = {
    "es": "A bilingual lesson mixing English and Spanish. Una lección "
          "bilingüe que mezcla inglés y español.",
    "fr": "A bilingual lesson mixing English and French. Une leçon "
          "bilingue qui mélange l'anglais et le français.",
    "it": "A bilingual lesson mixing English and Italian. Una lezione "
          "bilingue che mescola inglese e italiano.",
    "pt": "A bilingual lesson mixing English and Portuguese. Uma aula "
          "bilíngue que mistura inglês e português.",
    "ru": "A bilingual lesson mixing English and Russian. Двуязычный "
          "урок, в котором смешиваются английский и русский.",
    "zh": "A bilingual lesson mixing English and Chinese. 一节混合英语"
          "和中文的双语课。",
    "hi": "A bilingual lesson mixing English and Hindi. अंग्रेज़ी और "
          "हिंदी को मिलाने वाला द्विभाषी पाठ।",
}


# Which pairs priming actually helps -- measured, not assumed
# (tests/reports/verify_codeswitch_2026-08-31.md). Span survival with vs
# without priming: es 86->96, ru 62->88 (unprimed, Whisper *translates*
# Russian spans into English). But fr 82->52, pt 74->40, it 92->80,
# zh 69->56: for those, the prompt induces Whisper's known
# hallucination/repetition failure ("My favorite phrase is! My favorite
# phrase is...", empty transcripts, stray Arabic). So priming is a
# per-pair policy, applied only where the measurement says it helps.
PRIMING_HELPS = {"es", "ru"}


def bilingual_priming(code: str, native: str = "en") -> str | None:
    """The Whisper initial prompt for a <native>+<code> lesson, or None
    for pairs where priming measurably hurts (see PRIMING_HELPS).

    T16: the pair is (native, target) rather than (English, target). The
    prompts are all English+X, so a Russian speaker learning English gets
    the same ru prompt as an English speaker learning Russian; a pair
    with no English in it (Russian speaker, Spanish lessons) is unmeasured
    and gets no priming."""
    code, native = str(code).lower(), str(native or "en").lower()
    if code == native:
        return None
    if native == "en":
        other = code
    elif code == "en":
        other = native
    else:
        return None
    if other not in PRIMING_HELPS:
        return None
    return PRIMING.get(other)


class MultilingualWhisperMLX(WhisperSTTServiceMLX):
    """MLX Whisper with detection turned back on, and the result kept.

    This overrides `run_stt` rather than wrapping it because the language
    pipecat reports is baked into the middle of that method: it yields a
    TranscriptionFrame carrying `self._settings.language`, and there is no hook
    between the transcribe call and the frame. The two hallucination guards
    below are carried over from pipecat's implementation deliberately; if that
    upstream method grows a third, this needs it too.
    """

    # Set (e.g. by tutor mode) to condition every transcription on a
    # bilingual prompt; None leaves Whisper unconditioned.
    initial_prompt: str | None = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe, detect the language, and report both."""
        try:
            import mlx_whisper

            await self.start_processing_metrics()

            audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            model_path = assert_given(self._settings.model)
            if model_path is None:
                raise ValueError("Whisper model must be specified")
            temperature = assert_given(self._settings.temperature)

            # language=None is the whole point: it asks Whisper to detect
            # rather than assume.
            chunk = await asyncio.to_thread(
                mlx_whisper.transcribe,
                audio_float,
                path_or_hf_repo=model_path,
                temperature=temperature,
                language=None,
                initial_prompt=self.initial_prompt,
            )

            text = ""
            no_speech_prob_threshold = assert_given(self._settings.no_speech_prob)
            for segment in chunk.get("segments", []):
                # Carried over from pipecat: this exact compression ratio is a
                # known hallucination signature.
                if segment.get("compression_ratio", None) == 0.5555555555555556:
                    continue
                if (
                    no_speech_prob_threshold is not None
                    and segment.get("no_speech_prob", 0.0) < no_speech_prob_threshold
                ):
                    text += f"{segment.get('text', '')} "

            await self.stop_processing_metrics()

            if not text.strip():
                return

            detected = chunk.get("language")
            language = None
            if detected:
                code = str(detected).lower()
                known = LANGUAGES.get(code)
                if known:
                    language = known.language
                else:
                    # Not a Kokoro language, but downstream may still speak
                    # it (Piper's ru/zh); report it and let the router's
                    # speakable() guard decide. Unknown codes stay None.
                    try:
                        language = Language(code)
                    except ValueError:
                        language = None

            log.info("heard (%s): %s", detected or "unknown", text.strip())
            await self._handle_transcription(text, True, language)
            yield TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), language)

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


class LanguageRouter(FrameProcessor):
    """Point the Kokoro voice at whatever language was just heard.

    Sits directly after the STT service, watches TranscriptionFrames, and
    pushes a TTSUpdateSettingsFrame downstream when the language changes. Every
    frame is passed through untouched.

    The update travels the same path as the transcript, so it reaches the TTS
    service ahead of the reply that transcript produces. The robot answers the
    turn it just heard in the right voice, not the turn after.
    """

    def __init__(self, *, initial: str = "en", voice: str | None = None,
                 enabled: bool = True,
                 extra: dict[str, Voice] | None = None):
        super().__init__()
        self._enabled = enabled
        self._current = initial
        # Languages a second engine can speak (T5: Piper's ru/zh), checked
        # after Kokoro's own. The TTS service downstream must know how to
        # honor them (DualEngineTTS does).
        self._extra = dict(extra or {})
        # An explicit --voice overrides the default for the starting language
        # only. Switching away and back returns to that override.
        self._overrides = {initial: voice} if voice else {}

    def speakable(self, code: str):
        """The Voice for a code across both engines, or None."""
        return LANGUAGES.get(code) or self._extra.get(code)

    def voice_for(self, code: str) -> str:
        if self._overrides.get(code):
            return self._overrides[code]
        return self.speakable(code).voice

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._enabled and isinstance(frame, TranscriptionFrame):
            await self._maybe_switch(frame)

        await self.push_frame(frame, direction)

    async def _maybe_switch(self, frame: TranscriptionFrame) -> None:
        code = _base_code(frame.language)
        if code is None or code == self._current:
            return
        spoken = self.speakable(code)
        if spoken is None:
            if code in BROKEN_LANGUAGES:
                # Worth saying out loud: the robot is about to answer in
                # English and the reason is not obvious from the outside.
                log.info("heard %s, which no local engine speaks "
                         "intelligibly here; staying in %s",
                         BROKEN_LANGUAGES[code].name,
                         self.speakable(self._current).name)
            return
        if len(frame.text.strip()) < MIN_CHARS_TO_SWITCH:
            log.debug("ignoring %s detected from a short utterance: %r",
                      code, frame.text.strip())
            return
        voice = self.voice_for(code)
        log.info("switching speech to %s (voice %s)", spoken.name, voice)
        self._current = code
        await self.push_frame(TTSUpdateSettingsFrame(
            delta=KokoroTTSService.Settings(
                voice=voice, language=spoken.language)))


def _base_code(language) -> str | None:
    """Reduce a Language enum (or a raw code) to the base code LANGUAGES uses."""
    if language is None:
        return None
    value = getattr(language, "value", language)
    return str(value).split("-")[0].lower()
