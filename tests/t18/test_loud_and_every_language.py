"""2026-09-25: as loud as the speaker allows, and every language Gemini
Live speaks (a visitor asked for Arabic and heard "I do not speak Arabic
yet"). No keys, no models; the pipecat part skips in the light venv."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voice"))

from loudness import CEILING_DB, SoftClip, louder_by_db   # noqa: E402
from tutor_mode import (                                   # noqa: E402
    _LANGUAGE_NAMES, _SCRIPT_HINTS, cloud_language_names, normalize_language,
)

SR = 24000


def _speechlike(seconds: float = 1.0) -> np.ndarray:
    """A voiced-speech stand-in: a 150 Hz buzz with harmonics, syllable
    envelope, normalised to 0 dBFS peak like Gemini's audio."""
    t = np.arange(int(SR * seconds)) / SR
    x = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 12))
    x *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3 * t))
    return (x / np.abs(x).max()).astype(np.float32)


def _rms_db(x):
    return 20 * np.log10(np.sqrt((x ** 2).mean()))


# -- loudness -------------------------------------------------------------------

def test_soft_clip_is_louder_and_never_over_the_ceiling():
    x = _speechlike()
    y = SoftClip(12).process(x)
    assert np.abs(y).max() <= 10 ** (CEILING_DB / 20) + 1e-6
    assert _rms_db(y) - _rms_db(x) > 3.0


def test_more_drive_is_louder():
    x = _speechlike()
    levels = [_rms_db(SoftClip(d).process(x)) for d in (3, 6, 9, 12)]
    assert levels == sorted(levels)
    assert louder_by_db(12) > louder_by_db(9) > louder_by_db(6) > 0


def test_zero_drive_is_a_passthrough():
    x = _speechlike()
    assert np.array_equal(SoftClip(0).process(x), x)


def test_pcm16_round_trip_keeps_length():
    x = (_speechlike(0.1) * 32767).astype(np.int16).tobytes()
    assert len(SoftClip(12).process_pcm16(x)) == len(x)


def test_processor_changes_tts_audio_only():
    pytest.importorskip("pipecat")
    import asyncio
    from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
    from pipecat.processors.frame_processor import FrameDirection

    proc = SoftClip(12).as_processor()
    pushed = []

    async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)
    proc.push_frame = fake_push
    raw = (_speechlike(0.1) * 0.5 * 32767).astype(np.int16).tobytes()

    async def drive():
        await proc.process_frame(TTSAudioRawFrame(
            audio=raw, sample_rate=SR, num_channels=1),
            FrameDirection.DOWNSTREAM)
        await proc.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    asyncio.run(drive())
    assert isinstance(pushed[-1], TTSStoppedFrame)
    out = np.frombuffer(pushed[0].audio, dtype=np.int16)
    assert out.size == len(raw) // 2
    assert np.abs(out).max() > np.abs(np.frombuffer(raw, np.int16)).max()


# -- every language -------------------------------------------------------------

def test_gemini_lives_ninety_nine_languages_are_known():
    # zh-Hans/zh-Hant fold into zh and pt-BR/pt-PT into pt: 96 codes.
    assert len(_LANGUAGE_NAMES) >= 96
    for code in ("ar", "am", "sw", "yo", "zu", "ka", "hy", "fil", "ceb",
                 "qu", "mi", "wo", "ja", "ko", "he", "fa"):
        assert code in _LANGUAGE_NAMES, code


@pytest.mark.parametrize("said,code", [
    ("Arabic", "ar"), ("arabic", "ar"), ("ar", "ar"), ("ar-EG", "ar"),
    ("Arabic (Egyptian)", "ar"), ("Farsi", "fa"), ("Tagalog", "fil"),
    ("Mandarin", "zh"), ("Chinese", "zh"), ("zh-Hans", "zh"),
    ("pt-BR", "pt"), ("Brazilian Portuguese", "pt"), ("Swahili", "sw"),
    ("Yoruba", "yo"), ("Georgian", "ka"), ("nb", "no"),
])
def test_normalize_language_takes_every_form(said, code):
    assert normalize_language(said) == code


def test_normalize_language_still_rejects_nonsense():
    assert normalize_language("Klingon") is None
    assert normalize_language("") is None


def test_the_cloud_prompt_names_arabic():
    names = cloud_language_names()
    assert "Arabic" in names and "Zulu" in names and "Mandarin Chinese" in names
    assert names.count(",") >= 90


def test_non_latin_scripts_get_a_hint():
    for code in ("ar", "fa", "ur", "am", "ka", "hy", "th", "km", "ta", "bn"):
        assert code in _SCRIPT_HINTS, code
