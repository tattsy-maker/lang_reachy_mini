"""T17.1 (let them finish) and T17.10 (pace): the turn-patience settings
the agent hands Gemini, the briefing's side of it, the wpm line and the
WSOLA stretcher. No keys, no models; the pipecat parts skip in the
light venv."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voice"))

from pace import Stretcher, pace_line, wpm            # noqa: E402
from turns import (                                    # noqa: E402
    DEFAULT_PATIENCE_MS, MIN_PATIENCE_MS, PATIENCE_RULE, vad_settings,
)
from tutor.store import LearnerStore                   # noqa: E402
from tutor_mode import build_briefing                  # noqa: E402


# -- T17.1 ------------------------------------------------------------------------

def test_vad_settings_carry_the_patience():
    s = vad_settings(1800)
    assert s["silence_duration_ms"] == 1800
    assert s["end_sensitivity"] == "END_SENSITIVITY_LOW"
    assert s["prefix_padding_ms"] > 0
    assert vad_settings(0) is None and vad_settings(None) is None, "0 = Gemini's default"
    assert vad_settings(10)["silence_duration_ms"] == MIN_PATIENCE_MS
    assert DEFAULT_PATIENCE_MS >= 1500, "a thinking pause is over a second"


def test_gemini_params_are_built_from_the_settings():
    pytest.importorskip("pipecat", reason="GeminiVADParams comes from pipecat")
    from turns import to_gemini_vad_params
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
    vad = to_gemini_vad_params(vad_settings(2000))
    assert vad.silence_duration_ms == 2000
    assert vad.end_sensitivity.value == "END_SENSITIVITY_LOW"
    settings = GeminiLiveLLMService.Settings(vad=vad)
    assert settings.vad is vad
    assert to_gemini_vad_params(None) is None


def test_briefing_tells_the_model_to_wait(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    text = build_briefing(store.create("Ana", "fr"), "")
    assert PATIENCE_RULE in text
    assert "searching for a word, wait" in text


# -- T17.10 -------------------------------------------------------------------------

def test_wpm_and_the_pace_line():
    assert wpm("one two three four", 2.0) == 120
    assert wpm("", 2.0) is None and wpm("a", 0) is None
    assert pace_line("one two three four", 2.0) == "pace: 4 words in 2.0s (120 wpm)"


def _tone(seconds=2.0, sr=24000):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * 220 * t)
            * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


@pytest.mark.parametrize("factor", [1.2, 1.5, 0.8])
def test_stretcher_changes_length_not_level(factor):
    x = _tone()
    s = Stretcher(factor, 24000)
    out = [s.process(x[i:i + 480]) for i in range(0, len(x), 480)]
    out.append(s.flush())
    y = np.concatenate(out)
    ratio = len(y) / len(x)
    assert abs(ratio - factor) < 0.05, ratio
    level = np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2))
    assert 0.85 < level < 1.15, level


def test_stretcher_at_one_is_a_passthrough():
    x = _tone(0.5)
    s = Stretcher(1.0)
    assert np.array_equal(s.process(x), x)
    assert s.flush().size == 0


def test_stretcher_processor_flushes_on_tts_stop():
    pytest.importorskip("pipecat")
    import asyncio
    from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
    from pipecat.processors.frame_processor import FrameDirection

    proc = Stretcher(1.25, 24000).as_processor()
    pushed = []

    async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)
    proc.push_frame = fake_push

    async def drive():
        x = (_tone(1.0) * 32767).astype(np.int16).tobytes()
        step = 480 * 2
        for i in range(0, len(x), step):
            await proc.process_frame(TTSAudioRawFrame(
                audio=x[i:i + step], sample_rate=24000, num_channels=1),
                FrameDirection.DOWNSTREAM)
        await proc.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    asyncio.run(drive())
    audio = [f for f in pushed if isinstance(f, TTSAudioRawFrame)]
    total = sum(len(f.audio) // 2 for f in audio)
    assert abs(total / 24000 - 1.25) < 0.06, total / 24000
    assert isinstance(pushed[-1], TTSStoppedFrame), "the stop frame follows the flushed tail"
