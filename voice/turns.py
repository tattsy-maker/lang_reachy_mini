"""Turn patience (T17.1): let the visitor finish.

On 2026-09-05 the mother was cut off twice when she paused to think
("you didn't let me finish"), and her closing wish was "pause and give
me more time when I am searching for words". The model apologised and
could change nothing: in cloud mode turn-taking is Gemini Live's
server-side voice activity detection, and the agent passed it no
settings, so it ran on Google's defaults (tuned for fluent speakers).

pipecat 1.6 exposes the knobs (``GeminiVADParams``). This module holds
the policy as plain values so the light test venv can check it without
pipecat; ``agent.py`` turns them into the service's settings object.

    --turn-patience-ms 1800     (BOOTH_TURN_PATIENCE_MS in the booth script)

The price is lag: every reply now starts after that much silence, so
the ``turn: first sound`` median rises by about the same amount. That
is deliberate; a learner searching for a word needs the silence more
than the reply needs the speed.
"""

from __future__ import annotations

DEFAULT_PATIENCE_MS = 1800
# Below this the request is meaningless: Gemini's own default is already
# shorter than a thinking pause.
MIN_PATIENCE_MS = 500
# The audio kept from before the detected start of speech, so a soft
# first syllable is not lost.
PREFIX_PADDING_MS = 300


def vad_settings(patience_ms: int | None) -> dict | None:
    """The Gemini VAD settings for a patience value, as plain values
    (enum *names* from google.genai.types), or None for "leave Gemini's
    defaults alone" (``patience_ms`` of 0 or None)."""
    if not patience_ms or patience_ms <= 0:
        return None
    patience_ms = max(MIN_PATIENCE_MS, int(patience_ms))
    return {
        # LOW end sensitivity: needs more silence before calling the
        # turn over; the silence itself is the number below.
        "end_sensitivity": "END_SENSITIVITY_LOW",
        "silence_duration_ms": patience_ms,
        "prefix_padding_ms": PREFIX_PADDING_MS,
    }


def to_gemini_vad_params(settings: dict | None):
    """``vad_settings`` -> a pipecat ``GeminiVADParams`` (needs pipecat and
    google-genai; only the agent calls this)."""
    if settings is None:
        return None
    from google.genai.types import EndSensitivity
    from pipecat.services.google.gemini_live.llm import GeminiVADParams
    return GeminiVADParams(
        end_sensitivity=EndSensitivity(settings["end_sensitivity"]),
        silence_duration_ms=settings["silence_duration_ms"],
        prefix_padding_ms=settings["prefix_padding_ms"],
    )


# The briefing's side of it: even with a patient VAD, the model must not
# fill a pause with a reply of its own.
PATIENCE_RULE = (
    "When the student pauses mid-sentence, or is clearly searching for a "
    "word, wait. Do not finish their sentence, do not answer a question "
    "they have not finished asking. If you did cut in, apologise in three "
    "words and let them go on.")
