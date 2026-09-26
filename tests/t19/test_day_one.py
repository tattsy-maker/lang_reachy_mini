"""2026-09-25 evening, after the Faire's first day: a gentle wake-up, the
greeting ("I am Reachy", never "you are Reachy"), look only when asked,
calm recorded moves, a quick one-word answer that Gemini notices, the
newcomer asked instead of carried along, and a call-out to onlookers.
No keys, no hardware. Run in the voice venv (numpy, scipy; the agent
test also needs pipecat and skips without it)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voice"))
sys.path.insert(0, str(REPO / "tests" / "t15"))

import moves                                                # noqa: E402
import session                                              # noqa: E402
import tutor_mode                                           # noqa: E402
from session import ACTIVE, WATCHING, Caller                # noqa: E402
from turns import DEFAULT_ONSET_MS, vad_settings            # noqa: E402


# -- the greeting ------------------------------------------------------------------

def test_the_greeting_says_i_am_reachy_and_what_it_can_do():
    quick = tutor_mode.stranger_briefing("quick", "French")
    assert '"Hi! I am Reachy, a friendly language tutor."' in quick
    assert "you are Reachy" not in quick
    assert "a quick quiz" in quick and "beginner" in quick
    assert "I am Reachy, a friendly" in tutor_mode.STRANGER_BRIEFING
    assert "I am Reachy, a friendly" in tutor_mode.QUICK_UNSURE_BRIEFING


def test_the_quick_lesson_asks_the_level_and_the_kind_of_practice():
    lesson = tutor_mode._QUICK_LESSON
    assert "Always know their level before you teach" in lesson
    assert "useful words" in lesson and "conversation" in lesson
    assert "English is their own language" in lesson
    assert "tone" in lesson, "the Mandarin tones feedback"


def test_booth_persona_plays_along_and_knows_how_it_works():
    persona = tutor_mode.BOOTH_PERSONA
    assert "zero by zero" in persona.lower()
    assert "joke" in persona
    assert "Gemini Live" in persona and "Pollen Robotics" in persona
    assert "I will know" not in persona, "the family dropped that line"


# -- look only when asked ------------------------------------------------------------

def test_look_is_only_for_when_the_visitor_asks():
    text = (REPO / "voice" / "agent.py").read_text()
    start = text.index("VISION_FACE_LOOK = ")
    vision = text[start:text.index('"""', text.index('"""', start) + 3)]
    assert "want to check who" not in vision
    assert "Never call it on your own" in vision
    assert "never describe or bring up a picture from earlier" in vision
    assert "call look" not in session.BOOTH_NOTE
    assert "one person at a time" in session.BOOTH_NOTE


# -- English in English ------------------------------------------------------------

class Params:
    def __init__(self, **arguments):
        self.arguments = arguments
        self.result = None

    async def result_callback(self, result):
        self.result = result


def test_english_asked_in_english_is_questioned_first(tmp_path):
    pytest.importorskip("pipecat")
    from tutor.store import LearnerStore
    holder = tutor_mode.CurrentLearner()
    tools = tutor_mode.build_enrollment_tools(
        LearnerStore(tmp_path), holder, face_source=None)
    start = next(t.handler for t in tools if t.name == "start_lesson")
    p = Params(language="English", level="beginner", native_language="English")
    asyncio.run(start(p))
    assert "own language" in p.result["note"] and "Spanish" in p.result["note"]
    p = Params(language="Spanish", level="beginner", native_language="English")
    asyncio.run(start(p))
    assert "useful words" in p.result["note"]


# -- turns: a quick "hola" is a turn ------------------------------------------------

def test_a_short_word_is_enough_to_start_a_turn():
    assert DEFAULT_ONSET_MS <= 150
    assert vad_settings(1200)["prefix_padding_ms"] == DEFAULT_ONSET_MS
    assert vad_settings(1200, onset_ms=300)["prefix_padding_ms"] == 300
    booth = (REPO / "start_booth.sh").read_text()
    assert 'BOOTH_TURN_PATIENCE_MS:-1200' in booth
    assert '--turn-onset-ms "$TURN_ONSET_MS"' in booth


# -- calm moves ----------------------------------------------------------------

def test_a_slowed_clip_takes_longer_and_plays_the_same_path():
    from reachy_target import _Slowed

    class Clip:
        duration = 2.0
        sound_path = "x.wav"

        def evaluate(self, t):
            return np.eye(4) * (1 + t), np.array([t, -t]), t

    slow = _Slowed(Clip(), 0.5)
    assert slow.duration == 4.0 and slow.sound_path is None
    assert slow.evaluate(3.0)[2] == pytest.approx(1.5)
    assert slow.evaluate(10.0)[2] < 2.0, "never past the clip's end"
    assert moves.LIBRARY["sway"].pass_secs == pytest.approx(1.9 / 0.5 + 1.7)
    assert moves.LIBRARY["spin"].pass_secs == pytest.approx(6.0)


def test_every_clip_plays_inside_the_calm_limits():
    measured = 0
    for spec in moves.LIBRARY.values():
        path = moves.clip_path(spec) if spec.dataset else None
        if path is None:
            continue
        played = moves.measure(path, spec.speed)
        for key, limit in moves.CALM.items():
            assert played[key] <= limit * 1.02, (spec.name, key, played[key])
        measured += 1
    if not measured:
        pytest.skip("move datasets not cached (moves.py --preload)")


def test_the_antenna_wiggle_is_no_longer_a_saw_blade():
    driver = (REPO / "reachy_driver.py").read_text()
    agent = (REPO / "voice" / "agent.py").read_text()
    assert "goto, 0.25," not in driver
    assert "posture(duration=0.25, antenna" not in agent


# -- waking up ------------------------------------------------------------------

def test_wake_up_is_slow_and_the_vendor_snap_is_off(monkeypatch):
    pytest.importorskip("pipecat")
    import agent

    calls = []

    class Robot:
        async def call(self, name, **args):
            calls.append((name, args))

    async def no_sleep(_):
        return None

    monkeypatch.setattr(agent.asyncio, "sleep", no_sleep)
    asyncio.run(agent.wake_gently(Robot()))
    assert calls[0] == ("home", {"duration": agent.WAKE_RISE_SECS})
    assert agent.WAKE_RISE_SECS >= 3.0
    stretches = [a for n, a in calls[1:] if n == "goto_posture"]
    assert stretches and all(a["duration"] >= 0.9 for a in stretches)
    assert stretches[-1]["antenna_left"] == 0.0
    booth = (REPO / "start_booth.sh").read_text()
    assert booth.count("reachy-mini-daemon --no-wake-up-on-start") == 2


# -- the newcomer is asked --------------------------------------------------------

def test_a_newcomer_in_a_guest_lesson_is_asked_once(tmp_path):
    from test_identity import Face, blend, make, unit

    runner, store, holder = make(tmp_path, swap_secs=3.0, onboarding="quick")
    kid1, kid2, kid3 = unit(0), unit(1), unit(2)

    async def main():
        runner._recent_vectors = [kid1]
        await runner.start_session(now=10.0)
        holder.guest = {"target_language": "es", "level": "beginner",
                        "native_language": "en"}
        for t in (12.0, 13.0, 14.0, 15.5, 17.0):
            await runner.observe(Face(embedding=blend(kid2, 0.9, seed=int(t))),
                                 t, (480, 640))
        await runner._cue_task
        asked = [c for c in runner.cues if "go on with Spanish" in c]
        assert len(asked) == 1, runner.cues
        # a third child 10 s later: still inside the gap, not asked again
        for t in (27.0, 28.0, 29.0, 30.5, 32.0):
            await runner.observe(Face(embedding=blend(kid3, 0.9, seed=int(t))),
                                 t, (480, 640))
        assert len([c for c in runner.cues if "go on with" in c]) == 1
        assert runner.machine.state == ACTIVE and runner.went_neutral == 0

    asyncio.run(main())


def test_no_question_before_a_lesson_has_started(tmp_path):
    from test_identity import Face, blend, make, unit

    runner, store, holder = make(tmp_path, swap_secs=3.0, onboarding="quick")

    async def main():
        runner._recent_vectors = [unit(0)]
        await runner.start_session(now=10.0)
        for t in (12.0, 13.0, 14.0, 15.5, 17.0):
            await runner.observe(Face(embedding=blend(unit(1), 0.9,
                                                      seed=int(t))),
                                 t, (480, 640))
        assert runner._cue_task is None
        assert not any("go on with" in c for c in runner.cues)

    asyncio.run(main())


# -- calling out to onlookers -------------------------------------------------------

def test_the_caller_greets_a_newcomer_once_and_not_a_poster():
    c = Caller(every_secs=40, fresh_secs=10, hold_secs=1.0)
    assert not c.on_frame(True, 0.0), "one frame is not someone looking"
    assert c.on_frame(True, 1.5), "an onlooker at startup counts"
    for t in np.arange(2.0, 200.0, 0.5):      # a poster that never leaves
        assert not c.on_frame(True, float(t)), t
    for t in np.arange(200.0, 215.0, 0.5):     # it goes; someone else comes
        c.on_frame(False, float(t))
    assert not c.on_frame(True, 215.0)
    assert c.on_frame(True, 216.5)


def test_the_caller_waits_between_calls_and_while_someone_is_served():
    c = Caller(every_secs=40, fresh_secs=10, hold_secs=1.0)
    c.on_frame(True, 0.0)
    assert c.on_frame(True, 1.0)
    for t in np.arange(1.5, 12.0, 0.5):
        c.on_frame(False, float(t))
    c.on_frame(True, 12.0)
    assert not c.on_frame(True, 13.5), "only 13 s since the last call"
    # a bystander watching a lesson is not fresh when the lesson ends
    c2 = Caller(every_secs=40)
    for t in np.arange(0.0, 60.0, 0.5):
        assert not c2.on_frame(True, float(t), allowed=False)
    assert not c2.on_frame(True, 60.5, allowed=True)
    assert not Caller(every_secs=0).on_frame(True, 5.0), "0 = off"


def test_the_runner_calls_out_only_when_nobody_is_being_served(tmp_path):
    from test_identity import make

    class Far:
        def __init__(self, width):
            self.bbox = (100, 100, 100 + width, 100 + width)
            self.score = 0.8
            self.embedding = None

    runner, store, holder = make(tmp_path, call_out_every=40)

    async def main():
        await runner.onlookers([Far(20)], 1000, visitor=False, now=0.0)
        await runner.onlookers([Far(20)], 1000, visitor=False, now=2.0)
        assert not runner.cues, "2 % of the frame is too far to be anyone"
        await runner.onlookers([Far(40)], 1000, visitor=False, now=20.0)
        await runner.onlookers([Far(40)], 1000, visitor=False, now=21.5)
        assert runner.cues == [session.CALL_OUT_CUE]
        runner.machine.state = ACTIVE
        for t in (80.0, 81.5, 83.0):
            await runner.onlookers([Far(40)], 1000, visitor=False, now=t)
        assert len(runner.cues) == 1, "not while someone is being taught"

    assert runner.machine.state == WATCHING
    asyncio.run(main())
