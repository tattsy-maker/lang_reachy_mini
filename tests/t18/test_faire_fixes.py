"""2026-09-25 afternoon at the Faire: barge-in over Gemini Live, a guest
lesson that survives a new face, voice that cannot hold an empty session
forever, one "still there?" per absence. No keys, no hardware; the
pipecat strategy test skips in the light venv."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voice"))
sys.path.insert(0, str(REPO / "tests" / "t18"))

from barge_in import EchoGate, level_db                    # noqa: E402
from session import ACTIVE, SessionMachine                  # noqa: E402

FRAME = 0.02


def _feed(gate, level, secs):
    opened = False
    for _ in range(int(round(secs / FRAME))):
        opened |= gate.feed(level, FRAME)
    return opened


# -- the echo gate ---------------------------------------------------------------

def test_echo_alone_never_opens_the_gate():
    gate = EchoGate(margin_db=10)
    rng = np.random.default_rng(0)
    for _ in range(500):                                # 10 s of robot speech
        assert not gate.feed(-25 + rng.normal(0, 3), FRAME)
    assert gate.echo_reference() is not None and not gate.open


def test_a_voice_over_the_echo_opens_it_after_the_onset():
    gate = EchoGate(margin_db=10, onset_ms=200)
    _feed(gate, -28, 2.0)                               # echo reference
    assert not _feed(gate, -10, 0.1), "100 ms is a click, not a visitor"
    assert _feed(gate, -10, 0.3)
    assert gate.open and gate.last_trigger["level"] == pytest.approx(-10)
    gate.bot_stopped()
    assert not gate.open, "every reply starts muted again"


def test_the_gate_stays_shut_until_it_knows_the_echo():
    gate = EchoGate()
    assert not _feed(gate, -5, 0.3), "no reference yet: stay muted"


def test_the_floor_keeps_a_quiet_robot_from_opening_on_room_noise():
    gate = EchoGate(margin_db=10, floor_db=-40)
    _feed(gate, -70, 2.0)                               # nearly silent echo
    assert not _feed(gate, -45, 0.5)                    # hall noise
    assert _feed(gate, -30, 0.5)                        # someone talking


def test_level_db():
    loud = (np.ones(320) * 16384).astype(np.int16).tobytes()
    assert level_db(loud) == pytest.approx(-6.0, abs=0.1)
    assert level_db(b"") == -120.0


def test_strategy_mutes_during_robot_speech_until_a_barge_in():
    pytest.importorskip("pipecat")
    from barge_in import make_strategy
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame,
    )
    s = make_strategy(margin_db=10)

    def frame(amp):
        pcm = (np.sin(np.linspace(0, 60, 320)) * amp).astype(np.int16)
        return InputAudioRawFrame(audio=pcm.tobytes(), sample_rate=16000,
                                  num_channels=1)

    async def run():
        out = [await s.process_frame(frame(300))]      # robot quiet: open
        out.append(await s.process_frame(BotStartedSpeakingFrame()))
        for _ in range(100):                           # 2 s of echo
            out.append(await s.process_frame(frame(1000)))
        for _ in range(15):                            # 300 ms of a visitor
            out.append(await s.process_frame(frame(20000)))
        after = await s.process_frame(frame(20000))
        stopped = await s.process_frame(BotStoppedSpeakingFrame())
        return out, after, stopped
    out, after, stopped = asyncio.run(run())
    assert out[0] is False and out[1] is True and all(out[2:102])
    assert after is False and s.barge_ins == 1
    assert stopped is False


# -- the session machine -----------------------------------------------------------

def _active(m, now=0.0):
    m.on_face(True, now)
    m.session_started(now)
    return m


def test_voice_holds_a_session_only_so_long_after_the_last_face():
    m = _active(SessionMachine(absent_secs=20, voice_hold_secs=45))
    t = 0.0
    while t < 120:
        t += 5.0
        m.on_voice(t)                                  # hall chatter, no face
        if m.on_face(False, t) == "end":
            break
    assert t <= 45 + 20 + 5, f"chatter held an empty session until {t}s"
    m2 = _active(SessionMachine(absent_secs=20))       # default: forever
    for t in range(5, 120, 5):
        m2.on_voice(float(t))
        assert m2.on_face(False, float(t)) != "end"


def test_still_there_is_asked_once_per_absence():
    m = _active(SessionMachine(absent_secs=20, voice_hold_secs=45))
    asks = 0
    t = 0.0
    while t < 60:
        t += 1.0
        if t in (16.0, 30.0):
            m.on_voice(t)                              # a voice, no face
        advice = m.on_face(False, t)
        if advice == "ask":
            asks += 1
            m.session_asked()
        if advice == "end":
            break
    assert asks == 1
    m.on_face(True, t)                                 # the face is back
    assert m._asked is False, "a face re-arms the question"
    assert m.state == ACTIVE


# -- a guest lesson survives a new face ------------------------------------------------

def test_a_new_face_carries_on_a_guest_lesson_but_still_swaps_a_learner(tmp_path):
    sys.path.insert(0, str(REPO / "tests" / "t15"))
    from test_identity import Face, blend, make, observe, unit
    from session import WATCHING

    runner, store, holder = make(tmp_path, swap_secs=3.0, onboarding="quick")
    kid1, kid2 = unit(0), unit(1)
    runner._recent_vectors = [kid1]
    asyncio.run(runner.start_session(now=10.0))
    assert runner.machine.state == ACTIVE and holder.learner is None
    holder.guest = {"target_language": "en", "level": "beginner",
                    "native_language": "en"}
    for t in (12.0, 13.0, 14.0, 15.5, 17.0):
        observe(runner, Face(embedding=blend(kid2, 0.9, seed=int(t))), now=t)
    assert runner.machine.state == ACTIVE, "the guest lesson was reset"
    assert holder.guest is not None and runner.went_neutral == 0
    assert not any("Hello" in c or "walked up" in c for c in runner.cues[1:])
    # the newcomer is now the session's face
    observe(runner, Face(embedding=blend(kid2, 0.9, seed=99)), now=18.0)
    assert runner._other_since is None

    # an enrolled learner is still protected by the swap
    runner2, store2, holder2 = make(tmp_path / "b", swap_secs=3.0,
                                    onboarding="quick")
    ana, ben = unit(2), unit(3)
    store2.create("Ana", "fr", embedding=[float(x) for x in ana])
    runner2._recent_vectors = [ana]
    asyncio.run(runner2.start_session(now=10.0))
    assert holder2.learner is not None
    for t in (12.0, 13.0, 14.0, 15.5):
        observe(runner2, Face(embedding=blend(ben, 0.9, seed=int(t))), now=t)
    assert runner2.machine.state == WATCHING
