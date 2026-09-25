"""T13.2: the presence policy on the fake clock, and the runner's cues."""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from session import (  # noqa: E402
    ACTIVE, WATCHING, Attractor, SessionMachine, SessionRunner,
    STILL_THERE_CUE, WALKAWAY_CUE,
)
from tutor_mode import CurrentLearner  # noqa: E402
from tutor.store import LearnerStore  # noqa: E402


def active_machine():
    m = SessionMachine(stable_secs=2.0, absent_secs=60.0)
    m.on_face(True, 0.0)
    assert m.on_face(True, 2.0) == "start"
    m.session_started(2.0)
    return m


def test_voice_extends_presence_but_never_starts_a_session():
    m = SessionMachine(stable_secs=2.0, absent_secs=60.0)
    m.on_voice(0.0)
    m.on_voice(5.0)
    assert m.state == WATCHING and m.on_face(False, 6.0) is None

    m = active_machine()
    assert m.on_face(False, 50.0) == "ask"      # gone 48 s: still there?
    m.session_asked()
    m.on_voice(55.0)                            # they answered from the kettle
    assert m.on_face(False, 90.0) is None       # 35 s since the voice: alive
    assert m.on_face(False, 100.0) == "ask"     # 45 s: asks again
    m.session_asked()
    assert m.on_face(False, 114.0) is None
    assert m.on_face(False, 55.0 + 60.0) == "end"


def test_ask_fires_once_then_end():
    m = active_machine()
    assert m.on_face(False, 41.0) is None       # 39 s: not yet
    assert m.on_face(False, 42.5) == "ask"      # 40.5 s: two thirds of 60
    assert m.on_face(False, 43.0) == "ask"      # repeats until confirmed
    m.session_asked()
    assert m.on_face(False, 50.0) is None       # asked once, not again
    assert m.on_face(False, 61.9) is None
    assert m.on_face(False, 62.0) == "end"


def test_face_reappearing_resets_the_ask():
    m = active_machine()
    assert m.on_face(False, 45.0) == "ask"
    m.session_asked()
    assert m.on_face(True, 46.0) is None        # back in frame
    assert m.on_face(False, 80.0) is None       # 34 s: quiet
    assert m.on_face(False, 87.0) == "ask"      # asks again for this absence
    assert m.seconds_absent(87.0) == 41.0


def test_attractor_waits_fires_repeats_and_is_silenced():
    a = Attractor(after_secs=120, every_secs=180)
    assert not a.on_face(False, 0.0)
    assert not a.on_face(False, 100.0)
    assert a.on_face(False, 120.0)              # first fire
    assert not a.on_face(False, 200.0)          # too soon for a second
    assert a.on_face(False, 300.0)              # every 180 s
    assert not a.on_face(True, 301.0)           # somebody: silenced
    assert not a.on_face(False, 302.0)          # clock restarts
    assert not a.on_face(False, 400.0)
    assert a.on_face(False, 422.0)
    assert not Attractor(after_secs=0).on_face(False, 999.0), "0 = off"


# -- runner cues (fakes for pipecat and the robot) -----------------------------

class FakeContext:
    def __init__(self):
        self.history = []

    def set_messages(self, messages):
        self.history.append(messages)


class Runner(SessionRunner):
    def __init__(self, **kw):
        super().__init__(task=None, languages="Spanish", base_prompt="BASE.",
                         **kw)
        self.cues, self.performed, self.went_neutral = [], [], 0

    async def _queue_user_turn(self, text):
        self.cues.append(text)

    async def _robot_neutral(self):
        self.went_neutral += 1

    async def _perform(self, name, seconds):
        self.performed.append((name, seconds))


def unit(axis=0, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def test_runner_asks_then_says_goodbye_and_saves(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    runner = Runner(source="unused", store=store, holder=holder,
                    context=FakeContext(), robot=None, save_wait_secs=0.3)
    sam = store.create("Sam", "es", embedding=[float(x) for x in unit()])
    runner._recent_vectors = [unit()]
    asyncio.run(runner.start_session(now=10.0))
    assert holder.learner.id == sam.id

    asyncio.run(runner.ask_still_there())
    assert runner.cues[-1] == STILL_THERE_CUE
    assert runner.machine._asked is True
    assert "still there" in STILL_THERE_CUE and "one short sentence" in STILL_THERE_CUE

    asyncio.run(runner.end_session())
    assert runner.cues[-1] == WALKAWAY_CUE
    assert "goodbye" in WALKAWAY_CUE and "save_session_notes" in WALKAWAY_CUE
    assert "Do not speak" not in WALKAWAY_CUE, "the silent save is gone"
    assert runner.went_neutral == 1 and runner.machine.state == WATCHING


def test_note_voice_reaches_the_machine():
    runner = Runner(source="unused", store=None, holder=CurrentLearner(),
                    context=FakeContext(), robot=None)
    assert runner._voice_at is None
    runner.note_voice()
    assert runner._voice_at is not None
