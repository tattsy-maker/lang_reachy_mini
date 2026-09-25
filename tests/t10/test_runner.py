"""T10: the runner's start/end choreography, with the pipecat and robot
seams overridden. Real store, real holder, fake conversation."""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from session import SessionRunner  # noqa: E402
from tutor_mode import CurrentLearner  # noqa: E402
from tutor.store import LearnerStore  # noqa: E402


class FakeContext:
    def __init__(self):
        self.history = []

    def set_messages(self, messages):
        self.history.append(messages)

    def system_text(self):
        return self.history[-1][0]["content"]


class Runner(SessionRunner):
    def __init__(self, **kwargs):
        super().__init__(task=None, languages="Spanish, French",
                         base_prompt="BASE.", **kwargs)
        self.cues = []
        self.went_neutral = 0

    async def _queue_user_turn(self, text):
        self.cues.append(text)

    async def _robot_neutral(self):
        self.went_neutral += 1


def unit(dim=8, axis=0):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def make_runner(tmp_path, **kwargs):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    runner = Runner(source="unused", store=store, holder=holder,
                    context=FakeContext(), robot=None,
                    save_wait_secs=0.4, **kwargs)
    return runner, store, holder


def test_known_face_starts_with_their_briefing(tmp_path):
    runner, store, holder = make_runner(tmp_path)
    maria = store.create("Maria", "es", level="intermediate", tier="family",
                         embedding=[float(x) for x in unit()])
    runner._recent_vectors = [unit()]
    asyncio.run(runner.start_session(now=10.0))

    assert holder.learner.id == maria.id
    prompt = runner.context.system_text()
    assert prompt.startswith("BASE.") and "Maria" in prompt
    assert runner.cues and "walked up" in runner.cues[-1]
    assert runner.machine.state == "active"


def test_unknown_face_starts_stranger_flow(tmp_path):
    runner, store, holder = make_runner(tmp_path)
    runner._recent_vectors = [unit()]
    asyncio.run(runner.start_session(now=10.0))
    assert holder.learner is None
    assert "do not recognize" in runner.context.system_text()


def test_walkaway_cues_notes_then_resets(tmp_path):
    runner, store, holder = make_runner(tmp_path)
    sam = store.create("Sam", "fr", embedding=[float(x) for x in unit()])
    runner._recent_vectors = [unit()]
    asyncio.run(runner.start_session(now=10.0))
    assert holder.learner.id == sam.id

    asyncio.run(runner.end_session())
    assert any("save_session_notes" in c for c in runner.cues), \
        "walk-away never asked for notes"
    # reset is complete: nothing leaks into the next visitor
    assert holder.learner is None and holder.candidate is None
    assert holder.saved_ids == set()
    assert "nobody is in front of you" in runner.context.system_text().lower()
    assert runner.went_neutral == 1
    assert runner.machine.state == "watching"


def test_walkaway_save_already_done_skips_the_cue(tmp_path):
    runner, store, holder = make_runner(tmp_path)
    sam = store.create("Sam", "fr", embedding=[float(x) for x in unit()])
    runner._recent_vectors = [unit()]
    asyncio.run(runner.start_session(now=10.0))
    holder.saved_ids.add(sam.id)          # the model already said goodbye
    cues_before = len(runner.cues)
    asyncio.run(runner.end_session())
    assert len(runner.cues) == cues_before, \
        "no notes cue should fire when notes are already saved"


def test_two_visitors_do_not_leak_briefings(tmp_path):
    runner, store, holder = make_runner(tmp_path)
    store.create("Maria", "es", embedding=[float(x) for x in unit(axis=0)])
    store.create("Sam", "fr", embedding=[float(x) for x in unit(axis=1)])

    runner._recent_vectors = [unit(axis=0)]
    asyncio.run(runner.start_session(now=10.0))
    assert "Maria" in runner.context.system_text()
    holder.saved_ids.add("maria")
    asyncio.run(runner.end_session())

    runner._recent_vectors = [unit(axis=1)]
    asyncio.run(runner.start_session(now=200.0))
    prompt = runner.context.system_text()
    assert "Sam" in prompt and "Maria" not in prompt, \
        "the second visitor's briefing leaked the first visitor"
