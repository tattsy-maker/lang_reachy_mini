"""2026-09-25: the quick start (one question -- which language, what
level -- then a lesson; nothing stored unless the visitor asks to be
remembered, and then only name and goal) and a livelier attractor (30 s,
then every 60 s, a dozen more moves, never the same twice, cut short by a
visitor walking up). The tool test skips without pipecat."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voice"))
sys.path.insert(0, str(REPO / "tests" / "t13"))

from intake import INTAKE_FIELDS, QUICK_FIELDS, Intake     # noqa: E402
from moves import (ATTRACT_MOVES, LIBRARY, attract_passes,  # noqa: E402
                   describe, names)
from session import SessionRunner                           # noqa: E402
from tutor.store import LearnerStore                        # noqa: E402
from tutor_mode import (                                    # noqa: E402
    QUICK_STRANGER_BRIEFING, STRANGER_BRIEFING, CurrentLearner,
    build_unsure_briefing, normalize_level, stranger_briefing,
)


# -- the intake ------------------------------------------------------------------

def test_quick_intake_needs_only_name_and_goal_after_the_lesson_started():
    it = Intake(quick=True)
    assert it.missing() == list(QUICK_FIELDS)
    it.answers.update(target_language="ar", level="beginner",
                      native_language="en")
    assert it.missing() == ["name", "goal"]
    assert not it.complete
    it.answer("name", "Layla")
    it.answer("goal", "travel to Cairo")
    assert it.complete
    kw = it.profile_kwargs()
    assert kw["corrections"] == "blocking" and kw["explain_in"] == "native"
    assert it.plan(0.0) is None, "nobody was asked for minutes"
    assert "Then back to the lesson" in Intake(quick=True).status()["note"]


def test_full_intake_is_unchanged():
    it = Intake()
    assert it.missing() == list(INTAKE_FIELDS)
    assert "No lesson yet" in it.status()["note"]


# -- the prompts ------------------------------------------------------------------

def test_quick_briefing_asks_one_question_and_never_offers_to_remember():
    text = stranger_briefing("quick", "unused")
    assert text is QUICK_STRANGER_BRIEFING
    assert "Reachy" in text and "start_lesson" in text
    assert "which language they would like to practise" in text
    assert "beginner or already speak some" in text
    assert "rest of the day" not in text
    assert "do not offer to remember them" in text
    assert "Only if they ask you to remember them" in text
    assert stranger_briefing("full", "Spanish") == STRANGER_BRIEFING.format(
        languages="Spanish")


def test_quick_unsure_briefing_leads_a_no_into_the_quick_start(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    sam = store.create("Sam", "es")
    quick = build_unsure_briefing(sam, quick=True)
    assert "Sam, is that you?" in quick and "start_lesson" in quick
    assert "rest of the day" not in quick
    assert "rest of the day" in build_unsure_briefing(sam)


@pytest.mark.parametrize("said,level", [
    ("beginner", "beginner"), ("Advanced", "advanced"), ("a little", "intermediate"),
    ("I speak some", "intermediate"), ("fluent", "advanced"), ("B2", "intermediate"),
    ("never studied it", "beginner"), ("", "beginner"),
])
def test_normalize_level(said, level):
    assert normalize_level(said) == level


# -- the tool ---------------------------------------------------------------------

class Params:
    def __init__(self, **arguments):
        self.arguments = arguments
        self.result = None

    async def result_callback(self, result):
        self.result = result


def test_start_lesson_then_remember_me_asks_two_questions(tmp_path):
    pytest.importorskip("pipecat", reason="FunctionSchema comes from pipecat")
    from tutor_mode import build_enrollment_tools, build_tutor_tools
    fake = types.ModuleType("face_id")
    fake.capture_embedding = lambda source, frames=None, **kw: [0.1, 0.2, 0.3]
    saved = sys.modules.get("face_id")
    sys.modules["face_id"] = fake
    rebriefs = []
    try:
        store = LearnerStore(tmp_path / "learners")
        holder = CurrentLearner()
        intake = Intake(quick=True)
        tools = {t.name: t for t in build_enrollment_tools(
            store, holder, face_source=None, intake=intake,
            rebrief=rebriefs.append)}
        tools.update({t.name: t for t in build_tutor_tools(store, holder)})

        p = Params(language="Arabic", level="beginner", native_language="English")
        asyncio.run(tools["start_lesson"].handler(p))
        assert p.result["lesson"] == "Arabic" and p.result["taught_in"] == "English"
        assert "Arabic script" in p.result["rules"]
        assert "Speak English for everything" in p.result["rules"]
        assert holder.guest == {"target_language": "ar", "level": "beginner",
                                "native_language": "en"}
        assert store.list() == [], "a guest lesson stores nothing"

        p = Params(practiced="x")
        asyncio.run(tools["save_session_notes"].handler(p))
        assert p.result["saved"] is False and "guest" in p.result["reason"]

        p = Params()
        asyncio.run(tools["enroll_new_learner"].handler(p))
        assert p.result["enrolled"] is False
        assert p.result["missing"] == ["name", "goal"]

        for field, value in (("name", "Layla"), ("goal", "travel")):
            p = Params(field=field, value=value)
            asyncio.run(tools["intake_answer"].handler(p))
            assert p.result["ok"]
        p = Params()
        asyncio.run(tools["enroll_new_learner"].handler(p))
        assert p.result["enrolled"] is True
        assert p.result["target_language"] == "ar"
        assert "carry on with the lesson" in p.result["note"]
        assert rebriefs == [], "the guest lesson goes on, no fresh session"
        (layla,) = store.list()
        assert layla.corrections == "blocking" and layla.level == "beginner"

        p = Params(language="French", level="advanced")
        asyncio.run(tools["start_lesson"].handler(p))
        assert "error" in p.result, "an enrolled learner uses set_target_language"
    finally:
        if saved is not None:
            sys.modules["face_id"] = saved
        else:
            sys.modules.pop("face_id", None)


# -- the attractor ----------------------------------------------------------------

def test_attract_moves_exist_and_stay_out_of_the_models_list():
    assert len(ATTRACT_MOVES) >= 12 and len(set(ATTRACT_MOVES)) == len(ATTRACT_MOVES)
    assert all(m in LIBRARY for m in ATTRACT_MOVES)
    model = names()
    assert "look_around" not in model and "look_around:" not in describe()
    assert "dance" in model and "peekaboo" in model
    # 1.9 s played at half speed (2026-09-25 evening) + 1.7 s a pass
    assert attract_passes("sway") == (1, pytest.approx(5.5))
    assert attract_passes("dance") == (2, pytest.approx(9.8))  # full speed
    for jerky in ("electric", "stumble", "chicken", "robot", "jackson"):
        assert jerky not in ATTRACT_MOVES, jerky
    assert attract_passes("dance_long")[0] == 1


class FakeContext:
    def set_messages(self, messages):
        pass


class Runner(SessionRunner):
    def __init__(self, **kw):
        super().__init__(task=None, languages="Spanish", base_prompt="BASE.",
                         **kw)
        self.performed, self.went_neutral = [], 0

    async def _queue_user_turn(self, text):
        pass

    async def _robot_neutral(self):
        self.went_neutral += 1

    async def _perform(self, name, seconds, repeat=1):
        self.performed.append((name, repeat))


def test_attractor_varies_and_a_visitor_cuts_it_short(tmp_path):
    runner = Runner(source="unused", store=LearnerStore(tmp_path / "l"),
                    holder=CurrentLearner(), context=FakeContext(),
                    robot=None, attract_secs=30, attract_every=60)
    t = 0.0
    while t <= 30 + 60 * 20:
        asyncio.run(runner.observe(None, t))
        t += 5.0
    played = [n for n, _ in runner.performed]
    assert len(played) == 21, played        # at 30 s, then every 60 s
    assert all(a != b for a, b in zip(played, played[1:])), "no repeats"
    assert len(set(played)) >= 8, "variety"
    assert all(r >= 1 for _, r in runner.performed)
    # a visitor arrives two seconds into the last move: it is stopped
    runner._recent_vectors = [np.eye(8, dtype=np.float32)[0]]
    asyncio.run(runner.start_session(now=runner._attract_until - 5.0))
    assert runner.went_neutral == 1


class GlanceRunner(Runner):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.glances, self.timeline = [], []

    async def _perform(self, name, seconds, repeat=1):
        self.performed.append((name, repeat))
        self.timeline.append(("move", self._now, seconds))

    async def _glance(self, cmd):
        self.glances.append(cmd)
        self.timeline.append(("glance", self._now, cmd["duration"]))


def _drive(runner, until, step=0.5, start=0.0):
    t = start
    while t <= until:
        runner._now = t
        asyncio.run(runner.observe(None, t))
        t += step


def test_idle_is_mostly_dancing_back_to_back(tmp_path):
    """The booth's idle defaults (5 s, then every 12 s): moves follow
    one another without overlapping, and glances only fill the gaps."""
    runner = GlanceRunner(source="unused", store=LearnerStore(tmp_path / "l"),
                          holder=CurrentLearner(), context=FakeContext(),
                          robot=None, attract_secs=5, attract_every=12,
                          glance_every=4)
    _drive(runner, 300)
    moves = [(t, d) for kind, t, d in runner.timeline if kind == "move"]
    assert moves[0][0] <= 6.0, "first dance about 5 s after the frame empties"
    for (t0, d0), (t1, _) in zip(moves, moves[1:]):
        assert t1 >= t0 + d0 - 1e-6, "a move started on top of another"
    busy = sum(d for _, d in moves)
    assert busy / 300 > 0.6, f"only {busy / 300:.0%} of idle time moving"
    for kind, t, _ in runner.timeline:
        if kind == "glance":
            assert not any(t0 <= t < t0 + d0 for t0, d0 in moves), \
                "a glance during a dance"


def test_glances_stop_when_a_face_appears():
    from session import Glancer
    import random
    g = Glancer(every_secs=4, rng=random.Random(1))
    assert g.on_face(False, 0.0) is None           # arms
    assert g.on_face(False, 2.0) is None
    cmd = g.on_face(False, 3.0)
    assert cmd is not None and abs(cmd["head_yaw"]) <= 0.62
    assert "body_yaw" not in cmd, "the base stays put"
    assert g.on_face(True, 3.5) is None
    assert g.on_face(False, 4.0) is None, "re-arms after a face"
    assert Glancer(0).on_face(False, 100.0) is None, "0 = off"
