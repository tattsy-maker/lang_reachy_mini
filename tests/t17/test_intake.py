"""T17.4: the intake as a first meeting, enforced in code. The Intake
validates one answer at a time and knows what to ask next; the
enrollment tool refuses until it is complete; the agreed minutes time a
wrap-up cue; a returning learner gets the two session questions."""

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

from intake import (                                       # noqa: E402
    INTAKE_FIELDS, Intake, SessionPlan, normalize_corrections, parse_minutes,
    time_cue,
)
from session import ACTIVE, WALKUP_CUE, WALKUP_CUE_KNOWN, SessionRunner  # noqa: E402
from tutor.store import LearnerStore                       # noqa: E402
from tutor_mode import (                                   # noqa: E402
    ENROLLMENT_SCRIPT, STRANGER_BRIEFING, CurrentLearner, build_briefing,
)


def full_interview(intake: Intake):
    for f, v in (("name", "Солнышко"), ("target_language", "English"),
                 ("native_language", "Russian"), ("explain_in", "both"),
                 ("level", "beginner"), ("goal", "conversation"),
                 ("minutes", "15 minutes"), ("today", "small talk"),
                 ("corrections", "every mistake")):
        out = intake.answer(f, v)
        assert out["ok"], out
    return intake


# -- the Intake ----------------------------------------------------------------------

def test_answers_are_validated_and_normalized():
    i = Intake()
    assert i.answer("name", "  Igor  ")["value"] == "Igor"
    assert i.answer("target_language", "English")["value"] == "en"
    assert not i.answer("target_language", "Klingon")["ok"]
    assert i.answer("native_language", "Russian")["value"] == "ru"
    assert i.answer("explain_in", "both")["value"] == "both"
    assert not i.answer("level", "fluent")["ok"]
    assert i.answer("level", "Beginner")["value"] == "beginner"
    assert i.answer("goal", "for my job interviews", note="I have interviews")["value"] == "work"
    assert i.goal_note == "I have interviews"
    assert not i.answer("minutes", "a while")["ok"]
    assert i.answer("minutes", "about 20 min")["value"] == 20
    assert i.answer("today", "breakfast words")["value"] == "breakfast words"
    assert i.answer("corrections", "only what matters")["value"] == "blocking"
    assert not i.answer("shoe_size", "42")["ok"]
    assert i.complete and i.next_question() is None
    kw = i.profile_kwargs()
    assert kw == {"level": "beginner", "goal": "work",
                  "goal_note": "I have interviews", "native_language": "ru",
                  "explain_in": "both", "corrections": "blocking"}


def test_explain_in_accepts_a_language_name():
    i = Intake()
    i.answer("target_language", "en")
    assert i.answer("explain_in", "English")["value"] == "target"
    j = Intake()
    j.answer("target_language", "en")
    assert j.answer("explain_in", "Russian")["value"] == "native"
    assert j.answers["native_language"] == "ru", "naming the language names their own"


def test_missing_follows_the_interview_order_and_names_the_next_question():
    i = Intake()
    assert i.missing() == list(INTAKE_FIELDS)
    assert i.next_question() == "ask their name"
    i.answer("name", "Igor")
    assert "which language" in i.next_question()
    i.answer("target_language", "English")
    assert "own language" in i.next_question() and "English" in i.next_question()
    for f, v in (("native_language", "ru"), ("explain_in", "native")):
        i.answer(f, v)
    assert "beginner, intermediate or advanced" in i.next_question()
    status = i.status()
    assert status["missing"][0] == "level" and "No lesson yet" in status["note"]
    full_interview(i)
    assert i.status() == {"missing": [], "note": "every question is answered: call enroll_new_learner"}
    j = Intake()
    j.answer("name", "A")
    assert not j.complete and j.missing_required()[0] == "target_language"
    i.reset()
    assert not i.answers and i.goal_note == ""


def test_helpers():
    assert parse_minutes(15) == 15 and parse_minutes("10 minutes") == 10
    assert parse_minutes("twenty") == 20 and parse_minutes("десять минут") == 10
    assert parse_minutes("0") is None and parse_minutes("999") is None
    assert parse_minutes("no idea") is None
    assert normalize_corrections("every") == "every"
    assert normalize_corrections("at the end please") == "end"
    assert normalize_corrections("whatever") is None


# -- the plan and the time cue ------------------------------------------------------

def test_session_plan_warns_once_at_the_fraction():
    plan = SessionPlan(minutes=10, focus="ordering food", started_at=100.0)
    assert plan.remaining_minutes(100.0) == 10.0
    assert not plan.due_warning(100.0 + 7 * 60)
    assert plan.due_warning(100.0 + 8 * 60), "80% of ten minutes"
    assert not plan.due_warning(100.0 + 9 * 60), "only once"
    cue = time_cue(plan, 100.0 + 8 * 60)
    assert "2 minutes of the 10 agreed remain" in cue and "stop here or carry on" in cue
    assert "10 minutes" in plan.spoken_plan_note() and "ordering food" in plan.spoken_plan_note()
    assert "1 minute of" in time_cue(SessionPlan(5, "x", 0.0), 4 * 60 + 10)


class FakeContext:
    def __init__(self):
        self.history = []

    def set_messages(self, messages):
        self.history.append(messages)


class Runner(SessionRunner):
    def __init__(self, **kw):
        super().__init__(task=None, languages="French", base_prompt="BASE.",
                         **kw)
        self.cues = []

    async def _queue_user_turn(self, text):
        self.cues.append(text)

    async def _robot_neutral(self):
        pass


class Face:
    def __init__(self, embedding=None, bbox=(100, 100, 200, 200)):
        self.embedding = embedding
        self.bbox = bbox


def test_runner_injects_the_time_cue_and_greets_a_known_learner_with_the_questions(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    runner = Runner(source="unused", store=store, holder=holder,
                    context=FakeContext(), robot=None, save_wait_secs=0.1)
    ana = np.zeros(8, dtype=np.float32)
    ana[0] = 1.0
    store.create("Ana", "fr", embedding=[float(x) for x in ana])
    runner._recent_vectors = [ana]
    asyncio.run(runner.start_session(now=10.0))
    assert runner.machine.state == ACTIVE and holder.learner is not None
    assert runner.cues[-1] == WALKUP_CUE_KNOWN
    assert "set_session_plan" in WALKUP_CUE_KNOWN and "set_session_plan" not in WALKUP_CUE
    holder.plan = SessionPlan(minutes=5, focus="greetings", started_at=10.0)
    asyncio.run(runner.observe(Face(embedding=ana), 10.0 + 3 * 60, (480, 640)))
    assert not any("agreed remain" in c for c in runner.cues)
    asyncio.run(runner.observe(Face(embedding=ana), 10.0 + 4 * 60 + 1, (480, 640)))
    assert any("agreed remain" in c for c in runner.cues), "the wrap-up cue"
    n = len(runner.cues)
    asyncio.run(runner.observe(Face(embedding=ana), 10.0 + 4 * 60 + 30, (480, 640)))
    assert len(runner.cues) == n, "once"
    asyncio.run(runner.end_session())
    assert holder.plan is None, "the plan does not outlive the session"


def test_a_stranger_gets_the_plain_walkup_cue(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    runner = Runner(source="unused", store=store, holder=holder,
                    context=FakeContext(), robot=None)
    v = np.zeros(8, dtype=np.float32)
    v[3] = 1.0
    runner._recent_vectors = [v]
    asyncio.run(runner.start_session(now=1.0))
    assert runner.cues[-1] == WALKUP_CUE


# -- the prompts ---------------------------------------------------------------------

def test_enrollment_script_is_the_interview():
    text = STRANGER_BRIEFING.format(languages="Spanish, French")
    for n, needle in ((1, "their name"), (2, "which language"),
                      (3, "own language"), (4, "beginner, intermediate or advanced"),
                      (5, "why they are learning"), (6, "how many minutes"),
                      (7, "how they want to be corrected")):
        assert f"{n}." in text and needle in text, f"question {n} missing"
    assert "intake_answer" in ENROLLMENT_SCRIPT
    assert "Never invent an answer" in ENROLLMENT_SCRIPT
    assert "enroll_new_learner (no arguments)" in ENROLLMENT_SCRIPT
    assert "plan of two or three steps" in ENROLLMENT_SCRIPT


def test_briefing_carries_the_plan_and_the_correction_tool_rule(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    ana = store.create("Ana", "fr")
    plan = SessionPlan(minutes=12, focus="the metro", started_at=0.0)
    text = build_briefing(ana, "", plan=plan)
    assert "They have 12 minutes and want: the metro" in text
    assert "call the matching tool in the same turn" in text
    assert "They have" not in build_briefing(ana, "")


# -- the tools ---------------------------------------------------------------------

class Params:
    def __init__(self, **arguments):
        self.arguments = arguments
        self.result = None

    async def result_callback(self, result):
        self.result = result


def test_enroll_refuses_until_the_interview_is_complete(tmp_path):
    pytest.importorskip("pipecat", reason="FunctionSchema comes from pipecat")
    from tutor_mode import build_enrollment_tools, build_tutor_tools
    fake = types.ModuleType("face_id")
    fake.capture_embedding = lambda source, frames=None, **kw: [0.1, 0.2, 0.3]
    saved = sys.modules.get("face_id")
    sys.modules["face_id"] = fake
    try:
        store = LearnerStore(tmp_path / "learners")
        holder = CurrentLearner()
        intake = Intake()
        tools = {t.name: t for t in build_enrollment_tools(
            store, holder, face_source=None, intake=intake)}
        assert holder.intake is intake
        # the 2026-09-05 one-sentence enrollment, refused
        p = Params(name="Солнышко", target_language="English", level="beginner",
                   goal="conversation")
        asyncio.run(tools["enroll_new_learner"].handler(p))
        assert p.result["enrolled"] is False and "not finished" in p.result["error"]
        assert p.result["missing"] == list(INTAKE_FIELDS)
        assert p.result["next"] == "ask their name"
        assert store.list() == []
        # one answer at a time
        p = Params(field="name", value="Солнышко")
        asyncio.run(tools["intake_answer"].handler(p))
        assert p.result["ok"] and p.result["missing"][0] == "target_language"
        p = Params(field="level", value="fluent")
        asyncio.run(tools["intake_answer"].handler(p))
        assert not p.result["ok"] and "beginner" in p.result["error"]
        # everything but explain_in: enrollment goes ahead, explanations
        # default to their own language (the live 2026-09-05 case)
        probe = Intake()
        for f, v in (("name", "Ana"), ("target_language", "es"),
                     ("native_language", "en"), ("level", "beginner"),
                     ("goal", "travel"), ("minutes", 5), ("today", "x"),
                     ("corrections", "end")):
            probe.answer(f, v)
        assert probe.complete and probe.missing() == ["explain_in"]
        assert probe.status()["next"].startswith("ask whether you should explain")
        assert probe.profile_kwargs()["explain_in"] == "native"
        assert not Intake().complete
        full_interview(intake)
        p = Params()
        asyncio.run(tools["enroll_new_learner"].handler(p))
        assert p.result["enrolled"] is True
        assert p.result["explain_in"] == "both" and p.result["corrections"] == "every"
        assert p.result["minutes"] == 15 and "small talk" in p.result["note"]
        learner = store.load(p.result["id"])
        assert learner.native_language == "ru" and learner.target_language == "en"
        assert learner.explain_in == "both" and learner.corrections == "every"
        assert holder.plan is not None and holder.plan.minutes == 15
        # answering again after enrollment is refused
        p = Params(field="level", value="advanced")
        asyncio.run(tools["intake_answer"].handler(p))
        assert "already enrolled" in p.result["error"]
        # a returning learner's plan
        tutor = {t.name: t for t in build_tutor_tools(store, holder)}
        p = Params(minutes="10", focus="the past tense")
        asyncio.run(tutor["set_session_plan"].handler(p))
        assert p.result["minutes"] == 10 and holder.plan.focus == "the past tense"
        p = Params(minutes="lots", focus="x")
        asyncio.run(tutor["set_session_plan"].handler(p))
        assert "error" in p.result
    finally:
        if saved is None:
            del sys.modules["face_id"]
        else:
            sys.modules["face_id"] = saved


def test_enroll_without_an_intake_keeps_the_old_argument_form(tmp_path):
    pytest.importorskip("pipecat", reason="FunctionSchema comes from pipecat")
    from tutor_mode import build_enrollment_tools
    fake = types.ModuleType("face_id")
    fake.capture_embedding = lambda source, frames=None, **kw: [0.1, 0.2, 0.3]
    saved = sys.modules.get("face_id")
    sys.modules["face_id"] = fake
    try:
        store = LearnerStore(tmp_path / "learners")
        holder = CurrentLearner()
        enroll = {t.name: t for t in build_enrollment_tools(
            store, holder, face_source=None)}["enroll_new_learner"]
        p = Params(name="Igor", target_language="English", level="beginner",
                   goal="work", native_language="Russian")
        asyncio.run(enroll.handler(p))
        assert p.result["enrolled"] and store.load(p.result["id"]).explain_in == "native"
    finally:
        if saved is None:
            del sys.modules["face_id"]
        else:
            sys.modules["face_id"] = saved


def test_a_bare_yes_is_not_a_name_or_a_focus():
    """2026-09-23: the answer to "Would you like me to remember you?" was
    stored as today's focus, and the plan told the model they want "yes"."""
    from intake import Intake
    intake = Intake()
    for field in ("today", "name"):
        out = intake.answer(field, "Yes.")
        assert not out["ok"] and "yes/no" in out["error"]
        assert field not in intake.answers
    assert intake.answer("today", "greetings for a trip")["ok"]
    assert intake.answer("name", "Luca")["ok"]
