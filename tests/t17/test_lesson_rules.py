"""T17.5 (one language policy), T17.6 (spoken exercises), T17.7
(corrections and the next rung), T17.8 (fooling around) and T17.9 (the
closing, recorded by code): the prompt rules, the quiz script, the
sightseeing budget, the feedback catcher and the tools."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voice"))

from budget import SightseeingBudget                       # noqa: E402
from tutor.store import LearnerStore                       # noqa: E402
from tutor.wishes import count_wishes, read_feedback, record_feedback  # noqa: E402
from tutor_mode import (                                   # noqa: E402
    BOOTH_PERSONA, WISH_QUESTION, CurrentLearner, build_briefing,
    catch_feedback, quiz_script, wish_followup,
)


# -- T17.5: one language policy ------------------------------------------------------

def test_explain_policy_follows_the_profile(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    mother = store.create("Солнышко", "en", native_language="ru", explain_in="both")
    text = build_briefing(mother, "")
    assert "They asked for both languages: explain in English first, then repeat the key point in Russian" in text
    native = store.create("Igor", "en", native_language="ru")
    assert "Every explanation, instruction and aside is in Russian" in build_briefing(native, "")
    target = store.create("Lena", "en", native_language="ru", explain_in="target")
    assert "They asked to be taught in English" in build_briefing(target, "")
    same = store.create("Odd", "ru", native_language="ru")
    assert "They asked to be taught in Russian" in build_briefing(same, ""), "native == target"


def test_drift_and_no_answer_in_the_question_rules(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    mother = store.create("Солнышко", "en", native_language="ru")
    text = build_briefing(mother, "")
    assert "If Солнышко drifts into Russian mid-lesson, answer that once in Russian, then set the next task in English" in text
    assert "a tutor who follows the student out of the lesson language is not tutoring" in text
    assert "Never ask Солнышко to say in English something you have just said in English" in text
    assert "never ask a question whose answer is in the question" in text


def test_script_rule_names_the_script(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    ru = store.create("John", "ru")
    text = build_briefing(ru, "")
    assert "Write every Russian word in its own script (Cyrillic: спасибо, never spasibo), never in Latin transliteration" in text
    zh = store.create("Li", "zh")
    assert "characters: 谢谢" in build_briefing(zh, "")
    es = store.create("Ana", "es")
    assert "Write every Spanish word in its own script, never in Latin transliteration" in build_briefing(es, "")


# -- T17.6: spoken exercises -----------------------------------------------------------

def test_quiz_script_speaks_the_gap_and_letters_the_options():
    item = quiz_script("Все даты поставок указаны ___ в договоре.",
                       ["заранее", "полностью", "вчера"], "полностью", "ru")
    say = item["say"]
    assert say.startswith("Вставьте слово: Все даты поставок указаны пропуск в договоре.")
    assert "Варианты: a) заранее, b) полностью, c) вчера." in say
    assert say.endswith("Какой подходит?") and "___" not in say
    assert item["answer"] == "полностью" and item["has_gap"]
    en = quiz_script("This film left a (blank) impression.", ["deep", "loudly"], "deep")
    assert "left a blank impression" in en["say"] and "The options are a) deep, b) loudly." in en["say"]
    assert "Which one fits?" in en["say"]
    choice = quiz_script("Which word is an adjective?", ["quickly", "green", "run"], "green", "en")
    assert not choice["has_gap"] and choice["say"].startswith("Which word is an adjective?")
    dots = quiz_script("Мы готовы ... ваше предложение", ["обсудить", "вчера"], None, "ru")
    assert "готовы пропуск ваше" in dots["say"] and dots["answer"] is None
    assert quiz_script("I ___ tea", [], None, "xx")["say"].startswith("Fill the gap: I blank tea"), "unknown language: English words"


def test_briefing_allows_only_formats_that_work_by_ear(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    text = build_briefing(store.create("Ana", "fr"), "")
    assert "Exercises are heard, not read" in text
    assert "call quiz and read its script word for word" in text
    assert "do you know what an adjective is?" in text


# -- T17.7: corrections -----------------------------------------------------------------

def test_corrections_rule_follows_the_preference(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    every = store.create("A", "en", native_language="ru")
    text = build_briefing(every, "")
    assert "Correct every mistake as it happens" in text
    assert "say the corrected sentence once" in text
    assert "offer one richer or more natural way to say it" in text and "the next rung" in text
    blocking = store.create("B", "en", corrections="blocking")
    assert "Correct only what gets in the way of being understood" in build_briefing(blocking, "")
    end = store.create("C", "en", corrections="end")
    assert "Do not interrupt to correct" in build_briefing(end, "")
    assert "Speak slowly and clearly" in build_briefing(every, ""), "beginner pace (T17.10)"


# -- T17.8: fooling around ---------------------------------------------------------------

def test_sightseeing_budget_counts_refuses_and_resets():
    lesson = {"on": True}
    b = SightseeingBudget(limit=2, window_secs=120.0, in_lesson=lambda: lesson["on"],
                          clock=lambda: 0.0)
    assert b.allow("look", now=10.0) is None
    assert b.allow("move_head", now=20.0) is None
    over = b.allow("turn_body", now=30.0)
    assert over["skipped"] and "back to the lesson" in over["reason"] or "lesson" in over["reason"]
    assert b.refused == 1 and b.used(now=30.0) == 2
    assert b.allow("nod", now=31.0) is None, "only the sightseeing tools count"
    assert b.allow("look", now=131.0) is None, "the window slid"
    lesson["on"] = False
    assert b.allow("look", now=132.0) is None and b.allow("look", now=133.0) is None, \
        "no budget outside a lesson"
    lesson["on"] = True
    b.reset()
    assert b.used(now=134.0) == 0 and b.allow("look", now=134.0) is None


def test_prompts_steer_back(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    text = build_briefing(store.create("Ana", "fr"), "")
    assert "A joke, a dare or an off-topic request gets one playful answer" in text
    assert "the third in a row gets a light no" in text
    assert "makes no sense in context, say so and ask again" in text
    assert "never about your own camera or hearing" in text
    assert "I am a tutor, not a periscope" in BOOTH_PERSONA


# -- T17.9: the closing ---------------------------------------------------------------------

def test_closing_question_asks_for_feedback_or_ideas():
    assert "what should I do better" in WISH_QUESTION
    assert "robot you had bought" in WISH_QUESTION
    assert WISH_QUESTION in BOOTH_PERSONA
    assert "kind improve" in BOOTH_PERSONA
    assert 'Never say the "Go and practice" line before they have answered' in BOOTH_PERSONA


def test_wish_followup_opens_the_window_and_the_catcher_records(tmp_path):
    holder = CurrentLearner()
    note = wish_followup(holder, ask_wish=True, farewell=True)
    assert note and WISH_QUESTION in note and "kind improve or wish" in note
    assert holder.awaiting_feedback_since is not None
    path = tmp_path / "feedback.md"
    since = holder.awaiting_feedback_since
    assert not catch_feedback(holder, "yes", since + 2, path=path), "too short to be an answer"
    assert catch_feedback(holder, "pause and give me more time to think", since + 5, path=path)
    text = read_feedback(path)
    assert "[answer]: pause and give me more time to think" in text
    assert holder.wish_recorded and holder.awaiting_feedback_since is None
    assert not catch_feedback(holder, "and another thing about the pace", since + 6, path=path), "once"
    assert text.count("- ") == 1


def test_catcher_is_quiet_before_the_question_and_after_the_window(tmp_path):
    holder = CurrentLearner()
    path = tmp_path / "feedback.md"
    assert not catch_feedback(holder, "I would like more tests", 10.0, path=path)
    holder.awaiting_feedback_since = 10.0
    assert not catch_feedback(holder, "I would like more tests", 10.0 + 200, path=path)
    assert holder.awaiting_feedback_since is None, "the window closed"
    assert not path.exists()
    holder.reset()
    assert holder.awaiting_feedback_since is None


def test_record_feedback_file_format(tmp_path):
    path = tmp_path / "feedback.md"
    record_feedback("tests with options", kind="improve", name="John", path=path,
                    date="2026-09-05 19:20")
    record_feedback("teach me while I cook", kind="wish", path=path,
                    date="2026-09-05 19:41")
    record_feedback("  caught   words ", kind="nonsense", path=path,
                    date="2026-09-05 19:42")
    text = read_feedback(path)
    assert text.startswith("# Visitor feedback")
    assert "- 2026-09-05 19:20 (John) [improve]: tests with options\n" in text
    assert "- 2026-09-05 19:41 [wish]: teach me while I cook\n" in text
    assert "[answer]: caught words" in text, "an unknown kind is just an answer"
    with pytest.raises(ValueError):
        record_feedback("  ", path=path)


class Params:
    def __init__(self, **arguments):
        self.arguments = arguments
        self.result = None

    async def result_callback(self, result):
        self.result = result


def test_tools_quiz_and_record_wish_with_a_kind(tmp_path):
    pytest.importorskip("pipecat", reason="FunctionSchema comes from pipecat")
    from tutor_mode import build_tutor_tools
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    holder.learner = store.create("John", "ru", native_language="en")
    wishes = tmp_path / "booth" / "wishes.md"
    tools = {t.name: t for t in build_tutor_tools(store, holder, wishes_path=wishes,
                                                  ask_wish=True)}
    p = Params(sentence="Мы готовы ___ ваше предложение", options=["обсудить", "вчера"],
               answer="обсудить")
    asyncio.run(tools["quiz"].handler(p))
    assert p.result["say"].startswith("Fill the gap: Мы готовы blank ваше предложение")
    assert "word for word" in p.result["note"]
    p = Params(sentence="no gap here", options=[])
    asyncio.run(tools["quiz"].handler(p))
    assert "error" in p.result
    holder.awaiting_feedback_since = 1.0
    p = Params(wish="tests with options and a marked gap", kind="improve")
    asyncio.run(tools["record_wish"].handler(p))
    assert p.result["recorded"] and p.result["kind"] == "improve"
    feedback = tmp_path / "booth" / "feedback.md"
    assert "[improve]: tests with options and a marked gap" in read_feedback(feedback)
    assert count_wishes(wishes) == 0, "feedback on the robot is not a product wish"
    assert holder.wish_recorded and holder.awaiting_feedback_since is None
    holder.wish_recorded = False
    p = Params(wish="teach me while I cook")
    asyncio.run(tools["record_wish"].handler(p))
    assert p.result["kind"] == "wish" and count_wishes(wishes) == 1
    assert "[wish]: teach me while I cook" in read_feedback(feedback)
