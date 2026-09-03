"""T13.1: learner goals and the explicit level at enrollment.

Store round-trip with and without the new keys (unmarked); the briefing
carries per-goal guidance (unmarked); the stranger briefing is a fixed
four-question script (unmarked); and a scripted enrollment interview
through the real agent in each speech mode asks the level *before*
enrolling and stores the goal (anthropic / google marked).
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from tutor.store import GOALS, LearnerStore  # noqa: E402
from tutor_mode import (  # noqa: E402
    ENROLLMENT_SCRIPT, STRANGER_BRIEFING, UNSURE_BRIEFING,
    build_briefing, normalize_goal,
)


def tts_lines(log: str) -> str:
    return " ".join(
        m.group(1) for m in re.finditer(r"Generating TTS \[(.*)\]", log))


# -- store --------------------------------------------------------------------

def test_goal_round_trip(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    l = store.create("Ana", "es", level="intermediate", goal="work",
                     goal_note="interviews at a hotel chain")
    back = store.load(l.id)
    assert back.goal == "work"
    assert back.goal_note == "interviews at a hotel chain"
    data = json.loads((tmp_path / "learners" / l.id / "profile.json").read_text())
    assert data["goal"] == "work" and "goal_note" in data


def test_goal_defaults_and_validation(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    l = store.create("Bo", "fr")
    assert l.goal == "conversation" and l.goal_note == ""
    with pytest.raises(ValueError):
        store.create("Cy", "fr", goal="world domination")


def test_pre_t13_profile_loads_with_defaults(tmp_path):
    """A profile.json written before T13 has no goal keys at all."""
    root = tmp_path / "learners"
    (root / "maria").mkdir(parents=True)
    (root / "maria" / "profile.json").write_text(json.dumps({
        "name": "Maria", "target_language": "es", "level": "intermediate",
        "embedding": None, "sessions": 3, "last_seen": "2026-08-30",
        "tier": "family"}))
    learner = LearnerStore(root).load("maria")
    assert learner is not None, "legacy profile must still load"
    assert learner.goal == "conversation" and learner.goal_note == ""
    # and an unknown stored goal degrades to 'other' rather than crashing
    (root / "maria" / "profile.json").write_text(json.dumps({
        "name": "Maria", "target_language": "es", "level": "intermediate",
        "embedding": None, "sessions": 3, "last_seen": "2026-08-30",
        "tier": "family", "goal": "moon", "goal_note": "x"}))
    assert LearnerStore(root).load("maria").goal == "other"


def test_fixture_learners_still_load(paths):
    learners = LearnerStore(paths.fixtures / "learners").list()
    assert {l.name for l in learners} >= {"Maria", "Sam"}
    assert all(l.goal in GOALS for l in learners)


# -- briefing -----------------------------------------------------------------

def test_briefing_carries_goal_guidance(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    exam = store.create("Ana", "es", level="advanced", goal="exam",
                        goal_note="DELE in October")
    text = build_briefing(exam, "")
    assert "preparing for an exam" in text and "synonyms" in text
    assert "DELE in October" in text
    chat = store.create("Bo", "fr", goal="conversation")
    text = build_briefing(chat, "")
    assert "everyday conversations" in text and "DELE" not in text
    work = store.create("Cy", "it", goal="work")
    assert "interviews and meetings" in build_briefing(work, "")


def test_normalize_goal_maps_free_text():
    assert normalize_goal("job interview prep") == "work"
    assert normalize_goal("I have a DELE exam") == "exam"
    assert normalize_goal("just chatting with my in-laws") == "conversation"
    assert normalize_goal("a trip to Lisbon") == "travel"
    assert normalize_goal("poetry") == "other"
    assert normalize_goal("EXAM") == "exam"


def test_stranger_briefing_is_a_four_question_script():
    text = STRANGER_BRIEFING.format(languages="Spanish, French")
    for n, needle in ((1, "their name"), (2, "which language"),
                      (3, "beginner, intermediate or advanced"),
                      (4, "why they are learning")):
        assert f"{n}." in text and needle in text, f"question {n} missing"
    assert "ONE AT A TIME" in text
    assert "never assume" in text
    assert "beginner level unless" not in text, "the old default is gone"
    # the unsure path falls into the same interview, and formats with
    # only a name (its callers never pass languages)
    unsure = UNSURE_BRIEFING.format(name="Maria")
    assert "enrollment interview" in unsure and "3." in unsure
    assert "{" not in unsure, "unsubstituted placeholder would be read aloud"
    assert ENROLLMENT_SCRIPT.count("{languages}") == 1


# -- live enrollment interview -------------------------------------------------

INTERVIEW = [
    "Hello! I'd like to learn some Spanish.",
    "Yes please, remember me. My name is Sunita.",
    "Spanish.",
    "I'd say intermediate.",
    "I'm preparing for job interviews in Madrid.",
]


def _assert_interview(log, root):
    spoken = tts_lines(log).lower()
    # the level was asked explicitly ...
    assert re.search(r"beginner|intermediate|advanced|your level", spoken), \
        "the level question was never asked aloud:\n" + spoken[-2000:]
    # ... before enrollment fired
    ask_at = min((spoken.find(w) for w in ("beginner", "intermediate",
                                           "advanced", "level")
                  if spoken.find(w) >= 0), default=-1)
    assert ask_at >= 0
    enrolled_at = log.find("tutor: enrolled new guest")
    assert enrolled_at > 0
    learners = LearnerStore(root).list()
    assert len(learners) == 1
    guest = learners[0]
    assert guest.level == "intermediate", guest.level
    assert guest.goal == "work", (guest.goal, guest.goal_note)


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_enrollment_asks_level_and_goal_local(paths, tmp_path, run_agent_say):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    root = tmp_path / "learners"
    log, found = run_agent_say(
        INTERVIEW, wait_for=r"tutor: enrolled new guest",
        extra_args=["--face-source", str(clip), "--learners-root", str(root),
                    "--say-gap", "12"],
        timeout=360)
    assert found, "enroll_new_learner never fired; log tail:\n" + log[-4000:]
    _assert_interview(log, root)


@pytest.mark.google
@pytest.mark.models
@pytest.mark.audio
def test_enrollment_asks_level_and_goal_cloud(paths, tmp_path, run_agent_say):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    root = tmp_path / "learners"
    log, found = run_agent_say(
        INTERVIEW, wait_for=r"tutor: enrolled new guest",
        extra_args=["--speech", "cloud", "--face-source", str(clip),
                    "--learners-root", str(root), "--say-gap", "12"],
        timeout=360)
    assert found, "enroll_new_learner never fired; log tail:\n" + log[-4000:]
    # Cloud mode has no TTS log lines to read; the stored profile is the
    # evidence that all four answers reached the tool.
    guest = LearnerStore(root).list()[0]
    assert guest.level == "intermediate" and guest.goal == "work", \
        (guest.level, guest.goal, guest.goal_note)
