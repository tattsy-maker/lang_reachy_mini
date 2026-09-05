"""T16 -- any native language: the learner is taught *in* their own
language and may be learning English. No models, no camera, no keys."""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (REPO, os.path.join(REPO, "voice")):
    if path not in sys.path:
        sys.path.insert(0, path)

from tutor.store import LearnerStore, OPTIONAL_KEYS, PROFILE_KEYS   # noqa: E402
from tutor_mode import (                                            # noqa: E402
    BOOTH_PERSONA, ENROLLMENT_SCRIPT, STRANGER_BRIEFING, build_briefing,
    build_unsure_briefing, native_language_of, normalize_language,
    voice_cue,
)


# -- the store ----------------------------------------------------------------

def test_native_language_is_stored_and_defaults_to_english(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    igor = store.create("Igor", "en", level="beginner", native_language="RU")
    assert store.load(igor.id).native_language == "ru", "lower-cased"
    maria = store.create("Maria", "es")
    assert store.load(maria.id).native_language == "en"
    assert "native_language" in PROFILE_KEYS
    assert OPTIONAL_KEYS["native_language"] == "en"
    raw = json.loads((store.root / igor.id / "profile.json").read_text())
    assert raw["native_language"] == "ru" and raw["target_language"] == "en"


def test_profile_without_native_language_still_loads(tmp_path):
    """Every profile written before T16 (and both fixtures) lacks the key."""
    root = tmp_path / "learners"
    (root / "sam").mkdir(parents=True)
    (root / "sam" / "profile.json").write_text(json.dumps({
        "name": "Sam", "target_language": "fr", "level": "beginner",
        "embedding": None, "sessions": 1, "last_seen": "2026-08-30",
        "tier": "guest"}))
    learner = LearnerStore(root).load("sam")
    assert learner is not None and learner.native_language == "en"
    assert native_language_of(learner) == "en"
    # and an empty/None value degrades to English rather than to ""
    (root / "sam" / "profile.json").write_text(json.dumps({
        "name": "Sam", "target_language": "fr", "level": "beginner",
        "embedding": None, "sessions": 1, "last_seen": "2026-08-30",
        "tier": "guest", "native_language": None}))
    assert LearnerStore(root).load("sam").native_language == "en"


# -- the briefing -------------------------------------------------------------

def test_briefing_explains_in_the_native_language(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    igor = store.create("Igor", "en", level="beginner", native_language="ru")
    text = build_briefing(igor, "")
    assert "beginner English" in text
    assert "Their own language is Russian" in text
    assert "brief Russian explanation" in text
    assert "if Igor asks something in Russian, answer it" in text
    assert "say what to express in Russian and have Igor say it in English" in text
    assert "English words embedded in a Russian sentence" in text
    assert "set_native_language" in text
    assert "{" not in text and "}" not in text, "unsubstituted placeholder"
    # the intermediate and advanced guidance name the native language too
    for level, needle in (("intermediate", "Explain in Russian only when"),
                          ("advanced", "Use Russian only as a last resort")):
        learner = store.create(f"I{level}", "en", level=level,
                               native_language="ru")
        assert needle in build_briefing(learner, ""), level


def test_english_native_briefing_is_unchanged(tmp_path):
    """The default keeps every pre-T16 lesson exactly as it was."""
    store = LearnerStore(tmp_path / "learners")
    maria = store.create("Maria", "es", level="beginner")
    text = build_briefing(maria, "")
    assert "brief English explanation" in text
    assert "say what to express in English and have Maria say it in Spanish" in text
    assert "Their own language is English" in text


def test_briefing_survives_native_equal_to_target(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    odd = store.create("Odd", "ru", level="beginner", native_language="ru")
    text = build_briefing(odd, "")
    assert "{" not in text and "beginner Russian" in text


# -- the other prompts and cues ------------------------------------------------

def test_stranger_and_enrollment_prompts_allow_english_as_a_target():
    text = STRANGER_BRIEFING.format(languages="Spanish, French")
    assert "English is a fine answer" in text
    assert "which language you should explain things in" in text
    assert "native_language" in ENROLLMENT_SCRIPT
    assert "carry on in that language" in text, "a Russian hello is answered in Russian"
    assert "otherwise in the student's own language" in BOOTH_PERSONA


def test_unsure_and_voice_cues_speak_the_native_language(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    igor = store.create("Igor", "en", native_language="ru")
    assert "Open in Russian by asking" in build_unsure_briefing(igor)
    assert "in Russian" in voice_cue("challenge", igor, store)
    assert "in Russian" in voice_cue("downgrade", igor, store)
    confirmed = voice_cue("confirmed", igor, store)
    assert "in English" in confirmed and "explained in Russian" in confirmed
    maria = store.create("Maria", "es")
    assert "Open in English by asking" in build_unsure_briefing(maria)


def test_normalize_language_accepts_the_native_forms():
    assert normalize_language("Russian") == "ru"
    assert normalize_language("ru") == "ru"
    assert normalize_language("English") == "en"
    assert normalize_language("Klingon") is None


# -- Whisper priming works on the pair, not on "English + X" ------------------

def test_priming_is_symmetric_in_the_pair():
    pytest.importorskip("pipecat", reason="multilingual imports pipecat")
    from multilingual import PRIMING, bilingual_priming
    assert bilingual_priming("ru") == PRIMING["ru"]              # en speaker, ru lessons
    assert bilingual_priming("en", "ru") == PRIMING["ru"]        # ru speaker, en lessons
    assert bilingual_priming("es", "ru") is None, "unmeasured pair: no priming"
    assert bilingual_priming("fr", "en") is None, "priming hurts fr (T7)"
    assert bilingual_priming("en", "en") is None


# -- the tools ---------------------------------------------------------------

def test_set_native_language_tool_and_enrollment_field(tmp_path):
    pytest.importorskip("pipecat", reason="FunctionSchema comes from pipecat")
    import asyncio
    import types
    from tutor_mode import (CurrentLearner, build_enrollment_tools,
                            build_tutor_tools)

    class Params:
        def __init__(self, **arguments):
            self.arguments = arguments
            self.result = None

        async def result_callback(self, result):
            self.result = result

    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    tools = {t.name: t for t in build_tutor_tools(store, holder)}
    assert "set_native_language" in tools

    # enrollment with a native language and no camera: the handler
    # imports face_id lazily, so a stub module stands in for the capture
    fake = types.ModuleType("face_id")
    fake.capture_embedding = lambda source, frames=None, **kw: [0.1, 0.2, 0.3]
    saved = sys.modules.get("face_id")
    sys.modules["face_id"] = fake
    try:
        enroll = {t.name: t for t in build_enrollment_tools(
            store, holder, face_source=None)}["enroll_new_learner"]
        p = Params(name="Igor", target_language="English", level="beginner",
                   goal="work", native_language="Russian")
        asyncio.run(enroll.handler(p))
    finally:
        if saved is None:
            del sys.modules["face_id"]
        else:
            sys.modules["face_id"] = saved
    assert p.result["enrolled"] and p.result["native_language"] == "ru"
    assert "explaining in Russian" in p.result["note"]
    assert store.load(p.result["id"]).native_language == "ru"

    p = Params(language="Ukrainian")
    asyncio.run(tools["set_native_language"].handler(p))
    assert p.result["native_language"] == "Ukrainian" and p.result["was"] == "ru"
    assert store.load(holder.learner.id).native_language == "uk"
    assert holder.learner.native_language == "uk"
    assert "keep practising English" in p.result["note"]

    p = Params(language="Klingon")
    asyncio.run(tools["set_native_language"].handler(p))
    assert "error" in p.result
