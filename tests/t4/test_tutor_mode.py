"""T4: tutor mode — briefing content (unmarked, fast) and live `--say`
runs through the real agent (anthropic-marked, stub-free: --no-robot).

Live assertions are deliberately cheap (spec: language of output, section
headers in notes — no LLM judging). The reply text is read from pipecat's
`Generating TTS [...]` log lines.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from tutor_mode import build_briefing, load_learner  # noqa: E402
from tutor.store import LearnerStore  # noqa: E402

NEXT_TIME_HOOK = "open by asking Maria about her weekend trip to Barcelona"

SEED_NOTE = f"""- **Practiced:** preterite of ir; travel vocabulary
- **Struggled with:** fuimos vs eramos; corrected to fuimos for completed events
- **Wins:** rolled r in carretera
- **Next time:** {NEXT_TIME_HOOK}"""

# Cheap language evidence: common function words, counted with word
# boundaries so "la" doesn't match inside "later".
SPANISH_TOKENS = ("hola", "el", "la", "los", "que", "muy", "bien", "tu",
                  "de", "en", "es", "para", "gracias", "cómo", "qué", "y")
ENGLISH_TOKENS = ("the", "you", "is", "are", "hello", "it", "to", "that",
                  "means", "say", "in", "and", "for", "we")


def tts_lines(log: str) -> str:
    return " ".join(
        m.group(1) for m in re.finditer(r"Generating TTS \[(.*)\]", log))


def count_tokens(text: str, tokens) -> int:
    text = text.lower()
    return sum(len(re.findall(rf"\b{t}\b", text)) for t in tokens)


def seed_store(root, **overrides) -> LearnerStore:
    store = LearnerStore(root)
    fields = dict(name="Maria", target_language="es", level="intermediate",
                  tier="family")
    fields.update(overrides)
    name = fields.pop("name")
    language = fields.pop("target_language")
    learner = store.create(name, language, **fields)
    store.append_session(learner.id, SEED_NOTE, date="2026-08-30")
    return store


# -- briefing content (no model, no audio) ----------------------------------

def test_briefing_carries_profile_and_notes(tmp_path):
    store = seed_store(tmp_path / "learners")
    learner, notes, _ = load_learner(str(store.root), "maria")
    briefing = build_briefing(learner, notes)
    assert "Maria" in briefing and "Spanish" in briefing
    assert "intermediate" in briefing
    assert "session number 2" in briefing
    assert NEXT_TIME_HOOK in briefing, "recent notes must be inlined"
    assert "save_session_notes" in briefing


def test_briefing_first_session_has_no_notes(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    learner = store.create("Sam", "fr", level="beginner")
    briefing = build_briefing(learner, "")
    assert "first session" in briefing and "French" in briefing
    assert "beginner" in briefing


def test_load_learner_by_display_name_and_ambiguity(tmp_path):
    root = tmp_path / "learners"
    store = LearnerStore(root)
    store.create("Maria", "es")
    learner, _, _ = load_learner(str(root), "MARIA")
    assert learner.id == "maria"
    store.create("Maria", "fr")
    with pytest.raises(SystemExit, match="ambiguous"):
        load_learner(str(root), "Maria")
    with pytest.raises(SystemExit, match="no learner"):
        load_learner(str(root), "nobody")


# -- live runs through the real agent ----------------------------------------

@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_mini_lesson_speaks_spanish_and_picks_up_the_notes(
        tmp_path, run_agent_say):
    store = seed_store(tmp_path / "learners")
    log, found = run_agent_say(
        ["Hola, estoy lista para practicar.",
         "Sí. Ayer yo fuimos a un restaurante.",   # deliberate error
         "Ah sí, yo fui. Gracias."],
        # wait until the third turn is injected, then let its reply finish
        wait_for=r"injecting utterance 3/3",
        settle=20,
        extra_args=["--learner", "maria", "--learners-root", str(store.root),
                    "--say-gap", "10"],
        timeout=300)
    assert found, "third turn never ran; log tail:\n" + log[-4000:]
    spoken = tts_lines(log)
    assert "Generating TTS" in log, "no reply reached synthesis"
    assert count_tokens(spoken, SPANISH_TOKENS) > count_tokens(
        spoken, ENGLISH_TOKENS), f"replies not mostly Spanish: {spoken!r}"
    assert "barcelona" in spoken.lower(), (
        "opening did not pick up the seeded 'Next time' note: "
        f"{spoken!r}")
    assert re.search(r"\bfui\b", spoken.lower()), (
        f"the 'yo fuimos' error was never corrected to 'fui': {spoken!r}")


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_goodbye_saves_well_formed_notes(tmp_path, run_agent_say):
    store = seed_store(tmp_path / "learners")
    log, found = run_agent_say(
        ["Hola. Perdona, me tengo que ir. ¡Adiós, hasta la próxima!"],
        wait_for=r"tutor: saved session notes",
        extra_args=["--learner", "maria", "--learners-root", str(store.root)],
        timeout=300)
    assert found, ("save_session_notes never fired on goodbye; log tail:\n"
                   + log[-4000:])
    learner = store.load("maria")
    assert learner.sessions == 2, "session bookkeeping did not advance"
    notes = store.read_notes("maria", max_sessions=1)
    assert "Session 2" in notes
    for section in ("Practiced", "Struggled with", "Wins", "Next time"):
        assert f"**{section}:**" in notes, f"notes entry missing {section}"
    # the seeded entry must still be intact underneath
    assert NEXT_TIME_HOOK in store.read_notes("maria")


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_clear_evidence_updates_stored_level(tmp_path, run_agent_say):
    store = seed_store(tmp_path / "learners", level="beginner")
    log, found = run_agent_say(
        ["Para ser honesta, llevo diez años hablando español a diario y "
         "trabajo como traductora profesional. Este nivel me queda muy "
         "corto. Por favor actualiza mi nivel a avanzado."],
        wait_for=r"tutor: level for maria changed",
        extra_args=["--learner", "maria", "--learners-root", str(store.root)],
        timeout=300)
    assert found, ("update_learner_level never fired; log tail:\n"
                   + log[-4000:])
    assert store.load("maria").level == "advanced"


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_beginner_gets_english_scaffolding(tmp_path, run_agent_say):
    root = tmp_path / "learners"
    LearnerStore(root).create("Sam", "es", level="beginner")
    log, found = run_agent_say(
        ["Hi! How do I say hello in Spanish?"],
        extra_args=["--learner", "sam", "--learners-root", str(root)],
        timeout=300)
    assert found, "no reply reached synthesis; log tail:\n" + log[-4000:]
    spoken = tts_lines(log)
    assert count_tokens(spoken, ENGLISH_TOKENS) >= 2, (
        f"beginner reply lacks English scaffolding: {spoken!r}")
    assert "hola" in spoken.lower(), f"never taught 'hola': {spoken!r}"
