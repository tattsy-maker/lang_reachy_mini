"""Tutor mode (T4): the learner briefing and the memory tools.

Turns the general voice agent into *this learner's* tutor: the briefing is
appended to the agent's system prompt (target language and level from the
learner's profile, recent session notes inlined), and two new tools let the
model write back to the learner store — ``save_session_notes`` at the end
of a session and ``update_learner_level`` when the evidence is clear.

Named ``tutor_mode`` rather than the plan's ``tutor.py`` because the
learner store already owns the ``tutor`` package name at the repo root, and
this module must import it.
"""

from __future__ import annotations

import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tutor.store import Learner, LearnerStore, LEVELS  # noqa: E402

logger = logging.getLogger("tutor_mode")

# Kept in sync with multilingual.LANGUAGES (which is authoritative but pulls
# in pipecat, so it is only consulted lazily -- the briefing logic here must
# stay importable in the light test venv). Includes the two spec languages
# that have no local voice yet.
_LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French",
                   "it": "Italian", "pt": "Portuguese", "hi": "Hindi",
                   "ru": "Russian", "zh": "Mandarin Chinese"}

DEFAULT_LEARNERS_ROOT = os.path.join(_REPO, "learners")

# How many past sessions the briefing carries. Enough to pick up threads,
# small enough not to bloat a latency-sensitive prompt (spec risk section:
# the briefing must not blow the ~2.7s booth turn budget).
BRIEFING_SESSIONS = 3

_LEVEL_GUIDANCE = {
    "beginner": (
        "The student is a beginner. Use short, simple {language} phrases, "
        "spoken plainly, and give a brief English explanation alongside "
        "anything new. Celebrate small wins."),
    "intermediate": (
        "The student is intermediate. Speak mostly {language} at a "
        "comfortable everyday level. Explain in English only when the "
        "student is genuinely stuck."),
    "advanced": (
        "The student is advanced. Stay in {language} the whole time, idioms "
        "welcome, natural pace. Use English only as a last resort."),
}

_BRIEFING = """

You are in tutor mode. You are a friendly, patient language tutor, and this \
is a lesson.

Your student is {name}: {level} {language}, {session_line}

Tutoring rules. Where they conflict with the general language rule above, \
these win:

- Teach in {language}. The target language comes from {name}'s profile, not \
from what you hear. If {name} asks something in English, answer it, then \
steer back into {language}. Never switch the lesson to another language.
- {level_guidance}
- Correct every mistake, briefly and kindly. Give the right form, let the \
lesson move on. Never let an error slide, and never lecture about grammar \
for more than one sentence.
- Nod for right answers. Shake your head gently for wrong ones.
- Ask one question at a time, so {name} talks more than you do.
- Open by greeting {name} by name in {language}, then pick up exactly where \
the notes below leave off.

When {name} says goodbye or the session is clearly over, say a short warm \
goodbye and call save_session_notes exactly once, honestly filled in: what \
was practiced, what {name} struggled with including the correction, what \
clicked, and what to open with next time. If this session showed clearly \
that {name}'s stored level is wrong, also call update_learner_level.

{notes_section}"""


def language_name(code: str) -> str:
    """Human name for a language code, falling back to the code itself."""
    code = code.lower()
    try:
        from multilingual import LANGUAGES
        if code in LANGUAGES:
            return LANGUAGES[code].name
    except ImportError:
        pass
    return _LANGUAGE_NAMES.get(code, code)


def build_briefing(learner: Learner, notes: str) -> str:
    """The tutor briefing to append to the agent's system prompt."""
    language = language_name(learner.target_language)
    if learner.sessions:
        session_line = (f"session number {learner.sessions + 1} together. "
                        "Your notes from past sessions, newest first, are "
                        "below.")
        notes_section = notes.strip()
    else:
        session_line = ("your first session together. There are no notes "
                        "yet.")
        notes_section = ("No notes yet. Start by getting to know "
                         f"{learner.name} a little: ask, in simple "
                         f"{language}, what they would like to practice.")
    guidance = _LEVEL_GUIDANCE.get(learner.level,
                                   _LEVEL_GUIDANCE["intermediate"])
    return _BRIEFING.format(
        name=learner.name,
        level=learner.level,
        language=language,
        session_line=session_line,
        level_guidance=guidance.format(language=language),
        notes_section=notes_section,
    )


def load_learner(root: str, name_or_id: str) -> tuple[Learner, str, LearnerStore]:
    """Resolve a --learner argument to (learner, briefing notes, store).

    Accepts a folder id or a display name (case-insensitive). Exits with a
    clear message on no match or an ambiguous one — a wrong greeting is the
    worst failure mode, so never guess between two Marias.
    """
    store = LearnerStore(root)
    learner = store.load(name_or_id)
    if learner is None:
        matches = store.find_by_name(name_or_id)
        if len(matches) > 1:
            raise SystemExit(
                "--learner %r is ambiguous; use one of the ids: %s"
                % (name_or_id, ", ".join(m.id for m in matches)))
        learner = matches[0] if matches else None
    if learner is None:
        known = ", ".join(l.id for l in store.list()) or "none yet"
        raise SystemExit("no learner %r under %s (known: %s)"
                         % (name_or_id, root, known))
    notes = store.read_notes(learner.id, max_sessions=BRIEFING_SESSIONS)
    return learner, notes, store


def build_tutor_tools(store: LearnerStore, learner: Learner) -> list:
    """The memory tools, same FunctionSchema style as the motion tools."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    saved = {"done": False}

    async def save_session_notes(params):
        if saved["done"]:
            # The prompt says "exactly once"; make a second call harmless
            # rather than trusting the model with duplicate entries.
            await params.result_callback(
                {"saved": False, "reason": "notes already saved this session"})
            return
        a = params.arguments
        body = "\n".join([
            f"- **Practiced:** {a.get('practiced', '').strip()}",
            f"- **Struggled with:** {a.get('struggled_with', '').strip()}",
            f"- **Wins:** {a.get('wins', '').strip()}",
            f"- **Next time:** {a.get('next_time', '').strip()}",
        ])
        updated = store.append_session(learner.id, body)
        saved["done"] = True
        logger.info("tutor: saved session notes for %s (now %d sessions)",
                    learner.id, updated.sessions)
        await params.result_callback(
            {"saved": True, "session": updated.sessions})

    async def update_learner_level(params):
        level = str(params.arguments.get("level", "")).strip().lower()
        if level not in LEVELS:
            await params.result_callback(
                {"error": f"level must be one of {list(LEVELS)}"})
            return
        current = store.load(learner.id)
        if current is None:
            await params.result_callback({"error": "learner vanished"})
            return
        old = current.level
        current.level = level
        store.save(current)
        learner.level = level
        logger.info("tutor: level for %s changed %s -> %s",
                    learner.id, old, level)
        await params.result_callback({"level": level, "was": old})

    text = {"type": "string"}
    return [
        FunctionSchema(
            name="save_session_notes",
            description="Save your end-of-session notes to the student's "
                        "file. Call exactly once, when the student says "
                        "goodbye or the session is clearly over. Be honest "
                        "and specific; you will rely on these notes next "
                        "time.",
            properties={
                "practiced": {"type": "string",
                              "description": "topics and vocabulary covered"},
                "struggled_with": {"type": "string",
                                   "description": "specific mistakes, each "
                                                  "with its correction"},
                "wins": {"type": "string",
                         "description": "what clicked this session"},
                "next_time": {"type": "string",
                              "description": "what to open with next session"},
            },
            required=["practiced", "struggled_with", "wins", "next_time"],
            handler=save_session_notes,
        ),
        FunctionSchema(
            name="update_learner_level",
            description="Change the student's stored level. Only when this "
                        "session gave clear evidence the stored level is "
                        "wrong.",
            properties={"level": {"type": "string",
                                  "enum": list(LEVELS),
                                  "description": "the new level"}},
            required=["level"],
            handler=update_learner_level,
        ),
    ]
