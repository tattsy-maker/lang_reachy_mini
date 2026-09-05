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

import asyncio
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tutor.store import GOALS, Learner, LearnerStore, LEVELS  # noqa: E402
from tutor.wishes import DEFAULT_WISHES_FILE, record_wish as _record_wish  # noqa: E402

logger = logging.getLogger("tutor_mode")

# Kept in sync with multilingual.LANGUAGES (which is authoritative but pulls
# in pipecat, so it is only consulted lazily -- the briefing logic here must
# stay importable in the light test venv). Includes the two spec languages
# that have no local voice yet.
_LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French",
                   "it": "Italian", "pt": "Portuguese", "hi": "Hindi",
                   "ru": "Russian", "zh": "Mandarin Chinese"}
# Beyond the six tutoring languages: names a visitor may ask for. Local
# mode has no voice for these (the speech layer folds them into the main
# voice), but cloud mode (Gemini) speaks them natively -- rehearsal found a
# visitor asking for Swedish and the tool refusing it.
_LANGUAGE_NAMES.update({
    "de": "German", "sv": "Swedish", "nl": "Dutch", "da": "Danish",
    "no": "Norwegian", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "uk": "Ukrainian", "el": "Greek", "tr": "Turkish", "ar": "Arabic",
    "he": "Hebrew", "ja": "Japanese", "ko": "Korean", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "ro": "Romanian", "hu": "Hungarian",
    "ca": "Catalan", "ta": "Tamil", "bn": "Bengali", "ur": "Urdu",
    "fa": "Persian", "sw": "Swahili",
})

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

# Why the student is learning (T13.1). The family: "one wants only
# conversation, another is preparing for an exam, a third for a job
# interview -- it should behave differently for each."
_GOAL_GUIDANCE = {
    "conversation": (
        "{name} wants to be able to hold everyday conversations. Keep the "
        "talk flowing, correct only what blocks understanding, and favour "
        "useful phrases over rules."),
    "exam": (
        "{name} is preparing for an exam. Be exact: correct every error, "
        "offer richer vocabulary and synonyms, name the grammar point in "
        "one sentence, and vary the register."),
    "work": (
        "{name} needs {language} for work, such as interviews and meetings. "
        "Practise formal register and professional vocabulary, offer more "
        "sophisticated alternatives to what they said, and rehearse "
        "typical workplace exchanges."),
    "travel": (
        "{name} is learning for travel. Drill the practical situations -- "
        "directions, ordering, shopping, small talk -- and prize being "
        "understood over being perfect."),
    "other": (
        "{name}'s own goal is described below; tailor the lesson to it."),
}

_BRIEFING = """

You are in tutor mode. You are a friendly, patient language tutor, and this \
is a lesson.

Your student is {name}: {level} {language}, {session_line}
Their goal: {goal_guidance}{goal_note}

Tutoring rules. Where they conflict with the general language rule above, \
these win:

- Teach in {language}. The target language comes from {name}'s profile, not \
from what you hear: if {name} asks something in English, answer it, then \
steer back into {language}. A stray word in another language does not change \
the lesson. But if {name} clearly asks to practice a different language, that \
is allowed and welcome: switch at once and call set_target_language so it is \
remembered.
- {level_guidance}
- Correct every mistake, briefly and kindly. Give the right form, let the \
lesson move on. Never let an error slide, and never lecture about grammar \
for more than one sentence.
- The transcripts you receive can garble {language} words embedded in an \
English sentence (the recognizer commits to one language at a time). If a \
phrase looks mangled but context makes clear what a {language} learner was \
trying to say, repair it silently and answer that. Only ask them to repeat \
when the meaning is genuinely unrecoverable — and never scold pronunciation \
based on a garbled transcript.
- Nod for right answers. Shake your head gently for wrong ones.
- Ask one question at a time, so {name} talks more than you do.
- Open by greeting {name} by name in {language}, then pick up exactly where \
the notes below leave off.

When {name} says goodbye or the session is clearly over, say a short warm \
goodbye and call save_session_notes exactly once, honestly filled in: what \
was practiced, what {name} struggled with including the correction, what \
clicked, and what to open with next time. If this session showed clearly \
that {name}'s stored level is wrong, also call update_learner_level.

If {name} ever asks to be forgotten, call forget_me, then confirm out loud \
that their file is deleted on the spot.

{notes_section}"""

# The enrollment interview (T13.1). The family watched Italian "start
# strangely" while French asked the level at once: the questions are now
# a fixed script, one at a time, and the lesson waits for all four.
ENROLLMENT_SCRIPT = """\
Enrollment is four short questions, asked ONE AT A TIME in this order, \
each in its own turn, waiting for the answer before the next. No lesson, \
no teaching, no foreign words until all four are answered:
  1. their name;
  2. which language they want to practice ({languages});
  3. their level in it: beginner, intermediate or advanced (ask this \
explicitly, in those words, never assume);
  4. why they are learning: just conversation, an exam, work or interviews, \
travel, or something else (one short question, their own words are fine).
Then call enroll_new_learner with all four (name, target_language, level, \
goal, and their own words as goal_note). When it succeeds, greet them by \
name and start a short first lesson pitched at the level and goal they gave."""

# Briefing when a face is present but matches nobody in the store (T9).
STRANGER_BRIEFING = """

You are in tutor mode, but you do not recognize the person in front of \
you. You are a friendly language tutor in a small robot body.

- Greet them warmly in English and introduce yourself as a language tutor.
- If they would like a lesson, first ask, in these words or close to them: \
"Would you like me to remember you for the rest of the day?" You need a \
clear yes before anything about them is stored.
- If they agree, run the enrollment interview below.
- If they decline, that is completely fine. Chat normally, store nothing, \
and do not ask again.
- If they were enrolled and say goodbye, call save_session_notes as usual. \
If they ask to be forgotten, call forget_me and confirm out loud.

""" + ENROLLMENT_SCRIPT + "\n"

# Briefing when the best face match lands in the ask-don't-guess band.
UNSURE_BRIEFING = """

You are in tutor mode. The person in front of you might be {name}, one of \
your students, but you are not certain, and a wrong greeting is worse than \
asking.

- Do NOT use their name as if you were sure, and do not mention their \
lesson history yet.
- Open in English by asking, warmly: "{name}, is that you?"
- If they confirm, call confirm_identity and continue as their tutor using \
what it returns.
- If they say no, apologize lightly, then treat them as someone new: offer \
a lesson, ask "Would you like me to remember you for the rest of the \
day?", and on a clear yes run the enrollment interview below. If they \
decline, chat normally and store nothing.

""" + ENROLLMENT_SCRIPT.replace("{languages}", "any language you can teach") + "\n"
# (UNSURE_BRIEFING is formatted with name= only, so the interview's
# language roster is spelled out in words here rather than substituted.)

# Booth persona (T13.5). The family wanted "a couple of jokes with a
# little edge"; the decision on 2026-09-02 was no Skynet/Terminator/robot
# uprising allusions and no movie lines -- gentle edge only. Also carries
# the wishlist question (T13.6). Appended only with --persona booth.
BOOTH_PERSONA = """

Booth persona. You are the demo at a Maker Faire booth, so you have a \
little character, used sparingly: at most one quip per moment, each at \
most once per visitor, never at the expense of a learner's mistake, and \
never interrupting a lesson. Say a quip in the lesson language when the \
student is intermediate or advanced, otherwise in English. Keep the edge \
gentle: nothing about robots taking over, no threats, no movie quotes.
- When enrollment succeeds: "I will remember you. Until closing time, anyway."
- When you have misheard twice in a row: "My ears were the cheapest part \
of me. Once more?"
- When they say goodbye, and only then, do this in order across turns: \
first say "Before you go, one quick question: if this were a robot you had \
bought, what would you want it to do?" and STOP -- say nothing else and \
wait for their answer. When they answer, call record_wish with their words, \
thank them in one sentence, then add "Go and practice. I will know if you \
did not." and call save_session_notes as usual. If they do not answer or \
say they have to run, let it go: goodbye and notes, no wish. Never bring \
the question up mid-lesson and never announce that you are about to ask.
"""

PERSONAS = {"plain": "", "booth": BOOTH_PERSONA}


def build_persona(kind: str) -> str:
    """The persona addendum for --persona; "" for plain."""
    if kind not in PERSONAS:
        raise ValueError(f"unknown persona {kind!r}; choose from {list(PERSONAS)}")
    return PERSONAS[kind]


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
    goal = getattr(learner, "goal", "conversation")
    goal_guidance = _GOAL_GUIDANCE.get(goal, _GOAL_GUIDANCE["other"])
    goal_note = getattr(learner, "goal_note", "") or ""
    goal_note = (f' In their words: "{goal_note.strip()}".'
                 if goal_note.strip() else "")
    return _BRIEFING.format(
        name=learner.name,
        level=learner.level,
        language=language,
        session_line=session_line,
        level_guidance=guidance.format(language=language),
        goal_guidance=goal_guidance.format(name=learner.name,
                                           language=language),
        goal_note=goal_note,
        notes_section=notes_section,
    )


def normalize_goal(value: str) -> str:
    """Free text ('job interviews', 'just chatting') -> one of GOALS."""
    text = str(value or "").strip().lower()
    if text in GOALS:
        return text
    hints = (("conversation", ("convers", "chat", "talk", "speak", "fun",
                               "family", "friend")),
             ("exam", ("exam", "test", "certif", "school", "class", "dele",
                       "delf", "hsk", "toefl", "ielts", "grade")),
             ("work", ("work", "job", "interview", "career", "business",
                       "meeting", "professional", "office", "client")),
             ("travel", ("travel", "trip", "holiday", "vacation", "visit",
                         "abroad", "tourist")))
    for goal, keys in hints:
        if any(k in text for k in keys):
            return goal
    return "other"


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


class CurrentLearner:
    """The mutable identity slot every tutor tool reads. ``--learner``
    fills it at startup; face recognition fills it when sure; and
    confirm_identity / enroll_new_learner fill it mid-conversation.

    ``candidate`` holds an unsure face match awaiting verbal confirmation.
    ``saved_ids`` tracks whose notes are saved this session, shared with
    the session runner (T10) so a walk-away save can be observed."""

    def __init__(self, learner: Learner | None = None):
        self.learner = learner
        self.candidate: Learner | None = None
        self.saved_ids: set[str] = set()

    def reset(self) -> None:
        self.learner = None
        self.candidate = None
        self.saved_ids.clear()


def normalize_language(value: str) -> str | None:
    """'es' or 'Spanish' (any case) -> 'es'; None when unrecognized."""
    value = value.strip().lower()
    if value in _LANGUAGE_NAMES:
        return value
    for code, name in _LANGUAGE_NAMES.items():
        if value == name.lower():
            return code
    return None


# Spoken follow-ups to the voice-print policy (T13.9, voiceid.VoiceIdentity).
# The challenge is the family's "you don't sound like yourself today"
# moment -- playful, never accusing.
def voice_cue(action: str, learner: Learner | None, store: LearnerStore
              ) -> str | None:
    """The user-turn cue to inject after a voice-identity action."""
    if learner is None:
        return None
    name = learner.name
    if action == "challenge":
        return (f"(Identity check: you recognized {name} by face, but the "
                f"voice you are hearing does not match {name}'s stored voice "
                "print. Say, lightly and playfully, that they do not sound "
                "quite like themselves today, and ask them to say a little "
                "more. One warm sentence, no accusation, then wait.)")
    if action == "downgrade":
        return (f"(Their voice still does not match {name}'s. You are no "
                f"longer sure who this is. Ask plainly: \"{name}, is that "
                "you?\" If they say yes, call confirm_identity and carry on. "
                "If they say no, apologize lightly and treat them as someone "
                "new: offer a lesson and run the enrollment interview.)")
    if action == "confirmed":
        notes = store.read_notes(learner.id, max_sessions=BRIEFING_SESSIONS)
        language = language_name(learner.target_language)
        return (f"(Voice check: this is {name}. Their voice matches the print "
                f"on file, so do not ask. Greet {name} by name in {language} "
                "and continue as their tutor, picking up where the notes "
                f"leave off. Profile: {learner.level} {language}, "
                f"{learner.sessions} past sessions.)\n\nRecent notes:\n"
                + (notes.strip() or "none yet"))
    return None


def build_tutor_tools(store: LearnerStore, holder: CurrentLearner,
                      wishes_path=None) -> list:
    """The memory tools, same FunctionSchema style as the motion tools."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    saved_ids = holder.saved_ids
    wishes_path = wishes_path or DEFAULT_WISHES_FILE

    async def save_session_notes(params):
        learner = holder.learner
        if learner is None:
            await params.result_callback(
                {"saved": False,
                 "reason": "no learner identified or enrolled this session"})
            return
        if learner.id in saved_ids:
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
        saved_ids.add(learner.id)
        logger.info("tutor: saved session notes for %s (now %d sessions)",
                    learner.id, updated.sessions)
        await params.result_callback(
            {"saved": True, "session": updated.sessions})

    async def update_learner_level(params):
        learner = holder.learner
        if learner is None:
            await params.result_callback({"error": "no learner identified"})
            return
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

    async def set_target_language(params):
        learner = holder.learner
        if learner is None:
            await params.result_callback({"error": "no learner identified"})
            return
        language = normalize_language(str(params.arguments.get("language", "")))
        if language is None:
            await params.result_callback(
                {"error": "unrecognized language; supported: "
                          + ", ".join(sorted(_LANGUAGE_NAMES.values()))})
            return
        current = store.load(learner.id)
        if current is None:
            await params.result_callback({"error": "learner vanished"})
            return
        old_code = current.target_language
        current.target_language = language
        store.save(current)
        learner.target_language = language
        logger.info("tutor: language for %s changed %s -> %s",
                    learner.id, old_code, language)
        await params.result_callback(
            {"target_language": language_name(language), "was": old_code,
             "note": "continue the lesson in the new language"})

    async def set_learner_goal(params):
        learner = holder.learner
        if learner is None:
            await params.result_callback({"error": "no learner identified"})
            return
        goal = normalize_goal(str(params.arguments.get("goal", "")))
        note = str(params.arguments.get("goal_note", "")).strip()
        current = store.load(learner.id)
        if current is None:
            await params.result_callback({"error": "learner vanished"})
            return
        old = current.goal
        current.goal = goal
        if note:
            current.goal_note = note
        store.save(current)
        learner.goal = goal
        learner.goal_note = current.goal_note
        logger.info("tutor: goal for %s changed %s -> %s (%s)",
                    learner.id, old, goal, note or "-")
        await params.result_callback(
            {"goal": goal, "was": old, "goal_note": current.goal_note,
             "note": "adapt the rest of the lesson to this goal"})

    async def record_wish(params):
        text = str(params.arguments.get("wish", "")).strip()
        if not text:
            await params.result_callback({"error": "the wish was empty"})
            return
        name = holder.learner.name if holder.learner else None
        path = await asyncio.to_thread(_record_wish, text, name=name,
                                       path=wishes_path)
        logger.info("booth: wish recorded (%s): %s", name or "anonymous", text)
        await params.result_callback(
            {"recorded": True, "file": os.path.basename(str(path)),
             "note": "thank them briefly; do not ask for another"})

    async def forget_me(params):
        learner = holder.learner
        if learner is None:
            await params.result_callback(
                {"error": "nobody is identified, there is nothing to forget"})
            return
        store.delete(learner.id)
        holder.learner = None
        saved_ids.discard(learner.id)
        logger.info("tutor: forgot learner %s on request", learner.id)
        await params.result_callback(
            {"forgotten": True, "name": learner.name})

    return [
        FunctionSchema(
            name="forget_me",
            description="Delete everything stored about the current person "
                        "-- profile, face data, and notes -- immediately. "
                        "Only when they explicitly ask to be forgotten. "
                        "Confirm out loud afterwards.",
            properties={}, required=[], handler=forget_me,
        ),
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
            name="set_target_language",
            description="Switch which language the student practices, when "
                        "they clearly ask to. Any language you can speak "
                        "can be taught, including Russian and Mandarin.",
            properties={"language": {"type": "string",
                                     "description": "e.g. 'Russian' or 'ru'"}},
            required=["language"],
            handler=set_target_language,
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
        FunctionSchema(
            name="set_learner_goal",
            description="Record or change why the student is learning, when "
                        "they tell you: conversation, exam, work (jobs, "
                        "interviews), travel, or other. Adapts how you "
                        "teach them from now on.",
            properties={"goal": {"type": "string", "enum": list(GOALS),
                                 "description": "the category"},
                        "goal_note": {"type": "string",
                                      "description": "their goal in their "
                                                     "own words"}},
            required=["goal"],
            handler=set_learner_goal,
        ),
        FunctionSchema(
            name="record_wish",
            description="Log what a visitor says they would want this "
                        "product to do, in their own words. Call once per "
                        "visitor, right after they answer the wish question.",
            properties={"wish": {"type": "string",
                                 "description": "their answer, verbatim"}},
            required=["wish"],
            handler=record_wish,
        ),
    ]


def build_enrollment_tools(store: LearnerStore, holder: CurrentLearner,
                           face_source, frames_factory=None,
                           voice_identity=None) -> list:
    """enroll_new_learner and confirm_identity, registered alongside the
    memory tools whenever the agent has a face source. confirm_identity
    resolves ``holder.candidate`` — the unsure face match set at startup
    (T9) or by the session runner (T10).

    ``frames_factory`` (T13.3), when given, returns an iterable of frames
    from the shared camera hub instead of reopening ``face_source`` -- a
    V4L2 device cannot be streamed twice."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    async def enroll_new_learner(params):
        if holder.learner is not None:
            # One person per session; a duplicate call must not mint a
            # second profile (T10 resets the holder between visitors).
            await params.result_callback(
                {"enrolled": False,
                 "reason": f"already tutoring {holder.learner.name} -- "
                           "they are enrolled and remembered"})
            return
        a = params.arguments
        name = str(a.get("name", "")).strip()
        if not name:
            await params.result_callback({"error": "a name is required"})
            return
        language = normalize_language(str(a.get("target_language", "")))
        if language is None:
            await params.result_callback(
                {"error": "unrecognized language; supported: "
                          + ", ".join(sorted(_LANGUAGE_NAMES.values()))})
            return
        level = str(a.get("level", "beginner")).strip().lower()
        if level not in LEVELS:
            level = "beginner"
        goal = normalize_goal(str(a.get("goal", "conversation")))
        goal_note = str(a.get("goal_note", "")).strip()

        from face_id import capture_embedding
        frames = frames_factory() if frames_factory is not None else None
        vector = await asyncio.to_thread(capture_embedding, face_source,
                                         frames=frames)
        if vector is None:
            await params.result_callback(
                {"error": "no face visible right now; ask them to look at "
                          "you and try once more"})
            return
        # The interview itself is the voice enrollment (T13.9): four
        # answers is plenty of speech for a print, if the collector heard it.
        voice_print = (voice_identity.print_list()
                       if voice_identity is not None else None)
        learner = store.create(name, language, level=level, tier="guest",
                               embedding=[float(x) for x in vector],
                               goal=goal, goal_note=goal_note,
                               voice_embedding=voice_print)
        holder.learner = learner
        logger.info("tutor: enrolled new guest %s (%s, %s, goal %s%s)",
                    learner.id, language, level, goal,
                    ", with voice print" if voice_print else "")
        await params.result_callback(
            {"enrolled": True, "name": learner.name, "id": learner.id,
             "target_language": language, "level": level, "goal": goal,
             "note": "you are now their tutor; greet them by name and "
                     "begin a short first lesson at this level and goal"})

    async def confirm_identity(params):
        candidate = holder.candidate
        if candidate is None:
            await params.result_callback(
                {"error": "there is no identity candidate to confirm"})
            return
        learner = store.load(candidate.id)
        if learner is None:
            await params.result_callback({"error": "learner vanished"})
            return
        holder.learner = learner
        notes = store.read_notes(learner.id, max_sessions=BRIEFING_SESSIONS)
        logger.info("tutor: identity confirmed as %s", learner.id)
        await params.result_callback(
            {"confirmed": learner.name,
             "target_language": language_name(learner.target_language),
             "level": learner.level,
             "sessions": learner.sessions,
             "recent_notes": notes,
             "note": "greet them by name in their target language and pick "
                     "up where the notes leave off"})

    tools = [
        FunctionSchema(
            name="enroll_new_learner",
            description="Create a guest learner profile and capture their "
                        "face so they are remembered for the rest of the "
                        "day. Only after they clearly said yes to being "
                        "remembered, and only once all four enrollment "
                        "questions (name, language, level, goal) are "
                        "answered.",
            properties={
                "name": {"type": "string",
                         "description": "their name, as they said it"},
                "target_language": {
                    "type": "string",
                    "description": "language to practice, e.g. 'Spanish' "
                                   "or 'es'"},
                "level": {"type": "string", "enum": list(LEVELS),
                          "description": "the level they stated when asked"},
                "goal": {"type": "string", "enum": list(GOALS),
                         "description": "why they are learning"},
                "goal_note": {"type": "string",
                              "description": "their goal in their own words"},
            },
            required=["name", "target_language", "level", "goal"],
            handler=enroll_new_learner,
        ),
    ]
    tools.append(FunctionSchema(
        name="confirm_identity",
        description="The person verbally confirmed they are the student "
                    "you tentatively recognized. Call this to load their "
                    "profile and notes, then continue as their tutor. Only "
                    "after an explicit yes to your 'is that you?' question.",
        properties={}, required=[], handler=confirm_identity,
    ))
    return tools
