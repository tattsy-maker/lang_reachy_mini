"""The intake (T17.4): a first meeting with a teacher, enforced in code.

On 2026-09-05 the mother said "Yes, let's. My name is Солнышко" and ten
seconds after the greeting the model called ``enroll_new_learner`` with
target English, level beginner, goal conversation and English as the
language to explain in -- three of four answers invented, the fifth
never asked. The family's debrief: ask the level every time, ask how
long they have and what they want today, structure the lesson, and make
the start feel like meeting a teacher.

A prompt cannot enforce that; a tool can. The model now fills an
``Intake`` one field at a time through ``intake_answer(field, value)``,
and ``enroll_new_learner`` refuses until every field has been answered,
saying which are still missing and what to ask next. The order below is
the interview order; the model asks in the language the visitor speaks.

Two of the fields are not profile facts but *this session's* plan
(``minutes``, ``today``); they become a ``SessionPlan``, which also
times the "about two minutes left" cue the runner injects at
``warn_fraction`` of the agreed time. A returning learner gets the same
two questions from the greeting cue and ``set_session_plan``.

Pure Python, no pipecat: unit-tested in the light venv.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EXPLAIN_IN = ("native", "target", "both")
CORRECTIONS = ("every", "blocking", "end")

# Interview order. Fields that are naturally one question share a turn
# (own language + explain-in; minutes + today).
INTAKE_FIELDS = ("name", "target_language", "native_language", "explain_in",
                 "level", "goal", "minutes", "today", "corrections")

# ``explain_in`` is asked but never blocks: when the visitor named their
# own language and the model forgot to record the explain-in choice
# (live, 2026-09-05: "I speak English ... explain in English" became
# native_language only), enrollment goes ahead with "native", which is
# what T16 always did. Every other field is required.
OPTIONAL_FIELDS = ("explain_in",)
DEFAULTS = {"explain_in": "native"}
MIN_MINUTES, MAX_MINUTES = 1, 180

# What to ask for each field, for the model (it phrases it in the
# visitor's language). ``{target}`` is filled once the target is known.
# Replies that answer a yes/no question, never a name or a lesson focus.
_BARE_REPLIES = {"yes", "yeah", "yep", "sure", "ok", "okay", "no", "nope",
                 "да", "нет", "sí", "si", "oui", "non", "ja", "nein"}

QUESTIONS = {
    "name": "ask their name",
    "target_language": "ask which language they want to practise (English "
                       "is a fine answer from someone whose own language is "
                       "different)",
    "native_language": "ask what their own language is, and whether you "
                       "should explain things in it, in {target}, or in "
                       "both (record native_language and explain_in)",
    "explain_in": "ask whether you should explain in their own language, "
                  "in {target}, or in both",
    "level": "ask their level in {target}, in these words: beginner, "
             "intermediate or advanced (never assume it)",
    "goal": "ask why they are learning {target}: just conversation, an "
            "exam, work or interviews, travel, or something else",
    "minutes": "ask how many minutes they have today and what they want "
               "to get out of it (record minutes and today)",
    "today": "ask what they want to get out of today's lesson",
    "corrections": "ask how they want to be corrected: every mistake as "
                   "it happens, only what gets in the way of being "
                   "understood, or at the end",
}

_CORRECTION_WORDS = (
    ("every", ("every", "all", "always", "each", "as it happens", "immediately")),
    ("blocking", ("block", "only", "important", "understand", "big", "major")),
    ("end", ("end", "after", "later", "afterwards", "summary")),
)


def normalize_corrections(value: str) -> str | None:
    text = str(value or "").strip().lower()
    if text in CORRECTIONS:
        return text
    for kind, words in _CORRECTION_WORDS:
        if any(w in text for w in words):
            return kind
    return None


def parse_minutes(value) -> int | None:
    """'15', 15, '15 minutes', 'about 20 min' -> 15/20; None if no number."""
    if isinstance(value, (int, float)):
        n = int(value)
    else:
        m = re.search(r"\d+", str(value or ""))
        if not m:
            words = {"five": 5, "ten": 10, "fifteen": 15, "twenty": 20,
                     "thirty": 30, "forty": 40, "sixty": 60,
                     "пять": 5, "десять": 10, "пятнадцать": 15,
                     "двадцать": 20, "тридцать": 30}
            text = str(value or "").lower()
            n = next((v for w, v in words.items() if w in text), None)
            if n is None:
                return None
        else:
            n = int(m.group())
    if n < MIN_MINUTES or n > MAX_MINUTES:
        return None
    return n


class Intake:
    """The interview's answers, validated one field at a time."""

    def __init__(self):
        self.answers: dict = {}
        self.goal_note = ""

    # -- answering ---------------------------------------------------------

    def answer(self, field_name: str, value, note: str | None = None) -> dict:
        """Record one answer. Returns ``{"ok": True, "field": f, "value":
        v}`` or ``{"ok": False, "field": f, "error": why}``; either way
        the caller appends ``missing``/``next`` for the model."""
        from tutor_mode import normalize_goal, normalize_language
        f = str(field_name or "").strip().lower()
        if f not in INTAKE_FIELDS:
            return {"ok": False, "field": f,
                    "error": f"unknown field; one of {', '.join(INTAKE_FIELDS)}"}
        raw = value
        if f in ("name", "today"):
            text = " ".join(str(raw or "").split())
            if not text:
                return {"ok": False, "field": f, "error": "empty"}
            if text.lower().strip(" .!?,") in _BARE_REPLIES:
                # 2026-09-23: the "remember you?" yes went in as today's
                # focus, and the lesson plan read "want: yes".
                return {"ok": False, "field": f,
                        "error": f"{text!r} is a yes/no, not an answer to "
                                 f"this; {QUESTIONS[f]}"}
            self.answers[f] = text
            return {"ok": True, "field": f, "value": text}
        if f in ("target_language", "native_language"):
            code = normalize_language(str(raw or ""))
            if code is None:
                return {"ok": False, "field": f,
                        "error": f"unrecognized language {raw!r}"}
            self.answers[f] = code
            return {"ok": True, "field": f, "value": code}
        if f == "explain_in":
            text = str(raw or "").strip().lower()
            if text in EXPLAIN_IN:
                self.answers[f] = text
                return {"ok": True, "field": f, "value": text}
            code = normalize_language(text)
            if code is None:
                return {"ok": False, "field": f,
                        "error": "one of native, target, both, or a language"}
            if code == self.answers.get("target_language"):
                self.answers[f] = "target"
            else:
                # "explain in Russian" from someone who never named their
                # own language: that names it.
                self.answers.setdefault("native_language", code)
                self.answers[f] = "native" if code == self.answers[
                    "native_language"] else "both"
            return {"ok": True, "field": f, "value": self.answers[f]}
        if f == "level":
            from tutor.store import LEVELS
            text = str(raw or "").strip().lower()
            if text not in LEVELS:
                return {"ok": False, "field": f,
                        "error": f"one of {', '.join(LEVELS)}"}
            self.answers[f] = text
            return {"ok": True, "field": f, "value": text}
        if f == "goal":
            self.answers[f] = normalize_goal(str(raw or ""))
            self.goal_note = " ".join(str(note or raw or "").split())
            return {"ok": True, "field": f, "value": self.answers[f]}
        if f == "minutes":
            n = parse_minutes(raw)
            if n is None:
                return {"ok": False, "field": f,
                        "error": f"a number of minutes, {MIN_MINUTES}-{MAX_MINUTES}"}
            self.answers[f] = n
            return {"ok": True, "field": f, "value": n}
        if f == "corrections":
            kind = normalize_corrections(str(raw or ""))
            if kind is None:
                return {"ok": False, "field": f,
                        "error": f"one of {', '.join(CORRECTIONS)}"}
            self.answers[f] = kind
            return {"ok": True, "field": f, "value": kind}
        return {"ok": False, "field": f, "error": "unhandled"}   # pragma: no cover

    # -- state ---------------------------------------------------------------

    def missing(self) -> list[str]:
        return [f for f in INTAKE_FIELDS if f not in self.answers]

    def missing_required(self) -> list[str]:
        return [f for f in self.missing() if f not in OPTIONAL_FIELDS]

    @property
    def complete(self) -> bool:
        """Every required answer is in (``explain_in`` may default)."""
        return not self.missing_required()

    def next_question(self) -> str | None:
        """What to ask next, in the interview order, for the model."""
        missing = self.missing()
        if not missing:
            return None
        from tutor_mode import language_name
        target = language_name(self.answers.get("target_language", "en")) \
            if "target_language" in self.answers else "the language"
        f = missing[0]
        return QUESTIONS[f].format(target=target)

    def status(self) -> dict:
        """The tail every tool result carries."""
        missing = self.missing()
        out = {"missing": missing}
        if missing and not self.complete:
            out["next"] = self.next_question()
            out["note"] = ("if the visitor already said the answer to a "
                           "missing question, record it now with another "
                           "intake_answer call; otherwise ask the next "
                           "question, one question, in the visitor's "
                           "language. No lesson yet.")
        elif missing:
            out["next"] = self.next_question()
            out["note"] = ("only explain_in is unanswered: ask it if they "
                           "have not said, else call enroll_new_learner "
                           "(explanations default to their own language)")
        else:
            out["note"] = "every question is answered: call enroll_new_learner"
        return out

    def profile_kwargs(self) -> dict:
        """Keyword arguments for ``LearnerStore.create``."""
        a = self.answers
        return dict(level=a["level"], goal=a["goal"],
                    goal_note=self.goal_note,
                    native_language=a["native_language"],
                    explain_in=a.get("explain_in", DEFAULTS["explain_in"]),
                    corrections=a["corrections"])

    def plan(self, now: float) -> "SessionPlan":
        return SessionPlan(minutes=self.answers["minutes"],
                           focus=self.answers["today"], started_at=now)

    def reset(self) -> None:
        self.answers.clear()
        self.goal_note = ""


@dataclass
class SessionPlan:
    """This session's agreed length and focus, and the one time cue."""
    minutes: int
    focus: str
    started_at: float
    warned: bool = False
    warn_fraction: float = 0.8
    extra: dict = field(default_factory=dict)

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started_at)

    def remaining_minutes(self, now: float) -> float:
        return max(0.0, self.minutes - self.elapsed(now) / 60.0)

    def due_warning(self, now: float) -> bool:
        """True once, when ``warn_fraction`` of the agreed time is used."""
        if self.warned:
            return False
        if self.elapsed(now) >= self.warn_fraction * self.minutes * 60.0:
            self.warned = True
            return True
        return False

    def spoken_plan_note(self) -> str:
        return (f"They have {self.minutes} minutes and want: {self.focus}. "
                "Before the first task, say a plan of two or three steps "
                "that fits that, in one or two sentences, then start.")


def time_cue(plan: SessionPlan, now: float) -> str:
    left = max(1, round(plan.remaining_minutes(now)))
    unit = "minute" if left == 1 else "minutes"
    return (f"(About {left} {unit} of the {plan.minutes} agreed remain. "
            "Finish the current exercise, sum up in two sentences what "
            "they did well and one thing to work on, then ask whether "
            "they want to stop here or carry on. If they leave, notes "
            "and the closing question as usual.)")
