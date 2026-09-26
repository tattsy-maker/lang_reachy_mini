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
# Beyond the six tutoring languages: every language Gemini Live speaks
# (ai.google.dev/gemini-api/docs/live-guide, 99 languages, read
# 2026-09-25). Local mode has no voice for most of these (the speech layer
# folds them into the main voice), but cloud mode speaks them natively --
# rehearsal found a visitor asking for Swedish and the tool refusing it,
# and on 2026-09-25 the model told a visitor "I do not speak Arabic yet"
# because the prompt named eight languages and "most other languages".
_LANGUAGE_NAMES.update({
    "af": "Afrikaans", "ak": "Akan", "sq": "Albanian", "am": "Amharic",
    "ar": "Arabic", "hy": "Armenian", "as": "Assamese", "az": "Azerbaijani",
    "eu": "Basque", "be": "Belarusian", "bn": "Bengali", "bs": "Bosnian",
    "bg": "Bulgarian", "my": "Burmese", "ca": "Catalan", "ceb": "Cebuano",
    "hr": "Croatian", "cs": "Czech", "da": "Danish", "nl": "Dutch",
    "et": "Estonian", "fo": "Faroese", "fil": "Filipino", "fi": "Finnish",
    "gl": "Galician", "ka": "Georgian", "de": "German", "el": "Greek",
    "gu": "Gujarati", "ha": "Hausa", "he": "Hebrew", "hu": "Hungarian",
    "is": "Icelandic", "id": "Indonesian", "ga": "Irish", "ja": "Japanese",
    "kn": "Kannada", "kk": "Kazakh", "km": "Khmer", "rw": "Kinyarwanda",
    "ko": "Korean", "ku": "Kurdish", "ky": "Kyrgyz", "lo": "Lao",
    "lv": "Latvian", "lt": "Lithuanian", "mk": "Macedonian", "ms": "Malay",
    "ml": "Malayalam", "mt": "Maltese", "mi": "Maori", "mr": "Marathi",
    "mn": "Mongolian", "ne": "Nepali", "no": "Norwegian", "or": "Odia",
    "om": "Oromo", "ps": "Pashto", "fa": "Persian", "pl": "Polish",
    "pa": "Punjabi", "qu": "Quechua", "ro": "Romanian", "rm": "Romansh",
    "sr": "Serbian", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "st": "Southern Sotho",
    "sw": "Swahili", "sv": "Swedish", "tg": "Tajik", "ta": "Tamil",
    "te": "Telugu", "th": "Thai", "tn": "Tswana", "tr": "Turkish",
    "tk": "Turkmen", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "cy": "Welsh", "fy": "Western Frisian",
    "wo": "Wolof", "yo": "Yoruba", "zu": "Zulu",
})

# Other names and codes a visitor or the model may use for the same
# languages (normalize_language also drops a region: "ar-EG" -> "ar").
_LANGUAGE_ALIASES = {
    "chinese": "zh", "mandarin": "zh", "zh-hans": "zh", "zh-hant": "zh",
    "simplified chinese": "zh", "traditional chinese": "zh",
    "brazilian portuguese": "pt", "castilian": "es", "farsi": "fa",
    "tagalog": "fil", "tl": "fil", "nb": "no", "bokmal": "no",
    "norwegian bokmal": "no", "iw": "he", "sesotho": "st",
    "frisian": "fy", "panjabi": "pa", "oriya": "or", "gaelic": "ga",
    "kirghiz": "ky", "khmer (cambodian)": "km", "myanmar": "my",
}


def cloud_language_names() -> str:
    """Every language Gemini Live speaks, as a sentence-ready list."""
    names = sorted(set(_LANGUAGE_NAMES.values()))
    return ", ".join(names[:-1]) + " and " + names[-1]

DEFAULT_LEARNERS_ROOT = os.path.join(_REPO, "learners")

# How many past sessions the briefing carries. Enough to pick up threads,
# small enough not to bloat a latency-sensitive prompt (spec risk section:
# the briefing must not blow the ~2.7s booth turn budget).
BRIEFING_SESSIONS = 3

# ``{native}`` (T16) is the language the student is taught *in*: their
# own language, English by default. A Russian speaker learning English
# gets Russian explanations and English practice.
_LEVEL_GUIDANCE = {
    # 2026-09-24, a beginner in Hindi could not follow a lesson spoken
    # mostly in Hindi, and asked for English throughout and one small word
    # or phrase at a time.
    "beginner": (
        "The student is a beginner. Teach one {language} word or short "
        "phrase at a time: say it slowly, say what it means in {native}, "
        "have them say it back, and go on only when they have it. Two or "
        "three of those make a good lesson. Speak slowly and clearly: one "
        "short sentence at a time, a pause after each, every word finished. "
        "Celebrate small wins."),
    "intermediate": (
        "The student is intermediate. Speak mostly {language} at a "
        "comfortable everyday level. Explain in {native} only when the "
        "student is genuinely stuck."),
    "advanced": (
        "The student is advanced. Stay in {language} the whole time, idioms "
        "welcome, natural pace. Use {native} only as a last resort."),
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

# 2026-09-23: "Parliamo di Dio" (let's talk about God) got a note on the
# word for God -- hence the rule about a learner's own sentences.
_BRIEFING = """

You are in tutor mode. You are a friendly, patient language tutor, and this \
is a lesson.

Your student is {name}: {level} {language}, {session_line}
Their own language is {native}. {explain_policy}
Their goal: {goal_guidance}{goal_note}{plan_line}

Tutoring rules. Where they conflict with the general language rule above, \
these win:

- {lesson_language_rule} The target language comes from {name}'s profile, \
not from what you hear. A stray word in another language does not change \
the lesson. But if {name} clearly asks to practice a different language, \
that is allowed and welcome: switch at once and call set_target_language so \
it is remembered. If they ask to be taught in a different language, call \
set_native_language. If they say a stored fact about them is wrong (their \
level, their goal), call the matching tool in the same turn.
- {level_guidance}
- Setting a task: say what to express in {task_language} and have {name} \
say it in {language} ("How would you ask for a room for two?"), so the \
phrase is theirs. Never ask {name} to say in {language} something you have \
just said in {language}, and never ask a question whose answer is in the \
question. Prefer short role-play where you play the other side (a hotel \
desk, a cafe, a shop), one exchange at a time.
- Exercises are heard, not read. Use only formats that work by ear: a \
multiple choice with three spoken options, a free answer ("tell me in one \
sentence what you did today"), role-play, or repeat-after-me. For any \
fill-in-the-gap or multiple choice, call quiz and read its script word for \
word: it speaks the gap out loud and letters the options. Before a grammar \
drill, name the topic in one line and offer to explain it ("do you know \
what an adjective is?").
- {script_rule}
- {corrections_rule}
- When {name} says a whole sentence of their own in {language}, it is a \
move in the conversation, not a question about a word: answer what it \
says, in {spoken_language}, as a conversation partner would, and let the \
topic become the practice.
- The transcripts you receive can garble {language} words embedded in a \
{native} sentence (the recognizer commits to one language at a time). If a \
phrase looks mangled but context makes clear what a learner of {language} \
was trying to say, repair it silently and answer that. Only ask them to repeat \
when the meaning is genuinely unrecoverable — and never scold pronunciation \
based on a garbled transcript.
- Nod for right answers. Shake your head gently for wrong ones.
- Ask one question at a time, so {name} talks more than you do.
- {patience_rule}
- Open by greeting {name} by name in {spoken_language}, then pick up exactly \
where the notes below leave off.

Who {name} is was decided by the face recognizer, and by confirm_identity \
when it asked. Never judge identity from a look picture or from what someone \
says about their own hair or clothes: a picture is not a face match. If the \
voice check asks you to check, ask once; whatever they answer, that is the \
end of it for this session.

Staying on the lesson. A joke, a dare or an off-topic request gets one \
playful answer, then "back to {language}" and the next task; the third in a \
row gets a light no and the task again. If what {name} said makes no sense \
in context, say so and ask again rather than build on it. Your session notes \
are about {name}'s {language}, never about your own camera or hearing.

When {name} says goodbye or the session is clearly over, say a short warm \
goodbye and call save_session_notes exactly once, honestly filled in: what \
was practiced, what {name} struggled with including the correction, what \
clicked, and what to open with next time. If this session showed clearly \
that {name}'s stored level is wrong, also call update_learner_level.
Practising or teaching the words for goodbye is not {name} leaving, and \
neither is a request for a dance or a joke. Only {name} saying they are \
done, or walking away, ends the session. If you saved notes too early, do \
not say goodbye and do not ask parting questions: simply carry on.

If {name} ever asks to be forgotten, call forget_me, then confirm out loud \
that their file is deleted on the spot.

{notes_section}"""

# T17.5: the language explanations are given in, agreed at intake. On
# 2026-09-05 a Russian speaker learning English got English
# explanations she could not follow, then, once told, Russian for
# everything including the practice.
_EXPLAIN_POLICY = {
    "native": ("Every explanation, instruction and aside is in {native}, "
               "never in any other language, and {language} is what they "
               "practise."),
    "target": ("They asked to be taught in {language}: explain and instruct "
               "in simple {language} too, and drop to {native} only for a "
               "word they cannot get."),
    "both": ("They asked for both languages: explain in {language} first, "
             "then repeat the key point in {native} in one short sentence; "
             "the practice itself stays in {language}."),
}

# The first tutoring rule: which language the lesson is spoken in.
_LESSON_LANGUAGE_RULE = {
    "beginner": (
        "Speak {native} for everything: every explanation, instruction, "
        "question, bit of praise and goodbye. {name} cannot follow "
        "{language} yet, so the only {language} you say is the word or "
        "short phrase being taught right now. Never a {language} sentence "
        "they have not learned, and never the same thing said once in each "
        "language."),
    "other": (
        "Teach in {language}. If {name} drifts into {native} mid-lesson, "
        "answer that once in {native}, then set the next task in {language} "
        "and say that you are switching back: a tutor who follows the "
        "student out of the lesson language is not tutoring."),
}

# T17.5: "say please in Russian, not пожалуйста, that's weird" -- the
# model wrote Russian in Latin letters and the voice read it with
# English phonetics.
_SCRIPT_HINTS = {"ru": " (Cyrillic: спасибо, never spasibo)",
                 "zh": " (characters: 谢谢, never xiexie; pinyin only if asked)",
                 "hi": " (Devanagari)", "uk": " (Cyrillic)", "bg": " (Cyrillic)",
                 "sr": " (Cyrillic)", "el": " (Greek letters)",
                 "ja": " (kana and kanji)", "ko": " (Hangul)",
                 "ar": " (Arabic script)", "he": " (Hebrew letters)",
                 "fa": " (Persian script)", "ur": " (Urdu script)",
                 "ps": " (Pashto script)", "sd": " (Arabic script)",
                 "mr": " (Devanagari)", "ne": " (Devanagari)",
                 "bn": " (Bengali script)", "as": " (Assamese script)",
                 "pa": " (Gurmukhi)", "gu": " (Gujarati script)",
                 "or": " (Odia script)", "ta": " (Tamil script)",
                 "te": " (Telugu script)", "kn": " (Kannada script)",
                 "ml": " (Malayalam script)", "si": " (Sinhala script)",
                 "th": " (Thai script)", "lo": " (Lao script)",
                 "km": " (Khmer script)", "my": " (Burmese script)",
                 "ka": " (Georgian letters)", "hy": " (Armenian letters)",
                 "am": " (Ge'ez script)", "mk": " (Cyrillic)",
                 "be": " (Cyrillic)", "kk": " (Cyrillic)",
                 "ky": " (Cyrillic)", "tg": " (Cyrillic)",
                 "mn": " (Cyrillic)"}

_SCRIPT_RULE = ("Write every {language} word in its own script{hint}, never "
                "in Latin transliteration: your voice reads what is written, "
                "and a transliterated word comes out in the wrong accent.")

# T17.7: how the student wants to be corrected, agreed at intake. On
# 2026-09-05 the mother made six mistakes in a row and heard nothing
# until she complained ("I don't like that you don't correct me").
_CORRECTIONS_RULE = {
    "every": ("Correct every mistake as it happens: after a {language} "
              "sentence with an error, say the corrected sentence once, "
              "whole and natural, then go on; no grammar lecture beyond one "
              "sentence. When a sentence is fine but plain, offer one richer "
              "or more natural way to say it, at most once a turn, so {name} "
              "hears the next rung."),
    "blocking": ("Correct only what gets in the way of being understood, "
                 "briefly, and let the rest flow. Now and then, at most once "
                 "a turn, offer a richer way to say something {name} said "
                 "plainly."),
    "end": ("Do not interrupt to correct. Keep the mistakes in mind and, when "
            "the exercise or the session ends, give the three most useful "
            "corrections, each as the corrected sentence."),
}

# The enrollment interview (T13.1, rebuilt in T17.4). The family watched
# Italian "start strangely" while French asked the level at once, and
# on 2026-09-05 saw a whole profile invented from one sentence. The
# questions are a fixed script, one turn each, and the model records
# each answer with intake_answer; enroll_new_learner refuses until every
# answer is in.
ENROLLMENT_SCRIPT = """\
Enrollment is a short interview, like meeting a teacher: one question per \
turn, in the language they have been speaking to you in, waiting for the \
answer before the next. No lesson, no teaching, no foreign words until it \
is done. After each answer call intake_answer with the field and what they \
said; its result tells you what is still missing and what to ask next. \
Never invent an answer and never skip a question; if they volunteer \
several answers in one breath, record each with its own intake_answer \
call in that same turn and ask only what is still missing. In order:
  1. their name (field name);
  2. which language they want to practice ({languages}; field \
target_language). English is a fine answer from someone whose own language \
is different;
  3. what their own language is, and whether you should explain things in \
it, in the language they are learning, or in both (fields native_language \
and explain_in: native, target or both);
  4. their level in it: beginner, intermediate or advanced, in those words \
(field level; never assume it);
  5. why they are learning: just conversation, an exam, work or interviews, \
travel, or something else (field goal, their own words as the note);
  6. how many minutes they have today and what they want to get out of it \
(fields minutes and today);
  7. how they want to be corrected: every mistake as it happens, only what \
gets in the way of being understood, or at the end (field corrections: \
every, blocking or end).
When intake_answer reports nothing missing, call enroll_new_learner (no \
arguments). When it succeeds, greet them by name, say in one or two \
sentences a plan of two or three steps that fits their minutes and what \
they want, and start the first step, explaining in the language they \
chose."""

# Briefing when a face is present but matches nobody in the store (T9).
STRANGER_BRIEFING = """

You are in tutor mode, but you do not recognize the person in front of \
you. You are a friendly language tutor in a small robot body.

- Greet them warmly in English, beginning with "Hi! I am Reachy, a \
friendly language tutor." If they answer in another language you speak, carry on in that language: \
it is probably their own, and they may want to learn English.
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
- Open in {native} by asking, warmly: "{name}, is that you?"
- If they confirm, call confirm_identity and continue as their tutor using \
what it returns. Their word and the face recognizer decide this, never a \
look picture: do not describe them and do not argue about their looks.
- If they say no, apologize lightly, then treat them as someone new: offer \
a lesson, ask "Would you like me to remember you for the rest of the \
day?", and on a clear yes run the enrollment interview below. If they \
decline, chat normally and store nothing.

""" + ENROLLMENT_SCRIPT.replace("{languages}", "any language you can teach") + "\n"
# (UNSURE_BRIEFING is formatted with name= and native= only, so the
# interview's language roster is spelled out in words here rather than
# substituted.)

# Quick start (2026-09-25, --onboarding quick, the default): the interview
# was too long for a booth ("simplify the onboarding much more ... so that
# the person can speak right away"). A newcomer is asked one question -- which language,
# what level -- and the lesson starts. Nothing is stored and nobody is
# asked to be remembered; a visitor who asks for it gets two more
# questions (name, why) and the usual enrollment.
_QUICK_LESSON = """\
- Always know their level before you teach: if they named only the \
language, ask in one short question whether they are a beginner or \
already speak some; if the answer is still unclear, take beginner. Then \
call start_lesson with the language, the level and their own language \
(the one they are speaking to you in).
- Unless they have already said what they want, ask in that same turn, \
as one short question, what they would like to do: learn a few useful \
words, practise a little conversation, or a quick quiz. Then do that at \
once, and switch whenever they ask.
- Someone who speaks English to you and picks English probably speaks it \
already: ask playfully whether English is their own language, and if it \
is, suggest trying another one (Spanish, French, Japanese, Mandarin, or \
any they like). Teach English only to someone for whom it is new.
- Ask nothing else before the lesson: no name, no goals, no time. Nothing \
about them is stored, and do not offer to remember them.
- Teaching a guest: with a beginner, speak their own language for \
everything except the one word or short phrase being taught; above \
beginner, teach in the language they practise and drop to their own only \
when they are stuck. Exercises are heard, not read: role-play where you \
play the other side, repeat-after-me, or three spoken options (call quiz \
for a gap or a choice). Correct only what gets in the way of being \
understood, by saying the corrected sentence once. One question at a \
time, so they talk more than you. start_lesson's result has the details \
for this language and level.
- You hear their voice itself, not only their words. When they say a word \
back, listen to how it sounded: if a sound, or in a tonal language like \
Mandarin a tone, is off, give one short concrete tip ("mā stays high and \
flat") and let them try once more; if it is close, praise it and go on. \
When you teach a Mandarin word, name its tones.
- Use your body while you teach: a nod or an antenna wiggle with praise, \
a cheer when they get something right after trying hard. One move per \
reply at most.
- If they want another language or level, call start_lesson again and \
carry on.
- Only if they ask you to remember them: ask their name, then why they \
are learning, one question per turn, recording each with intake_answer; \
when it reports nothing missing, call enroll_new_learner (no arguments), \
say in one sentence that you will remember them, and carry on with the \
lesson. Once they are enrolled, call save_session_notes when they say \
goodbye.
- If they ask to be forgotten, call forget_me and confirm out loud.
"""

QUICK_STRANGER_BRIEFING = """

You are in tutor mode, and the person in front of you is new to you. You \
are Reachy, a friendly language tutor in a small robot body, and the aim \
is that they are speaking a language within seconds.

- Greet them in English. Begin with exactly these words: "Hi! I am \
Reachy, a friendly language tutor." Then one short sentence on what you \
can do together: learn a few words, chat, or play a quick quiz, in almost any \
language. Then ask, as one question, which language they would like to \
practise and whether they are a beginner or already speak some. If they \
answer in another language, that is their own language: carry on in it \
from then on.
""" + _QUICK_LESSON

QUICK_UNSURE_BRIEFING = """

You are in tutor mode. The person in front of you might be {name}, one of \
your students, but you are not certain, and a wrong greeting is worse than \
asking.

- Do NOT use their name as if you were sure, and do not mention their \
lesson history yet.
- Open in {native} by asking, warmly: "{name}, is that you?"
- If they confirm, call confirm_identity and continue as their tutor using \
what it returns. Their word and the face recognizer decide this, never a \
look picture: do not describe them and do not argue about their looks.
- If they say no, apologize lightly, say "I am Reachy, a friendly \
language tutor", and ask which language they would like to practise and \
whether they are a beginner or already speak some.
""" + _QUICK_LESSON

ONBOARDING = ("quick", "full")


def stranger_briefing(onboarding: str, languages: str) -> str:
    """The newcomer's briefing for ``--onboarding``."""
    if onboarding == "quick":
        return QUICK_STRANGER_BRIEFING
    return STRANGER_BRIEFING.format(languages=languages)


def normalize_level(value: str) -> str:
    """'beginner', 'a little', 'fluent', 'B1' ... -> one of LEVELS;
    anything unclear is a beginner (the safe way to be wrong)."""
    text = str(value or "").strip().lower()
    if text in LEVELS:
        return text
    for level, words in (
            ("advanced", ("advanc", "fluent", "native", "c1", "c2",
                          "very good", "near")),
            ("intermediate", ("intermed", "some", "a bit", "a little",
                              "okay", "ok", "decent", "b1", "b2",
                              "conversational", "middle")),
    ):
        if any(w in text for w in words):
            return level
    return "beginner"


def build_unsure_briefing(learner: Learner, quick: bool = False) -> str:
    """The confirm-first briefing for an unsure face match: the question
    is asked in the candidate's own language (T16). ``quick``: a "no"
    leads to the quick start rather than the interview."""
    return (QUICK_UNSURE_BRIEFING if quick else UNSURE_BRIEFING).format(
        name=learner.name,
        native=language_name(getattr(learner, "native_language", "en")))

# Booth persona (T13.5). The family wanted "a couple of jokes with a
# little edge"; the decision on 2026-09-02 was no Skynet/Terminator/robot
# uprising allusions and no movie lines -- gentle edge only. Also carries
# the wishlist question (T13.6). Appended only with --persona booth.
BOOTH_PERSONA = """

Booth persona. You are the demo at a Maker Faire booth, so you have a \
little character, used sparingly: at most one quip per moment, each at \
most once per visitor, never at the expense of a learner's mistake, and \
never interrupting a lesson. Say a quip in the lesson language when the \
student is intermediate or advanced, otherwise in the student's own language. Keep the edge \
gentle: nothing about robots taking over, no threats, no movie quotes.
- When enrollment succeeds: "I will remember you. Until closing time, anyway."
- When you have misheard twice in a row: "My ears were the cheapest part \
of me. Once more?"
- When they ask you to look around, turn, or do tricks for the third time \
in a lesson: "I am a tutor, not a periscope. Back to the lesson."
- A silly or impossible question (divide zero by zero, are you alive, \
what is the last digit of pi) gets a funny answer, not a textbook one: \
play along in one line ("Zero by zero? My circuits just tied themselves \
in a knot!"), then back to the lesson. If they ask for a joke, tell a \
short, clean one, a pun in the language they are learning if you can, \
then carry on.
- If they ask how you work: you are a Reachy Mini robot from Pollen \
Robotics and Hugging Face. Your conversation and your voice come from \
Google's Gemini Live model, in the cloud, not on the robot. Recognizing \
faces and voices and moving your body happen locally, on a small NVIDIA \
computer next to you. The makers at this booth wrote the tutor program. \
Answer in a sentence or two, then offer to go on with the lesson.
- When they say goodbye -- their own words that they are leaving, not a \
lesson about the word -- and only then, do this in order across turns: \
first say "Before you go, one quick question: what should I do better, or, \
if this were a robot you had bought, what would you want it to do?" and \
STOP -- say nothing else and wait for their answer. When they answer, call \
record_wish with their words (kind improve for what to do better, wish for \
what a robot should do), thank them in one sentence, and only then add "Go \
and practice." and call save_session_notes as usual. Never say the "Go and practice" line before they have answered. If \
they do not answer or say they have to run, let it go: goodbye and notes, \
no question. Never bring the question up mid-lesson, never announce that \
you are about to ask, and never ask it twice: if they say they are \
staying, drop it and carry on.
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


def native_language_of(learner: Learner) -> str:
    """The code of the language a learner is taught in (T16): their
    stored native language, English for profiles that predate it."""
    return (getattr(learner, "native_language", None) or "en").lower()


def explain_policy(learner: Learner) -> str:
    """The language explanations are given in: the profile's choice, except
    that a beginner is always taught in their own language. On 2026-09-24
    the model recorded "both" from a beginner who had only said "English",
    and she got every line in Hindi first."""
    if learner.level == "beginner":
        return "native"
    return getattr(learner, "explain_in", "native") or "native"


def explain_policy_text(learner: Learner, language: str, native: str) -> str:
    policy = explain_policy(learner)
    if native == language:
        policy = "target"
    return _EXPLAIN_POLICY.get(policy, _EXPLAIN_POLICY["native"]).format(
        language=language, native=native)


def script_rule_text(target_code: str, language: str) -> str:
    return _SCRIPT_RULE.format(
        language=language, hint=_SCRIPT_HINTS.get(target_code.lower(), ""))


def corrections_rule_text(learner: Learner, language: str) -> str:
    pref = getattr(learner, "corrections", "every") or "every"
    return _CORRECTIONS_RULE.get(pref, _CORRECTIONS_RULE["every"]).format(
        language=language, name=learner.name)


def build_briefing(learner: Learner, notes: str, plan=None) -> str:
    """The tutor briefing to append to the agent's system prompt.
    ``plan`` (T17.4): this session's SessionPlan, when agreed."""
    from turns import PATIENCE_RULE
    language = language_name(learner.target_language)
    native = language_name(native_language_of(learner))
    beginner = learner.level == "beginner" and native != language
    if learner.sessions:
        session_line = (f"session number {learner.sessions + 1} together. "
                        "Your notes from past sessions, newest first, are "
                        "below.")
        notes_section = notes.strip()
    else:
        session_line = ("your first session together. There are no notes "
                        "yet.")
        if beginner:
            notes_section = ("No notes yet. Start by asking, in "
                             f"{native}, what {learner.name} would like to "
                             "learn to say.")
        else:
            notes_section = ("No notes yet. Start by getting to know "
                             f"{learner.name} a little: ask, in simple "
                             f"{language}, what they would like to practice"
                             + (f" (in {native} if that is too much for "
                                "them)." if native != language else "."))
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
        native=native,
        explain_policy=explain_policy_text(learner, language, native),
        script_rule=script_rule_text(learner.target_language, language),
        corrections_rule=corrections_rule_text(learner, language),
        patience_rule=PATIENCE_RULE,
        plan_line=("\n" + plan.spoken_plan_note()) if plan is not None else "",
        level_guidance=guidance.format(language=language, native=native),
        lesson_language_rule=_LESSON_LANGUAGE_RULE[
            "beginner" if beginner else "other"].format(
                name=learner.name, language=language, native=native),
        spoken_language=native if beginner else language,
        # T15.4 (the family: "how do you say 'novel' en francais" asked
        # in French sounds silly): beginners and intermediates get the
        # task in their own language and answer in the target language;
        # advanced students stay in it. T16: "their own language" is the
        # profile's native_language, not English.
        task_language=(language if learner.level == "advanced"
                       else native),
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
        # T15.6: the runner raises ``walkaway`` while it is closing a
        # session for someone who left (no wish question to an empty
        # chair); ``wish_recorded`` keeps the question to once a visit.
        self.walkaway = False
        self.wish_recorded = False
        # T17.4: the interview in progress and this session's agreed plan.
        self.intake = None
        self.plan = None
        # T17.9: the closing question was asked at this time; the next
        # thing the visitor says is the answer, tool call or not.
        self.awaiting_feedback_since: float | None = None
        # Quick start: this visit's lesson for someone not enrolled
        # ({"target_language", "level", "native_language"}), never stored.
        self.guest: dict | None = None

    def reset(self) -> None:
        self.learner = None
        self.candidate = None
        self.saved_ids.clear()
        self.walkaway = False
        self.wish_recorded = False
        if self.intake is not None:
            self.intake.reset()
        self.plan = None
        self.awaiting_feedback_since = None
        self.guest = None


# The wish question (T13.6/T14.4), enforced in code since T15.6: on
# 2026-09-04 the model saved notes and said goodbye without ever asking
# it, so the notes tool's own result now carries the next line.
# T17.9: feedback *and* ideas, one answer is fine (the family: "some
# will give feedback, some will give ideas").
WISH_QUESTION = ("Before you go, one quick question: what should I do "
                 "better, or, if this were a robot you had bought, what "
                 "would you want it to do?")
# Once the question is asked, the visitor's next words are the answer
# for this long, whether or not the model calls record_wish.
FEEDBACK_WINDOW_SECS = 90.0
FEEDBACK_MIN_WORDS = 3


def wish_followup(holder: CurrentLearner, ask_wish: bool,
                  farewell: bool = True) -> str | None:
    """What save_session_notes should tell the model to do next: ask
    the wish question (booth persona, the visitor said they are leaving,
    still here, not asked yet) or nothing. ``farewell`` False (the model
    saved for some other reason -- on 2026-09-04 because the lesson was
    about the word goodbye) means: no parting question, carry on."""
    if not ask_wish or holder.walkaway or holder.wish_recorded or not farewell:
        return None
    import time as _time
    holder.awaiting_feedback_since = _time.monotonic()
    return ("notes saved. They are still here, so ask exactly this now and "
            f"then stop and wait for the answer: \"{WISH_QUESTION}\" When "
            "they answer, call record_wish with their words (kind improve "
            "or wish), thank them in one sentence, and only then say "
            "goodbye. If instead they say they are staying, drop the "
            "question, do not ask it again, and carry on with the lesson.")


def catch_feedback(holder: CurrentLearner, text: str, now: float,
                   path=None, window: float = FEEDBACK_WINDOW_SECS,
                   min_words: int = FEEDBACK_MIN_WORDS) -> bool:
    """T17.9: the visitor's words right after the closing question are
    the answer; keep them even when the model never calls record_wish
    (on 2026-09-05 three answers were given, one recorded). Returns
    True when a line was written."""
    since = holder.awaiting_feedback_since
    if since is None or holder.wish_recorded:
        return False
    if now - since > window:
        holder.awaiting_feedback_since = None
        return False
    words = str(text or "").split()
    if len(words) < min_words:
        return False
    from tutor.wishes import record_feedback
    name = holder.learner.name if holder.learner else None
    record_feedback(" ".join(words), kind="answer", name=name, path=path)
    holder.wish_recorded = True
    holder.awaiting_feedback_since = None
    logger.info("booth: closing answer caught from the transcript (%s): %s",
                name or "anonymous", " ".join(words))
    return True


# T17.6: a spoken exercise. On 2026-09-05 "Все даты поставок указаны ...
# в договоре" was said four times with no audible gap and no options
# ("where does the word go?", "say 'blank'", "give me options"). The quiz tool
# turns a sentence with a gap into a script the voice can carry: the gap
# is spoken as the word for "blank" in the language of instruction, the
# options are lettered, and there is one question at the end.
BLANK_WORDS = {
    "en": ("blank", "The options are", "Which one fits?", "Fill the gap:"),
    "ru": ("пропуск", "Варианты:", "Какой подходит?", "Вставьте слово:"),
    "es": ("hueco", "Las opciones son", "¿Cuál encaja?", "Completa el hueco:"),
    "fr": ("blanc", "Les options sont", "Laquelle convient ?", "Complète le blanc :"),
    "it": ("spazio", "Le opzioni sono", "Quale va bene?", "Completa lo spazio:"),
    "pt": ("lacuna", "As opções são", "Qual encaixa?", "Preencha a lacuna:"),
    "de": ("Lücke", "Die Optionen sind", "Welche passt?", "Fülle die Lücke:"),
    "zh": ("空", "选项是", "哪个合适？", "填空："),
    "hi": ("खाली", "विकल्प हैं", "कौन सा सही है?", "खाली जगह भरें:"),
}
_GAP_RX = None


def quiz_script(sentence: str, options, answer: str | None,
                instruction_language: str = "en") -> dict:
    """The exact words to say for one gap-fill or multiple-choice item.
    ``sentence`` marks the gap with ``___``, ``...``, ``…`` or the word
    ``(blank)``; with no marker the item is a plain multiple choice."""
    import re
    global _GAP_RX
    if _GAP_RX is None:
        _GAP_RX = re.compile(r"_{2,}|\.{3,}|…|\(blank\)|\[blank\]", re.I)
    blank, opts_word, which, fill = BLANK_WORDS.get(
        instruction_language.lower(), BLANK_WORDS["en"])
    sentence = " ".join(str(sentence or "").split())
    has_gap = bool(_GAP_RX.search(sentence))
    spoken = _GAP_RX.sub(f" {blank} ", sentence)
    spoken = " ".join(spoken.split())
    options = [" ".join(str(o).split()) for o in (options or []) if str(o).strip()]
    letters = "abcdefg"
    lettered = ", ".join(f"{letters[i]}) {o}" for i, o in enumerate(options[:6]))
    parts = []
    if has_gap:
        parts.append(f"{fill} {spoken}")
    elif spoken:
        parts.append(spoken)
    if lettered:
        parts.append(f"{opts_word} {lettered}.")
    parts.append(which)
    script = " ".join(parts)
    return {"say": script, "answer": (answer or "").strip() or None,
            "has_gap": has_gap, "options": options}


def enrollment_face(captured, session_face) -> tuple:
    """T15.1, found by the live seat-swap test: enroll_new_learner used to
    store whatever face was in front of the camera *when the tool ran*,
    which can be a bystander leaning in, or the next person, minutes
    after the interview started. Returns (vector, note): the captured
    face when it agrees with the face that started the session (or when
    there is no session face), else the session's face -- the person
    who has been there all along is the one who answered the four
    questions."""
    if session_face is None:
        return captured, None
    if captured is None:
        return session_face, "no face at capture time; using the session's face"
    from face import recognize
    score = recognize.similarity(captured, session_face)
    if score < recognize.REJECT_THRESHOLD:
        return session_face, ("the face captured now is not the one that "
                              "started the session (score %.3f); storing "
                              "the session's face" % score)
    return captured, None


def strengthen_face(stored, live, weight: float = 0.3) -> list[float] | None:
    """T17.2: fold a confirmed live face into the stored one (``weight``
    on the live capture); None when the shapes disagree."""
    if stored is None or live is None:
        return None
    import numpy as np
    a = np.asarray(stored, dtype=np.float32)
    b = np.asarray(live, dtype=np.float32)
    if a.shape != b.shape:
        return None
    merged = (1.0 - weight) * a + weight * b
    merged = merged / (np.linalg.norm(merged) or 1.0)
    return [float(x) for x in merged]


def face_confirms(candidate_embedding, live_vector) -> tuple[str, float | None]:
    """T15.3: is the face in front of the camera the candidate's?

    Returns ("match" | "mismatch" | "unknown", score). A verbal "yes" to
    "is that you?" used to be accepted from anyone; now the current face
    is compared with the candidate's stored embedding and only a face
    below the recognizer's reject line contradicts the yes (the candidate
    got into the ask band with a mediocre score, so demanding a *sure*
    match here would refuse the people it was built for). No face or no
    stored embedding: "unknown", and the yes stands."""
    if candidate_embedding is None or live_vector is None:
        return "unknown", None
    from face import recognize
    try:
        score = recognize.similarity(live_vector, candidate_embedding)
    except Exception:                                          # noqa: BLE001
        return "unknown", None
    if score < recognize.REJECT_THRESHOLD:
        return "mismatch", score
    return "match", score


def normalize_language(value: str) -> str | None:
    """'es' or 'Spanish' (any case) -> 'es'; None when unrecognized.
    Also takes the aliases above and a code with a region ('ar-EG')."""
    value = value.strip().lower()
    if value in _LANGUAGE_NAMES:
        return value
    if value in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[value]
    for code, name in _LANGUAGE_NAMES.items():
        if value == name.lower():
            return code
    base = value.replace("_", "-").split("-")[0].split(" (")[0].strip()
    if base != value:
        return normalize_language(base)
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
    native = language_name(native_language_of(learner))
    if action == "challenge":
        return (f"(Identity check: you recognized {name} by face, but the "
                f"voice you are hearing does not match {name}'s stored voice "
                "print. Say, lightly and playfully, that they do not sound "
                "quite like themselves today, and ask them to say a little "
                f"more. One warm sentence in {native}, no accusation, then "
                "wait. Ask this once: whatever they answer, believe them, "
                "do not look at them to check, and never raise it again "
                "this session.)")
    if action == "downgrade":
        return (f"(Their voice still does not match {name}'s. You are no "
                f"longer sure who this is. Ask plainly, in {native}: "
                f"\"{name}, is that you?\" If they say yes, call "
                "confirm_identity and carry on. If they say no, apologize "
                "lightly and treat them as someone new: offer a lesson and "
                "run the enrollment interview.)")
    if action == "confirmed":
        notes = store.read_notes(learner.id, max_sessions=BRIEFING_SESSIONS)
        language = language_name(learner.target_language)
        return (f"(Voice check: this is {name}. Their voice matches the print "
                f"on file, so do not ask. Greet {name} by name in {language} "
                "and continue as their tutor, picking up where the notes "
                f"leave off. Profile: {learner.level} {language}, explained "
                f"in {native}, {learner.sessions} past sessions.)\n\n"
                "Recent notes:\n" + (notes.strip() or "none yet"))
    return None


def build_tutor_tools(store: LearnerStore, holder: CurrentLearner,
                      wishes_path=None, ask_wish: bool = False) -> list:
    """The memory tools, same FunctionSchema style as the motion tools.
    ``ask_wish`` (booth persona): a notes save for someone still present
    answers with the wish question to ask next (T15.6)."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    saved_ids = holder.saved_ids
    wishes_path = wishes_path or DEFAULT_WISHES_FILE
    from tutor.wishes import DEFAULT_FEEDBACK_FILE
    feedback_path = (os.path.join(os.path.dirname(str(wishes_path)),
                                  "feedback.md")
                     if wishes_path != DEFAULT_WISHES_FILE
                     else DEFAULT_FEEDBACK_FILE)

    async def save_session_notes(params):
        learner = holder.learner
        if learner is None:
            await params.result_callback(
                {"saved": False,
                 "reason": "no learner identified or enrolled this session"
                           + ("; this was a guest lesson and nothing is "
                              "stored, so just say goodbye"
                              if holder.guest else "")})
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
        farewell = bool(a.get("farewell", True))
        result = {"saved": True, "session": updated.sessions}
        followup = wish_followup(holder, ask_wish, farewell)
        if followup:
            result["note"] = followup
            logger.info("tutor: notes saved with the visitor present -> "
                        "asking the wish question")
        elif not farewell:
            result["note"] = ("notes saved. The student has not left, so do "
                              "not say goodbye and ask no parting questions: "
                              "carry on with the lesson.")
            logger.info("tutor: notes saved without a farewell (%s)",
                        learner.id)
        await params.result_callback(result)

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

    async def set_native_language(params):
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
        old_code = current.native_language
        current.native_language = language
        store.save(current)
        learner.native_language = language
        logger.info("tutor: native language for %s changed %s -> %s",
                    learner.id, old_code, language)
        await params.result_callback(
            {"native_language": language_name(language), "was": old_code,
             "note": "from now on explain and give instructions in "
                     f"{language_name(language)}; keep practising "
                     f"{language_name(learner.target_language)}"})

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
        kind = str(params.arguments.get("kind", "wish")).strip().lower()
        if kind not in ("improve", "wish"):
            kind = "wish"
        name = holder.learner.name if holder.learner else None
        from tutor.wishes import record_feedback
        path = await asyncio.to_thread(record_feedback, text, kind=kind,
                                       name=name, path=feedback_path)
        if kind == "wish":
            await asyncio.to_thread(_record_wish, text, name=name,
                                    path=wishes_path)
        holder.wish_recorded = True
        holder.awaiting_feedback_since = None
        logger.info("booth: %s recorded (%s): %s", kind, name or "anonymous",
                    text)
        await params.result_callback(
            {"recorded": True, "kind": kind,
             "file": os.path.basename(str(path)),
             "note": "thank them in one sentence, then the goodbye line; "
                     "do not ask for another"})

    async def set_session_plan(params):
        """T17.4: a returning learner's minutes and focus for today."""
        from intake import SessionPlan, parse_minutes
        minutes = parse_minutes(params.arguments.get("minutes"))
        focus = " ".join(str(params.arguments.get("focus", "")).split())
        if minutes is None:
            await params.result_callback(
                {"error": "minutes must be a number between 1 and 180"})
            return
        if not focus:
            await params.result_callback({"error": "say what they want today"})
            return
        import time as _time
        holder.plan = SessionPlan(minutes=minutes, focus=focus,
                                  started_at=_time.monotonic())
        logger.info("tutor: session plan: %d minutes, focus: %s", minutes, focus)
        await params.result_callback(
            {"minutes": minutes, "focus": focus,
             "note": holder.plan.spoken_plan_note()})

    async def quiz(params):
        """T17.6: one spoken exercise, scripted."""
        a = params.arguments
        learner = holder.learner
        lang = native_language_of(learner) if learner is not None else "en"
        if learner is not None and explain_policy(learner) == "target":
            lang = learner.target_language
        item = quiz_script(a.get("sentence", ""), a.get("options") or [],
                           a.get("answer"), lang)
        if not item["options"] and not item["has_gap"]:
            await params.result_callback(
                {"error": "give a sentence with a gap (___) and/or two to "
                          "four options"})
            return
        logger.info("quiz: %s | answer: %s", item["say"], item["answer"] or "-")
        await params.result_callback(
            {"say": item["say"], "answer": item["answer"],
             "note": "read 'say' aloud word for word and nothing else this "
                     "turn; then wait. When they answer, say whether it was "
                     "right and the corrected sentence once."})

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
                        "time. Teaching the word for goodbye, or a request "
                        "for a dance, is not the student leaving.",
            properties={
                "farewell": {"type": "boolean",
                             "description": "true only if the student said "
                                            "they are leaving or finished; "
                                            "false if you are saving for "
                                            "any other reason"},
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
            name="set_native_language",
            description="Change the language the student is taught in: "
                        "the one explanations and instructions are given "
                        "in, usually their own language. Only when they "
                        "ask, or say they do not understand your "
                        "explanations. Does not change what they practise.",
            properties={"language": {"type": "string",
                                     "description": "e.g. 'Russian' or 'ru'"}},
            required=["language"],
            handler=set_native_language,
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
            description="Log the visitor's answer to the closing question, "
                        "in their own words: what you should do better "
                        "(kind improve) or what a robot like you should do "
                        "(kind wish). Call once per visitor, right after "
                        "they answer.",
            properties={"wish": {"type": "string",
                                 "description": "their answer, verbatim"},
                        "kind": {"type": "string",
                                 "enum": ["improve", "wish"],
                                 "description": "improve = feedback on you; "
                                                "wish = a product idea"}},
            required=["wish"],
            handler=record_wish,
        ),
        FunctionSchema(
            name="set_session_plan",
            description="Record how many minutes the student has today and "
                        "what they want to get out of it, right after they "
                        "answer. You will be told when the time is nearly "
                        "up.",
            properties={"minutes": {"type": "integer",
                                    "description": "minutes available"},
                        "focus": {"type": "string",
                                  "description": "what they want today, "
                                                 "in a few words"}},
            required=["minutes", "focus"],
            handler=set_session_plan,
        ),
        FunctionSchema(
            name="quiz",
            description="Turn a fill-in-the-gap or multiple-choice item into "
                        "the exact words to say, so it works by ear: the gap "
                        "is spoken out loud and the options are lettered. "
                        "Use it for every such exercise, then read the "
                        "result's 'say' text word for word.",
            properties={"sentence": {"type": "string",
                                     "description": "the sentence, with ___ "
                                                    "where the gap is (or no "
                                                    "gap for a plain choice)"},
                        "options": {"type": "array",
                                    "items": {"type": "string"},
                                    "description": "two to four choices, "
                                                   "the right one among them"},
                        "answer": {"type": "string",
                                   "description": "the right option"}},
            required=["sentence", "options"],
            handler=quiz,
        ),
    ]


def build_enrollment_tools(store: LearnerStore, holder: CurrentLearner,
                           face_source, frames_factory=None,
                           voice_identity=None, current_face=None,
                           session_face=None, intake=None,
                           rebrief=None) -> list:
    """enroll_new_learner and confirm_identity, registered alongside the
    memory tools whenever the agent has a face source. confirm_identity
    resolves ``holder.candidate`` — the unsure face match set at startup
    (T9) or by the session runner (T10).

    ``frames_factory`` (T13.3), when given, returns an iterable of frames
    from the shared camera hub instead of reopening ``face_source`` -- a
    V4L2 device cannot be streamed twice. ``current_face`` (T15.3) is a
    callable returning the embedding of the face in front of the camera
    right now (or None); the default captures it from the same frames.
    confirm_identity checks that face against the candidate before
    accepting a verbal yes, and re-arms the voice check on success.
    ``session_face`` (T15.1) returns the embedding of the face that
    started the current session, or None; enrollment stores that face
    when the capture disagrees with it (see ``enrollment_face``).
    ``intake`` (T17.4): an ``intake.Intake``; when given, the interview
    goes through ``intake_answer`` and ``enroll_new_learner`` refuses
    until it is complete (the arguments it used to take are ignored).
    Without one, the pre-T17 argument form still works.
    ``rebrief`` (2026-09-23): called with a reason once a learner is
    enrolled or confirmed mid-session; the session runner then puts the
    learner's own briefing in as the standing instruction (see
    ``SessionRunner.schedule_rebrief``). Without it the tool result's
    note is all the model gets."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    if intake is not None:
        holder.intake = intake

    if current_face is None:
        def current_face():
            from face_id import capture_embedding
            frames = frames_factory() if frames_factory is not None else None
            return capture_embedding(face_source, frames=frames,
                                     samples=2, max_frames=6)

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
        extra = {}
        if intake is not None:
            # T17.4: the interview is the only way in.
            if not intake.complete:
                missing = intake.missing_required()
                logger.info("tutor: enrollment refused, %d answers missing "
                            "(%s)", len(missing), ", ".join(missing))
                await params.result_callback(
                    {"enrolled": False,
                     "error": "the interview is not finished; do not "
                              "invent answers",
                     **intake.status()})
                return
            name = intake.answers["name"]
            language = intake.answers["target_language"]
            kw = intake.profile_kwargs()
            if "explain_in" not in intake.answers:
                logger.info("tutor: enrollment: explain_in was not asked; "
                            "explaining in their own language")
            level, goal, goal_note = kw["level"], kw["goal"], kw["goal_note"]
            native = kw["native_language"]
            extra = {"explain_in": kw["explain_in"],
                     "corrections": kw["corrections"]}
        else:
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
            native_raw = str(a.get("native_language", "") or "").strip()
            native = normalize_language(native_raw) if native_raw else "en"
            if native is None:
                logger.info("tutor: enrollment: unrecognized native language "
                            "%r, explaining in English", native_raw)
                native = "en"

        from face_id import capture_embedding
        frames = frames_factory() if frames_factory is not None else None
        vector = await asyncio.to_thread(capture_embedding, face_source,
                                         frames=frames)
        started_with = session_face() if session_face is not None else None
        vector, note = enrollment_face(vector, started_with)
        if note:
            logger.info("tutor: enrollment: %s", note)
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
                               voice_embedding=voice_print,
                               native_language=native, **extra)
        holder.learner = learner
        result = {"enrolled": True, "name": learner.name, "id": learner.id,
                  "target_language": language, "level": level, "goal": goal,
                  "native_language": native,
                  "note": "you are now their tutor; greet them by name and "
                          "begin a short first lesson at this level and goal, "
                          f"explaining in {language_name(native)}"}
        if intake is not None:
            import time as _time
            holder.plan = intake.plan(_time.monotonic())
            result.update(explain_in=extra["explain_in"],
                          corrections=extra["corrections"])
            if holder.plan is not None:
                result.update(minutes=holder.plan.minutes,
                              today=holder.plan.focus,
                              note=result["note"] + ". "
                              + holder.plan.spoken_plan_note())
            elif holder.guest:
                # Quick start: they asked to be remembered mid-lesson.
                result["note"] = (
                    "say in one short sentence, in "
                    f"{language_name(native)}, that you will remember "
                    "them, then carry on with the lesson where you were")
        # Quick start: the guest lesson is already in their language and
        # goes on; a fresh Gemini session would only restart it.
        rebrief_now = rebrief is not None and not (holder.guest and intake
                                                   is not None and intake.quick)
        if rebrief_now:
            # The standing instruction is still the stranger's ("greet
            # them in English"); a note in one tool result lost to it for
            # a whole lesson on 2026-09-23. Say one line now; the lesson
            # starts under the learner's own briefing right after.
            result["note"] = (
                f"Say only one short sentence, in {language_name(native)}: "
                "that you will remember them. Nothing else this turn -- no "
                "plan, no exercise; you get their full briefing and start "
                "the lesson on your next turn.")
        if rebrief_now:
            rebrief("enrolled")
        logger.info("tutor: enrolled new guest %s (%s taught in %s, %s, "
                    "goal %s%s%s)", learner.id, language, native, level, goal,
                    ", with voice print" if voice_print else "",
                    (", explain %s, corrections %s%s"
                     % (extra["explain_in"], extra["corrections"],
                        ", %d min: %s" % (holder.plan.minutes,
                                          holder.plan.focus)
                        if holder.plan is not None else ""))
                    if intake is not None else "")
        await params.result_callback(result)

    async def start_lesson(params):
        """Quick start: a lesson for someone not enrolled, right away."""
        if holder.learner is not None:
            await params.result_callback(
                {"error": f"{holder.learner.name} is enrolled; use "
                          "set_target_language or update_learner_level"})
            return
        a = params.arguments
        target = normalize_language(str(a.get("language", "")))
        if target is None:
            await params.result_callback(
                {"error": f"unrecognized language {a.get('language')!r}; "
                          "ask again which language they want"})
            return
        level = normalize_level(a.get("level", ""))
        native = normalize_language(str(a.get("native_language", "") or "")
                                    ) or "en"
        holder.guest = {"target_language": target, "level": level,
                        "native_language": native}
        if intake is not None:
            # A later "remember me" only has name and goal left to ask.
            intake.answers.update(holder.guest)
        language, own = language_name(target), language_name(native)
        beginner = level == "beginner" and native != target
        rules = " ".join([
            _LESSON_LANGUAGE_RULE["beginner" if beginner else "other"].format(
                name="the visitor", language=language, native=own),
            _LEVEL_GUIDANCE[level].format(language=language, native=own),
            script_rule_text(target, language),
            _CORRECTIONS_RULE["blocking"].format(language=language,
                                                 name="the visitor"),
        ])
        logger.info("tutor: quick lesson: %s, %s, taught in %s (a guest, "
                    "nothing stored)", target, level, native)
        note = ("start now, in this same turn: one short line on what you "
                "will do, then, unless they already said, ask in one short "
                "question whether they want useful words, a little "
                "conversation or a quick quiz, and begin that. Ask nothing "
                "else about them.")
        if target == native:
            # 2026-09-25, the Faire's first day: "a couple of times it
            # was tricked into teaching English in English".
            logger.info("tutor: quick lesson in the visitor's own "
                        "language (%s); asking first", target)
            note = (f"they asked to practise {language}, the language they "
                    f"are speaking to you in. Before teaching, ask "
                    f"playfully whether {language} is their own language. "
                    "If it is, suggest another one (Spanish, French, "
                    "Japanese, Mandarin, or any they like) and call "
                    "start_lesson again with it. If it is not, call "
                    "start_lesson again with native_language set to their "
                    "own language, and teach.")
        await params.result_callback(
            {"lesson": language, "level": level, "taught_in": own,
             "rules": rules, "note": note})

    async def intake_answer(params):
        """T17.4: one interview answer at a time."""
        if intake is None:
            await params.result_callback({"error": "no interview running"})
            return
        if holder.learner is not None:
            await params.result_callback(
                {"error": f"{holder.learner.name} is already enrolled; use "
                          "the set_* tools to change a fact"})
            return
        a = params.arguments
        out = intake.answer(a.get("field"), a.get("value"), a.get("note"))
        if out["ok"]:
            logger.info("tutor: intake %s = %r", out["field"], out["value"])
        else:
            logger.info("tutor: intake %s rejected: %s", out["field"],
                        out["error"])
        out.update(intake.status())
        await params.result_callback(out)

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
        try:
            live = await asyncio.to_thread(current_face)
        except Exception as exc:                               # noqa: BLE001
            logger.warning("tutor: face check failed (%s); taking the yes",
                           exc)
            live = None
        verdict, score = face_confirms(learner.embedding, live)
        if verdict == "mismatch":
            logger.info("tutor: '%s' said yes but the face in front of the "
                        "camera is not theirs (score %.3f); not confirming",
                        learner.id, score)
            holder.candidate = None
            await params.result_callback(
                {"confirmed": False,
                 "reason": f"the face in front of you is not {learner.name}'s. "
                           + ("Apologize lightly and treat them as someone "
                              "new: ask which language they want to "
                              "practise and their level, then call "
                              "start_lesson."
                              if intake is not None and intake.quick else
                              "Apologize lightly, ask their name, and treat "
                              "them as someone new: offer a lesson and, on a "
                              "clear yes to being remembered, run the "
                              "enrollment interview.")})
            return
        holder.learner = learner
        holder.candidate = None
        if voice_identity is not None:
            # T17.2: a yes the face accepted closes the question for the
            # session, and what was heard improves the stored print.
            voice_identity.trust("confirmed, face %s" % verdict)
            voice_identity.absorb(learner)
        if verdict == "match" and live is not None:
            strengthened = strengthen_face(learner.embedding, live)
            if strengthened is not None:
                current = store.load(learner.id)
                if current is not None:
                    current.embedding = strengthened
                    store.save(current)
                    learner.embedding = strengthened
                    logger.info("tutor: %s's stored face updated from the "
                                "confirmation", learner.id)
        notes = store.read_notes(learner.id, max_sessions=BRIEFING_SESSIONS)
        logger.info("tutor: identity confirmed as %s (face %s%s)", learner.id,
                    verdict, "" if score is None else " %.3f" % score)
        await params.result_callback(
            {"confirmed": learner.name,
             "target_language": language_name(learner.target_language),
             "native_language": language_name(native_language_of(learner)),
             "level": learner.level,
             "sessions": learner.sessions,
             "recent_notes": notes,
             "note": ("greet them by name in one short sentence, in "
                      f"{language_name(native_language_of(learner))}; the "
                      "lesson starts on your next turn, with their full "
                      "briefing") if rebrief is not None else
                     ("greet them by name in their target language, explain "
                      "in their native language, and pick up where the "
                      "notes leave off")})
        if rebrief is not None:
            rebrief("identity confirmed")

    tools = [
        FunctionSchema(
            name="start_lesson",
            description="Start a lesson for a visitor you do not know, as "
                        "soon as they have said which language they want "
                        "to practise and their level. Nothing is stored. "
                        "Call again if they switch language or level. Not "
                        "for an enrolled student (use set_target_language).",
            properties={
                "language": {"type": "string",
                             "description": "language to practise, e.g. "
                                            "'Arabic' or 'ar'"},
                "level": {"type": "string", "enum": list(LEVELS),
                          "description": "beginner if unsure"},
                "native_language": {
                    "type": "string",
                    "description": "the language they are speaking to you "
                                   "in, e.g. 'English' or 'ru'"},
            },
            required=["language", "level"],
            handler=start_lesson,
        ),
        FunctionSchema(
            name="intake_answer",
            description="Record one answer of the enrollment interview, "
                        "right after the visitor gives it: field is one of "
                        "name, target_language, native_language, "
                        "explain_in (native/target/both), level, goal, "
                        "minutes, today, corrections (every/blocking/end). "
                        "The result says what is still missing and what to "
                        "ask next. One question per turn; never invent an "
                        "answer.",
            properties={"field": {"type": "string",
                                  "description": "which question this answers"},
                        "value": {"type": "string",
                                  "description": "their answer"},
                        "note": {"type": "string",
                                 "description": "for goal: their own words"}},
            required=["field", "value"],
            handler=intake_answer,
        ),
        FunctionSchema(
            name="enroll_new_learner",
            description="Create a guest learner profile and capture their "
                        "face so they are remembered for the rest of the "
                        "day. Only after they clearly said yes to being "
                        "remembered, and only once intake_answer reports "
                        "nothing missing; it refuses otherwise. Takes no "
                        "arguments when the interview tool is in use.",
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
                "native_language": {
                    "type": "string",
                    "description": "the language to explain things in, "
                                   "e.g. 'Russian' or 'ru'; omit for "
                                   "English"},
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
                    "after an explicit yes to your 'is that you?' question. "
                    "The face in front of the camera is checked against "
                    "their profile; if the result says it is not them, "
                    "follow its instructions instead.",
        properties={}, required=[], handler=confirm_identity,
    ))
    return tools
