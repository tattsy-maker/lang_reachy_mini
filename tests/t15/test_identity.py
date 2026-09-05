"""T15: identity for the whole session (unit, no camera, no models).

The 2026-09-04 family session, as unit tests: the same face keeps a
session alive whatever the voice print says (T15.1/T15.2); a different
face held for ``swap_secs`` ends it and the newcomer gets their own;
a face gap forces a fresh look; a verbal "yes" is checked against the
face (T15.3); the briefing sets tasks in English below advanced
(T15.4); the wish question follows a notes save for someone still
present (T15.6); the collector stamps when the visitor stopped
speaking for the turn timer (T15.7).
"""

import asyncio
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voice"))

from session import ACTIVE, BOOTH_NOTE, WATCHING, SessionRunner  # noqa: E402
from tutor.store import LearnerStore  # noqa: E402
from tutor_mode import (  # noqa: E402
    WISH_QUESTION, CurrentLearner, build_briefing, enrollment_face,
    face_confirms, wish_followup,
)
from voiceid import VoiceCollector, VoiceIdentity  # noqa: E402


def unit(axis, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def blend(base, target, seed=1):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=base.shape).astype(np.float32)
    noise -= np.dot(noise, base) * base
    noise /= np.linalg.norm(noise)
    return target * base + np.sqrt(1 - target ** 2) * noise


class Face:
    """What recognize.analyze/detect hand the runner."""

    def __init__(self, embedding=None, bbox=(100, 100, 200, 200)):
        self.embedding = embedding
        self.bbox = bbox


class FakeContext:
    def __init__(self):
        self.history = []

    def set_messages(self, messages):
        self.history.append(messages)

    def system_text(self):
        return self.history[-1][0]["content"]


class Runner(SessionRunner):
    def __init__(self, **kw):
        super().__init__(task=None, languages="French", base_prompt="BASE.",
                         **kw)
        self.cues, self.went_neutral = [], 0

    async def _queue_user_turn(self, text):
        self.cues.append(text)

    async def _robot_neutral(self):
        self.went_neutral += 1


def make(tmp_path, **kw):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    kw.setdefault("save_wait_secs", 0.1)
    runner = Runner(source="unused", store=store, holder=holder,
                    context=FakeContext(), robot=None, **kw)
    return runner, store, holder


def observe(runner, face, now):
    asyncio.run(runner.observe(face, now, (480, 640)))


def start_with(runner, store, holder, vector, name="Ana", now=10.0):
    learner = store.create(name, "fr", embedding=[float(x) for x in vector])
    runner._recent_vectors = [vector]
    asyncio.run(runner.start_session(now=now))
    assert holder.learner.id == learner.id and runner.machine.state == ACTIVE
    return learner


# -- T15.1: the face is the arbiter ---------------------------------------------------

def test_the_same_face_overrules_the_voice_and_rearms_it(tmp_path):
    runner, store, holder = make(tmp_path)
    vid = VoiceIdentity(store, holder)
    runner.voice_identity = vid
    ana = unit(0)
    start_with(runner, store, holder, ana)
    observe(runner, Face(embedding=blend(ana, 0.8)), now=12.0)
    assert runner._same_face_at == 12.0
    vid.changed = True                      # what "speaker_changed" leaves
    asyncio.run(runner.speaker_changed(now=13.0))
    assert runner.machine.state == ACTIVE, "the voice ended a session the face vouched for"
    assert holder.learner is not None
    assert vid.changed is False, "the voice check was not re-armed"
    assert not any("save_session_notes" in c for c in runner.cues)


def test_a_different_face_for_swap_secs_ends_the_session_and_the_newcomer_starts(tmp_path):
    runner, store, holder = make(tmp_path, swap_secs=3.0, stable_secs=2.0)
    ana, ben = unit(0), unit(1)
    first = start_with(runner, store, holder, ana)
    store.create("Ben", "fr", embedding=[float(x) for x in ben])
    observe(runner, Face(embedding=blend(ana, 0.9)), now=11.0)
    for t in (12.0, 13.0, 14.0):
        observe(runner, Face(embedding=blend(ben, 0.9, seed=int(t))), now=t)
        assert runner.machine.state == ACTIVE, "too early: a glimpse is not a swap"
    observe(runner, Face(embedding=blend(ben, 0.9)), now=15.5)
    assert runner.machine.state == WATCHING, "the seat swap went unnoticed"
    assert any("save_session_notes" in c for c in runner.cues), \
        "the person who left got no notes"
    assert holder.learner is None and runner._session_face is None
    # the newcomer's own session, from a clean slate
    for t in (16.0, 17.0, 18.5):
        observe(runner, Face(embedding=blend(ben, 0.9, seed=int(t))), now=t)
    assert runner.machine.state == ACTIVE and holder.learner.id != first.id
    assert "Ben" in runner.context.system_text()


def test_a_bystander_glimpse_does_not_end_the_session(tmp_path):
    runner, store, holder = make(tmp_path, swap_secs=3.0)
    ana, other = unit(0), unit(2)
    start_with(runner, store, holder, ana)
    observe(runner, Face(embedding=blend(other, 0.9)), now=12.0)
    assert runner._other_since == 12.0
    observe(runner, Face(embedding=blend(ana, 0.9)), now=13.0)
    assert runner._other_since is None and runner.machine.state == ACTIVE
    assert runner._same_face_at == 13.0


def test_a_face_back_after_a_gap_is_checked_at_once(tmp_path):
    runner, store, holder = make(tmp_path, face_recheck_secs=2.0)
    ana = unit(0)
    start_with(runner, store, holder, ana)
    observe(runner, Face(embedding=blend(ana, 0.9)), now=11.0)
    assert not runner._recheck_due(12.0)
    observe(runner, None, now=12.0)
    observe(runner, None, now=13.0)
    observe(runner, Face(), now=14.0)          # detect-only frame, no identity
    assert runner._force_recheck and runner._recheck_due(14.0), \
        "somebody came back after a gap and nobody looked at who"


def test_voice_change_with_nobody_in_frame_still_ends_the_session(tmp_path):
    runner, store, holder = make(tmp_path)
    start_with(runner, store, holder, unit(0))
    for t in (20.0, 21.0):
        observe(runner, None, now=t)
    asyncio.run(runner.speaker_changed(now=22.0))
    assert runner.machine.state == WATCHING
    assert any("save_session_notes" in c for c in runner.cues)


def test_voice_change_with_an_unjudged_face_defers_to_the_face_check(tmp_path):
    runner, store, holder = make(tmp_path, face_vouch_secs=5.0)
    start_with(runner, store, holder, unit(0))
    observe(runner, Face(), now=11.0)          # a face, identity not yet run
    asyncio.run(runner.speaker_changed(now=20.0))   # the 10.0 vouch is stale
    assert runner.machine.state == ACTIVE and runner._force_recheck


def test_runner_prompts_carry_the_booth_note(tmp_path):
    runner, store, holder = make(tmp_path)
    assert "swap seats" in BOOTH_NOTE
    assert "swap seats" in runner.idle_prompt()
    start_with(runner, store, holder, unit(0))
    assert "swap seats" in runner.context.system_text()
    quiet = make(tmp_path / "q", booth_note=False)[0]
    assert "swap seats" not in quiet.idle_prompt()


# -- T15.2: the voice asks, never ends -------------------------------------------------

def test_speaker_change_needs_four_misses_in_a_row(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    voice = unit(0)
    holder.learner = store.create("A", "es", voice_embedding=[float(x) for x in voice])
    vid = VoiceIdentity(store, holder)
    assert vid.change_after == 4
    assert vid.on_sample(blend(voice, 0.9)) == "verified"
    for i in range(3):
        assert vid.on_sample(unit(3 + i)) is None
    assert vid.on_sample(unit(7)) == "speaker_changed"


def test_short_samples_feed_the_print_but_never_judge(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    voice = unit(0)
    holder.learner = store.create("A", "es", voice_embedding=[float(x) for x in voice])
    vid = VoiceIdentity(store, holder, decision_min_secs=3.0)
    assert vid.on_sample(unit(5), secs=1.8) is None, "a 1.8 s clip was judged"
    assert not vid.challenged and len(vid.samples) == 1
    assert vid.on_sample(unit(5), secs=None) == "challenge", "an injected file is judged"
    assert vid.on_sample(blend(voice, 0.9), secs=3.4) in ("verified", None)


def test_rearm_after_a_downgrade_listens_again(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    voice = unit(0)
    maria = store.create("Maria", "es", voice_embedding=[float(x) for x in voice])
    holder.learner = maria
    vid = VoiceIdentity(store, holder)
    assert vid.on_sample(unit(1)) == "challenge"
    assert vid.on_sample(unit(2)) == "downgrade"
    assert vid.on_sample(unit(2)) is None, "downgraded: deaf, as before"
    holder.learner, holder.candidate = maria, None   # confirm_identity did this
    assert vid.rearm() == "rearmed"
    assert vid.on_sample(unit(1)) == "challenge", "still deaf after the yes"


# -- T15.3: a yes is checked against the face --------------------------------------------

def test_face_confirms_match_mismatch_unknown():
    ana = unit(0)
    assert face_confirms(ana, blend(ana, 0.9))[0] == "match"
    assert face_confirms(ana, blend(ana, 0.3))[0] == "match", \
        "the ask band got them here; a mediocre score still confirms"
    assert face_confirms(ana, unit(1))[0] == "mismatch"
    assert face_confirms(ana, None) == ("unknown", None)
    assert face_confirms(None, ana) == ("unknown", None)


def test_enrollment_stores_the_face_that_started_the_session():
    """Found by the live seat-swap run: the tool captured Scott's face
    under Sunita's name because he had taken the seat by the time the
    interview ended."""
    ana, scott = unit(0), unit(1)
    assert enrollment_face(blend(ana, 0.9), ana)[1] is None
    vector, note = enrollment_face(scott, ana)
    assert vector is ana and "not the one that started" in note
    vector, note = enrollment_face(None, ana)
    assert vector is ana and "no face at capture" in note
    assert enrollment_face(scott, None) == (scott, None), "no session: as before"
    assert enrollment_face(None, None) == (None, None)


# -- T15.4: tasks in English below advanced -------------------------------------------------

def test_briefing_sets_tasks_in_english_below_advanced(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    for level, task_language in (("beginner", "English"),
                                 ("intermediate", "English"),
                                 ("advanced", "French")):
        learner = store.create(f"L{level}", "fr", level=level)
        text = build_briefing(learner, "")
        assert f"say what to express in {task_language}" in text, level
        assert "how do you say X" in text and "role-play" in text


def test_agent_prompt_no_longer_claims_it_cannot_see():
    source = (REPO / "voice" / "agent.py").read_text()
    assert "VISION_FACE_LOOK" in source and "Never say you cannot see" in source
    assert re.search(r"def vision_text\(args\)", source)


# -- T15.6: the wish question after a notes save ------------------------------------------

def test_wish_followup_only_when_present_and_unasked():
    holder = CurrentLearner()
    assert wish_followup(holder, ask_wish=False) is None
    note = wish_followup(holder, ask_wish=True)
    assert note and WISH_QUESTION in note and "record_wish" in note
    holder.wish_recorded = True
    assert wish_followup(holder, ask_wish=True) is None
    holder.reset()
    holder.walkaway = True
    assert wish_followup(holder, ask_wish=True) is None, "no question to an empty chair"
    holder.reset()
    assert holder.walkaway is False and wish_followup(holder, True)


def test_end_session_raises_the_walkaway_flag_while_closing(tmp_path):
    runner, store, holder = make(tmp_path)
    start_with(runner, store, holder, unit(0))
    seen = []
    orig = runner._queue_user_turn

    async def spy(text):
        seen.append(holder.walkaway)
        await orig(text)

    runner._queue_user_turn = spy
    asyncio.run(runner.end_session())
    assert seen == [True], "the notes cue went out without the walk-away flag"
    assert holder.walkaway is False, "the flag leaked into the next visitor"


# -- T15.7: when did the visitor stop speaking ----------------------------------------------

def test_collector_reports_when_the_visitor_stopped_speaking():
    stops = []
    collector = VoiceCollector(lambda v, s: None, silence_secs=0.5,
                               min_secs=1.5)
    collector.on_speech_end = stops.append
    rate = 16000
    rng = np.random.default_rng(0)
    loud = (rng.normal(size=rate // 10) * 3000).astype(np.int16).tobytes()
    quiet = np.zeros(rate // 10, dtype=np.int16).tobytes()
    for _ in range(8):                       # 0.8 s of speech
        collector.feed(loud, rate)
    assert stops == []
    for _ in range(6):                       # 0.6 s of silence closes it
        collector.feed(quiet, rate)
    assert len(stops) == 1, "no speech-end stamp for the turn timer"
    for _ in range(3):
        collector.feed(loud, rate)          # 0.3 s blip: below the floor
    for _ in range(6):
        collector.feed(quiet, rate)
    assert len(stops) == 1, "a 0.3 s blip should not count as a turn"
