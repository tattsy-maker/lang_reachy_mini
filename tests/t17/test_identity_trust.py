"""T17.2 (trust a confirmed identity) and T17.3 (bystanders do not
count): the voice print keeps its doubt to the log while the face
vouches and trusts after two overrules or a confirmation; a confirmed
visit improves the stored prints; the runner finds the session's face
among every face in the frame. Unit, no camera, no models."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voice"))

from session import ACTIVE, WATCHING, SessionRunner        # noqa: E402
from tutor.store import LearnerStore                       # noqa: E402
from tutor_mode import (                                   # noqa: E402
    CurrentLearner, build_briefing, build_unsure_briefing, strengthen_face,
    voice_cue,
)
from voiceid import VoiceIdentity                          # noqa: E402


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
    def __init__(self, embedding=None, bbox=(100, 100, 200, 200)):
        self.embedding = embedding
        self.bbox = bbox

    @property
    def centre(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class FakeContext:
    def __init__(self):
        self.history = []

    def set_messages(self, messages):
        self.history.append(messages)


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


def observe(runner, face, now, faces=None):
    asyncio.run(runner.observe(face, now, (480, 640), faces=faces))


def start_with(runner, store, holder, vector, name="Ana", now=10.0):
    learner = store.create(name, "fr", embedding=[float(x) for x in vector])
    runner._recent_vectors = [vector]
    asyncio.run(runner.start_session(now=now))
    assert holder.learner.id == learner.id and runner.machine.state == ACTIVE
    return learner


# -- T17.2: the voice asks the face first ---------------------------------------------

def test_a_mismatch_while_the_face_vouches_is_overruled_then_trusted(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    voice = unit(0)
    ana = store.create("Ana", "fr", voice_embedding=[float(x) for x in voice])
    holder.learner = ana
    vouch = {"yes": True}
    vid = VoiceIdentity(store, holder, face_vouches=lambda: vouch["yes"],
                        trust_after=2)
    other = unit(1)                       # somebody else's voice, or a silly one
    assert vid.on_sample(other, 4.0) == "overruled"
    assert holder.learner is ana and vid.challenged is False, "nothing was asked"
    assert vid.trusted is False
    assert vid.on_sample(other, 4.0) == "overruled"
    assert vid.trusted is True, "two overrules -> trusted for the session"
    assert vid.on_sample(other, 4.0) is None, "trusted: no more decisions"
    assert len(vid.samples) == 3, "samples still feed the print"
    vid.reset()
    assert vid.trusted is False


def test_without_a_face_the_voice_still_asks(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    ana = store.create("Ana", "fr", voice_embedding=[float(x) for x in unit(0)])
    holder.learner = ana
    vid = VoiceIdentity(store, holder, face_vouches=lambda: False)
    assert vid.on_sample(unit(1), 4.0) == "challenge"
    none = VoiceIdentity(store, holder)
    none.reset()
    assert none.on_sample(unit(1), 4.0) == "challenge", "no face source: as before"


def test_trust_and_absorb_after_a_confirmation(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    stored = unit(0)
    ana = store.create("Ana", "fr", voice_embedding=[float(x) for x in stored])
    holder.learner = ana
    vid = VoiceIdentity(store, holder)
    heard = blend(stored, 0.5, seed=3)
    vid.on_sample(heard, 4.0)
    vid.trust("confirmed")
    assert vid.trusted and vid.on_sample(unit(1), 4.0) is None
    assert vid.absorb(ana, weight=0.5)
    new = np.asarray(store.load(ana.id).voice_embedding, dtype=np.float32)
    assert abs(np.linalg.norm(new) - 1.0) < 1e-5
    assert float(np.dot(new, heard)) > float(np.dot(stored, heard)), \
        "the stored print moved toward what was heard"
    assert ana.voice_embedding == store.load(ana.id).voice_embedding
    # no samples: nothing to absorb
    fresh = VoiceIdentity(store, holder)
    assert fresh.absorb(ana) is False


def test_strengthen_face_moves_the_stored_face_toward_the_live_one():
    stored, live = unit(0), blend(unit(0), 0.6, seed=5)
    merged = np.asarray(strengthen_face(stored, live, weight=0.3))
    assert abs(np.linalg.norm(merged) - 1.0) < 1e-5
    assert np.dot(merged, live) > np.dot(stored, live)
    assert strengthen_face(stored, None) is None
    assert strengthen_face(stored, unit(0, dim=4)) is None, "shape mismatch"


def test_runner_face_vouches_and_binds_itself_to_the_voice(tmp_path):
    runner, store, holder = make(tmp_path, face_vouch_secs=5.0)
    vid = VoiceIdentity(store, holder)
    runner.voice_identity = vid
    ana = unit(0)
    start_with(runner, store, holder, ana)
    assert runner.face_vouches(now=10.5)
    observe(runner, Face(embedding=blend(ana, 0.8)), now=12.0)
    assert vid.face_vouches is not None, "the runner handed the voice its predicate"
    assert vid.face_vouches() or runner.face_vouches(now=12.1)
    assert runner.face_vouches(now=20.0) is False, "too long ago"
    asyncio.run(runner.end_session())
    assert runner.face_vouches(now=12.0) is False, "nobody there"


def test_a_good_session_writes_the_face_back(tmp_path):
    runner, store, holder = make(tmp_path)
    ana = unit(0)
    learner = start_with(runner, store, holder, ana)
    seen = blend(ana, 0.7, seed=9)
    for i in range(7):
        observe(runner, Face(embedding=seen), now=12.0 + 2 * i)
    asyncio.run(runner.end_session())
    stored = np.asarray(store.load(learner.id).embedding, dtype=np.float32)
    assert np.dot(stored, seen) > np.dot(ana, seen), "the profile moved toward what it saw"
    assert abs(np.linalg.norm(stored) - 1.0) < 1e-5


def test_a_short_session_leaves_the_face_alone(tmp_path):
    runner, store, holder = make(tmp_path)
    ana = unit(0)
    learner = start_with(runner, store, holder, ana)
    observe(runner, Face(embedding=blend(ana, 0.7)), now=12.0)
    asyncio.run(runner.end_session())
    assert store.load(learner.id).embedding == [float(x) for x in ana]


def test_confirm_identity_trusts_absorbs_and_strengthens(tmp_path):
    pytest.importorskip("pipecat", reason="FunctionSchema comes from pipecat")
    from tutor_mode import build_enrollment_tools

    class Params:
        def __init__(self, **arguments):
            self.arguments = arguments
            self.result = None

        async def result_callback(self, result):
            self.result = result

    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    face, voice = unit(0), unit(2)
    ana = store.create("Ana", "fr", embedding=[float(x) for x in face],
                       voice_embedding=[float(x) for x in voice])
    holder.candidate = ana
    vid = VoiceIdentity(store, holder)
    vid.samples.append(blend(voice, 0.5, seed=4))
    live = blend(face, 0.6, seed=8)
    tools = {t.name: t for t in build_enrollment_tools(
        store, holder, face_source=None, voice_identity=vid,
        current_face=lambda: live)}
    p = Params()
    asyncio.run(tools["confirm_identity"].handler(p))
    assert p.result["confirmed"] == "Ana" and holder.learner.id == ana.id
    assert vid.trusted, "a confirmed yes closes the question"
    saved = store.load(ana.id)
    assert saved.voice_embedding != [float(x) for x in voice], "print absorbed"
    assert np.dot(np.asarray(saved.embedding), live) > np.dot(face, live), \
        "face strengthened from the confirmation"


# -- the prompts ---------------------------------------------------------------------

def test_prompts_keep_identity_with_the_recognizer(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    ana = store.create("Ana", "fr")
    text = build_briefing(ana, "")
    assert "Never judge identity from a look picture" in text
    assert "that is the end of it for this session" in text
    assert "a picture is not a face match" in text
    unsure = build_unsure_briefing(ana)
    assert "never a look picture" in unsure
    cue = voice_cue("challenge", ana, store)
    assert "Ask this once" in cue and "never raise it again" in cue


# -- T17.3: every face in the frame -------------------------------------------------

def test_the_session_face_counts_even_when_it_is_not_the_largest(tmp_path):
    runner, store, holder = make(tmp_path, swap_secs=3.0)
    ana, kid = unit(0), unit(1)
    start_with(runner, store, holder, ana)
    big_kid = Face(embedding=blend(kid, 0.9), bbox=(0, 0, 400, 400))
    small_ana = Face(embedding=blend(ana, 0.8), bbox=(500, 100, 600, 200))
    for i in range(4):                     # 8 s of a big bystander
        observe(runner, big_kid, now=12.0 + 2 * i, faces=[big_kid, small_ana])
    assert runner.machine.state == ACTIVE, "a bystander in front ended the lesson"
    assert runner._other_since is None
    assert runner._session_bbox == small_ana.bbox, "the tracker follows hers"
    assert not any("save_session_notes" in c for c in runner.cues)


def test_a_bystander_alone_for_swap_secs_still_swaps(tmp_path):
    runner, store, holder = make(tmp_path, swap_secs=3.0, stable_secs=2.0)
    ana, kid = unit(0), unit(1)
    start_with(runner, store, holder, ana)
    big_kid = Face(embedding=blend(kid, 0.9), bbox=(0, 0, 400, 400))
    observe(runner, big_kid, now=12.0, faces=[big_kid])
    observe(runner, big_kid, now=14.0, faces=[big_kid])
    assert runner.machine.state == ACTIVE
    observe(runner, big_kid, now=15.5, faces=[big_kid])
    assert runner.machine.state == WATCHING, "nobody but the newcomer for 3 s"
    assert any("save_session_notes" in c for c in runner.cues)


def test_between_recognitions_the_nearest_box_is_followed(tmp_path):
    runner, store, holder = make(tmp_path)
    ana = unit(0)
    start_with(runner, store, holder, ana)
    hers = Face(embedding=blend(ana, 0.8), bbox=(500, 100, 600, 200))
    observe(runner, hers, now=12.0, faces=[hers])
    assert runner._session_bbox == hers.bbox
    big = Face(bbox=(0, 0, 400, 400))
    near = Face(bbox=(510, 105, 610, 205))
    picked = runner._pick_face([big, near], now=12.5)
    assert picked is near
    far = Face(bbox=(1500, 900, 1600, 1000))
    assert runner._pick_face([big, far], now=12.5) is big, "nothing near: the largest"
    assert runner._pick_face([], now=12.5) is None


def test_recognize_finds_every_face_in_a_frame(paths):
    """Two fixture portraits side by side: analyze_all returns both,
    largest first, each with an embedding. Runs under voice/.venv."""
    pytest.importorskip("cv2")
    import json
    import subprocess
    import cv2
    faces = sorted((paths.fixtures / "faces").glob("*.jp*g"))
    if len(faces) < 2:
        pytest.skip("need two fixture portraits")
    if not paths.voice_py.exists():
        pytest.skip("voice/.venv missing (the face models live there)")
    a = cv2.imread(str(faces[0]))
    b = cv2.imread(str(faces[1]))
    h = min(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0] * 0.6), int(h * 0.6)))
    canvas = np.zeros((h, a.shape[1] + b.shape[1] + 40, 3), dtype=np.uint8)
    canvas[:h, :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1] + 40:] = b
    out = Path(str(paths.tests / "reports" / "t17_two_faces.jpg"))
    cv2.imwrite(str(out), canvas)
    code = ("import sys, json, cv2; sys.path.insert(0, %r); "
            "from face import recognize; "
            "fs = recognize.analyze_all(cv2.imread(%r)); "
            "print(json.dumps([{'area': f.area, 'dim': len(f.embedding)} for f in fs]))"
            % (str(paths.repo), str(out)))
    res = subprocess.run([str(paths.voice_py), "-c", code], capture_output=True,
                         text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert len(got) == 2, got
    assert got[0]["area"] >= got[1]["area"] and got[1]["dim"] == 512
