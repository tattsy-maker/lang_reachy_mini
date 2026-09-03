"""T9: conversational enrollment.

The face pipeline is fed the fixture video; the dialog is scripted with
--say. The unsure band is manufactured deterministically: an embedding
blended to sit at cosine 0.35 from the video's own face — inside the
ask-don't-guess band (0.25–0.45) by construction.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutor.store import LearnerStore  # noqa: E402


def tts_lines(log: str) -> str:
    return " ".join(
        m.group(1) for m in re.finditer(r"Generating TTS \[(.*)\]", log))


def video_face_embedding(paths, tmp_path) -> np.ndarray:
    """Embedding of the fixture clip's face, via the recognize CLI."""
    clip = cv2.VideoCapture(str(paths.fixtures / "video" / "sunita_clip.avi"))
    ok, frame = clip.read()
    clip.release()
    assert ok
    still = tmp_path / "frame0.jpg"
    cv2.imwrite(str(still), frame)
    out = subprocess.run(
        [str(paths.voice_py), str(paths.repo / "face" / "recognize.py"),
         "embed", str(still)],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    vector = json.loads(out.stdout)["embedding"]
    assert vector, "no face in the fixture clip's first frame"
    v = np.asarray(vector, dtype=np.float32)
    return v / np.linalg.norm(v)


def blend_to_similarity(base: np.ndarray, target: float) -> list[float]:
    """A unit vector at exactly ``target`` cosine similarity to ``base``."""
    rng = np.random.default_rng(20260831)
    noise = rng.normal(size=base.shape).astype(np.float32)
    noise -= np.dot(noise, base) * base
    noise /= np.linalg.norm(noise)
    v = target * base + np.sqrt(1 - target ** 2) * noise
    return [float(x) for x in v]


def face_id(paths, source, root) -> dict:
    out = subprocess.run(
        [str(paths.voice_py), str(paths.repo / "voice" / "face_id.py"),
         "--source", str(source), "--learners-root", str(root)],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


# -- the identification states, no LLM involved ------------------------------

@pytest.mark.models
def test_face_id_unknown_known_unsure(paths, tmp_path):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    root = tmp_path / "learners"
    store = LearnerStore(root)

    assert face_id(paths, clip, root)["status"] == "unknown"

    base = video_face_embedding(paths, tmp_path)
    sure = store.create("Sunita", "es", tier="family",
                        embedding=[float(x) for x in base])
    result = face_id(paths, clip, root)
    assert result["status"] == "known" and result["learner"] == "sunita"

    store.delete(sure.id)
    store.create("Maria", "es", embedding=blend_to_similarity(base, 0.35))
    result = face_id(paths, clip, root)
    assert result["status"] == "unsure" and result["learner"] == "maria", \
        f"blended embedding should land in the ask band: {result}"


# -- scripted dialogs through the real agent ---------------------------------

@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_unknown_face_enrolls_by_voice(paths, tmp_path, run_agent_say):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    root = tmp_path / "learners"
    # T13.1 made enrollment a four-question interview (name, language,
    # level, goal), so the scripted visitor answers all four.
    log, found = run_agent_say(
        ["Hello there! I'd love to learn some Spanish.",
         "Yes please, remember me! My name is Sunita, and Spanish please.",
         "I'm a beginner.",
         "Just for fun, chatting with friends."],
        wait_for=r"tutor: enrolled new guest",
        extra_args=["--face-source", str(clip),
                    "--learners-root", str(root), "--say-gap", "12"],
        timeout=360)
    assert found, "enroll_new_learner never fired; log tail:\n" + log[-4000:]
    assert "remember you" in tts_lines(log).lower(), \
        "consent question was never asked aloud"

    learners = LearnerStore(root).list()
    assert len(learners) == 1
    guest = learners[0]
    assert guest.tier == "guest" and guest.name.lower() == "sunita"
    assert guest.embedding and len(guest.embedding) == 512

    # the same video must now be recognized as her
    result = face_id(paths, clip, root)
    assert result["status"] == "known" and result["learner"] == guest.id


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_no_thanks_stores_nothing(paths, tmp_path, run_agent_say):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    root = tmp_path / "learners"
    log, found = run_agent_say(
        ["Hi! No thank you, please don't remember me. "
         "Just tell me how to say good morning in French."],
        settle=12,
        extra_args=["--face-source", str(clip),
                    "--learners-root", str(root)],
        timeout=300)
    assert found, "no reply at all; log tail:\n" + log[-4000:]
    assert "enrolled new guest" not in log
    assert LearnerStore(root).list() == [], \
        "a profile was stored without consent"


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_unsure_band_asks_then_confirms(paths, tmp_path, run_agent_say):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    root = tmp_path / "learners"
    store = LearnerStore(root)
    base = video_face_embedding(paths, tmp_path)
    store.create("Maria", "es", level="intermediate", tier="family",
                 embedding=blend_to_similarity(base, 0.35))
    log, found = run_agent_say(
        ["Hello robot!",
         "Yes, it's me, Maria!"],
        wait_for=r"tutor: identity confirmed as maria",
        extra_args=["--face-source", str(clip),
                    "--learners-root", str(root), "--say-gap", "12"],
        timeout=300)
    assert "face: unsure" in log, "startup did not land in the unsure band"
    assert found, "confirm_identity never fired; log tail:\n" + log[-4000:]
    assert "maria" in tts_lines(log).lower(), \
        "the confirmation question never said the candidate's name"


@pytest.mark.anthropic
@pytest.mark.audio
def test_forget_me_deletes_on_the_spot(tmp_path, run_agent_say):
    root = tmp_path / "learners"
    store = LearnerStore(root)
    store.create("Sam", "fr", level="beginner")
    log, found = run_agent_say(
        ["Please forget me completely. Delete everything you know about "
         "me, right now."],
        wait_for=r"tutor: forgot learner sam",
        also_wait_for=r"Generating TTS",  # then the spoken confirmation
        extra_args=["--learner", "sam", "--learners-root", str(root)],
        timeout=300)
    assert found, "forget_me never fired; log tail:\n" + log[-4000:]
    assert store.list() == [], "the folder is still there"
    assert re.search(r"Generating TTS", log), \
        "no spoken confirmation after forgetting"
