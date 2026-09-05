"""T14 through the real agent: sight (T14.1) in both speech modes, and
one launch serving two visits over the cloud voice with a fresh Gemini
session each time and a readable transcript (T14.3, T14.6)."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutor.store import LearnerStore  # noqa: E402

VOICES = Path(__file__).resolve().parents[1] / "fixtures" / "voices"


def tts_lines(log: str) -> str:
    return " ".join(
        m.group(1) for m in re.finditer(r"Generating TTS \[(.*)\]", log))


def two_visits_clip(paths, tmp_path, gap_secs: float = 8.0,
                    visit_loops: int = 4) -> Path:
    """The fixture clip looped ``visit_loops`` times (a ~24 s visit),
    ``gap_secs`` of nobody, then the same again -- built on the fly (tens
    of MB as MJPG; not worth committing)."""
    import cv2
    import numpy as np
    src = str(paths.fixtures / "video" / "sunita_clip.avi")
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]
    out_path = tmp_path / "two_visits.avi"
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"),
                          fps, (w, h))
    gap = np.full((h, w, 3), 18, dtype=np.uint8)
    for _ in range(visit_loops):
        for f in frames:
            out.write(f)
    for _ in range(int(fps * gap_secs)):
        out.write(gap)
    for _ in range(visit_loops):
        for f in frames:
            out.write(f)
    out.release()
    return out_path


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_look_tool_local(paths, tmp_path, run_agent_say):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    log, found = run_agent_say(
        ["Take a look through your camera and tell me what you see."],
        wait_for=r"look: one \d+x\d+ frame",
        also_wait_for=r"Generating TTS",
        extra_args=["--face-source", str(clip),
                    "--learners-root", str(tmp_path / "learners")],
        settle=10, timeout=300)
    assert found, "look never fired or nothing was said:\n" + log[-4000:]
    spoken = tts_lines(log).lower()
    assert spoken, "no description spoken"
    assert not re.search(r"cannot see|can't see|no camera", spoken), spoken


@pytest.mark.google
@pytest.mark.models
@pytest.mark.audio
def test_look_tool_cloud(paths, tmp_path, run_agent_say):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    log, found = run_agent_say(
        ["Take a look through your camera and tell me what you see."],
        wait_for=r"look: one \d+x\d+ frame",
        also_wait_for=r"said: ",
        extra_args=["--speech", "cloud", "--face-source", str(clip),
                    "--learners-root", str(tmp_path / "learners")],
        settle=8, timeout=300)
    assert found, "look never fired or nothing was said:\n" + log[-4000:]
    said = " ".join(re.findall(r"said: (.*)", log)).lower()
    assert not re.search(r"cannot see|can't see|no camera", said), said


@pytest.mark.google
@pytest.mark.models
@pytest.mark.audio
def test_two_visits_over_the_cloud_voice(paths, tmp_path, run_agent_say):
    """One launch, the two-visit clip (a ~24 s visit, 8 s of nobody, the
    visit again): the stranger enrolls, walks away (notes saved), comes
    back and is greeted as a known learner from a fresh Gemini session --
    the 2026-09-03 failure mode, fixed. Timeline at 2 fps: session starts
    ~1 s in; the interview answer lands at 8 s; goodbye at 20 s; gone at
    ~24 s, ended ~28 s; back at ~32 s."""
    clip = two_visits_clip(paths, tmp_path)
    root = tmp_path / "learners"
    log, found = run_agent_say(
        ["Hi! Yes, please remember me. I'm Sunita, Spanish, beginner, "
         "just for conversation.",
         "Bye for now!"],
        wait_for=r"session: recognized sunita",
        extra_args=["--speech", "cloud", "--session", "--face-source", str(clip),
                    "--learners-root", str(root),
                    "--stable-secs", "0.5", "--absent-secs", "4",
                    "--say-delay", "8", "--say-gap", "12",
                    "--voice-source", str(VOICES / "af_heart" / "af_heart_1.wav")],
        settle=12, timeout=420)
    assert "tutor: enrolled new guest" in log, "first visit never enrolled:\n" + log[-4000:]
    assert log.count("cloud brain: fresh Gemini session") >= 2, \
        "no fresh Gemini session per visit:\n" + log[-3000:]
    assert found, "the second visit was not recognized:\n" + log[-4000:]
    assert re.search(r"said: ", log), "no spoken transcript in the log (T14.6)"
    learner = LearnerStore(root).load("sunita")
    assert learner is not None and learner.sessions >= 1, "notes never saved"
