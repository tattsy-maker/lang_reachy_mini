"""T10: one full simulated visitor through the real agent — fixture video
supplies the face, --say the conversation, the clip running out the
disappearance. Ends with a well-formed notes entry and a reset."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutor.store import LearnerStore  # noqa: E402


def video_embedding(paths):
    out = subprocess.run(
        [str(paths.voice_py), str(paths.repo / "face" / "recognize.py"),
         "embed", str(paths.fixtures / "faces" / "sunita_b.jpg")],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)["embedding"]


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_full_visitor_lifecycle(paths, tmp_path, run_agent_say):
    root = tmp_path / "learners"
    store = LearnerStore(root)
    store.create("Sunita", "es", level="intermediate", tier="family",
                 embedding=video_embedding(paths))
    clip = paths.fixtures / "video" / "sunita_clip.avi"

    log, found = run_agent_say(
        ["¡Hola! Sí, soy yo. Quiero practicar un poco."],
        wait_for=r"session: ended and reset",
        extra_args=["--session", "--face-source", str(clip),
                    "--learners-root", str(root),
                    "--stable-secs", "0.5", "--absent-secs", "10",
                    "--say-delay", "5"],
        timeout=300)
    assert "session: started" in log, "no session ever started:\n" + log[-4000:]
    assert re.search(r"session: recognized sunita", log), \
        "the seeded face was not recognized:\n" + log[-4000:]
    assert found, "the session never ended/reset; log tail:\n" + log[-4000:]

    learner = store.load("sunita")
    assert learner.sessions == 1, "walk-away save did not land"
    notes = store.read_notes("sunita", max_sessions=1)
    for section in ("Practiced", "Struggled with", "Wins", "Next time"):
        assert f"**{section}:**" in notes, f"notes missing {section}"
