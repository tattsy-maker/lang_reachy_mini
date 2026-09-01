"""T8: cloud speech mode (Gemini Live).

The google-marked tests skip until GOOGLE_API_KEY appears in the
environment or voice/.env — they are written to run the moment it does.
Local mode's regression is the rest of the suite (t0/t4/t9/t10), which
must stay green with the --speech flag merely existing.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutor.store import LearnerStore  # noqa: E402


@pytest.mark.models
def test_cloud_mode_fails_fast_and_clearly_without_key(paths):
    # Blank (not absent) keys: load_env_file never overrides an existing
    # env var, so this keeps a real key in voice/.env from leaking in.
    out = subprocess.run(
        [str(paths.voice_py), "agent.py", "--speech", "cloud", "--no-robot"],
        cwd=paths.voice_dir, capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
             "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
    assert out.returncode != 0
    blob = out.stdout + out.stderr
    assert "GOOGLE_API_KEY" in blob, \
        "the missing-key failure must name the fix"
    assert "Traceback" not in blob, "should exit cleanly, not crash"


@pytest.mark.google
@pytest.mark.audio
def test_cloud_say_turn_gets_an_audio_reply(run_agent_say):
    log, found = run_agent_say(
        ["Hello! Please answer with one short sentence."],
        wait_for=r"Bot started speaking",
        extra_args=["--speech", "cloud"],
        timeout=180)
    assert found, "no audio reply in cloud mode; log tail:\n" + log[-4000:]


@pytest.mark.google
@pytest.mark.audio
def test_cloud_goodbye_saves_notes(tmp_path, run_agent_say):
    root = tmp_path / "learners"
    store = LearnerStore(root)
    store.create("Maria", "es", level="intermediate", tier="family")
    log, found = run_agent_say(
        ["Hola. Me tengo que ir, ¡adiós, hasta la próxima!"],
        wait_for=r"tutor: saved session notes",
        extra_args=["--speech", "cloud", "--learner", "maria",
                    "--learners-root", str(root)],
        timeout=180)
    assert found, ("save_session_notes never fired in cloud mode:\n"
                   + log[-4000:])
    notes = store.read_notes("maria", max_sessions=1)
    for section in ("Practiced", "Struggled with", "Wins", "Next time"):
        assert f"**{section}:**" in notes


@pytest.mark.google
@pytest.mark.audio
def test_cloud_motion_tool_fires_on_stub(stub_robot, run_agent_say):
    log, found = run_agent_say(
        ["Please nod twice right now."],
        wait_for=r"nod",
        no_robot=False,
        extra_args=["--speech", "cloud",
                    "--broker", stub_robot.broker],
        timeout=180)
    assert found and re.search(r"nod", log), \
        "the nod tool never fired in cloud mode:\n" + log[-4000:]
