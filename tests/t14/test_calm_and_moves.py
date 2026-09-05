"""T14.2 (calm tracking), T14.5 (longer dances), T14.4 (the wish question)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from moves import LIBRARY, MAX_PERFORM_SECS  # noqa: E402
from tracking import FaceTracker  # noqa: E402
from tutor_mode import BOOTH_PERSONA  # noqa: E402

W, H = 1920, 1080


class FakeRobot:
    def __init__(self):
        self.sent = []

    def posture(self, duration=0.5, **dofs):
        self.sent.append(dict(dofs, duration=duration))


def box(cx, cy, size=200):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def test_tracker_defaults_are_calm():
    t = FaceTracker(FakeRobot(), clock=lambda: 0.0)
    assert t.gain <= 0.5
    assert t.dead_band >= math.radians(5.9)
    assert t.min_interval >= 0.8
    assert t.max_step <= math.radians(8.01)
    assert t.move_secs >= 0.6


def test_tracker_sends_at_most_one_move_per_second_while_a_face_wanders():
    """A face drifting around the frame at 2 fps for a minute (the 2026-09-03
    session had ~135 commands a minute from tracker + sway)."""
    robot = FakeRobot()
    t = FaceTracker(robot, clock=lambda: 0.0)
    now = 0.0
    for i in range(120):                       # 60 s at 2 fps
        now += 0.5
        cx = W / 2 + 500 * math.sin(i / 7.0)   # slow wander, ±500 px
        cy = H / 2 + 150 * math.cos(i / 11.0)
        t.observe(box(cx, cy), W, H, now=now)
    assert len(robot.sent) <= 60, len(robot.sent)
    for cmd in robot.sent:
        assert cmd["duration"] >= 0.6
        if "body_yaw" not in cmd and "head_yaw" in cmd:
            pass


def test_small_sway_of_the_head_does_not_move_the_robot():
    robot = FakeRobot()
    t = FaceTracker(robot, clock=lambda: 0.0)
    for i in range(20):
        t.observe(box(W / 2 + 40 * (-1) ** i, H / 2 + 30), W, H, now=i * 1.0)
    assert robot.sent == [], "a 40 px wobble is inside the dead-band"


# -- longer dances ------------------------------------------------------------

def test_dances_default_to_about_thirty_seconds():
    for name in ("dance", "dance_groovy", "dance_pendulum"):
        spec = LIBRARY[name]
        assert spec.default_secs >= 25
        passes = spec.passes_for(None)
        assert 20 <= passes * spec.seconds <= 40, (name, passes)
    assert LIBRARY["cheer"].passes_for(None) == 1, "emotions play once"
    assert LIBRARY["dance"].passes_for(60) * 9 <= MAX_PERFORM_SECS + 9
    assert LIBRARY["dance"].passes_for(500) * 9 <= MAX_PERFORM_SECS + 9, "capped"
    assert LIBRARY["dance"].passes_for(5) == 1


def test_stub_plays_a_move_several_times(paths):
    if not paths.robot_py.exists():
        pytest.skip("./.venv missing")
    out = subprocess.run(
        [str(paths.robot_py), "controller.py", "--stub", "perform", "cheer",
         "--repeat", "3"],
        cwd=paths.repo, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stderr.count('"phase": "playing"') == 3, out.stderr[-1500:]
    assert '"of": 3' in out.stderr


# -- the wish question --------------------------------------------------------

def test_wish_question_waits_for_an_answer():
    text = BOOTH_PERSONA
    assert "Before you go, one quick question" in text
    assert "STOP" in text and "wait for their answer" in text
    assert "When they answer, call record_wish" in text
    assert "Never bring the question up mid-lesson" in text
    assert "never announce" in text
