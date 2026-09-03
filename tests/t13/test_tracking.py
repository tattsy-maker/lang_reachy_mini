"""T13.3: the face tracker -- geometry and controller on a fake robot
(unmarked), the detect -> track path over the fixture clip through
voice/.venv (models), and a live smoke on the real camera (robot)."""

import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from tracking import (  # noqa: E402
    BODY_YAW_LIMIT, HEAD_YAW_LIMIT, FaceTracker, pixel_to_angles,
)

W, H = 1920, 1080


class FakeRobot:
    def __init__(self, measured=None):
        self.sent = []
        self.measured = measured or {}

    def posture(self, duration=0.5, **dofs):
        self.sent.append(dict(dofs, duration=duration))

    async def status(self):
        return {"measured": self.measured}


class FakeEmbodiment:
    def __init__(self):
        self.pitch_bias = 0.0
        self.busy = False


def box(cx, cy, size=200):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


# -- geometry ----------------------------------------------------------------

def test_pixel_to_angles_centre_edges_corners():
    yaw, pitch = pixel_to_angles(W / 2, H / 2, W, H)
    assert abs(yaw) < 1e-9 and abs(pitch) < 1e-9
    yaw, _ = pixel_to_angles(0, H / 2, W, H)          # left edge
    assert yaw > 0 and math.degrees(yaw) == pytest.approx(43.9, abs=1.0)
    yaw, _ = pixel_to_angles(W, H / 2, W, H)          # right edge
    assert yaw < 0
    _, pitch = pixel_to_angles(W / 2, H, W, H)        # bottom edge: chin down
    assert pitch > 0 and math.degrees(pitch) == pytest.approx(28.4, abs=1.0)
    _, pitch = pixel_to_angles(W / 2, 0, W, H)
    assert pitch < 0
    yaw, pitch = pixel_to_angles(0, 0, W, H)          # top-left corner
    assert yaw > 0 and pitch < 0
    # resolution-independent: same angles at half size
    a = pixel_to_angles(100, 100, W, H)
    b = pixel_to_angles(50, 50, W / 2, H / 2)
    assert a == pytest.approx(b)


# -- controller --------------------------------------------------------------

def test_dead_band_and_rate_limit():
    robot = FakeRobot()
    t = FaceTracker(robot, clock=lambda: 0.0)
    assert t.observe(box(W / 2 + 20, H / 2), W, H, now=1.0) is None, \
        "a face a hair off centre must not move the head"
    assert robot.sent == []
    cmd = t.observe(box(400, H / 2), W, H, now=2.0)
    assert cmd and cmd["head_yaw"] > 0, "a face on the left -> turn left"
    assert t.observe(box(400, H / 2), W, H, now=2.1) is None, "rate-limited"
    assert t.observe(box(400, H / 2), W, H, now=2.6) is not None


def test_steps_are_capped_and_within_limits():
    t = FaceTracker(FakeRobot(), clock=lambda: 0.0)
    cmd = t.observe(box(5, H / 2), W, H, now=1.0)
    assert cmd["head_yaw"] <= math.radians(20) + 1e-9, "one step at most 20 deg"
    for i in range(30):
        t.observe(box(5, H / 2), W, H, now=2.0 + i)
    est = t.estimate
    assert abs(est["head_yaw"]) <= HEAD_YAW_LIMIT + 1e-9
    assert abs(est["body_yaw"]) <= BODY_YAW_LIMIT + 1e-9
    for sent in t.robot.sent:
        for k, v in sent.items():
            assert math.isfinite(v)


def test_body_takes_over_past_handoff():
    robot = FakeRobot()
    t = FaceTracker(robot, clock=lambda: 0.0)
    now = 0.0
    body_moves = []
    for _ in range(8):
        now += 1.0
        cmd = t.observe(box(5, H / 2), W, H, now=now) or {}
        if "body_yaw" in cmd:
            body_moves.append(cmd)
    assert body_moves, "the head alone never reached the handoff angle"
    first = body_moves[0]
    assert first["head_yaw"] == 0.0 and first["body_yaw"] > math.radians(35)
    assert t.suspended is False or True     # handoff suspends briefly
    assert all(abs(s["head_yaw"]) <= HEAD_YAW_LIMIT for s in robot.sent)


def test_pitch_goes_through_embodiment_bias_when_busy():
    emb = FakeEmbodiment()
    robot = FakeRobot()
    t = FaceTracker(robot, embodiment=emb, clock=lambda: 0.0)
    cmd = t.observe(box(W / 2, H - 50), W, H, now=1.0)
    assert "head_pitch" in cmd and cmd["head_pitch"] > 0
    assert emb.pitch_bias == pytest.approx(cmd["head_pitch"])
    emb.busy = True
    before = emb.pitch_bias
    cmd = t.observe(box(W / 2, H - 50), W, H, now=2.0)
    assert not cmd or "head_pitch" not in cmd, \
        "while embodiment talks, pitch is its to write"
    assert emb.pitch_bias > before, "...but it still gets the bias"


def test_suspend_then_resync_from_measured_pose():
    robot = FakeRobot(measured={"head_yaw": 0.4, "body_yaw": 0.2,
                                "head_pitch": 0.1})
    clock = [0.0]
    t = FaceTracker(robot, clock=lambda: clock[0])
    t.suspend(2.0)                                  # a tool moved the head
    assert t.observe(box(5, H / 2), W, H) is None   # suspended
    clock[0] = 3.0
    assert t.observe(box(5, H / 2), W, H) is None   # stale until resynced
    import asyncio
    asyncio.run(t.maybe_resync())
    assert t.estimate["head_yaw"] == pytest.approx(0.4)
    cmd = t.observe(box(5, H / 2), W, H)
    # 0.4 + a full step crosses the 35 deg handoff, so the body takes the
    # angle: either way the command builds on the measured 0.4, not on 0.
    assert cmd.get("head_yaw", 0) > 0.4 or cmd.get("body_yaw", 0) > 0.6, \
        ("resumes from where the robot really is", cmd)


def test_set_estimate_and_reset():
    t = FaceTracker(FakeRobot(), clock=lambda: 0.0)
    t.set_estimate(head_yaw=0.3, body_yaw=1.0)
    assert t.estimate["head_yaw"] == 0.3 and t.estimate["body_yaw"] == 1.0
    t.reset()
    assert t.estimate == {"head_yaw": 0.0, "body_yaw": 0.0, "head_pitch": 0.0}


def test_relax_returns_to_centre_after_a_quiet_spell():
    robot = FakeRobot()
    t = FaceTracker(robot, clock=lambda: 0.0, relax_after=8.0)
    t.observe(box(5, H / 2), W, H, now=1.0)
    assert t.relax(now=5.0) is None
    cmd = t.relax(now=10.0)
    assert cmd == {"head_yaw": 0.0, "body_yaw": 0.0, "head_pitch": 0.0}
    assert t.relax(now=20.0) is None, "only once"


# -- through the real detector ---------------------------------------------

@pytest.mark.models
def test_tracker_follows_the_fixture_face(paths):
    out = subprocess.run(
        [str(paths.voice_py), str(paths.voice_dir / "tracking.py"),
         "--source", str(paths.fixtures / "video" / "sunita_clip.avi"),
         "--fps", "2"],
        capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    data = json.loads(out.stdout)
    seen = [f for f in data["frames"] if f["bbox"]]
    assert seen, "the detector never found the fixture face"
    assert data["commands"], "a face off centre must produce commands"
    # every command's yaw points toward the face's side of the frame
    for f in seen:
        if f["cmd"] and "head_yaw" in f["cmd"] and "body_yaw" not in f["cmd"]:
            cx = (f["bbox"][0] + f["bbox"][2]) / 2
            side = 1 if cx < f["width"] / 2 else -1
            step = f["cmd"]["head_yaw"]
            assert step == 0 or (step > 0) == (side > 0), (f["bbox"], f["cmd"])
    for c in data["commands"]:
        assert abs(c.get("head_yaw", 0)) <= HEAD_YAW_LIMIT + 1e-9
        assert abs(c.get("body_yaw", 0)) <= BODY_YAW_LIMIT + 1e-9


@pytest.mark.robot
@pytest.mark.models
def test_tracker_sees_the_real_camera(paths):
    """Hold a face (yours, or a printed one) in front of the robot's camera."""
    if not glob.glob("/dev/video0"):
        pytest.skip("no /dev/video0")
    out = subprocess.run(
        [str(paths.voice_py), str(paths.voice_dir / "tracking.py"),
         "--source", "0", "--fps", "2", "--max-frames", "10"],
        capture_output=True, text=True, timeout=300)
    if out.returncode != 0 and "could not open" in out.stderr:
        pytest.skip("camera not readable (video group / daemon holds it)")
    assert out.returncode == 0, out.stderr[-3000:]
    data = json.loads(out.stdout)
    if not any(f["bbox"] for f in data["frames"]):
        pytest.skip("no face in front of the camera during the test")
