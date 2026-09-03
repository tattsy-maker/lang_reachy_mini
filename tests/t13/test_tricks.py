"""T13.4: the move library, play_move on the stub (hosted, no broker),
the shared FrameHub, and one recorded move on metal (robot)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moves import ATTRACT_MOVES, LIBRARY  # noqa: E402
from face.camera import FrameHub  # noqa: E402


def test_library_is_sane():
    assert {"dance", "spin", "wiggle"} <= set(LIBRARY)
    assert all(m in LIBRARY for m in ATTRACT_MOVES)
    for m in LIBRARY.values():
        assert m.seconds > 0 and m.description
        assert (m.dataset is None) == (m.move is None)


def robot_cli(paths, *args, timeout=120):
    return subprocess.run(
        [str(paths.robot_py), "controller.py", "--stub", *args],
        cwd=paths.repo, capture_output=True, text=True, timeout=timeout)


def test_stub_lists_and_plays_a_recorded_move(paths):
    if not paths.robot_py.exists():
        pytest.skip("./.venv missing")
    out = robot_cli(paths, "moves")
    assert out.returncode == 0, out.stderr[-2000:]
    names = {m["name"] for m in json.loads(out.stdout)["moves"]}
    assert names == set(LIBRARY)

    out = robot_cli(paths, "perform", "dance")
    assert out.returncode == 0, out.stderr[-2000:]
    reply = json.loads(out.stdout)
    assert reply["motion_id"].startswith("m-") and "posture" in reply
    assert "playing" in out.stderr, "no progress event for the move"


def test_stub_spin_and_wiggle_are_built_in(paths):
    if not paths.robot_py.exists():
        pytest.skip("./.venv missing")
    out = robot_cli(paths, "perform", "spin")
    assert out.returncode == 0, out.stderr[-2000:]
    assert '"phase": "back"' in out.stderr, "spin never swept back"
    assert json.loads(out.stdout)["posture"]["body_yaw"] == pytest.approx(0.0)
    out = robot_cli(paths, "perform", "wiggle")
    assert out.returncode == 0, out.stderr[-2000:]


def test_stub_rejects_unknown_move_with_the_list(paths):
    if not paths.robot_py.exists():
        pytest.skip("./.venv missing")
    out = robot_cli(paths, "perform", "moonwalk")
    assert out.returncode != 0
    assert "moonwalk" in out.stderr and "dance" in out.stderr


# -- the shared camera --------------------------------------------------------

def test_frame_hub_serves_many_readers_and_ends(paths):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    hub = FrameHub(clip, fps=30.0).start()   # fast pacing: the clip is short
    try:
        a = list(hub.frames())
        assert a, "first reader got nothing"
        assert hub.exhausted
        b = list(hub.frames())               # a late reader: source is done
        assert len(b) <= 1, "a late reader sees at most the final frame"
        seq, last = hub.latest()
        assert seq == len(a) or seq >= len(a)
        assert last is not None and last.shape == a[-1].shape
    finally:
        hub.close()


def test_frame_hub_readers_skip_rather_than_lag(paths):
    import threading
    import time
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    hub = FrameHub(clip, fps=20.0).start()
    slow, fast = [], []

    def reader(store, delay):
        for frame in hub.frames():
            store.append(frame)
            time.sleep(delay)

    ts = [threading.Thread(target=reader, args=(fast, 0.0)),
          threading.Thread(target=reader, args=(slow, 0.2))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    hub.close()
    assert fast and slow
    assert len(slow) < len(fast), "a slow reader must skip frames, not queue"


# -- metal --------------------------------------------------------------------

@pytest.mark.robot
def test_recorded_move_plays_on_metal(paths):
    """Plays the 'cheer' emotion on the real robot (hosted driver). Skips
    if another process already owns the robot (only one may)."""
    if subprocess.run(["pgrep", "-f", "controller.py serve"],
                      capture_output=True).returncode == 0:
        pytest.skip("a 'controller.py serve' owns the robot; stop it first")
    cached = subprocess.run([str(paths.robot_py), "moves.py", "--cached"],
                            cwd=paths.repo, capture_output=True, text=True)
    if cached.returncode != 0:
        pytest.skip("move datasets not cached: .venv/bin/python moves.py --preload")
    out = subprocess.run(
        [str(paths.robot_py), "controller.py", "perform", "cheer"],
        cwd=paths.repo, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "playing" in out.stderr
