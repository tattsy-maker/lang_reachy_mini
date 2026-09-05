"""T15.9: recorded moves own the body (found live on 2026-09-04).

Robot venv: the driver refuses a posture nudge while a recorded move
plays (the stub's 0.2 s move stands in for a 27 s dance) and accepts it
afterwards; the target's goto interpolates only the joint groups the
caller named. Voice venv: the embodiment sends nothing while held, and
the motion tools answer "skipped" during a dance.
"""

import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def run_in(python, script, cwd):
    out = subprocess.run([str(python), "-c", textwrap.dedent(script)],
                         cwd=cwd, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    return out.stdout


def test_driver_refuses_a_nudge_while_a_recorded_move_plays(paths):
    if not paths.robot_py.exists():
        pytest.skip("./.venv missing")
    out = run_in(paths.robot_py, """
        import asyncio, sys
        sys.path.insert(0, ".")
        from reachy_driver import ReachyMiniDriver
        from stub_target import StubTarget

        async def main():
            d = ReachyMiniDriver(target=StubTarget())
            events = []
            async def on_event(name, payload): events.append(name)
            d.set_event_callback(on_event)
            d._target.connect()
            await d.set_motors(True)
            move = await d.play_move("dance", repeat=1)
            assert move["accepted"], move
            nudge = await d.goto_posture(duration=0.3, head_pitch=0.1)
            assert nudge["accepted"] is False, nudge
            assert "recorded move" in nudge["reason"], nudge
            assert d._motion["motion_id"] == move["motion_id"], "the dance was cut"
            nod = await d.nod(times=1)
            assert nod["accepted"] is False, nod
            await asyncio.sleep(0.6)                     # the stub move ends
            assert d._motion["status"] == "succeeded", d._motion
            later = await d.goto_posture(duration=0.2, head_pitch=0.1)
            assert later["accepted"] is True, later
            # another move, or home, may still interrupt a move
            first = await d.play_move("dance", repeat=1)
            second = await d.play_move("cheer", repeat=1)
            assert second["accepted"] and d._motion["motion_id"] == second["motion_id"]
            await asyncio.sleep(0.3)
            move = await d.play_move("dance", repeat=1)
            home = await d.home(duration=0.2)
            assert home["accepted"], home
            await asyncio.sleep(0.5)
            print("driver-ok")
        asyncio.run(main())
    """, paths.repo)
    assert "driver-ok" in out


def test_target_goto_interpolates_only_the_named_groups(paths):
    if not paths.robot_py.exists():
        pytest.skip("./.venv missing")
    out = run_in(paths.robot_py, """
        import sys
        sys.path.insert(0, ".")
        import reachy_target
        from reachy_target import ReachyMiniTarget

        class Mini:
            def __init__(self): self.calls = []
            def set_target(self, **kw): self.calls.append(("set", kw))
            def goto_target(self, **kw): self.calls.append(("goto", kw))

        t = ReachyMiniTarget.__new__(ReachyMiniTarget)
        import threading
        t._lock = threading.RLock()
        t._cmd = {n: 0.0 for n in reachy_target.DOF_NAMES}
        t._cmd["body_yaw"] = 0.3
        mini = Mini()
        t._require = lambda: mini
        t.goto(0.8, head_pitch=-0.1, antenna_left=0.4)
        kind, kw = mini.calls[-1]
        assert kind == "goto" and kw["head"] is not None and kw["antennas"] is not None
        assert kw["body_yaw"] is None, "a head nudge re-drove the base: %r" % kw
        t.goto(0.8, antenna_left=0.2)
        kw = mini.calls[-1][1]
        assert kw["head"] is None and kw["antennas"] is not None and kw["body_yaw"] is None
        t.goto(1.4, body_yaw=0.5, head_yaw=0.0)
        kw = mini.calls[-1][1]
        assert kw["body_yaw"] == 0.5 and kw["head"] is not None and kw["antennas"] is None
        print("target-ok")
    """, paths.repo)
    assert "target-ok" in out


def test_a_recorded_move_ends_with_a_body_settle(paths):
    """T15.11: measured live, the base servo hunts after a move until it
    gets a clean interpolated target; every play_move now ends with one."""
    if not paths.robot_py.exists():
        pytest.skip("./.venv missing")
    out = run_in(paths.robot_py, """
        import sys, threading
        sys.path.insert(0, ".")
        import reachy_target
        from reachy_target import ReachyMiniTarget

        class Mini:
            def __init__(self): self.calls = []
            def play_move(self, move, **kw): self.calls.append(("play", move, kw))
            def goto_target(self, **kw): self.calls.append(("goto", kw))
            def set_target(self, **kw): self.calls.append(("set", kw))

        t = ReachyMiniTarget.__new__(ReachyMiniTarget)
        t._lock = threading.RLock()
        t._cmd = {n: 0.0 for n in reachy_target.DOF_NAMES}
        t._cmd["body_yaw"] = 0.25
        t._measured = dict(t._cmd, body_yaw=0.27, head_yaw=0.1)   # where the move left it
        mini = Mini()
        t._require = lambda: mini
        t._recorded_move = lambda dataset, move: "the-move"
        t.refresh = lambda: None
        posture = t.play_move("dance")
        kinds = [c[0] for c in mini.calls]
        assert kinds == ["play", "goto"], mini.calls
        assert mini.calls[0][2]["initial_goto_duration"] == 1.0
        settle = mini.calls[1][1]
        assert settle["body_yaw"] == 0.25 and settle["duration"] == 0.6, settle
        assert posture["body_yaw"] == 0.25 and posture["head_yaw"] == 0.1
        print("settle-ok")
    """, paths.repo)
    assert "settle-ok" in out


@pytest.mark.models
def test_embodiment_holds_and_tools_skip_during_a_dance(paths):
    out = run_in(paths.voice_py, """
        import asyncio, sys
        sys.path.insert(0, ".")
        from embodiment import Embodiment
        from agent import build_tools

        class Robot:
            def __init__(self): self.calls = []
            def posture(self, duration, **dofs): self.calls.append(("posture", dofs))
            def perform(self, name, repeat=1): self.calls.append(("perform", name, repeat))
            def nod(self, times=2): self.calls.append(("nod", times))
            def shake(self, times=2): self.calls.append(("shake", times))
            def home(self, duration=1.0): self.calls.append(("home",))

        class Params:
            def __init__(self, **arguments):
                self.arguments = arguments
                self.result = None
                self.function_name = "x"; self.tool_call_id = "1"
            async def result_callback(self, r): self.result = r

        async def main():
            robot = Robot()
            e = Embodiment(robot, sway=False)
            e._posture(0.5, head_pitch=0.1)
            assert robot.calls == [("posture", {"head_pitch": 0.1})]
            e.hold(0.4)
            e._posture(0.5, head_pitch=0.2)
            assert len(robot.calls) == 1, "embodiment moved while held"
            await asyncio.sleep(0.5)
            e._posture(0.5, head_pitch=0.3)
            assert len(robot.calls) == 2

            tools = {t.name: t.handler for t in build_tools(robot, None, e)}
            p = Params(move="cheer"); await tools["perform"](p)
            assert p.result["performing"] == "cheer" and e.holding
            assert ("perform", "cheer", 1) in robot.calls
            n = len(robot.calls)
            for name, args in (("move_head", {"yaw_degrees": 10}), ("nod", {}),
                               ("shake_head", {}), ("wiggle_antennas", {}),
                               ("turn_body", {"degrees": 20})):
                q = Params(**args); await tools[name](q)
                assert q.result.get("skipped"), (name, q.result)
            assert len(robot.calls) == n, "a tool moved the robot during the dance"
            print("voice-ok")
        asyncio.run(main())
    """, paths.voice_dir)
    assert "voice-ok" in out
