"""T0: the stub-robot fixture serves, answers an attached client, and the
teardown (SIGINT) is exercised by the fixture itself at session end."""

import subprocess


def test_stub_serve_answers_attached_client(paths, stub_robot):
    out = subprocess.run(
        [str(paths.robot_py), "controller.py",
         "--attach", stub_robot.broker, "slots"],
        cwd=paths.repo, capture_output=True, text=True, timeout=90)
    assert out.returncode == 0, f"slots failed:\n{out.stdout}\n{out.stderr}"
    # the slots table's footer proves a real command round-trip, not just a
    # transport connect
    assert "motors_enabled=" in out.stdout
