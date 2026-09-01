"""T11: the startup script — static checks anywhere, live smoke on metal."""

import os
import re
import subprocess
import time
from pathlib import Path

import pytest


def test_script_is_executable_and_valid_bash(paths):
    script = paths.repo / "start_booth.sh"
    assert script.exists() and os.access(script, os.X_OK)
    out = subprocess.run(["bash", "-n", str(script)],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr


def test_signage_covers_the_disclosures(paths):
    text = (paths.repo / "booth" / "SIGNAGE.md").read_text().lower()
    for required in ("face", "deleted at the end of the day", "forget me",
                     "not* a photo", "gemini", "claude"):
        assert required in text, f"signage missing disclosure: {required!r}"


@pytest.mark.robot
@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_booth_script_comes_up_and_sigints_clean(paths):
    if not os.access("/dev/video0", os.R_OK):
        pytest.skip("no read access to /dev/video0 "
                    "(needs 'video' group; see face/camera.py)")
    env = dict(os.environ, BOOTH_KEEP_GUESTS="1",
               BOOTH_EXTRA_AGENT="--deaf")
    proc = subprocess.Popen(["bash", str(paths.repo / "start_booth.sh")],
                            cwd=paths.repo, env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    ready = False
    deadline = time.monotonic() + 180
    lines = []
    try:
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            if "the booth is live" in line:
                ready = True
                break
        assert ready, "booth never reached ready:\n" + "".join(lines[-40:])
    finally:
        proc.send_signal(2)  # SIGINT, the mandated shutdown path
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
    tail = "".join(lines) + (proc.stdout.read() or "")
    assert "booth down" in tail, "shutdown path never completed:\n" + tail[-2000:]
