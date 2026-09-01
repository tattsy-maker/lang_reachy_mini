"""Shared plumbing for the language-tutor test suite (T0, see TASKS.md).

What lives here:

* **Marker-based skipping.** Tests marked ``robot`` / ``models`` /
  ``anthropic`` / ``google`` / ``audio`` skip with a printed reason when the
  prerequisite is absent, so the suite is green on any machine.
* **``stub_robot``** -- session fixture that serves the in-memory stub robot
  (``controller.py --stub serve``) on a pinned zenoh address and tears it
  down with SIGINT (SIGINT is mandatory; see CLAUDE.md).
* **``run_agent_say``** -- function fixture returning a helper that runs
  ``voice/agent.py --no-robot --say ...``, captures its log, waits for a
  pattern, and SIGINTs the agent. The repo's established way to drive a full
  agent turn without a microphone.

The zenoh port here is 7557, deliberately not the runbook's 7447, so the
suite never collides with a live ``serve`` on this machine.
"""

from __future__ import annotations

import glob
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent

# Tests import project modules (face.camera, later tutor.store) straight
# from the repo root — nothing here is pip-installed.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FIXTURES = TESTS_DIR / "fixtures"
REPORTS = TESTS_DIR / "reports"

ROBOT_PY = REPO / ".venv" / "bin" / "python"
VOICE_DIR = REPO / "voice"
VOICE_PY = VOICE_DIR / ".venv" / "bin" / "python"

TEST_ZENOH_LISTEN = "tcp/127.0.0.1:7557"
TEST_BROKER = "zenoh://127.0.0.1:7557"


# ---------------------------------------------------------------------------
# Marker prerequisites
# ---------------------------------------------------------------------------

def _env_or_dotenv(key: str) -> str | None:
    """A key from the environment, else from voice/.env (agent.py's source)."""
    if os.environ.get(key):
        return os.environ[key]
    dotenv = VOICE_DIR / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    return None


def _skip_reason(marker: str) -> str | None:
    """None if the marker's prerequisite is present, else a printable reason."""
    if marker == "robot":
        if not glob.glob("/dev/ttyACM*"):
            return "no robot on USB (/dev/ttyACM* absent)"
    elif marker == "models":
        if not VOICE_PY.exists():
            return "voice/.venv missing (big local models live there)"
    elif marker == "anthropic":
        if not _env_or_dotenv("ANTHROPIC_API_KEY"):
            return "no ANTHROPIC_API_KEY in environment or voice/.env"
    elif marker == "google":
        if not _env_or_dotenv("GOOGLE_API_KEY"):
            return "no GOOGLE_API_KEY in environment or voice/.env"
    elif marker == "audio":
        cards = Path("/proc/asound/cards")
        try:
            have_card = cards.exists() and cards.read_text().strip() not in ("", "--- no soundcards ---")
        except OSError:
            have_card = False
        if not have_card:
            return "no sound card (/proc/asound/cards empty)"
        if not os.access("/dev/snd", os.R_OK | os.X_OK):
            return "cannot access /dev/snd (audio group membership? see CLAUDE.md)"
    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker in ("robot", "models", "anthropic", "google", "audio"):
            if item.get_closest_marker(marker):
                reason = _skip_reason(marker)
                if reason:
                    item.add_marker(pytest.mark.skip(reason=reason))


# ---------------------------------------------------------------------------
# Paths, for tests (conftest isn't importable from per-task dirs)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def paths():
    return SimpleNamespace(
        repo=REPO, tests=TESTS_DIR, fixtures=FIXTURES, reports=REPORTS,
        robot_py=ROBOT_PY, voice_dir=VOICE_DIR, voice_py=VOICE_PY,
        broker=TEST_BROKER,
    )


# ---------------------------------------------------------------------------
# Log tailing for subprocesses
# ---------------------------------------------------------------------------

class LogTail:
    """Reads a process's merged output on a thread; lets callers wait for a
    pattern without risking a blocked pipe."""

    def __init__(self, stream):
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._pump, args=(stream,),
                                        daemon=True)
        self._thread.start()

    def _pump(self, stream):
        for line in stream:
            with self._lock:
                self._lines.append(line)

    def text(self) -> str:
        with self._lock:
            return "".join(self._lines)

    def wait_for(self, pattern: str, timeout: float,
                 proc: subprocess.Popen | None = None) -> bool:
        """True once ``pattern`` (a regex) appears in the output. Gives up
        early if ``proc`` exits — no point waiting out the timeout on a
        process that already died."""
        rx = re.compile(pattern)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if rx.search(self.text()):
                return True
            if proc is not None and proc.poll() is not None:
                time.sleep(0.5)  # let the reader thread drain the pipe
                break
            time.sleep(0.25)
        return rx.search(self.text()) is not None


def _sigint_and_wait(proc: subprocess.Popen, timeout: float = 20.0) -> None:
    """SIGINT (never SIGKILL first -- cleanup matters, see CLAUDE.md), then
    escalate only if the process ignores it."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Fixture: the stub robot, served on a pinned zenoh address
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def stub_robot():
    """Serve ``controller.py --stub`` on TEST_BROKER for the whole session.

    Yields a namespace with ``broker`` and ``log()``. Skips if the robot
    virtualenv (./.venv) hasn't been built on this machine.
    """
    if not ROBOT_PY.exists():
        pytest.skip("./.venv missing (robot virtualenv; see CLAUDE.md setup)")

    proc = subprocess.Popen(
        [str(ROBOT_PY), "controller.py", "--stub", "serve",
         "--zenoh-listen", TEST_ZENOH_LISTEN],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    tail = LogTail(proc.stdout)
    try:
        if not tail.wait_for(r"\[reachy\] serving", timeout=30):
            _sigint_and_wait(proc)
            pytest.fail("stub serve never reported readiness; log:\n"
                        + tail.text())
        yield SimpleNamespace(broker=TEST_BROKER, log=tail.text, proc=proc)
    finally:
        _sigint_and_wait(proc)


# ---------------------------------------------------------------------------
# Fixture: drive the voice agent by injected text (--say)
# ---------------------------------------------------------------------------

@pytest.fixture
def run_agent_say():
    """Returns ``run(utterances, ...) -> (log_text, pattern_found)``.

    Runs ``voice/agent.py --no-robot --say <u> ...`` from the voice venv,
    waits until ``wait_for`` (regex) shows up in the merged log (default: the
    pipecat line proving a reply reached synthesis), lets audio drain for
    ``settle`` seconds, then SIGINTs the agent. Never asserts on exit code:
    exit 1 after SIGINT is normal (CLAUDE.md).

    Startup without warmup is ~15-30s on this box; the default timeout leaves
    room for a cold model load.
    """
    if not VOICE_PY.exists():
        pytest.skip("voice/.venv missing (agent virtualenv; see CLAUDE.md)")

    procs: list[subprocess.Popen] = []

    def run(utterances, wait_for=r"Generating TTS \[", timeout=240.0,
            settle=8.0, extra_args=(), deaf=True, also_wait_for=None,
            no_robot=True):
        cmd = [str(VOICE_PY), "agent.py", "--no-warmup", "--say-delay", "2"]
        if no_robot:
            cmd.append("--no-robot")
        if deaf:
            cmd.append("--deaf")  # scripted runs must not hear the room
        for u in utterances:
            cmd += ["--say", u]
        cmd += list(extra_args)
        proc = subprocess.Popen(cmd, cwd=VOICE_DIR, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs.append(proc)
        tail = LogTail(proc.stdout)
        found = tail.wait_for(wait_for, timeout, proc=proc)
        if found and also_wait_for:
            # two-phase wait: e.g. a tool firing, then the spoken reply
            found = tail.wait_for(also_wait_for, 45.0, proc=proc)
        if found:
            time.sleep(settle)  # let the reply actually play out
        _sigint_and_wait(proc)
        return tail.text(), found

    yield run
    for proc in procs:
        _sigint_and_wait(proc)
