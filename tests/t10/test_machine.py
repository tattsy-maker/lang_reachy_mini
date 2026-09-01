"""T10: the session state machine, driven by a fake clock. No models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from session import ACTIVE, WATCHING, SessionMachine  # noqa: E402


def make():
    return SessionMachine(stable_secs=2.0, absent_secs=60.0)


def test_passer_by_never_starts():
    m = make()
    assert m.on_face(True, 0.0) is None
    assert m.on_face(True, 1.0) is None      # only 1s stable
    assert m.on_face(False, 1.5) is None     # gone again: reset
    assert m.on_face(True, 2.0) is None      # a new appearance starts over
    assert m.on_face(True, 3.9) is None
    assert m.state == WATCHING


def test_stable_face_starts_once_confirmed():
    m = make()
    m.on_face(True, 0.0)
    assert m.on_face(True, 2.0) == "start"
    # advice repeats until the caller confirms
    assert m.on_face(True, 2.5) == "start"
    m.session_started(2.5)
    assert m.state == ACTIVE
    assert m.on_face(True, 3.0) is None


def test_walk_away_ends_after_absence_and_only_then():
    m = make()
    m.on_face(True, 0.0)
    m.on_face(True, 2.0)
    m.session_started(2.0)
    assert m.on_face(False, 30.0) is None            # gone 28s: not yet
    assert m.on_face(True, 40.0) is None             # came back: timer resets
    assert m.on_face(False, 99.0) is None            # gone 59s
    assert m.on_face(False, 100.1) == "end"          # gone 60.1s
    m.session_ended()
    assert m.state == WATCHING


def test_two_visitors_in_a_row():
    m = make()
    for start in (0.0, 200.0):
        m.on_face(True, start)
        assert m.on_face(True, start + 2.0) == "start"
        m.session_started(start + 2.0)
        assert m.on_face(False, start + 100.0) == "end"
        m.session_ended()
    assert m.state == WATCHING
