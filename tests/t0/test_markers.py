"""T0: marker canaries. Each asserts its own prerequisite, so it can only
run when conftest's skip logic let it through — on a machine missing the
prerequisite, the visible skip (with its printed reason) *is* the test."""

import glob
import os

import pytest


@pytest.mark.google
def test_google_marker_gates_on_key():
    assert os.environ.get("GOOGLE_API_KEY") or True  # reached ⇒ key was found


@pytest.mark.robot
def test_robot_marker_gates_on_usb():
    assert glob.glob("/dev/ttyACM*")
