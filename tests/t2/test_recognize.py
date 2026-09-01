"""T2: face recognition core.

The decision band is unit-tested in this venv (match() needs only numpy).
Everything that touches InsightFace runs through face/recognize.py's CLI
under voice/.venv (where the face deps live) and is `models`-marked —
same subprocess pattern as the agent tests.

Threshold assertions use the *measured* fixture scores (see the constants
in face/recognize.py and progress/T2.md), not guessed margins.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from face.recognize import (  # noqa: E402  (numpy-only imports)
    ACCEPT_THRESHOLD,
    REJECT_THRESHOLD,
    Match,
    match,
    similarity,
)

PEOPLE = {"scott": ("scott_a", "scott_b"),
          "sunita": ("sunita_a", "sunita_b"),
          "tracy": ("tracy_a", "tracy_b")}


def run_cli(paths, *args) -> dict:
    out = subprocess.run(
        [str(paths.voice_py), str(paths.repo / "face" / "recognize.py"),
         *args],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


# -- decision band (no model needed) -----------------------------------------

def vector_at_similarity(base: np.ndarray, target: float) -> np.ndarray:
    """A unit vector whose cosine similarity to unit ``base`` is ``target``."""
    other = np.zeros_like(base)
    other[1] = 1.0
    v = target * base + np.sqrt(1 - target ** 2) * other
    return v / np.linalg.norm(v)


def test_match_decision_band():
    base = np.zeros(8, dtype=np.float32)
    base[0] = 1.0
    known = {"maria": base}
    mid = (ACCEPT_THRESHOLD + REJECT_THRESHOLD) / 2

    sure = match(vector_at_similarity(base, ACCEPT_THRESHOLD + 0.1), known)
    assert isinstance(sure, Match) and sure.name == "maria" and sure.sure

    unsure = match(vector_at_similarity(base, mid), known)
    assert unsure is not None and not unsure.sure, \
        "between the thresholds must be ask-don't-guess, not accept/reject"

    assert match(vector_at_similarity(base, REJECT_THRESHOLD - 0.1),
                 known) is None
    assert match(base, {}) is None


def test_match_picks_best_of_several():
    a = np.zeros(4); a[0] = 1.0
    b = np.zeros(4); b[1] = 1.0
    got = match(a * 0.9 + b * 0.1, {"a": a, "b": b})
    assert got.name == "a"
    assert similarity(a, a) == pytest.approx(1.0)


# -- against the real model ---------------------------------------------------

@pytest.mark.models
def test_fixture_matrix_separates_people(paths):
    result = run_cli(paths, "matrix", str(paths.fixtures / "faces"))
    assert all(result["faces"].values()), "a fixture photo lost its face"
    scores = result["scores"]

    def score(a, b):
        return scores.get(f"{a}|{b}", scores.get(f"{b}|{a}"))

    for person, (first, second) in PEOPLE.items():
        assert score(first, second) >= ACCEPT_THRESHOLD, \
            f"same-person pair {person} fell below accept"
    for pa, (a1, _) in PEOPLE.items():
        for pb, (b1, _) in PEOPLE.items():
            if pa < pb:
                assert score(a1, b1) < REJECT_THRESHOLD, \
                    f"different people {pa}/{pb} not rejected"


@pytest.mark.models
def test_no_face_image_yields_none(paths, tmp_path):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    blank[:, :, 1] = np.linspace(0, 255, 640, dtype=np.uint8)  # gradient
    cv2.imwrite(str(tmp_path / "landscape.jpg"), blank)
    result = run_cli(paths, "matrix", str(tmp_path))
    assert result["faces"] == {"landscape": False}
    assert result["scores"] == {}


@pytest.mark.models
def test_multiface_frame_uses_largest_face(paths, tmp_path):
    big = cv2.imread(str(paths.fixtures / "faces" / "sunita_b.jpg"))
    small = cv2.imread(str(paths.fixtures / "faces" / "scott_a.jpg"))
    small = cv2.resize(small, (big.shape[1] // 4, big.shape[0] // 4))
    composite = big.copy()
    composite[0:small.shape[0], 0:small.shape[1]] = small
    cv2.imwrite(str(tmp_path / "composite.jpg"), composite)
    for name in ("sunita_b", "scott_a"):
        cv2.imwrite(str(tmp_path / f"{name}.jpg"),
                    cv2.imread(str(paths.fixtures / "faces" / f"{name}.jpg")))

    scores = run_cli(paths, "matrix", str(tmp_path))["scores"]
    assert scores["composite|sunita_b"] >= ACCEPT_THRESHOLD, \
        "largest face (sunita) should dominate the composite's embedding"
    assert scores["composite|scott_a"] < REJECT_THRESHOLD, \
        "the small face (scott) leaked into the embedding"


@pytest.mark.models
def test_e2e_enroll_and_identify_in_video(paths):
    faces = paths.fixtures / "faces"
    result = run_cli(
        paths, "identify",
        "--enroll", f"{faces}/sunita_a.jpg,{faces}/sunita_b.jpg",
        "--source", str(paths.fixtures / "video" / "sunita_clip.avi"),
        "--name", "sunita")
    assert result["frames"] == 12          # T1's rate-cap arithmetic
    assert result["matched"] == result["frames"]
    assert result["min_score"] >= ACCEPT_THRESHOLD
