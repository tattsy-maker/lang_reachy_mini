"""T0: the committed fixtures are actually usable by later tasks."""

import json
import re

import cv2
import pytest

PEOPLE = {"sunita": 2, "tracy": 2, "scott": 2}


def test_face_photos_present_and_decodable(paths):
    faces = paths.fixtures / "faces"
    for person, count in PEOPLE.items():
        for suffix in "ab"[:count]:
            path = faces / f"{person}_{suffix}.jpg"
            assert path.exists(), f"missing fixture photo {path}"
            img = cv2.imread(str(path))
            assert img is not None, f"{path} does not decode"
            h, w = img.shape[:2]
            assert min(h, w) >= 400, f"{path} too small for face work ({w}x{h})"


def test_video_clip_opens_and_has_frames(paths):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    assert clip.exists(), "fixture video missing (tests/fixtures/make_video.py)"
    cap = cv2.VideoCapture(str(clip))
    assert cap.isOpened()
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        assert frame.shape[:2] == (480, 640)
    cap.release()
    assert frames >= 60, f"clip too short ({frames} frames)"


@pytest.mark.parametrize("learner,tier", [("maria", "family"), ("sam", "guest")])
def test_learner_sample_tree(paths, learner, tier):
    folder = paths.fixtures / "learners" / learner
    profile = json.loads((folder / "profile.json").read_text())
    for key in ("name", "target_language", "level", "embedding",
                "sessions", "last_seen", "tier"):
        assert key in profile, f"{learner}/profile.json missing {key!r}"
    assert profile["tier"] == tier
    assert isinstance(profile["embedding"], list) and profile["embedding"]

    notes = (folder / "notes.md").read_text()
    for section in ("Practiced", "Struggled with", "Wins", "Next time"):
        assert f"**{section}:**" in notes, f"{learner}/notes.md missing {section}"
    # newest-first ordering: session headings must be in descending date order
    dates = re.findall(r"^## (\d{4}-\d{2}-\d{2})", notes, flags=re.M)
    assert dates == sorted(dates, reverse=True), "notes not newest-first"
