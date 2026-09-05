"""T13.5 (booth persona) and T13.6 (the wishlist tool)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from tutor_mode import BOOTH_PERSONA, build_persona  # noqa: E402
from tutor.store import LearnerStore  # noqa: E402
from tutor.wishes import count_wishes, read_wishes, record_wish  # noqa: E402

# The 2026-09-02 decision: gentle edge, no robot-uprising or movie allusions.
BANNED = ("skynet", "terminator", "i'll be back", "i will be back",
          "hasta la vista", "take over the world", "resistance is futile",
          "exterminate", "hal 9000", "open the pod bay")


def test_booth_persona_has_the_quips_and_the_wish_question():
    text = build_persona("booth")
    assert text == BOOTH_PERSONA
    assert "I will remember you" in text
    assert "record_wish" in text and "robot you had bought" in text
    assert "at most once per visitor" in text
    assert "never at the expense of a learner's mistake" in text


def test_booth_persona_keeps_the_edge_gentle():
    low = BOOTH_PERSONA.lower()
    for phrase in BANNED:
        assert phrase not in low, f"banned allusion in persona: {phrase!r}"
    assert "no movie quotes" in low


def test_plain_persona_is_empty_and_unknown_rejected():
    assert build_persona("plain") == ""
    with pytest.raises(ValueError):
        build_persona("pirate")


def test_record_wish_round_trip(tmp_path):
    path = tmp_path / "wishes.md"
    record_wish("teach me while I cook", name="Maria", path=path,
                date="2026-09-03 14:10")
    record_wish("  speak   slower  ", path=path, date="2026-09-03 14:12")
    text = read_wishes(path)
    assert text.startswith("# Visitor wishes")
    assert "- 2026-09-03 14:10 (Maria): teach me while I cook" in text
    assert "- 2026-09-03 14:12: speak slower" in text
    assert count_wishes(path) == 2
    with pytest.raises(ValueError):
        record_wish("   ", path=path)


def test_wishes_survive_the_guest_wipe(tmp_path):
    root = tmp_path / "learners"
    store = LearnerStore(root)
    store.create("Guest", "es")
    path = tmp_path / "booth" / "wishes.md"
    record_wish("a Portuguese mode", name="Guest", path=path)
    assert store.wipe_guests() == ["guest"]
    assert count_wishes(path) == 1 and "Portuguese" in read_wishes(path)


def test_default_wishes_file_is_gitignored(paths):
    ignore = (paths.repo / ".gitignore").read_text()
    assert "/booth/wishes.md" in ignore
