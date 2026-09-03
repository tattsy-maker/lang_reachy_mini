"""T3: the learner store — CRUD, tiers, notes ordering, collisions,
corruption, and the scripted enroll→notes→wipe E2E."""

import json
import re
import subprocess

import pytest

from tutor.store import Learner, LearnerStore

NOTE = """- **Practiced:** greetings
- **Struggled with:** bonjour vs bonsoir
- **Wins:** full self-introduction
- **Next time:** numbers 1-10"""


@pytest.fixture
def store(tmp_path):
    return LearnerStore(tmp_path / "learners")


def test_create_load_roundtrip(store):
    created = store.create("Maria", "es", level="intermediate",
                           tier="family", embedding=[0.1, -0.2])
    loaded = store.load(created.id)
    assert loaded == created
    assert loaded.name == "Maria" and loaded.tier == "family"
    assert loaded.sessions == 0 and loaded.embedding == [0.1, -0.2]
    # profile.json holds exactly the spec's fields
    raw = json.loads((store.root / created.id / "profile.json").read_text())
    assert set(raw) == {"name", "target_language", "level", "embedding",
                        "sessions", "last_seen", "tier",
                        "goal", "goal_note",       # T13.1
                        "voice_embedding"}         # T13.9


def test_save_updates_profile(store):
    learner = store.create("Sam", "fr")
    learner.level = "intermediate"
    store.save(learner)
    assert store.load(learner.id).level == "intermediate"


def test_validation(store):
    with pytest.raises(ValueError):
        store.create("X", "es", level="fluent")
    with pytest.raises(ValueError):
        store.create("X", "es", tier="vip")


def test_delete_forget_me(store):
    learner = store.create("Sam", "fr")
    assert store.delete(learner.id) is True
    assert store.load(learner.id) is None
    assert store.delete(learner.id) is False  # already gone


def test_name_collision_disambiguates_folder_keeps_display_name(store):
    first = store.create("Maria", "es")
    second = store.create("Maria", "fr")
    third = store.create("maría", "it")  # accents slug to the same folder
    assert first.id == "maria"
    assert second.id == "maria-2"
    assert third.id == "maria-3"
    assert store.load(second.id).name == "Maria"
    assert store.load(third.id).name == "maría"
    assert len(store.find_by_name("MARIA")) == 2


def test_notes_newest_first_with_session_bookkeeping(store):
    learner = store.create("Sam", "fr")
    store.append_session(learner.id, "first session body",
                         date="2026-08-29")
    updated = store.append_session(learner.id, "second session body",
                                   date="2026-08-30")
    assert updated.sessions == 2 and updated.last_seen == "2026-08-30"

    notes = store.read_notes(learner.id)
    dates = re.findall(r"^## (\d{4}-\d{2}-\d{2}) — Session (\d)", notes,
                       flags=re.M)
    assert dates == [("2026-08-30", "2"), ("2026-08-29", "1")]
    assert notes.index("second session body") < notes.index(
        "first session body")


def test_read_notes_truncates_to_newest(store):
    learner = store.create("Sam", "fr")
    for day in ("2026-08-28", "2026-08-29", "2026-08-30"):
        store.append_session(learner.id, f"body {day}", date=day)
    top = store.read_notes(learner.id, max_sessions=2)
    assert "body 2026-08-30" in top and "body 2026-08-29" in top
    assert "body 2026-08-28" not in top


def test_corrupt_profile_skipped_never_deleted(store, caplog):
    good = store.create("Maria", "es")
    bad = store.create("Sam", "fr")
    profile = store.root / bad.id / "profile.json"
    profile.write_text("{this is not json")

    with caplog.at_level("WARNING"):
        listed = store.list()
    assert [l.id for l in listed] == [good.id]
    assert "corrupt" in caplog.text
    assert profile.exists(), "corrupt profile must be left for a human"
    # and the wipe must not touch it either -- unreadable might be family
    store.wipe_guests()
    assert profile.exists()


def test_wipe_guests_spares_family(store):
    family = store.create("Maria", "es", tier="family")
    guest_a = store.create("Sam", "fr")  # guest is the default
    guest_b = store.create("Ana", "pt", tier="guest")
    gone = store.wipe_guests()
    assert sorted(gone) == sorted([guest_a.id, guest_b.id])
    assert [l.id for l in store.list()] == [family.id]


def test_e2e_enroll_two_sessions_wipe(paths, tmp_path):
    """The task's scripted E2E, via the CLI for the wipe step."""
    root = tmp_path / "learners"
    store = LearnerStore(root)
    family = store.create("Maria", "es", level="intermediate", tier="family")
    guest = store.create("Sam", "fr", embedding=[0.5] * 8)
    for learner in (family, guest):
        store.append_session(learner.id, NOTE, date="2026-08-30")
        store.append_session(learner.id, NOTE, date="2026-08-31")

    out = subprocess.run(
        ["python3", str(paths.repo / "tutor" / "wipe_guests.py"),
         "--root", str(root)],
        capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert "deleted guest sam" in out.stdout

    survivors = LearnerStore(root).list()
    assert [l.id for l in survivors] == [family.id]
    assert survivors[0].sessions == 2
    assert "Session 2" in store.read_notes(family.id)
