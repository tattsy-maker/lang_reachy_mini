"""The learner store: one folder per learner, plain files, no database.

This is the stable API that T4 (briefing + memory tools), T9 (enrollment)
and T10 (session lifecycle) build against. Layout per the spec (section
4B)::

    learners/
      maria/
        profile.json    # structured facts, see PROFILE_KEYS
        notes.md        # tutor-written prose, newest session first
      maria-2/          # a second Maria: folder disambiguated, display
        ...             # name kept in profile.json

Usage::

    store = LearnerStore()                      # default root: ./learners
    learner = store.create("Maria", "es", level="intermediate",
                           tier="family", embedding=[...])
    learner = store.load(learner.id)            # None if absent/corrupt
    store.append_session(learner.id, body)      # prepends a notes entry,
                                                #   bumps sessions/last_seen
    store.read_notes(learner.id, max_sessions=3)
    store.list()                                # corrupt profiles are
                                                #   skipped with a warning,
                                                #   never deleted
    store.delete(learner.id)                    # "forget me"
    store.wipe_guests()                         # end of day: every guest
                                                #   folder goes, family stays

``id`` is the folder name (a slug of the display name, ``-2``/``-3``... on
collision); ``name`` is what was spoken at enrollment and what greetings
use. Levels: beginner / intermediate / advanced. Tiers: family (permanent)
/ guest (wiped at close of day).

Goals (T13.1, from the family debrief: "one wants conversation, another an
exam, a third a job interview"): ``goal`` is one of GOALS and ``goal_note``
is the learner's own words. Both are optional on disk -- a profile written
before T13 loads with ``goal="conversation"`` and an empty note. So is
``voice_embedding`` (T13.9): the speaker print, kept next to the face
signature and deleted with it.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("tutor.store")

LEVELS = ("beginner", "intermediate", "advanced")
TIERS = ("family", "guest")
GOALS = ("conversation", "exam", "work", "travel", "other")
PROFILE_KEYS = ("name", "target_language", "level", "embedding",
                "sessions", "last_seen", "tier", "goal", "goal_note",
                "voice_embedding")
# Keys a pre-T13 profile.json may lack, with the value they load as.
# ``voice_embedding`` (T13.9) is the ECAPA speaker print, 192 floats, or
# None until the tutor has heard enough of the learner to keep one.
OPTIONAL_KEYS = {"goal": "conversation", "goal_note": "",
                 "voice_embedding": None}

_NOTES_HEADER = "# {name} — session notes\n"
_ENTRY_RX = re.compile(r"^## \d{4}-\d{2}-\d{2}", re.M)


def _slugify(name: str) -> str:
    """A filesystem-safe folder name from a spoken display name."""
    ascii_name = (unicodedata.normalize("NFKD", name)
                  .encode("ascii", "ignore").decode())
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "learner"


def _today() -> str:
    return _dt.date.today().isoformat()


@dataclass
class Learner:
    id: str
    name: str
    target_language: str
    level: str = "beginner"
    embedding: list[float] | None = None
    sessions: int = 0
    last_seen: str = field(default_factory=_today)
    tier: str = "guest"
    goal: str = "conversation"
    goal_note: str = ""
    voice_embedding: list[float] | None = None

    def profile_dict(self) -> dict:
        return {key: getattr(self, key) for key in PROFILE_KEYS}


class LearnerStore:
    def __init__(self, root: str | Path = "learners"):
        self.root = Path(root)

    # -- paths -------------------------------------------------------------

    def _folder(self, learner_id: str) -> Path:
        return self.root / learner_id

    def _profile_path(self, learner_id: str) -> Path:
        return self._folder(learner_id) / "profile.json"

    def _notes_path(self, learner_id: str) -> Path:
        return self._folder(learner_id) / "notes.md"

    # -- CRUD --------------------------------------------------------------

    def create(self, name: str, target_language: str, *,
               level: str = "beginner", tier: str = "guest",
               embedding: list[float] | None = None,
               goal: str = "conversation", goal_note: str = "",
               voice_embedding: list[float] | None = None) -> Learner:
        """New learner folder; the id disambiguates on name collision."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        if goal not in GOALS:
            raise ValueError(f"goal must be one of {GOALS}, got {goal!r}")
        base = _slugify(name)
        learner_id, n = base, 1
        while self._folder(learner_id).exists():
            n += 1
            learner_id = f"{base}-{n}"
        learner = Learner(id=learner_id, name=name,
                          target_language=target_language, level=level,
                          embedding=list(embedding) if embedding else None,
                          tier=tier, goal=goal,
                          goal_note=str(goal_note or "").strip(),
                          voice_embedding=(list(voice_embedding)
                                           if voice_embedding else None))
        self._folder(learner_id).mkdir(parents=True)
        self.save(learner)
        self._notes_path(learner_id).write_text(
            _NOTES_HEADER.format(name=name))
        return learner

    def save(self, learner: Learner) -> None:
        self._profile_path(learner.id).write_text(
            json.dumps(learner.profile_dict(), indent=2, ensure_ascii=False)
            + "\n")

    def load(self, learner_id: str) -> Learner | None:
        """The learner, or None if absent or corrupt (corrupt profiles are
        warned about and left in place for a human — never deleted)."""
        path = self._profile_path(learner_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            fields = {key: data[key] for key in PROFILE_KEYS
                      if key not in OPTIONAL_KEYS}
            for key, default in OPTIONAL_KEYS.items():
                fields[key] = data.get(key, default)
            if fields["goal"] not in GOALS:
                fields["goal"] = "other"
            return Learner(id=learner_id, **fields)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("skipping corrupt profile %s (%s) — "
                           "left on disk for inspection", path, exc)
            return None

    def list(self) -> list[Learner]:
        if not self.root.exists():
            return []
        learners = (self.load(p.name) for p in sorted(self.root.iterdir())
                    if p.is_dir())
        return [learner for learner in learners if learner is not None]

    def find_by_name(self, name: str) -> list[Learner]:
        """All learners whose display name matches (case-insensitive)."""
        want = name.strip().lower()
        return [l for l in self.list() if l.name.strip().lower() == want]

    def delete(self, learner_id: str) -> bool:
        """'Forget me': the whole folder, on the spot. True if it existed."""
        folder = self._folder(learner_id)
        if not folder.is_dir():
            return False
        shutil.rmtree(folder)
        return True

    # -- notes -------------------------------------------------------------

    def append_session(self, learner_id: str, body: str,
                       date: str | None = None) -> Learner:
        """Prepend one session entry to notes.md (newest first) and do the
        bookkeeping: sessions += 1, last_seen = date. Returns the updated
        learner. ``body`` is the tutor's four-section prose."""
        learner = self.load(learner_id)
        if learner is None:
            raise KeyError(f"no learner {learner_id!r}")
        date = date or _today()
        learner.sessions += 1
        learner.last_seen = date
        self.save(learner)

        entry = (f"## {date} — Session {learner.sessions}\n\n"
                 + body.strip() + "\n")
        notes_path = self._notes_path(learner_id)
        text = (notes_path.read_text() if notes_path.exists()
                else _NOTES_HEADER.format(name=learner.name))
        match = _ENTRY_RX.search(text)
        if match:  # insert before the (previously) newest entry
            text = text[:match.start()] + entry + "\n" + text[match.start():]
        else:
            text = text.rstrip() + "\n\n" + entry
        notes_path.write_text(text)
        return learner

    def read_notes(self, learner_id: str,
                   max_sessions: int | None = None) -> str:
        """The notes file, optionally truncated to the newest N sessions
        (they are stored newest-first, so truncation keeps the top)."""
        path = self._notes_path(learner_id)
        if not path.exists():
            return ""
        text = path.read_text()
        if max_sessions is None:
            return text
        starts = [m.start() for m in _ENTRY_RX.finditer(text)]
        if len(starts) <= max_sessions:
            return text
        return text[:starts[max_sessions]].rstrip() + "\n"

    # -- tiers -------------------------------------------------------------

    def wipe_guests(self) -> list[str]:
        """Delete every guest-tier learner; family survives. Returns the
        deleted ids. Corrupt profiles are never wiped — a profile we cannot
        read might be family."""
        gone = []
        for learner in self.list():
            if learner.tier == "guest":
                self.delete(learner.id)
                gone.append(learner.id)
        return gone
