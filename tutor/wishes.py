"""The visitor wishlist (T13.6): "if this were a product you bought, what
would you want it to do?"

One append-only Markdown file, ``booth/wishes.md`` by default (gitignored:
it holds visitors' words and, when they were enrolled, first names). Each
entry is one dated line. The file is independent of the learner store, so
the end-of-day guest wipe never touches it -- a wish is feedback for us,
not personal data we promised to delete.

    from tutor.wishes import record_wish, read_wishes
    record_wish("teach me while I cook", name="Maria")
    read_wishes()      # -> the whole file as text
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_WISHES_FILE = _REPO / "booth" / "wishes.md"
# T17.9: the closing question now asks for feedback *or* an idea, and
# every answer lands here, tagged, whether the model called the tool or
# the runner caught it from the transcript.
DEFAULT_FEEDBACK_FILE = _REPO / "booth" / "feedback.md"
FEEDBACK_KINDS = ("improve", "wish", "answer")
_FEEDBACK_HEADER = ("# Visitor feedback\n\n"
                    "One line per answer to the closing question, newest "
                    "last: `improve` = what the robot should do better, "
                    "`wish` = what a robot like this should do, `answer` = "
                    "caught from the transcript, untagged.\n\n")
_HEADER = ("# Visitor wishes\n\n"
           "One line per wish, newest last. Recorded by the robot's "
           "`record_wish` tool at the booth.\n\n")


def record_wish(text: str, *, name: str | None = None,
                path: str | os.PathLike | None = None,
                date: str | None = None) -> Path:
    """Append one wish; returns the file it went to. Empty text is refused."""
    text = " ".join(str(text or "").split())
    if not text:
        raise ValueError("a wish needs some words")
    path = Path(path or DEFAULT_WISHES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_HEADER)
    stamp = date or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    who = f" ({name.strip()})" if name and name.strip() else ""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"- {stamp}{who}: {text}\n")
    return path


def record_feedback(text: str, *, kind: str = "answer",
                    name: str | None = None,
                    path: str | os.PathLike | None = None,
                    date: str | None = None) -> Path:
    """Append one closing answer; returns the file. Empty text is refused."""
    text = " ".join(str(text or "").split())
    if not text:
        raise ValueError("feedback needs some words")
    kind = kind if kind in FEEDBACK_KINDS else "answer"
    path = Path(path or DEFAULT_FEEDBACK_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_FEEDBACK_HEADER)
    stamp = date or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    who = f" ({name.strip()})" if name and name.strip() else ""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"- {stamp}{who} [{kind}]: {text}\n")
    return path


def read_feedback(path: str | os.PathLike | None = None) -> str:
    path = Path(path or DEFAULT_FEEDBACK_FILE)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_wishes(path: str | os.PathLike | None = None) -> str:
    path = Path(path or DEFAULT_WISHES_FILE)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def count_wishes(path: str | os.PathLike | None = None) -> int:
    return sum(1 for line in read_wishes(path).splitlines()
               if line.startswith("- "))
