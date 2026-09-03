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


def read_wishes(path: str | os.PathLike | None = None) -> str:
    path = Path(path or DEFAULT_WISHES_FILE)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def count_wishes(path: str | os.PathLike | None = None) -> int:
    return sum(1 for line in read_wishes(path).splitlines()
               if line.startswith("- "))
