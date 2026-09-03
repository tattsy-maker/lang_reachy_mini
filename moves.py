#!/usr/bin/env python3
"""The curated recorded-move library (T13.4): what "can you dance?" plays.

Pollen ships two HuggingFace datasets of recorded moves --
``pollen-robotics/reachy-mini-dances-library`` and
``pollen-robotics/reachy-mini-emotions-library`` -- that the vendor daemon
plays through ``ReachyMini.play_move``. This module picks the handful
worth exposing to a language model at a booth, gives each a short
description the model can choose by, and knows how to pre-fetch the
datasets so the venue's internet is never on the critical path.

Two names are not recorded moves at all: ``spin`` is a full body-yaw
sweep the driver performs itself, and ``wiggle`` is the antenna flick the
voice agent already had. Both live here so the model sees one list.

    python moves.py --list       # names, sources, descriptions
    python moves.py --preload    # fetch both datasets into the HF cache

Imports stdlib only at module level; the HuggingFace client is pulled in
lazily, so the voice agent (a different venv) can import the table.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

DANCES = "pollen-robotics/reachy-mini-dances-library"
EMOTIONS = "pollen-robotics/reachy-mini-emotions-library"
DATASETS = (DANCES, EMOTIONS)


@dataclass(frozen=True)
class MoveSpec:
    name: str            # what the model asks for
    dataset: str | None  # None = built into the driver
    move: str | None     # the dataset's file stem
    seconds: float       # rough duration, for pausing trackers and speech
    description: str


# Names are what the model sees; keep them plain English and few. Durations
# are approximate (recorded clips are 3-12 s) and only steer "how long to
# hold off the face tracker", so a little slack is fine.
LIBRARY: dict[str, MoveSpec] = {m.name: m for m in (
    MoveSpec("dance", EMOTIONS, "dance1", 9.0,
             "a short full-body dance; the default when asked to dance"),
    MoveSpec("dance_groovy", DANCES, "groovy_sway_and_roll", 8.0,
             "a groovier sway-and-roll dance"),
    MoveSpec("dance_pendulum", DANCES, "pendulum_swing", 7.0,
             "a slow pendulum swing side to side"),
    MoveSpec("spin", None, None, 6.0,
             "turn the whole body all the way round one way and back"),
    MoveSpec("dizzy", DANCES, "dizzy_spin", 6.0,
             "a wobbly, dizzy head spin"),
    MoveSpec("peekaboo", DANCES, "side_peekaboo", 5.0,
             "peek out to one side and back, playful"),
    MoveSpec("cheer", EMOTIONS, "cheerful1", 4.0,
             "a cheerful bounce; celebrate a right answer"),
    MoveSpec("amazed", EMOTIONS, "amazed1", 4.0,
             "a wide-eyed amazed reaction"),
    MoveSpec("curious", EMOTIONS, "curious1", 4.0,
             "lean in, curious"),
    MoveSpec("confused", EMOTIONS, "confused1", 4.0,
             "a puzzled head tilt"),
    MoveSpec("grateful", EMOTIONS, "grateful1", 4.0,
             "a small thankful bow"),
    MoveSpec("wiggle", None, None, 1.0,
             "flick both antennas, delighted"),
)}

# What the idle attractor plays when nobody has been in frame for a while
# (T13.4): short, visible from across a hall, no sound needed.
ATTRACT_MOVES = ("dance", "peekaboo", "dance_groovy")


def names() -> list[str]:
    return list(LIBRARY)


def describe() -> str:
    """One line per move, for a tool description."""
    return "; ".join(f"{m.name}: {m.description}" for m in LIBRARY.values())


def preload(datasets=DATASETS) -> dict[str, str | None]:
    """Fetch the datasets into the HuggingFace cache. Returns
    {dataset: local_path or None}; never raises -- venue internet is a
    known risk and the booth must start without it."""
    from huggingface_hub import snapshot_download
    out = {}
    for name in datasets:
        try:
            out[name] = snapshot_download(name, repo_type="dataset")
        except Exception as exc:                           # noqa: BLE001
            print(f"[moves] could not fetch {name}: {exc}", file=sys.stderr)
            out[name] = None
    return out


def cached(datasets=DATASETS) -> dict[str, bool]:
    """Which datasets are already in the local cache (no network)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    out = {}
    for name in datasets:
        try:
            snapshot_download(name, repo_type="dataset", local_files_only=True)
            out[name] = True
        except (LocalEntryNotFoundError, Exception):        # noqa: BLE001
            out[name] = False
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--cached", action="store_true",
                    help="report which datasets are cached, exit 1 if any "
                         "is missing")
    a = ap.parse_args()
    if a.list or not (a.preload or a.cached):
        for m in LIBRARY.values():
            src = f"{m.dataset}/{m.move}" if m.dataset else "built-in"
            print(f"{m.name:16s} {m.seconds:4.0f}s  {src:60s} {m.description}")
    if a.preload:
        for name, path in preload().items():
            print(f"{name}: {path or 'FAILED'}")
    if a.cached:
        status = cached()
        for name, ok in status.items():
            print(f"{name}: {'cached' if ok else 'MISSING'}")
        return 0 if all(status.values()) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
