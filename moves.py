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
    python moves.py --measure    # each clip's peak speeds, and its speed

Imports stdlib only at module level; the HuggingFace client is pulled in
lazily, so the voice agent (a different venv) can import the table.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

MAX_PERFORM_SECS = 60.0

DANCES = "pollen-robotics/reachy-mini-dances-library"
EMOTIONS = "pollen-robotics/reachy-mini-emotions-library"
DATASETS = (DANCES, EMOTIONS)


@dataclass(frozen=True)
class MoveSpec:
    name: str            # what the model asks for
    dataset: str | None  # None = built into the driver
    move: str | None     # the dataset's file stem
    seconds: float       # the clip's own length (recorded clips are 2-18 s)
    description: str
    default_secs: float = 0.0   # how long "perform" runs it unless told
                                # otherwise (0 = one pass); T14.5: dances
                                # loop to ~30 s, the family wanted longer
    attract_only: bool = False  # 2026-09-25: played by the idle attractor
                                # only; kept out of the model's list
    speed: float = 1.0          # playback rate; below 1 slows a clip that
                                # is faster than the CALM limits (--measure)

    @property
    def pass_secs(self) -> float:
        """One pass as played: the clip at its speed, plus the overhead."""
        return self.seconds / self.speed + (PASS_OVERHEAD_SECS
                                            if self.dataset else 0.0)

    def passes_for(self, seconds: float | None) -> int:
        """How many passes fill ``seconds`` (capped at MAX_PERFORM_SECS)."""
        want = seconds if seconds and seconds > 0 else (self.default_secs
                                                        or self.seconds)
        want = min(float(want), MAX_PERFORM_SECS)
        return max(1, int(round(want / self.pass_secs)))


# Names are what the model sees; keep them plain English and few. Durations
# are the clips' own (measured from their time stamps 2026-09-25; several
# had been guessed 2-4x too long), without the per-pass overhead below.
# Speeds (2026-09-25, evening): "bursty moves ... going down and up
# abruptly many times with the antennas moving very quickly like saw
# blades" scared a little girl on the Faire's first day. Each clip plays
# slowly enough to stay inside CALM; `moves.py --measure` recomputes them
# from the recorded data (polyrhythm's antennas peaked at 505 deg/s,
# sway's head at 235 mm/s).
LIBRARY: dict[str, MoveSpec] = {m.name: m for m in (
    MoveSpec("dance", EMOTIONS, "dance1", 3.2,
             "a full-body dance; the default when asked to dance", 30.0),
    MoveSpec("dance_groovy", DANCES, "groovy_sway_and_roll", 1.8,
             "a groovier sway-and-roll dance", 30.0, speed=0.65),
    MoveSpec("dance_pendulum", DANCES, "pendulum_swing", 1.8,
             "a slow pendulum swing side to side", 30.0),
    MoveSpec("spin", None, None, 6.0,
             "turn the whole body all the way round one way and back"),
    MoveSpec("dizzy", DANCES, "dizzy_spin", 1.8,
             "a wobbly, dizzy head spin"),
    MoveSpec("peekaboo", DANCES, "side_peekaboo", 5.0,
             "peek out to one side and back, playful", speed=0.7),
    MoveSpec("cheer", EMOTIONS, "cheerful1", 2.8,
             "a cheerful bounce; celebrate a right answer", speed=0.65),
    MoveSpec("amazed", EMOTIONS, "amazed1", 3.4,
             "a wide-eyed amazed reaction"),
    MoveSpec("curious", EMOTIONS, "curious1", 11.8,
             "lean in, curious"),
    MoveSpec("confused", EMOTIONS, "confused1", 7.9,
             "a puzzled head tilt", speed=0.8),
    MoveSpec("grateful", EMOTIONS, "grateful1", 2.5,
             "a small thankful bow"),
    MoveSpec("wiggle", None, None, 1.0,
             "flick both antennas, delighted"),
    # 2026-09-25: "a few different dance moves to make it funner" when
    # nobody is in view. Durations measured from the clips' own time
    # stamps; the dance-library ones are ~2 s loops the attractor repeats.
    MoveSpec("dance_long", EMOTIONS, "dance2", 17.3,
             "a long dance, as if music were playing", attract_only=True,
             speed=0.55),
    MoveSpec("dance_wiggly", EMOTIONS, "dance3", 18.4,
             "an energetic dance with wiggles", attract_only=True,
             speed=0.7),
    MoveSpec("beckon", EMOTIONS, "come1", 3.2,
             "invite someone to come closer", attract_only=True,
             speed=0.8),
    MoveSpec("welcome", EMOTIONS, "welcoming2", 4.3,
             "a friendly welcoming gesture", attract_only=True,
             speed=0.8),
    MoveSpec("knock_knock", EMOTIONS, "toc-toc-toc", 13.4,
             "knock, knock, anyone there?", attract_only=True),
    MoveSpec("polyrhythm", DANCES, "polyrhythm_combo", 2.9,
             "a three-against-two sway and nod", attract_only=True,
             speed=0.45),
    MoveSpec("spirals", DANCES, "interwoven_spirals", 4.0,
             "interwoven spirals on three axes", attract_only=True,
             speed=0.65),
    MoveSpec("sway", DANCES, "side_to_side_sway", 1.9,
             "a smooth side-to-side sway", attract_only=True,
             speed=0.5),
    MoveSpec("head_roll", DANCES, "head_tilt_roll", 1.8,
             "a slow ear-to-shoulder head roll", attract_only=True,
             speed=0.9),
    MoveSpec("look_around", EMOTIONS, "proud1", 3.8,
             "look all around with a satisfied air", attract_only=True),
)}

# What the idle attractor plays when nobody has been in frame for a while
# (T13.4): visible from across a hall, no sound needed. 2026-09-25: a
# dozen more for variety; each plays for about ATTRACT_PLAY_SECS (short
# loops repeated), and never the same one twice in a row. Smooth ones
# only: the sharp clips (electric jolt, stumble, chicken peck, grid snap,
# jackson square) went out the same afternoon after the robot "went up
# and down abruptly" and scared a girl.
ATTRACT_MOVES = ("dance", "peekaboo", "dance_groovy", "dance_pendulum",
                 "dance_long", "dance_wiggly", "beckon", "welcome",
                 "knock_knock", "polyrhythm", "spirals", "sway",
                 "head_roll", "look_around")
ATTRACT_PLAY_SECS = 8.0
# The calm limits a played clip keeps under (99th-percentile speeds over
# 0.1 s): head travel, head rotation, antenna swing. Chosen from the
# measured clips the family did not mind -- "dance" (100 mm/s, 250 deg/s
# antennas) and "look_around" -- and well under the sharp ones removed.
CALM = {"head_mm_s": 120.0, "head_deg_s": 110.0, "antenna_deg_s": 250.0}
# What every recorded pass costs beyond its clip: reachy_target.play_move
# takes 1.0 s to reach the first frame and ends with a 0.6 s base settle.
# Measured 2026-09-25: "stumble x4" (4 x 1.8 s) was still playing 11.8 s
# in, and the idle glances were refused until it ended.
PASS_OVERHEAD_SECS = 1.7


def attract_passes(name: str) -> tuple[int, float]:
    """(passes, seconds) for one attractor turn of ``name``: short loops
    repeat to fill about ATTRACT_PLAY_SECS, long clips play once; the
    seconds include each pass's overhead (built-in moves have none)."""
    spec = LIBRARY[name]
    per_pass = spec.pass_secs
    passes = max(1, int(round(ATTRACT_PLAY_SECS / per_pass)))
    return passes, passes * per_pass


def names() -> list[str]:
    """The moves the model may ask for (not the attractor's extras)."""
    return [m.name for m in LIBRARY.values() if not m.attract_only]


def describe() -> str:
    """One line per move, for a tool description."""
    return "; ".join(f"{m.name}: {m.description}" for m in LIBRARY.values()
                     if not m.attract_only)


def clip_path(spec: MoveSpec) -> str | None:
    """The recorded clip's JSON in the local HF cache, or None."""
    if spec.dataset is None:
        return None
    from huggingface_hub import snapshot_download
    try:
        root = snapshot_download(spec.dataset, repo_type="dataset",
                                 local_files_only=True)
    except Exception:                                      # noqa: BLE001
        return None
    import os
    path = os.path.join(root, spec.move + ".json")
    return path if os.path.exists(path) else None


def measure(path: str, speed: float = 1.0) -> dict:
    """Peak speeds of a recorded clip played at ``speed``: head travel
    (mm/s), head rotation and antenna swing (deg/s), the 99th percentile
    of a 0.1 s moving average, so one noisy sample does not count."""
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation
    with open(path) as fh:
        d = json.load(fh)
    t = np.asarray(d["time"], dtype=float) / speed
    frames = d["set_target_data"]
    head = np.asarray([f["head"] for f in frames], dtype=float)
    ant = np.asarray([f["antennas"] for f in frames], dtype=float)
    dt = np.diff(t)
    dt[dt <= 0] = 1e-3

    def peak(v):
        return float(np.percentile(
            np.convolve(v / dt, np.ones(5) / 5, mode="same"), 99))
    rot = Rotation.from_matrix(head[:, :3, :3])
    return {
        "head_mm_s": peak(np.linalg.norm(np.diff(head[:, :3, 3], axis=0),
                                         axis=1) * 1000.0),
        "head_deg_s": float(np.degrees(peak((rot[1:] * rot[:-1].inv())
                                            .magnitude()))),
        "antenna_deg_s": float(np.degrees(max(
            peak(np.abs(np.diff(ant[:, 0]))),
            peak(np.abs(np.diff(ant[:, 1])))))),
    }


def calm_speed(stats: dict) -> float:
    """The playback speed (a multiple of 0.05, at most 1) that brings a
    clip measured at speed 1 inside CALM."""
    import math
    ratio = min([1.0] + [CALM[k] / v for k, v in stats.items() if v > 0])
    return max(0.3, math.floor(ratio * 20 + 1e-9) / 20)


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
    ap.add_argument("--measure", action="store_true",
                    help="each recorded clip's peak speeds at its speed, "
                         "and the speed CALM would give it")
    ap.add_argument("--cached", action="store_true",
                    help="report which datasets are cached, exit 1 if any "
                         "is missing")
    a = ap.parse_args()
    if a.measure:
        print(f"{'name':16s} {'mm/s':>6s} {'deg/s':>6s} {'ant':>6s} "
              f"{'speed':>6s} {'calm':>6s}")
        for m in LIBRARY.values():
            path = clip_path(m)
            if path is None:
                continue
            played = measure(path, m.speed)
            print(f"{m.name:16s} {played['head_mm_s']:6.0f} "
                  f"{played['head_deg_s']:6.0f} {played['antenna_deg_s']:6.0f} "
                  f"{m.speed:6.2f} {calm_speed(measure(path)):6.2f}")
        return 0
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
