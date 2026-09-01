"""Who is standing in front of the robot? (T9)

Bridges the face modules (T1 camera, T2 recognition) to the learner store
(T3) for the voice agent: look at a frame source briefly, find the largest
face, and answer with one of four statuses:

    known    a stored learner matched at or above the accept threshold
    unsure   best match landed in the ask-don't-guess band -- the caller
             must confirm ("Maria, is that you?") before greeting
    unknown  a face is there but nobody in the store is close
    noface   no face appeared in the sampling window

Also the enrollment capture: ``capture_embedding`` watches the same source
and averages several face embeddings, exactly what ``enroll_new_learner``
saves into a new guest profile.

CLI (for tests; run from voice/.venv)::

    python voice/face_id.py --source clip.avi --learners-root learners/
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from face.camera import Camera  # noqa: E402
from face import recognize  # noqa: E402
from tutor.store import Learner, LearnerStore  # noqa: E402

logger = logging.getLogger("face_id")


def parse_source(source: str):
    """A --face-source value: a V4L2 index if it looks like one, else a path."""
    return int(source) if str(source).isdigit() else source


@dataclass
class Identification:
    status: str                    # known | unsure | unknown | noface
    learner: Learner | None = None
    score: float | None = None


def identify_from_source(source, store: LearnerStore, *,
                         samples: int = 3, max_frames: int = 12,
                         fps: float = 4.0) -> Identification:
    """Sample the source until ``samples`` face embeddings are collected
    (or ``max_frames`` frames pass), average them, and match against every
    stored learner that has an embedding."""
    vectors = []
    seen = 0
    for frame in Camera(parse_source(source), fps=fps).frames():
        seen += 1
        vector = recognize.embed(frame)
        if vector is not None:
            vectors.append(vector)
            if len(vectors) >= samples:
                break
        if seen >= max_frames:
            break
    if not vectors:
        return Identification("noface")

    face = recognize.enroll_from_vectors(vectors)
    # Only embeddings of the model's own dimension count; anything else is
    # a placeholder or a different model's leftovers, not a person.
    known = {l.id: l.embedding for l in store.list()
             if l.embedding and len(l.embedding) == len(face)}
    found = recognize.match(face, known)
    if found is None:
        return Identification("unknown")
    learner = store.load(found.name)
    status = "known" if found.sure else "unsure"
    logger.info("face: %s -> %s (score %.3f)", status,
                learner.id if learner else "?", found.score)
    return Identification(status, learner, round(found.score, 4))


def capture_embedding(source, *, samples: int = 4,
                      max_frames: int = 20, fps: float = 4.0):
    """The enrollment capture: 3-5 snapshots averaged into one embedding.
    Returns None if the face never showed up."""
    vectors = []
    seen = 0
    for frame in Camera(parse_source(source), fps=fps).frames():
        seen += 1
        vector = recognize.embed(frame)
        if vector is not None:
            vectors.append(vector)
            if len(vectors) >= samples:
                break
        if seen >= max_frames:
            break
    if not vectors:
        return None
    logger.info("enrollment capture: %d snapshots averaged", len(vectors))
    return recognize.enroll_from_vectors(vectors)


def main() -> int:
    import argparse
    import contextlib
    import json

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--learners-root", required=True)
    args = ap.parse_args()

    with contextlib.redirect_stdout(sys.stderr):  # model chatter, see T2
        result = identify_from_source(args.source,
                                      LearnerStore(args.learners_root))
    print(json.dumps({
        "status": result.status,
        "learner": result.learner.id if result.learner else None,
        "score": result.score,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
