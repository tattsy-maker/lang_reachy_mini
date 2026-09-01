"""Face recognition core (T2): enroll, match, reject — no agent wiring.

Built on InsightFace's ``buffalo_l`` model pack through ONNX Runtime. The
model downloads once into ``~/.insightface/models/buffalo_l`` (~280 MB) and
is cached thereafter. Which execution provider actually runs is logged at
startup — this box's history (CLAUDE.md, the CTranslate2 story) says never
to trust a library's GPU reputation, so ``provider_report()`` is part of
the API and the T2 progress log records the measured answer.

The public pieces T9/T10 build against:

* ``embed(image) -> vector | None`` — the largest face in a BGR image (the
  spec's closest-person rule), as a unit-length 512-float vector; ``None``
  when no face is found.
* ``enroll(images) -> vector | None`` — average of the per-image
  embeddings, re-normalized; ``None`` if no image had a face.
* ``match(vector, known) -> Match | None`` — best cosine match against
  ``known`` (a ``{name: vector}`` dict). ``None`` means nobody is close
  enough to mention (below REJECT_THRESHOLD). A returned ``Match`` carries
  ``sure``: ``True`` at or above ACCEPT_THRESHOLD, ``False`` in the unsure
  band, where the caller must *ask* ("Maria, is that you?") rather than
  greet — a wrong greeting is this feature's worst failure mode.

Thresholds are asserted against measured fixture scores in
``tests/t2/`` — see the T2 progress log for the measured distributions.

CLI (run from ``voice/.venv``, where the face dependencies live)::

    python face/recognize.py matrix DIR         # pairwise scores, JSON
    python face/recognize.py identify --enroll a.jpg,b.jpg --source clip.avi
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("face.recognize")

# Cosine-similarity decision band, calibrated on the fixture portraits
# (2026-08-31, buffalo_l, CPUExecutionProvider). Measured same-person
# pairs: 0.66 (Sunita ~2004 vs 2018, the hardest cross-year pair), 0.83
# (Scott 2014 vs 2019), 0.98 (Tracy, same sitting). Different-person
# pairs: -0.001 to 0.063. The gap is wide; the band sits inside it with a
# margin on both sides — below-threshold same-person cases fall in the
# ask-don't-guess band rather than being greeted wrong.
ACCEPT_THRESHOLD = 0.45   # at or above: greet by name
REJECT_THRESHOLD = 0.25   # below: a stranger
# between the two: ask, don't guess

_app = None


def get_app():
    """The shared FaceAnalysis instance (loads the model pack on first use)."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name="buffalo_l",
                            allowed_modules=["detection", "recognition"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("insightface ready; providers: %s", provider_report())
    return _app


def provider_report() -> list[str]:
    """Which ONNX Runtime execution providers are actually available."""
    import onnxruntime
    return onnxruntime.get_available_providers()


def embed(image) -> np.ndarray | None:
    """Unit-length embedding of the largest face in a BGR image, or None."""
    faces = get_app().get(image)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                  * (f.bbox[3] - f.bbox[1]))
    return np.asarray(largest.normed_embedding, dtype=np.float32)


def enroll(images) -> np.ndarray | None:
    """Average embedding across several images of the same person."""
    vectors = [v for v in (embed(img) for img in images) if v is not None]
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def similarity(a, b) -> float:
    """Cosine similarity between two embeddings (any norm)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class Match:
    name: str
    score: float
    sure: bool          # False = the unsure band: ask, don't greet


def match(vector, known: dict) -> Match | None:
    """Best match for ``vector`` among ``known`` {name: embedding}.

    None below REJECT_THRESHOLD (a stranger). ``sure`` is False in the
    ask-don't-guess band between the thresholds.
    """
    if not known:
        return None
    name, score = max(((n, similarity(vector, v)) for n, v in known.items()),
                      key=lambda pair: pair[1])
    if score < REJECT_THRESHOLD:
        return None
    return Match(name=name, score=score, sure=score >= ACCEPT_THRESHOLD)


# ---------------------------------------------------------------------------
# CLI, so the light test venv can drive all of this through voice/.venv
# ---------------------------------------------------------------------------

def _cmd_matrix(args) -> dict:
    import cv2
    folder = Path(args.dir)
    images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    vectors = {}
    faces = {}
    for path in images:
        vector = embed(cv2.imread(str(path)))
        faces[path.stem] = vector is not None
        if vector is not None:
            vectors[path.stem] = vector
    names = sorted(vectors)
    scores = {f"{a}|{b}": round(similarity(vectors[a], vectors[b]), 4)
              for i, a in enumerate(names) for b in names[i + 1:]}
    return {"provider": provider_report(), "faces": faces, "scores": scores}


def _cmd_identify(args) -> dict:
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from face.camera import Camera

    enroll_images = [cv2.imread(p) for p in args.enroll.split(",")]
    known = {args.name: enroll(enroll_images)}
    if known[args.name] is None:
        return {"error": "no face in any enrollment image"}

    frames = matches = 0
    scores = []
    for frame in Camera(args.source, fps=args.fps).frames():
        frames += 1
        vector = embed(frame)
        if vector is None:
            continue
        found = match(vector, known)
        if found:
            matches += 1
            scores.append(round(found.score, 4))
    return {"name": args.name, "frames": frames, "matched": matches,
            "scores": scores,
            "min_score": min(scores) if scores else None}


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="face recognition core")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("matrix", help="pairwise scores for a dir of images")
    m.add_argument("dir")
    i = sub.add_parser("identify",
                       help="enroll from images, identify in a video/dir")
    i.add_argument("--enroll", required=True,
                   help="comma-separated image paths")
    i.add_argument("--source", required=True, help="video file or image dir")
    i.add_argument("--name", default="person")
    i.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args()

    # InsightFace and ONNX Runtime chat on stdout while loading; keep stdout
    # pure JSON for callers by diverting everything else to stderr.
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        result = (_cmd_matrix(args) if args.cmd == "matrix"
                  else _cmd_identify(args))
    print(json.dumps(result, indent=2))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
