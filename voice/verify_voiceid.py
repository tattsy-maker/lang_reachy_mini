#!/usr/bin/env python3
"""The speaker-verification gate (T13.9): measure before trusting.

Embeds every wav under ``tests/fixtures/voices/<speaker>/`` (or any
directory laid out the same way -- point it at real family recordings
when you have them), scores all pairs, and reports the same-speaker and
different-speaker distributions next to the thresholds in voiceid.py, so
the numbers decide the thresholds and not the other way round. Same
method as T2's face calibration and T5/T7's round-trip gates.

    voice/.venv/bin/python voice/verify_voiceid.py            # fixtures
    voice/.venv/bin/python voice/verify_voiceid.py --dir DIR  # your own
    ... --out tests/reports                                   # write files

Exit status is 0 when the measured band is clean (every same-speaker
pair >= ACCEPT and every different-speaker pair < REJECT), 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from itertools import combinations
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

import voiceid  # noqa: E402

DEFAULT_DIR = _REPO / "tests" / "fixtures" / "voices"


def measure(root: Path) -> dict:
    vectors = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for wav in sorted(folder.glob("*.wav")):
            vectors[(folder.name, wav.name)] = voiceid.embed_wav(wav)
    same, diff = [], []
    for (a, b) in combinations(sorted(vectors), 2):
        s = round(voiceid.similarity(vectors[a], vectors[b]), 4)
        (same if a[0] == b[0] else diff).append(
            {"a": f"{a[0]}/{a[1]}", "b": f"{b[0]}/{b[1]}", "score": s})
    speakers = sorted({k[0] for k in vectors})
    # Enrollment-style check: print = mean of all but one clip, matched
    # against the held-out clip of every speaker.
    holdout = []
    for spk in speakers:
        clips = [k for k in vectors if k[0] == spk]
        for held in clips:
            rest = [vectors[k] for k in clips if k != held]
            if not rest:
                continue
            print_vec = voiceid.average(rest)
            for other in speakers:
                probes = [k for k in vectors if k[0] == other]
                probe = held if other == spk else probes[0]
                holdout.append({"print": spk, "probe": f"{probe[0]}/{probe[1]}",
                                "same": other == spk,
                                "score": round(voiceid.similarity(
                                    print_vec, vectors[probe]), 4)})
    def rng(rows):
        scores = [r["score"] for r in rows]
        return {"min": min(scores), "max": max(scores),
                "mean": round(sum(scores) / len(scores), 4),
                "n": len(scores)} if scores else None
    return {
        "date": _dt.date.today().isoformat(),
        "dir": str(root),
        "device": voiceid.provider_report(),
        "model": voiceid.MODEL_SOURCE,
        "speakers": speakers,
        "clips": len(vectors),
        "thresholds": {"accept": voiceid.ACCEPT_THRESHOLD,
                       "reject": voiceid.REJECT_THRESHOLD},
        "same_speaker": rng(same),
        "different_speaker": rng(diff),
        "holdout_same": rng([r for r in holdout if r["same"]]),
        "holdout_different": rng([r for r in holdout if not r["same"]]),
        "pairs_same": same,
        "pairs_different": diff,
    }


def clean(report: dict) -> bool:
    s, d = report["same_speaker"], report["different_speaker"]
    t = report["thresholds"]
    return bool(s and d and s["min"] >= t["accept"] and d["max"] < t["reject"])


def markdown(report: dict) -> str:
    t = report["thresholds"]
    lines = [f"# Speaker verification gate — {report['date']}", "",
             f"Model `{report['model']}` on `{report['device']}`; "
             f"{report['clips']} clips, speakers: {', '.join(report['speakers'])}.",
             "", "| Pairs | n | min | mean | max |", "|---|---|---|---|---|"]
    for name in ("same_speaker", "different_speaker", "holdout_same",
                 "holdout_different"):
        r = report[name]
        if r:
            lines.append(f"| {name} | {r['n']} | {r['min']} | {r['mean']} | {r['max']} |")
    lines += ["", f"Thresholds: accept ≥ {t['accept']}, reject < {t['reject']}. "
              f"Band is {'CLEAN' if clean(report) else 'NOT clean'}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out", default=None, help="write report files here")
    a = ap.parse_args()
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        report = measure(Path(a.dir))
    print(markdown(report))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = out / f"verify_voiceid_{report['date']}"
        stem.with_suffix(".json").write_text(json.dumps(report, indent=1))
        stem.with_suffix(".md").write_text(markdown(report))
        print(f"wrote {stem}.json / .md", file=sys.stderr)
    return 0 if clean(report) else 1


if __name__ == "__main__":
    sys.exit(main())
