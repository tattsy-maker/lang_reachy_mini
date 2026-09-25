#!/usr/bin/env python3
"""Corrections have a number (T17.7): count the student's mistakes and
the tutor's corrections in one session log.

    voice/.venv/bin/python tests/t17/judge_corrections.py booth/logs/2026-09-05_family/agent.log
    ... --learner learner            # one visitor (the store id in the log)
    ... --model claude-haiku-4-5-20251001 --out tests/reports/judge_corrections_2026-09-05.md

Reads the log's ``heard:`` lines (T17.7; on older logs pipecat's
``[Transcription:user]`` DEBUG lines) and ``said:`` lines, splits them
into sessions at ``session: started``, and asks a small Claude model to
judge each student utterance in the target language: was it a mistake,
and did the tutor's next reply correct it (a recast or an explicit
correction)? Prints per-session counts and writes a Markdown report.
Needs ANTHROPIC_API_KEY (environment or voice/.env).

Gemini's input transcription guesses the wrong language on short
utterances ("Não dá", "barnet ditt ja"): the judge is told to ignore
lines that are plainly transcription noise.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "voice"))

HEARD_RX = re.compile(r"^(\S+ \S+) INFO\s+agent: heard: (.*)$")
DEBUG_HEARD_RX = re.compile(r"^(\S+ \S+) \| DEBUG .*\[Transcription:user\] \[(.*)\]\s*$")
SAID_RX = re.compile(r"^(\S+ \S+) INFO\s+agent: said: (.*)$")
START_RX = re.compile(r"^(\S+ \S+) INFO\s+session: session: started")
ENROLL_RX = re.compile(r"tutor: enrolled new guest (\S+) \((\w+) taught in (\w+)")
RECOG_RX = re.compile(r"session: recognized (\S+)|identity confirmed as (\S+)")

JUDGE_PROMPT = """\
You are grading a language tutor's session transcript. The student is \
learning {target} and speaks {native}. Below are the student's utterances \
(HEARD) and the tutor's replies (SAID), in order.

For EACH heard line that is an attempt to speak {target} (ignore lines in \
{native}, one-word answers like yes/no, and lines that are plainly speech \
recognition noise in a third language), decide:
  mistake: true if the utterance has a grammar, word-choice or word-order \
error a teacher would fix (not pronunciation, not a filler);
  corrected: true if the tutor's NEXT reply repeats or gives the corrected \
form (a recast, "you could say ...", or an explicit correction). Only the \
next reply counts.

Answer with JSON only: {{"items": [{{"i": <heard index>, "text": "<the \
utterance>", "mistake": true/false, "corrected": true/false, "fix": "<the \
corrected sentence or empty>"}}]}} listing only utterances in {target}.

TRANSCRIPT:
{transcript}
"""


def parse(log_path: Path) -> list[dict]:
    """Sessions: [{started, learner, target, native, turns: [(kind, t, text)]}]."""
    sessions: list[dict] = []
    current = None
    for line in log_path.read_text(errors="replace").splitlines():
        if START_RX.match(line):
            current = {"started": line[:19], "learner": None, "target": None,
                       "native": None, "turns": []}
            sessions.append(current)
            continue
        if current is None:
            continue
        m = ENROLL_RX.search(line)
        if m:
            current["learner"], current["target"], current["native"] = m.groups()
            continue
        m = RECOG_RX.search(line)
        if m:
            current["learner"] = current["learner"] or (m.group(1) or m.group(2))
        for rx, kind in ((HEARD_RX, "heard"), (DEBUG_HEARD_RX, "heard"),
                         (SAID_RX, "said")):
            m = rx.match(line)
            if m:
                text = m.group(2).strip()
                if text:
                    current["turns"].append((kind, m.group(1), text))
                break
        m = re.search(r"tutor: native language for (\S+) changed \w+ -> (\w+)", line)
        if m:
            current["native"] = m.group(2)
        m = re.search(r"tutor: language for (\S+) changed \w+ -> (\w+)", line)
        if m:
            current["target"] = m.group(2)
    return [s for s in sessions if s["turns"]]


def transcript_text(session: dict) -> tuple[str, int]:
    lines, heard = [], 0
    for kind, _t, text in session["turns"]:
        if kind == "heard":
            lines.append(f"HEARD[{heard}]: {text}")
            heard += 1
        else:
            lines.append(f"SAID: {text}")
    return "\n".join(lines), heard


def judge(session: dict, model: str, client) -> dict:
    from tutor_mode import language_name
    text, n_heard = transcript_text(session)
    prompt = JUDGE_PROMPT.format(
        target=language_name(session["target"] or "en"),
        native=language_name(session["native"] or "en"),
        transcript=text)
    msg = client.messages.create(model=model, max_tokens=4000,
                                 messages=[{"role": "user", "content": prompt}])
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group()) if m else {"items": []}
    items = data.get("items", [])
    mistakes = [i for i in items if i.get("mistake")]
    corrected = [i for i in mistakes if i.get("corrected")]
    return {"learner": session["learner"] or "?", "started": session["started"],
            "target": session["target"], "native": session["native"],
            "heard": n_heard, "attempts": len(items),
            "mistakes": len(mistakes), "corrected": len(corrected),
            "items": items}


def report(results: list[dict], log_path: Path, model: str) -> str:
    out = [f"# Corrections judged: {log_path}", "",
           f"Judge: {model}, {_dt.datetime.now():%Y-%m-%d %H:%M}. A mistake "
           "is a grammar, word-choice or word-order error in the target "
           "language; corrected means the tutor's next reply gave the right "
           "form.", "",
           "| Session | Learner | Target/native | Heard | Attempts | Mistakes | Corrected | Rate |",
           "|---|---|---|---|---|---|---|---|"]
    for r in results:
        rate = ("%d%%" % round(100.0 * r["corrected"] / r["mistakes"])
                if r["mistakes"] else "-")
        out.append("| %s | %s | %s/%s | %d | %d | %d | %d | %s |" % (
            r["started"], r["learner"], r["target"], r["native"], r["heard"],
            r["attempts"], r["mistakes"], r["corrected"], rate))
    for r in results:
        missed = [i for i in r["items"] if i.get("mistake") and not i.get("corrected")]
        if missed:
            out += ["", f"## {r['learner']} ({r['started']}): mistakes left uncorrected", ""]
            for i in missed:
                out.append(f"- \"{i.get('text', '')}\" -> {i.get('fix', '') or '?'}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--learner", default=None, help="only this learner id")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--out", default=None, help="Markdown report path")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "voice"))
    from agent import load_env_file
    load_env_file()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("no ANTHROPIC_API_KEY (environment or voice/.env)", file=sys.stderr)
        return 2
    import anthropic
    client = anthropic.Anthropic()
    sessions = parse(Path(args.log))
    if args.learner:
        sessions = [s for s in sessions if s["learner"] == args.learner]
    results = [judge(s, args.model, client) for s in sessions]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    else:
        for r in results:
            print("%s %-28s %s/%s heard %2d attempts %2d mistakes %2d corrected %2d"
                  % (r["started"], r["learner"], r["target"], r["native"],
                     r["heard"], r["attempts"], r["mistakes"], r["corrected"]))
    if args.out:
        Path(args.out).write_text(report(results, Path(args.log), args.model))
        print("report:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
