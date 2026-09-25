"""T17 on the wire: the ``heard:`` line (T17.7), the corrections judge
over a log (T17.7), and Gemini's patience through a mid-sentence pause
(T17.1). The judge run and the probe need keys and the voice venv and
skip cleanly without them."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voice"))
sys.path.insert(0, str(REPO / "tests" / "t17"))

import judge_corrections as judge                          # noqa: E402

SESSION_LOG = REPO / "booth" / "logs" / "2026-09-05_family" / "agent.log"


# -- the heard line ------------------------------------------------------------------

def test_heard_logger_logs_and_hooks_either_direction(caplog):
    pytest.importorskip("pipecat")
    import logging
    from agent import HeardLogger
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    got = []
    proc = HeardLogger(on_heard=got.append)
    pushed = []

    async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append((frame, direction))
    proc.push_frame = fake_push

    async def drive():
        with caplog.at_level(logging.INFO, logger="agent"):
            await proc.process_frame(TranscriptionFrame(
                text="  Ты мне  не дал договорить. ", user_id="", timestamp="t"),
                FrameDirection.UPSTREAM)
            await proc.process_frame(TranscriptionFrame(
                text="I want to eat", user_id="", timestamp="t"),
                FrameDirection.DOWNSTREAM)
            await proc.process_frame(TranscriptionFrame(
                text="   ", user_id="", timestamp="t"), FrameDirection.UPSTREAM)
    asyncio.run(drive())
    assert got == ["Ты мне не дал договорить.", "I want to eat"]
    assert "heard: Ты мне не дал договорить." in caplog.text
    assert len(pushed) == 3 and pushed[0][1] == FrameDirection.UPSTREAM


# -- the judge's parser ----------------------------------------------------------------

SYNTHETIC = """\
2026-09-05 19:27:42,460 INFO    session: session: started
2026-09-05 19:27:48,790 INFO    agent: said: Hello there!
2026-09-05 19:27:56.484 | DEBUG    | pipecat.services.google.gemini_live.llm:_handle_msg_input_transcription:1976 - [Transcription:user] [Да, давай.]
2026-09-05 19:27:59,639 INFO    tutor_mode: tutor: enrolled new guest learner (en taught in en, beginner, goal conversation, with voice print)
2026-09-05 19:28:24,917 INFO    tutor_mode: tutor: native language for learner changed en -> ru
2026-09-05 19:38:09,000 INFO    agent: heard: Usually I eat in the on the breakfast.
2026-09-05 19:38:26,170 INFO    agent: said: Eggs are a great choice for breakfast!
2026-09-05 19:42:02,084 INFO    session: session: ended and reset; watching again
2026-09-05 19:45:00,000 INFO    session: session: started
2026-09-05 19:45:01,000 INFO    session: session: recognized john (score 0.9)
2026-09-05 19:45:05,000 INFO    agent: heard: Hello again
"""


def test_judge_parses_sessions_from_both_line_forms(tmp_path):
    log = tmp_path / "agent.log"
    log.write_text(SYNTHETIC)
    sessions = judge.parse(log)
    assert len(sessions) == 2
    first = sessions[0]
    assert first["learner"] == "learner" and first["target"] == "en"
    assert first["native"] == "ru", "the mid-session switch is applied"
    kinds = [k for k, _t, _x in first["turns"]]
    assert kinds == ["said", "heard", "heard", "said"]
    text, n_heard = judge.transcript_text(first)
    assert n_heard == 2 and "HEARD[1]: Usually I eat" in text
    assert sessions[1]["learner"] == "john"
    report = judge.report([{"learner": "learner", "started": "x", "target": "en",
                            "native": "ru", "heard": 2, "attempts": 1,
                            "mistakes": 1, "corrected": 0,
                            "items": [{"text": "Usually I eat in the on the breakfast.",
                                       "mistake": True, "corrected": False,
                                       "fix": "I usually eat breakfast."}]}],
                          log, "test-model")
    assert "| x | learner | en/ru | 2 | 1 | 1 | 0 | 0% |" in report
    assert "mistakes left uncorrected" in report


@pytest.mark.anthropic
@pytest.mark.models
def test_judge_the_last_family_session(paths):
    """The baseline number for T17.7: the 2026-09-05 log, if it is on
    this machine (booth logs are gitignored)."""
    if not SESSION_LOG.exists():
        pytest.skip(f"{SESSION_LOG} not on this machine")
    out = paths.reports / "judge_corrections_2026-09-05.md"
    res = subprocess.run(
        [str(paths.voice_py), str(REPO / "tests" / "t17" / "judge_corrections.py"),
         str(SESSION_LOG), "--json", "--out", str(out)],
        capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    data = json.loads(res.stdout[res.stdout.index("["):res.stdout.rindex("]") + 1])
    assert data, "no sessions judged"
    mother = [r for r in data if r["learner"] == "learner"]
    assert mother, [r["learner"] for r in data]
    assert mother[0]["mistakes"] >= 3, mother[0]
    assert mother[0]["corrected"] <= 2, "the baseline session had almost no corrections"
    assert out.exists()


# -- Gemini's patience -----------------------------------------------------------------

@pytest.mark.google
@pytest.mark.models
def test_gemini_waits_through_a_pause_with_our_settings(paths):
    res = subprocess.run(
        [str(paths.voice_py), str(REPO / "tests" / "t17" / "probe_patience.py"),
         "--patience-ms", "1800", "--pause", "1.2"],
        capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    got = json.loads(res.stdout.strip().splitlines()[-1])
    (paths.reports / "probe_patience_2026-09-05.json").write_text(
        json.dumps(got, ensure_ascii=False, indent=1))
    assert got["turns"] == 1, got
    assert got["transcripts"] and got["transcripts"][0].strip(), "no transcription heard"
