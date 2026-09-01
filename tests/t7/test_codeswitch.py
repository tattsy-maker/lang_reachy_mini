"""T7: mixed-language STT hardening — priming plumbing and the gate."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutor.store import LearnerStore  # noqa: E402

PLUMBING_SCRIPT = """
import asyncio, json, sys
sys.path.insert(0, "voice")
import numpy as np
import mlx_whisper
captured = {}
real = mlx_whisper.transcribe
def fake_transcribe(audio, **kw):
    captured.update(kw)
    return {"language": "en", "segments": [], "text": ""}
mlx_whisper.transcribe = fake_transcribe
from multilingual import MultilingualWhisperMLX, bilingual_priming
svc = MultilingualWhisperMLX(
    settings=MultilingualWhisperMLX.Settings(model="unused"))
svc.initial_prompt = bilingual_priming("es")
async def run():
    async for _ in svc.run_stt(np.zeros(16000, dtype=np.int16).tobytes()):
        pass
asyncio.run(run())
print(json.dumps({"initial_prompt": captured.get("initial_prompt")}))
"""


@pytest.mark.models
def test_priming_prompt_actually_reaches_whisper(paths):
    out = subprocess.run([str(paths.voice_py), "-c", PLUMBING_SCRIPT],
                         cwd=paths.repo, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.splitlines()[-1])
    assert got["initial_prompt"] and "español" in got["initial_prompt"], \
        "the bilingual prompt never reached mlx_whisper.transcribe"


@pytest.mark.models
def test_gate_runs_and_reports(paths):
    out = subprocess.run(
        [str(paths.voice_py),
         str(paths.repo / "voice" / "verify_codeswitch.py"),
         "--pairs", "es", "--limit", "2"],
        cwd=paths.repo / "voice", capture_output=True, text=True,
        timeout=600)
    assert "report:" in out.stdout, out.stderr[-2000:]
    report = json.loads(open(out.stdout.split("report:")[1].strip()).read())
    assert report[0]["pair"] == "en+es"
    row = report[0]["phrases"][0]
    assert row["unprimed"]["heard"] and row["primed"]["heard"]
    assert 0 <= row["primed"]["survival"] <= 100


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_agent_wires_priming_for_the_learner(tmp_path, run_agent_say):
    root = tmp_path / "learners"
    LearnerStore(root).create("Maria", "es", level="intermediate")
    log, found = run_agent_say(
        ["Hola, ¿practicamos un poco?"],
        extra_args=["--learner", "maria", "--learners-root", str(root)],
        timeout=300)
    assert found, "no reply; log tail:\n" + log[-4000:]
    assert "whisper priming: English + es" in log, \
        "the learner's bilingual priming was never installed"
