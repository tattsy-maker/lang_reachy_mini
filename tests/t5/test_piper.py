"""T5: Piper TTS + per-language engine routing.

Everything model-touching runs under voice/.venv as a subprocess (the
established pattern). The full round-trip gate was run and recorded on
2026-08-31 (ru 95.9%, zh 92.9%, both PASS — tests/reports/); the test
here re-runs it in smoke mode so the plumbing stays exercised.
"""

import json
import subprocess

import pytest

ROUTER_SCRIPT = """
import json, sys
sys.path.insert(0, "voice")
from multilingual import LanguageRouter
from piper_tts import PIPER_LANGUAGES, piper_available
router = LanguageRouter(initial="en", extra=piper_available())
print(json.dumps({code: (router.speakable(code).voice
                         if router.speakable(code) else None)
                  for code in ["es", "fr", "ru", "zh", "ja", "xx"]}))
"""

SYNTH_SCRIPT = """
import json, sys
sys.path.insert(0, "voice")
import numpy as np
from verify_language import synth_piper
out = {}
for code, text in [("ru", "Привет, как дела?"), ("zh", "你好，你怎么样？")]:
    audio = synth_piper(code, text)
    out[code] = {"secs": round(len(audio) / 16000, 2),
                 "rms": round(float(np.sqrt((audio ** 2).mean())), 4)}
print(json.dumps(out))
"""


def run_under_voice_venv(paths, script) -> dict:
    out = subprocess.run([str(paths.voice_py), "-c", script],
                         cwd=paths.repo, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.splitlines()[-1])


@pytest.mark.models
def test_router_maps_languages_to_engines(paths):
    voices = run_under_voice_venv(paths, ROUTER_SCRIPT)
    assert voices["es"] == "ef_dora"          # Kokoro keeps its languages
    assert voices["fr"] == "ff_siwis"
    assert voices["ru"] == "ru_RU-irina-medium"   # Piper takes ru/zh
    assert voices["zh"] == "zh_CN-huayan-medium"
    assert voices["ja"] is None               # nobody speaks ja locally
    assert voices["xx"] is None


@pytest.mark.models
def test_piper_produces_sane_audio(paths):
    audio = run_under_voice_venv(paths, SYNTH_SCRIPT)
    for code in ("ru", "zh"):
        assert 0.5 <= audio[code]["secs"] <= 10.0, \
            f"{code}: implausible duration {audio[code]['secs']}s"
        assert audio[code]["rms"] > 0.01, f"{code}: essentially silence"


@pytest.mark.models
def test_round_trip_gate_runs_and_reports(paths):
    out = subprocess.run(
        [str(paths.voice_py), str(paths.repo / "voice" /
                                  "verify_language.py"), "--limit", "2"],
        cwd=paths.repo / "voice", capture_output=True, text=True,
        timeout=600)
    # smoke mode may dip below the gate on 2 sentences; the report matters
    assert "report:" in out.stdout, out.stderr[-2000:]
    report = out.stdout.split("report:")[1].strip()
    results = json.loads(open(report).read())
    assert {r["language"] for r in results} == {"ru", "zh"}
    for r in results:
        assert r["sentences"] and all(row["heard"] for row in r["sentences"])
