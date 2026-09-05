"""T14.3: one visitor at a time -- speaker change from the voice print,
the runner ending a session on it, and the cloud brain reset (unit, on a
fake Gemini; plus a guard that the real service still has the fields)."""

import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from session import ACTIVE, WATCHING, CloudBrain, SessionRunner  # noqa: E402
from tutor.store import LearnerStore  # noqa: E402
from tutor_mode import CurrentLearner  # noqa: E402
from voiceid import VoiceIdentity  # noqa: E402


def unit(axis, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def blend(base, target, seed=1):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=base.shape).astype(np.float32)
    noise -= np.dot(noise, base) * base
    noise /= np.linalg.norm(noise)
    return target * base + np.sqrt(1 - target ** 2) * noise


# -- voice: recent window, speaker change -----------------------------------------

def test_decisions_use_the_recent_voice_not_the_session_mean(tmp_path):
    """The 2026-09-03 failure: after John's 15 samples, Rock's samples were
    averaged into a session mean that still matched John."""
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    john_voice = unit(0)
    john = store.create("John", "fr", voice_embedding=[float(x) for x in john_voice])
    holder.learner = john
    vid = VoiceIdentity(store, holder)
    for i in range(15):
        vid.on_sample(blend(john_voice, 0.9, seed=i))
    assert vid.verified
    rock_voice = unit(3)
    assert vid.on_sample(rock_voice) is None, "one odd sample: wait"
    assert vid.on_sample(blend(rock_voice, 0.95)) == "speaker_changed"
    assert vid.changed and holder.learner is john, \
        "the policy reports; the runner decides what to do"
    assert vid.on_sample(rock_voice) is None, "reported once"


def test_one_stray_sample_after_verification_is_still_ignored(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    voice = unit(0)
    holder.learner = store.create("A", "es", voice_embedding=[float(x) for x in voice])
    vid = VoiceIdentity(store, holder)
    assert vid.on_sample(blend(voice, 0.9)) == "verified"
    assert vid.on_sample(unit(5)) is None                  # a cough
    assert vid.on_sample(blend(voice, 0.9, seed=2)) is None
    assert vid.on_sample(blend(voice, 0.9, seed=3)) is None
    assert not vid.changed and vid.consecutive_mismatches == 0


# -- runner: a speaker change ends the session --------------------------------------

class FakeContext:
    def __init__(self):
        self.history = []

    def set_messages(self, messages):
        self.history.append(messages)

    def system_text(self):
        return self.history[-1][0]["content"]


class Runner(SessionRunner):
    def __init__(self, **kw):
        super().__init__(task=None, languages="French", base_prompt="BASE.",
                         **kw)
        self.cues, self.went_neutral = [], 0

    async def _queue_user_turn(self, text):
        self.cues.append(text)

    async def _robot_neutral(self):
        self.went_neutral += 1


def test_speaker_change_ends_the_session_and_watches_again(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    runner = Runner(source="unused", store=store, holder=holder,
                    context=FakeContext(), robot=None, save_wait_secs=0.2)
    john = store.create("John", "fr", embedding=[float(x) for x in unit(0)])
    runner._recent_vectors = [unit(0)]
    asyncio.run(runner.start_session(now=10.0))
    assert holder.learner.id == john.id and runner.machine.state == ACTIVE

    asyncio.run(runner.speaker_changed())
    assert any("save_session_notes" in c for c in runner.cues), \
        "John's notes were not asked for"
    assert holder.learner is None and runner.machine.state == WATCHING
    assert "nobody is in front of you" in runner.context.system_text().lower()
    # the face loop will start the newcomer's session; a stranger's face
    # gets the interview, not "already tutoring John"
    runner._recent_vectors = [unit(1)]
    asyncio.run(runner.start_session(now=20.0))
    assert "do not recognize" in runner.context.system_text()

    # while watching, a speaker change is a no-op
    asyncio.run(runner.end_session())
    cues = len(runner.cues)
    asyncio.run(runner.speaker_changed())
    assert len(runner.cues) == cues


def test_cue_seam_is_used_when_given(tmp_path):
    sent = []

    async def cue(text):
        sent.append(text)

    runner = SessionRunner(source="u", store=LearnerStore(tmp_path), holder=CurrentLearner(),
                           context=FakeContext(), task=None, base_prompt="B",
                           languages="x", cue=cue)
    asyncio.run(runner._queue_user_turn("hello"))
    assert sent == ["hello"]


# -- cloud brain ----------------------------------------------------------------------

class FakeGemini:
    def __init__(self):
        self._system_instruction_from_init = "old"
        self._session_resumption_handle = "resume-123"
        self._ready_for_realtime_input = True
        self._settings = type("S", (), {"system_instruction": "old"})()
        self.calls = []
        self.audio_paused = None

    def set_audio_input_paused(self, paused):
        self.audio_paused = paused
        self.calls.append(("audio_paused", paused))

    async def _disconnect(self):
        self.calls.append("disconnect")
        self._ready_for_realtime_input = False

    async def _connect(self, session_resumption_handle=None):
        self.calls.append(("connect", session_resumption_handle,
                           self._session_resumption_handle))
        self._ready_for_realtime_input = True


def test_cloud_brain_reset_gives_a_fresh_session_with_the_new_briefing():
    g = FakeGemini()
    ctx = FakeContext()
    brain = CloudBrain(g, ctx, ready_timeout=1.0)
    asyncio.run(brain.reset("NEW BRIEFING"))
    assert g._system_instruction_from_init == "NEW BRIEFING"
    assert g._settings.system_instruction == "NEW BRIEFING"
    assert g.calls == ["disconnect", ("connect", None, None),
                       ("audio_paused", False)], g.calls
    assert ctx.history[-1] == [{"role": "system", "content": "NEW BRIEFING"}]
    assert brain.resets == 1
    asyncio.run(brain.idle())
    assert g.audio_paused is True and g.calls.count("disconnect") == 1


def test_runner_resets_the_brain_on_start_and_end(tmp_path):
    g = FakeGemini()
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    ctx = FakeContext()
    runner = Runner(source="unused", store=store, holder=holder, context=ctx,
                    robot=None, save_wait_secs=0.1,
                    brain=CloudBrain(g, ctx, ready_timeout=1.0))
    runner._recent_vectors = [unit(0)]
    asyncio.run(runner.start_session(now=1.0))
    assert g.calls.count("disconnect") == 1
    assert "do not recognize" in g._system_instruction_from_init
    asyncio.run(runner.end_session())
    assert g.calls.count("disconnect") == 1, "leaving does not reconnect"
    assert g.audio_paused is True, "the room is not streamed while idle"
    assert "nobody is in front of you" in ctx.system_text().lower()
    runner._recent_vectors = [unit(1)]
    asyncio.run(runner.start_session(now=60.0))
    assert g.calls.count("disconnect") == 2 and g.audio_paused is False


@pytest.mark.models
def test_pipecat_gemini_service_still_has_the_private_fields(paths):
    """CloudBrain leans on pipecat 1.6 internals; fail loudly if they move."""
    code = ("from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService as G;"
            "import inspect; src=inspect.getsource(G);"
            "missing=[f for f in %r if f not in src]; print(missing)"
            % (list(CloudBrain.FIELDS),))
    out = subprocess.run([str(paths.voice_py), "-c", code], cwd=paths.voice_dir,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", "pipecat internals moved: " + out.stdout
