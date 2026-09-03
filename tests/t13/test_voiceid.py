"""T13.9: voice print as a second identity signal.

Unmarked: the fusion policy on fake vectors, the store field, the
collector's energy gate. models: the ECAPA gate over the fixture voices
(thresholds asserted against measured scores, T2-style). anthropic:
enrollment stores a print from --voice-source, and a face-known learner
with the wrong voice gets challenged, then asked "is that you?".
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from tutor.store import LearnerStore  # noqa: E402
from tutor_mode import CurrentLearner, voice_cue  # noqa: E402
from voiceid import (  # noqa: E402
    ACCEPT_THRESHOLD, EMBED_DIM, REJECT_THRESHOLD, VoiceCollector,
    VoiceIdentity, compare, similarity,
)

VOICES = Path(__file__).resolve().parents[1] / "fixtures" / "voices"


def unit(axis, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def blend(base, target, seed=1):
    """A unit vector at exactly ``target`` cosine to ``base``."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=base.shape).astype(np.float32)
    noise -= np.dot(noise, base) * base
    noise /= np.linalg.norm(noise)
    return target * base + np.sqrt(1 - target ** 2) * noise


# -- store ----------------------------------------------------------------------

def test_voice_embedding_round_trip_and_legacy(tmp_path):
    store = LearnerStore(tmp_path / "learners")
    l = store.create("Ana", "es", voice_embedding=[0.1] * 4)
    assert store.load(l.id).voice_embedding == [0.1] * 4
    l2 = store.create("Bo", "fr")
    assert store.load(l2.id).voice_embedding is None
    (tmp_path / "learners" / "old").mkdir()
    (tmp_path / "learners" / "old" / "profile.json").write_text(json.dumps({
        "name": "Old", "target_language": "es", "level": "beginner",
        "embedding": None, "sessions": 0, "last_seen": "2026-08-30",
        "tier": "family"}))
    assert store.load("old").voice_embedding is None


def test_thresholds_leave_an_ask_band():
    assert 0 < REJECT_THRESHOLD < ACCEPT_THRESHOLD < 1
    base = unit(0)
    assert compare(blend(base, 0.9), base).sure
    mid = compare(blend(base, (ACCEPT_THRESHOLD + REJECT_THRESHOLD) / 2), base)
    assert not mid.sure and not mid.rejected
    assert compare(blend(base, 0.1), base).rejected


# -- fusion policy ----------------------------------------------------------------

def make(tmp_path, *, learner=None, candidate=None):
    store = LearnerStore(tmp_path / "learners")
    holder = CurrentLearner()
    holder.learner, holder.candidate = learner, candidate
    return store, holder


def test_face_sure_voice_matches_verifies_once(tmp_path):
    store, holder = make(tmp_path)
    voice = unit(0)
    maria = store.create("Maria", "es", voice_embedding=[float(x) for x in voice])
    holder.learner = maria
    vid = VoiceIdentity(store, holder)
    assert vid.on_sample(blend(voice, 0.85)) == "verified"
    assert vid.on_sample(blend(voice, 0.80, seed=2)) is None
    # a later stray mismatch (a laugh, someone chiming in) is ignored
    assert vid.on_sample(unit(3)) is None
    assert holder.learner is maria


def test_face_sure_voice_wrong_challenges_then_downgrades(tmp_path):
    store, holder = make(tmp_path)
    maria = store.create("Maria", "es", voice_embedding=[float(x) for x in unit(0)])
    holder.learner = maria
    vid = VoiceIdentity(store, holder)
    assert vid.on_sample(unit(1)) == "challenge"
    assert holder.learner is maria, "one mismatch is a question, not a verdict"
    assert vid.on_sample(unit(2)) == "downgrade"
    assert holder.learner is None and holder.candidate is maria
    assert vid.on_sample(unit(2)) is None, "downgrade happens once"
    # the cues read as intended
    assert "playfully" in voice_cue("challenge", maria, store)
    assert "Maria, is that you?" in voice_cue("downgrade", maria, store)


def test_face_sure_voice_wrong_then_right_recovers(tmp_path):
    store, holder = make(tmp_path)
    voice = unit(0)
    maria = store.create("Maria", "es", voice_embedding=[float(x) for x in voice])
    holder.learner = maria
    vid = VoiceIdentity(store, holder)
    assert vid.on_sample(unit(1)) == "challenge"
    # they say more; the running print now leans to the real voice
    for _ in range(3):
        action = vid.on_sample(blend(voice, 0.95))
    assert action in ("verified", None) and holder.learner is maria
    assert not vid.downgraded


def test_face_unsure_voice_confirms(tmp_path):
    store, holder = make(tmp_path)
    voice = unit(0)
    maria = store.create("Maria", "es", level="advanced",
                         voice_embedding=[float(x) for x in voice])
    store.append_session(maria.id, "- **Next time:** the subjunctive")
    holder.candidate = maria
    vid = VoiceIdentity(store, holder)
    assert vid.on_sample(unit(5)) is None, "a strange voice settles nothing"
    assert holder.candidate is maria and holder.learner is None
    # next visitor-session: the same unsure face, and now the right voice
    vid2 = VoiceIdentity(store, holder)
    assert vid2.on_sample(blend(voice, 0.9)) == "confirmed"
    assert holder.learner is maria and holder.candidate is None
    cue = voice_cue("confirmed", maria, store)
    assert "do not ask" in cue and "subjunctive" in cue and "advanced" in cue


def test_face_sure_without_print_learns_one(tmp_path):
    store, holder = make(tmp_path)
    sam = store.create("Sam", "fr", tier="family")     # enrolled from photos
    holder.learner = sam
    vid = VoiceIdentity(store, holder, learn_after=2)
    assert vid.on_sample(unit(0)) is None
    assert vid.on_sample(blend(unit(0), 0.9)) == "learned"
    stored = store.load(sam.id).voice_embedding
    assert stored and len(stored) == 8
    assert similarity(stored, unit(0)) > 0.9
    assert vid.on_sample(unit(0)) == "verified", "and from then on it checks"


def test_reset_clears_session_state(tmp_path):
    store, holder = make(tmp_path)
    vid = VoiceIdentity(store, holder)
    holder.learner = store.create("A", "es", voice_embedding=[float(x) for x in unit(0)])
    vid.on_sample(unit(1)); vid.on_sample(unit(1))
    assert vid.downgraded
    vid.reset()
    assert not vid.downgraded and vid.samples == [] and vid.print is None


# -- the collector's gate (no model) -------------------------------------------

def pcm(seconds, level, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=int(seconds * sr)).astype(np.float32) * level
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


def test_collector_closes_a_sample_after_speech_then_silence():
    got = []
    c = VoiceCollector(lambda v, s: got.append(v), silence_secs=0.5,
                       min_secs=1.0)
    for _ in range(10):                 # 1 s of quiet: learn the floor
        assert c.feed(pcm(0.1, 0.001), 16000) is None
    for _ in range(15):                 # 1.5 s of "speech"
        out = c.feed(pcm(0.1, 0.1), 16000)
        assert out is None
    closed = None
    for _ in range(6):                  # 0.6 s of quiet closes it
        out = c.feed(pcm(0.1, 0.001), 16000)
        if out is not None:
            closed = out
    assert closed is not None and closed.size >= 16000 * 1.4


def test_collector_ignores_the_robots_own_voice_and_short_blips():
    c = VoiceCollector(lambda v, s: None, silence_secs=0.5, min_secs=1.0)
    c.set_bot_speaking(True)
    for _ in range(20):
        assert c.feed(pcm(0.1, 0.1), 16000) is None
    c.set_bot_speaking(False)
    for _ in range(3):                  # 0.3 s blip, under min_secs
        c.feed(pcm(0.1, 0.1), 16000)
    for _ in range(8):
        assert c.feed(pcm(0.1, 0.001), 16000) is None


def test_collector_resamples_stereo_48k():
    pytest.importorskip("scipy")        # resampling needs it (voice venv has it)
    c = VoiceCollector(lambda v, s: None)
    stereo = np.repeat(np.frombuffer(pcm(0.1, 0.1, sr=48000), np.int16), 2)
    assert c.feed(stereo.tobytes(), 48000, channels=2) is None
    assert c._speech and abs(c._speech[0].size - 1600) <= 2


# -- the measured gate ------------------------------------------------------------

@pytest.mark.models
def test_ecapa_gate_on_fixture_voices(paths):
    out = subprocess.run(
        [str(paths.voice_py), str(paths.voice_dir / "verify_voiceid.py"),
         "--dir", str(VOICES), "--out", str(paths.reports)],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, "band not clean:\n" + out.stdout + out.stderr[-2000:]
    date = re.search(r"gate — (\d{4}-\d{2}-\d{2})", out.stdout).group(1)
    report = json.loads(
        (paths.reports / f"verify_voiceid_{date}.json").read_text())
    assert report["same_speaker"]["min"] >= ACCEPT_THRESHOLD
    assert report["different_speaker"]["max"] < REJECT_THRESHOLD
    assert report["clips"] == 20 and len(report["speakers"]) == 5
    assert report["device"] in ("cuda:0", "cpu")


@pytest.mark.models
def test_voiceid_cli_embeds_a_clip(paths):
    out = subprocess.run(
        [str(paths.voice_py), str(paths.voice_dir / "voiceid.py"),
         str(VOICES / "am_adam" / "am_adam_1.wav")],
        capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    data = json.loads(out.stdout)
    assert len(data["embedding"]) == EMBED_DIM
    assert abs(np.linalg.norm(data["embedding"]) - 1) < 1e-3


# -- through the real agent -------------------------------------------------------

def tts_lines(log: str) -> str:
    return " ".join(
        m.group(1) for m in re.finditer(r"Generating TTS \[(.*)\]", log))


def video_embedding(paths):
    out = subprocess.run(
        [str(paths.voice_py), str(paths.repo / "face" / "recognize.py"),
         "embed", str(paths.fixtures / "faces" / "sunita_b.jpg")],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)["embedding"]


def clip_embedding(paths, wav):
    out = subprocess.run(
        [str(paths.voice_py), str(paths.voice_dir / "voiceid.py"), str(wav)],
        capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)["embedding"]


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_enrollment_stores_a_voice_print(paths, tmp_path, run_agent_say):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    root = tmp_path / "learners"
    log, found = run_agent_say(
        ["Hello! I'd like to learn some Spanish.",
         "Yes please, remember me. My name is Sunita.",
         "Spanish.", "Beginner.", "Just conversation."],
        wait_for=r"tutor: enrolled new guest",
        extra_args=["--face-source", str(clip), "--learners-root", str(root),
                    "--voice-source", str(VOICES / "af_heart" / "af_heart_1.wav"),
                    "--say-gap", "12"],
        timeout=400)
    assert found, "enrollment never fired; log tail:\n" + log[-4000:]
    assert "with voice print" in log
    guest = LearnerStore(root).list()[0]
    assert guest.voice_embedding and len(guest.voice_embedding) == EMBED_DIM
    assert similarity(guest.voice_embedding,
                      clip_embedding(paths, VOICES / "af_heart" / "af_heart_2.wav")
                      ) >= ACCEPT_THRESHOLD


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_known_face_wrong_voice_is_challenged_then_asked(paths, tmp_path,
                                                          run_agent_say):
    clip = paths.fixtures / "video" / "sunita_clip.avi"
    root = tmp_path / "learners"
    store = LearnerStore(root)
    store.create("Sunita", "es", level="intermediate", tier="family",
                 embedding=video_embedding(paths),
                 voice_embedding=clip_embedding(
                     paths, VOICES / "bm_george" / "bm_george_1.wav"))
    log, found = run_agent_say(
        ["Hola, quiero practicar un poco.",
         "Sí, claro. Hoy quiero hablar del mercado."],
        wait_for=r"voice: still no match .* downgraded to ask",
        # the downgrade cue is answered after this line; give the reply
        # time to be spoken before the SIGINT
        also_wait_for=r"Generating TTS",
        extra_args=["--face-source", str(clip), "--learners-root", str(root),
                    "--voice-source", str(VOICES / "af_heart" / "af_heart_3.wav"),
                    "--say-gap", "14"],
        settle=25, timeout=400)
    assert "face: known sunita" in log, "the seeded face was not recognized"
    assert "voice: sunita does not sound like themselves" in log, \
        "no challenge happened:\n" + log[-4000:]
    assert found, "never downgraded to the ask band:\n" + log[-4000:]
    # what was said after the downgrade must ask who this is (any language)
    after = log[log.find("downgraded to ask"):]
    spoken = tts_lines(after).lower()
    assert re.search(r"is that you|are you|eres|sunita", spoken), \
        "the robot never asked who it was talking to:\n" + spoken[-1500:]
