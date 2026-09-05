"""T15 through the real agent (skips without voice/.venv, a Gemini key or
a sound card): one cloud launch, a visitor enrolls, then somebody else
takes the seat mid-lesson without saying a word -- the face check ends
the first session (notes for the person who left) and greets the
newcomer as a stranger. Also: the turn timer inside the embodiment
processor, run in the voice venv where pipecat lives.
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutor.store import LearnerStore  # noqa: E402

VOICES = Path(__file__).resolve().parents[1] / "fixtures" / "voices"


def seat_swap_clip(paths, tmp_path, first_loops: int = 5,
                   second_secs: float = 45.0) -> Path:
    """The fixture clip (Sunita) looped ``first_loops`` times (~30 s),
    then Scott's portrait held for ``second_secs`` -- a seat swap with
    no gap, the 2026-09-04 case. Scott stays long enough for Sunita's
    goodbye and notes (the walk-away cue waits for the save) and his
    own greeting."""
    import cv2
    src = str(paths.fixtures / "video" / "sunita_clip.avi")
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]
    scott = cv2.imread(str(paths.fixtures / "faces" / "scott_a.jpg"))
    sh, sw = scott.shape[:2]
    crop_h = int(sw * h / w)                     # centre crop to the clip's aspect
    top = max(0, (sh - crop_h) // 2)
    scott = cv2.resize(scott[top:top + crop_h, :], (w, h))
    out_path = tmp_path / "seat_swap.avi"
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"),
                          fps, (w, h))
    for _ in range(first_loops):
        for f in frames:
            out.write(f)
    for _ in range(int(fps * second_secs)):
        out.write(scott)
    out.release()
    return out_path


@pytest.mark.google
@pytest.mark.models
@pytest.mark.audio
def test_seat_swap_mid_session_over_the_cloud_voice(paths, tmp_path,
                                                     run_agent_say):
    """Timeline at 2 fps: Sunita's session starts ~1 s in, the interview
    answer lands at 8 s and enrollment runs at ~18 s, Scott's face
    replaces hers at ~30 s and is the only face until the clip ends at
    ~75 s. Expected: a face-swap line
    within swap_secs, notes for Sunita, a stranger greeting for Scott
    from a fresh Gemini session, and no session ended by the voice."""
    clip = seat_swap_clip(paths, tmp_path)
    root = tmp_path / "learners"
    log, found = run_agent_say(
        ["Hi! Yes, please remember me. I'm Sunita, Spanish, beginner, "
         "just for conversation.",
         "Can we practise ordering coffee?"],
        wait_for=r"session: face swap",
        also_wait_for=r"session: face swap[\s\S]*face unknown -> stranger flow"
                      r"[\s\S]*cloud brain: fresh Gemini session",
        extra_args=["--speech", "cloud", "--session", "--face-source", str(clip),
                    "--learners-root", str(root), "--persona", "booth",
                    "--stable-secs", "0.5", "--absent-secs", "8",
                    "--say-delay", "8", "--say-gap", "8",
                    "--voice-source", str(VOICES / "af_heart" / "af_heart_1.wav")],
        settle=15, timeout=420)
    (paths.reports / "t15_seat_swap.log").write_text(log)   # the whole run, for reading
    assert "tutor: enrolled new guest" in log, "Sunita never enrolled:\n" + log[-4000:]
    assert found, "the seat swap was not caught:\n" + log[-5000:]
    swap_at = log.find("session: face swap")
    after = log[swap_at:]
    assert "goodbye + notes on sunita" in after, \
        "no notes for the person who left:\n" + after[-3000:]
    assert "session: face unknown -> stranger flow" in after, \
        "the newcomer was not treated as new:\n" + after[-3000:]
    assert "speaker changed with nobody in frame" not in log, \
        "the voice ended a session the face should have decided"
    assert "recognized sunita" not in after, \
        "the newcomer was recognized as the person who left (enrollment " \
        "stored the wrong face?):\n" + after[-3000:]
    # nothing spoke to the empty chair before the first walk-up (T15.5)
    first_start = log.find("session: session: started")
    assert "said:" not in log[:first_start], \
        "the robot spoke before anyone walked up:\n" + log[:first_start][-2000:]
    learner = LearnerStore(root).load("sunita")
    assert learner is not None and learner.sessions >= 1, "Sunita's notes never saved"


@pytest.mark.models
def test_embodiment_turn_timer_logs_first_sound_after_user_stop(paths):
    """In the voice venv: UserStopped -> BotStarted logs one ``turn:``
    line; a BotStarted with no preceding stop logs nothing (T15.7)."""
    script = textwrap.dedent("""
        import asyncio, logging, sys
        sys.path.insert(0, ".")
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
        from pipecat.frames.frames import (BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame, UserStoppedSpeakingFrame)
        from embodiment import Embodiment

        class Robot:
            def posture(self, duration, **dofs): pass

        async def main():
            e = Embodiment(Robot(), sway=False)
            e._react(BotStartedSpeakingFrame())      # no stop before it
            e._react(BotStoppedSpeakingFrame())
            assert e.bot_speaking is False
            e._react(UserStoppedSpeakingFrame())
            await asyncio.sleep(0.25)
            e._react(BotStartedSpeakingFrame())
            assert e.bot_speaking is True
            e.note_user_stopped(asyncio.get_event_loop().time() - 1.0)
            e._react(BotStartedSpeakingFrame())
        asyncio.run(main())
    """)
    out = subprocess.run([str(paths.voice_py), "-c", script],
                         cwd=paths.voice_dir, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = re.findall(r"turn: first sound ([0-9.]+)s after", out.stderr)
    assert len(lines) == 2, out.stderr[-2000:]
    assert 0.2 <= float(lines[0]) < 1.0 and float(lines[1]) >= 1.0
