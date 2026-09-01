"""Session lifecycle (T10): watch → greet → tutor → save on walk-away →
reset → watch, with nobody touching a keyboard.

Two layers:

* ``SessionMachine`` — the pure state machine. Two states (watching /
  active), fed ``on_face(present, now)`` about twice a second. A face must
  be stable for ``stable_secs`` before a session starts (a passer-by is
  not a visitor); a face gone ``absent_secs`` ends one. Fully
  unit-testable with a fake clock, no I/O, no models.
* ``SessionRunner`` — the agent-side driver. Reads frames from the face
  source (the T1 Camera), embeds them (T2) for presence *and* identity,
  and on the machine's say-so rewires the live conversation: sets the
  right briefing as a fresh system prompt (known face → the T4 briefing,
  unsure → confirm-first, unknown → stranger/enrollment flow), cues the
  model to greet, and on walk-away cues it to call save_session_notes,
  waits for the save, resets the context and the robot, and goes back to
  watching.

The two methods that touch pipecat (``_queue_user_turn``) and the robot
(``_robot_neutral``) are deliberately small and overridable, so the
runner's start/end choreography is testable without an LLM.

Spec note (§4A): recognition is *supposed* to be paused during
conversation. Presence still has to be tracked to detect the walk-away,
so during an active session frames are embedded only for "is a face
there" — the identity is never re-evaluated mid-session. Embedding costs
~95 ms on this box's CPU (measured, T2), which the GPU never sees.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

logger = logging.getLogger("session")

WATCHING = "watching"
ACTIVE = "active"

# What the model is told while nobody is around. It should never speak in
# this state -- there is nobody to speak to.
IDLE_NOTE = """

Right now nobody is in front of you. Stay quiet and wait. If somehow \
prompted anyway, answer briefly in English."""

WALKUP_CUE = ("(A visitor has just walked up to you. Follow your "
              "instructions and greet them now.)")

WALKAWAY_CUE = ("(The visitor has walked away and cannot hear you anymore. "
                "Do not speak. If you were tutoring someone this session, "
                "call save_session_notes now with your honest summary.)")


class SessionMachine:
    """Watch/active state with stability and walk-away timers."""

    def __init__(self, stable_secs: float = 2.0, absent_secs: float = 60.0):
        self.stable_secs = stable_secs
        self.absent_secs = absent_secs
        self.state = WATCHING
        self._first_seen: float | None = None
        self._last_seen: float | None = None

    def on_face(self, present: bool, now: float) -> str | None:
        """Feed one observation; returns "start", "end", or None.

        The caller confirms the transition with session_started() /
        session_ended() — until it does, the advice repeats.
        """
        if self.state == WATCHING:
            if not present:
                self._first_seen = None
                return None
            if self._first_seen is None:
                self._first_seen = now
                return None
            if now - self._first_seen >= self.stable_secs:
                return "start"
            return None
        # ACTIVE
        if present:
            self._last_seen = now
            return None
        if (self._last_seen is not None
                and now - self._last_seen >= self.absent_secs):
            return "end"
        return None

    def session_started(self, now: float) -> None:
        self.state = ACTIVE
        self._last_seen = now
        self._first_seen = None

    def session_ended(self) -> None:
        self.state = WATCHING
        self._first_seen = None
        self._last_seen = None


class SessionRunner:
    """Drives the machine from real frames and rewires the live agent."""

    def __init__(self, *, source, store, holder, context, task,
                 base_prompt: str, languages: str, robot=None,
                 stable_secs: float = 2.0, absent_secs: float = 60.0,
                 fps: float = 2.0, samples: int = 3,
                 save_wait_secs: float = 30.0):
        self.source = source
        self.store = store
        self.holder = holder
        self.context = context
        self.task = task
        self.base_prompt = base_prompt
        self.languages = languages
        self.robot = robot
        self.machine = SessionMachine(stable_secs, absent_secs)
        self.fps = fps
        self.samples = samples
        self.save_wait_secs = save_wait_secs
        self._recent_vectors: list = []

    # -- the two side-effect seams, overridable in tests -------------------

    async def _queue_user_turn(self, text: str) -> None:
        from pipecat.frames.frames import LLMMessagesAppendFrame
        await self.task.queue_frames([LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": text}], run_llm=True)])

    async def _robot_neutral(self) -> None:
        if self.robot is not None:
            try:
                self.robot.home(duration=1.0)
            except Exception as exc:                            # noqa: BLE001
                logger.warning("session: robot home failed: %s", exc)

    # -- choreography ------------------------------------------------------

    def idle_prompt(self) -> str:
        return self.base_prompt + IDLE_NOTE

    def _briefing_for(self, face_vector) -> str:
        """Identity resolution at session start (the once-per-session
        recognition pass) → the right system-prompt addendum."""
        from face import recognize
        from tutor_mode import (
            BRIEFING_SESSIONS, STRANGER_BRIEFING, UNSURE_BRIEFING,
            build_briefing,
        )
        known = {l.id: l.embedding for l in self.store.list()
                 if l.embedding and len(l.embedding) == len(face_vector)}
        found = recognize.match(face_vector, known)
        if found is None:
            logger.info("session: face unknown -> stranger flow")
            return STRANGER_BRIEFING.format(languages=self.languages)
        learner = self.store.load(found.name)
        if learner is None:
            return STRANGER_BRIEFING.format(languages=self.languages)
        if found.sure:
            logger.info("session: recognized %s (score %.3f)",
                        learner.id, found.score)
            self.holder.learner = learner
            notes = self.store.read_notes(learner.id,
                                          max_sessions=BRIEFING_SESSIONS)
            return build_briefing(learner, notes)
        logger.info("session: unsure about %s (score %.3f) -> will ask",
                    learner.id, found.score)
        self.holder.candidate = learner
        return UNSURE_BRIEFING.format(name=learner.name)

    async def start_session(self, now: float) -> None:
        from face import recognize
        face = recognize.enroll_from_vectors(self._recent_vectors)
        self.holder.reset()
        briefing = self._briefing_for(face)
        self.context.set_messages([
            {"role": "system", "content": self.base_prompt + briefing}])
        self.machine.session_started(now)
        logger.info("session: started")
        await self._queue_user_turn(WALKUP_CUE)

    async def end_session(self) -> None:
        learner = self.holder.learner
        if learner is not None and learner.id not in self.holder.saved_ids:
            logger.info("session: walk-away, asking for notes on %s",
                        learner.id)
            await self._queue_user_turn(WALKAWAY_CUE)
            deadline = time.monotonic() + self.save_wait_secs
            while (time.monotonic() < deadline
                   and learner.id not in self.holder.saved_ids):
                await asyncio.sleep(0.5)
            if learner.id not in self.holder.saved_ids:
                logger.warning("session: notes were never saved for %s",
                               learner.id)
        self.holder.reset()
        self.context.set_messages([
            {"role": "system", "content": self.idle_prompt()}])
        await self._robot_neutral()
        self.machine.session_ended()
        logger.info("session: ended and reset; watching again")

    # -- the frame loop ----------------------------------------------------

    async def run(self) -> None:
        from face.camera import Camera
        from face import recognize
        from face_id import parse_source

        camera = Camera(parse_source(self.source), fps=self.fps)
        frames = camera.frames()
        exhausted = False
        try:
            while True:
                frame = None
                if not exhausted:
                    frame = await asyncio.to_thread(next, frames, None)
                    if frame is None:
                        exhausted = True  # file source ran out: absent forever
                if frame is None:
                    await asyncio.sleep(1.0 / self.fps)
                    vector = None
                else:
                    vector = await asyncio.to_thread(recognize.embed, frame)

                now = time.monotonic()
                present = vector is not None
                if self.machine.state == WATCHING and present:
                    self._recent_vectors.append(vector)
                    self._recent_vectors = self._recent_vectors[-self.samples:]

                advice = self.machine.on_face(present, now)
                if advice == "start":
                    await self.start_session(now)
                elif advice == "end":
                    await self.end_session()
        finally:
            camera.close()
