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
(``_robot_neutral``, ``_perform``) are deliberately small and overridable,
so the runner's start/end choreography is testable without an LLM.

Presence policy (T13.2, from the family debrief "nobody could say how
long it takes to forget you"):

* the visitor's *speech* counts as presence (``on_voice``), so someone who
  steps out of frame but keeps talking is never timed out;
* at ``ask_fraction`` of the walk-away timer the robot asks once, out
  loud, whether they are still there ("ask" advice);
* when the timer expires the robot says a one-line goodbye *before* the
  notes save, so the save is never silent for someone still in earshot;
* every transition is logged as ``presence: face|voice|asking|gone``.

Also here: the idle ``Attractor`` (T13.4) -- with nobody in frame for a
while, play a short recorded move every few minutes so the booth reads
as alive from across the hall -- and the feed to the face tracker
(T13.3), which shares the runner's frames.

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

WALKAWAY_CUE = ("(The visitor seems to have left. Say one short goodbye "
                "sentence in case they can still hear you, and in this "
                "same turn, if you were tutoring someone this session, call "
                "save_session_notes with your honest summary.)")

STILL_THERE_CUE = ("(You have not seen or heard the visitor for a while. "
                   "Ask, in one short sentence in the lesson language, "
                   "whether they are still there. Nothing else.)")


class SessionMachine:
    """Watch/active state with stability, still-there and walk-away timers."""

    def __init__(self, stable_secs: float = 2.0, absent_secs: float = 60.0,
                 ask_fraction: float = 2.0 / 3.0):
        self.stable_secs = stable_secs
        self.absent_secs = absent_secs
        self.ask_fraction = ask_fraction
        self.state = WATCHING
        self._first_seen: float | None = None
        self._last_seen: float | None = None
        self._asked = False

    def on_face(self, present: bool, now: float) -> str | None:
        """Feed one observation; returns "start", "ask", "end", or None.

        The caller confirms a transition with session_started() /
        session_asked() / session_ended() — until it does, the advice
        repeats.
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
            self._asked = False
            return None
        if self._last_seen is None:
            return None
        gone = now - self._last_seen
        if gone >= self.absent_secs:
            return "end"
        if not self._asked and gone >= self.absent_secs * self.ask_fraction:
            return "ask"
        return None

    def on_voice(self, now: float) -> None:
        """The visitor spoke: that is presence too (T13.2). Voice never
        *starts* a session -- a face must -- but it keeps one alive."""
        if self.state == ACTIVE:
            self._last_seen = now
            self._asked = False

    def seconds_absent(self, now: float) -> float:
        if self.state != ACTIVE or self._last_seen is None:
            return 0.0
        return max(0.0, now - self._last_seen)

    def session_started(self, now: float) -> None:
        self.state = ACTIVE
        self._last_seen = now
        self._first_seen = None
        self._asked = False

    def session_asked(self) -> None:
        self._asked = True

    def session_ended(self) -> None:
        self.state = WATCHING
        self._first_seen = None
        self._last_seen = None
        self._asked = False


class Attractor:
    """Idle attractor timing (T13.4): with nobody in frame for
    ``after_secs``, fire, then again every ``every_secs`` while still
    empty. Any face silences it and restarts the clock. ``after_secs``
    of zero disables it."""

    def __init__(self, after_secs: float = 0.0, every_secs: float = 180.0):
        self.after_secs = after_secs
        self.every_secs = every_secs
        self._empty_since: float | None = None
        self._last_fired: float | None = None

    @property
    def enabled(self) -> bool:
        return self.after_secs > 0

    def on_face(self, present: bool, now: float) -> bool:
        if not self.enabled:
            return False
        if present:
            self._empty_since = None
            self._last_fired = None
            return False
        if self._empty_since is None:
            self._empty_since = now
            return False
        if now - self._empty_since < self.after_secs:
            return False
        if (self._last_fired is not None
                and now - self._last_fired < self.every_secs):
            return False
        self._last_fired = now
        return True


class SessionRunner:
    """Drives the machine from real frames and rewires the live agent."""

    def __init__(self, *, source, store, holder, context, task,
                 base_prompt: str, languages: str, robot=None, stt=None,
                 stable_secs: float = 2.0, absent_secs: float = 60.0,
                 fps: float = 2.0, samples: int = 3,
                 save_wait_secs: float = 30.0, hub=None, tracker=None,
                 attract_secs: float = 0.0, attract_every: float = 180.0,
                 voice_identity=None, cue=None, brain=None):
        self.source = source
        self.store = store
        self.holder = holder
        self.context = context
        self.task = task
        self.base_prompt = base_prompt
        self.languages = languages
        self.robot = robot
        self.stt = stt  # optional: bilingual priming per learner (T7)
        self.machine = SessionMachine(stable_secs, absent_secs)
        self.attractor = Attractor(attract_secs, attract_every)
        self.hub = hub            # shared camera (T13.3); else own Camera
        self.tracker = tracker    # FaceTracker fed from this loop, if any
        self.voice_identity = voice_identity   # T13.9, reset per visitor
        # T14.3: how cues reach the model and how the brain is reset per
        # visitor. ``cue(text)`` defaults to the aggregator path (local
        # mode); cloud mode passes the agent's say_cue. ``brain`` is a
        # CloudBrain (below) in cloud mode: it swaps Gemini's system
        # instruction and reconnects for a fresh server-side history.
        self.cue = cue
        self.brain = brain
        self.fps = fps
        self.samples = samples
        self.save_wait_secs = save_wait_secs
        self._recent_vectors: list = []
        self._voice_at: float | None = None
        self._voice_seen: float | None = None
        self._presence = "gone"

    # -- presence inputs from the pipeline ----------------------------------

    def note_voice(self) -> None:
        """Called (from the pipeline) whenever the visitor starts speaking."""
        self._voice_at = time.monotonic()

    # -- the two side-effect seams, overridable in tests -------------------

    async def _queue_user_turn(self, text: str) -> None:
        if self.cue is not None:
            await self.cue(text)
            return
        from pipecat.frames.frames import LLMMessagesAppendFrame
        await self.task.queue_frames([LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": text}], run_llm=True)])

    async def _set_system_prompt(self, prompt: str, fresh: bool = True) -> None:
        """This system prompt from now on: the local context always; the
        cloud brain too -- a fresh server-side session when a visitor
        starts (``fresh``), or just muted and idle when one leaves (the
        next start resets it anyway, and Gemini should not hear the room
        while nobody is there)."""
        self.context.set_messages([{"role": "system", "content": prompt}])
        if self.brain is not None:
            if fresh:
                await self.brain.reset(prompt)
            else:
                await self.brain.idle()

    async def _robot_neutral(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        if self.robot is not None:
            try:
                self.robot.home(duration=1.0)
            except Exception as exc:                            # noqa: BLE001
                logger.warning("session: robot home failed: %s", exc)

    async def _perform(self, name: str, seconds: float) -> None:
        """The idle attractor's move (overridable in tests)."""
        if self.tracker is not None:
            self.tracker.suspend(seconds)
        if self.robot is not None:
            try:
                self.robot.perform(name)
            except Exception as exc:                            # noqa: BLE001
                logger.warning("session: attractor move failed: %s", exc)

    def _log_presence(self, state: str) -> None:
        if state != self._presence:
            self._presence = state
            logger.info("presence: %s", state)

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
            self._prime_stt(learner)
            notes = self.store.read_notes(learner.id,
                                          max_sessions=BRIEFING_SESSIONS)
            return build_briefing(learner, notes)
        logger.info("session: unsure about %s (score %.3f) -> will ask",
                    learner.id, found.score)
        self.holder.candidate = learner
        return UNSURE_BRIEFING.format(name=learner.name)

    def _prime_stt(self, learner) -> None:
        if self.stt is None:
            return
        from multilingual import bilingual_priming
        prompt = bilingual_priming(learner.target_language)
        if prompt and hasattr(self.stt, "initial_prompt"):
            self.stt.initial_prompt = prompt
            logger.info("session: whisper priming English + %s",
                        learner.target_language)

    async def start_session(self, now: float) -> None:
        from face import recognize
        face = recognize.enroll_from_vectors(self._recent_vectors)
        self.holder.reset()
        if self.voice_identity is not None:
            self.voice_identity.reset()
        briefing = self._briefing_for(face)
        await self._set_system_prompt(self.base_prompt + briefing)
        self.machine.session_started(now)
        logger.info("session: started")
        await self._queue_user_turn(WALKUP_CUE)

    async def ask_still_there(self) -> None:
        """Two-thirds into the walk-away timer: one spoken check."""
        self._log_presence("asking")
        logger.info("session: nobody seen or heard for %.0fs, asking",
                    self.machine.seconds_absent(time.monotonic()))
        self.machine.session_asked()
        await self._queue_user_turn(STILL_THERE_CUE)

    async def end_session(self) -> None:
        self._log_presence("gone")
        learner = self.holder.learner
        if learner is not None and learner.id not in self.holder.saved_ids:
            logger.info("session: walk-away, goodbye + notes on %s",
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
        if self.voice_identity is not None:
            self.voice_identity.reset()
        if self.stt is not None and hasattr(self.stt, "initial_prompt"):
            self.stt.initial_prompt = None
        await self._set_system_prompt(self.idle_prompt(), fresh=False)
        await self._robot_neutral()
        self.machine.session_ended()
        logger.info("session: ended and reset; watching again")

    async def speaker_changed(self) -> None:
        """The voice print says somebody else is talking now (T14.3): end
        the current visitor's session (goodbye + notes for them) and go
        back to watching -- the face loop starts the newcomer's session
        by itself, with identity resolved afresh. Nothing to do while
        watching."""
        if self.machine.state != ACTIVE:
            return
        logger.info("session: speaker changed, ending %s's session",
                    self.holder.learner.id if self.holder.learner else "the")
        await self.end_session()

    # -- the frame loop ----------------------------------------------------

    async def run(self) -> None:
        import random
        from face import recognize
        from moves import ATTRACT_MOVES, LIBRARY

        if self.hub is not None:
            source = self.hub
            frames = self.hub.frames()
        else:
            from face.camera import Camera
            from face_id import parse_source
            source = Camera(parse_source(self.source), fps=self.fps)
            frames = source.frames()
        exhausted = False
        try:
            while True:
                frame = None
                if not exhausted:
                    frame = await asyncio.to_thread(next, frames, None)
                    if frame is None:
                        # A hub only hands out None when its source is
                        # done; a file source that ran out is absent
                        # forever (the simulated walk-away).
                        exhausted = True
                if frame is None:
                    await asyncio.sleep(1.0 / self.fps)
                    face = None
                elif self.machine.state == WATCHING:
                    face = await asyncio.to_thread(recognize.analyze, frame)
                else:
                    # Mid-session: position and presence only, never
                    # identity (spec section 4A).
                    face = await asyncio.to_thread(recognize.detect, frame)

                now = time.monotonic()
                present = face is not None
                if present and self.machine.state == WATCHING \
                        and face.embedding is not None:
                    self._recent_vectors.append(face.embedding)
                    self._recent_vectors = self._recent_vectors[-self.samples:]

                # Voice presence (T13.2): the pipeline stamps note_voice()
                # from another task; fold it into the machine here.
                if self._voice_at is not None \
                        and self._voice_at != self._voice_seen:
                    self._voice_seen = self._voice_at
                    self.machine.on_voice(self._voice_at)
                    if not present and self.machine.state == ACTIVE:
                        self._log_presence("voice")
                if present:
                    self._log_presence("face")

                if self.tracker is not None:
                    await self.tracker.maybe_resync()
                    if present:
                        h, w = frame.shape[:2]
                        self.tracker.observe(face.bbox, w, h, now)
                    else:
                        self.tracker.relax(now)

                advice = self.machine.on_face(present, now)
                if advice == "start":
                    await self.start_session(now)
                elif advice == "ask":
                    await self.ask_still_there()
                elif advice == "end":
                    await self.end_session()
                elif self.machine.state == WATCHING \
                        and self.attractor.on_face(present, now):
                    name = random.choice(ATTRACT_MOVES)
                    logger.info("attractor: nobody for %.0fs, playing %s",
                                self.attractor.after_secs, name)
                    await self._perform(name, LIBRARY[name].seconds)
        finally:
            if self.hub is None:
                source.close()


class CloudBrain:
    """Per-visitor reset of a Gemini Live session (T14.3, the old T12).

    Gemini keeps the conversation server-side and pipecat's service
    ignores later local context swaps: its system instruction is fixed
    at connect, ``_reconnect`` resumes the *same* server session through
    a resumption handle, and every connect replays the local history.
    So a new visitor needs all three undone: the service's stored system
    instruction replaced, the local history emptied, and a connect with
    no resumption handle. Uses the service's private fields, like the
    ``_create_single_response`` injection path the agent already relies
    on -- pinned to pipecat 1.6 and covered by a test that fails loudly
    if the fields move.
    """

    FIELDS = ("_system_instruction_from_init", "_session_resumption_handle",
              "_ready_for_realtime_input", "_disconnect", "_connect")

    def __init__(self, gemini, context, ready_timeout: float = 8.0):
        self.gemini = gemini
        self.context = context
        self.ready_timeout = ready_timeout
        self.resets = 0

    def _pause_audio(self, paused: bool) -> None:
        pause = getattr(self.gemini, "set_audio_input_paused", None)
        if pause is not None:
            pause(paused)

    async def idle(self) -> None:
        """Nobody in front of the robot: stop streaming the room to Gemini
        (no answers to background chatter, no billing for silence). The
        connection stays; the next visitor's reset replaces it."""
        self._pause_audio(True)
        logger.info("cloud brain: idle, microphone not streamed")

    async def reset(self, system_prompt: str) -> None:
        g = self.gemini
        g._system_instruction_from_init = system_prompt
        settings = getattr(g, "_settings", None)
        if settings is not None and hasattr(settings, "system_instruction"):
            try:
                settings.system_instruction = system_prompt
            except Exception:                              # noqa: BLE001
                pass
        # Empty local history: the system message alone is what the
        # adapter extracts as the instruction; there is nothing to replay.
        self.context.set_messages(
            [{"role": "system", "content": system_prompt}])
        await g._disconnect()
        g._session_resumption_handle = None
        await g._connect()
        self.resets += 1
        await self.wait_ready()
        self._pause_audio(False)
        logger.info("cloud brain: fresh Gemini session with a new briefing")

    async def wait_ready(self) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.ready_timeout
        while loop.time() < deadline:
            if getattr(self.gemini, "_ready_for_realtime_input", False):
                return True
            await asyncio.sleep(0.1)
        logger.warning("cloud brain: Gemini not ready %.0fs after reconnect",
                       self.ready_timeout)
        return False
