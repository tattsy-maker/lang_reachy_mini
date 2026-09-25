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

Identity during a session (T15.1, replacing the spec's original §4A
"recognize once, then presence only"): on 2026-09-04 one visitor took
another's seat mid-lesson and the robot tutored him under her name for
the rest of the session, because nothing looked at the face again after
the greeting and the voice check had been switched off by a verbal
"yes". So the face is now the arbiter for the whole session: every
``face_recheck_secs`` (and at once when a face reappears after a gap)
the largest face is embedded and compared with the face that started
the session. The same face keeps the session alive whatever the voice
print says; a different face held for ``swap_secs`` ends the session
(goodbye + notes for the person who left) and the newcomer gets their
own. Embedding costs ~95 ms on this box's CPU (measured, T2), which the
GPU never sees; at one in two seconds that is under 5% of a core.

Bystanders (T17.3): on 2026-09-05 a second face was "the largest" for
2-5 s about once a minute (34 near-zero ``other`` verdicts in forty
minutes, people leaning in from the next chair), and anyone who stayed
there 3 s would have ended the lesson. The runner now looks at *every*
face in the frame: the session's face anywhere in it means "same",
whichever is largest; ``other`` only when no face matches; and between
recognition passes the tracker follows the box nearest to where the
session's face was last seen, not the biggest one.

Also here since T17: the "time is nearly up" cue from the agreed session
plan (T17.4), the sightseeing budget reset per visitor (T17.8), and the
face write-back at the end of a session (T17.2: a strengthened reference
face goes into the profile so the next visit starts from a better one).
"""

from __future__ import annotations

import asyncio
import logging
import math
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

# T17.4: a returning learner gets the two session questions (how long,
# what today) that a newcomer answers in the intake.
WALKUP_CUE_KNOWN = ("(A visitor you know has just walked up to you. Follow "
                    "your instructions and greet them now. Then, in one "
                    "question, ask how many minutes they have today and "
                    "what they want to get out of it; call "
                    "set_session_plan with the answer, say a plan of two "
                    "or three steps in one or two sentences, and only "
                    "then begin.)")

WALKAWAY_CUE = ("(The visitor seems to have left. Say one short goodbye "
                "sentence in case they can still hear you, and in this "
                "same turn, if you were tutoring someone this session, call "
                "save_session_notes with your honest summary.)")

STILL_THERE_CUE = ("(You have not seen or heard the visitor for a while. "
                   "Ask, in one short sentence in the lesson language, "
                   "whether they are still there. Nothing else.)")

# T15.4, the family's own suggestion ("warn him in advance that many
# people will come"): every prompt the runner builds carries this.
BOOTH_NOTE = """

You are at a busy booth. People walk up, swap seats and leave without \
warning, often mid-sentence, so the person in front of you can change at \
any moment. If a voice or an answer does not fit the student you were \
talking to, do not assume: call look, or ask their name."""


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
                 voice_identity=None, cue=None, brain=None,
                 face_recheck_secs: float = 2.0, swap_secs: float = 3.0,
                 face_vouch_secs: float = 5.0, speaking=None,
                 booth_note: bool = True, budget=None,
                 snapshot_dir: str | None = None):
        self.source = source
        # 2026-09-23: every session start leaves a photo of who was
        # greeted (the box drawn in), next to the look frames.
        self.snapshot_dir = snapshot_dir
        self._last_frame = None
        self._snapshot_faces = None
        self._ignored_logged_at = -math.inf
        self.store = store
        self.holder = holder
        self.context = context
        self.task = task
        self.base_prompt = base_prompt + (BOOTH_NOTE if booth_note else "")
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
        # T15.1: identity for the whole session. ``face_recheck_secs``
        # between recognition passes while active; a different face for
        # ``swap_secs`` ends the session; a face match within
        # ``face_vouch_secs`` overrules the voice print.
        self.face_recheck_secs = face_recheck_secs
        self.swap_secs = swap_secs
        self.face_vouch_secs = face_vouch_secs
        self._session_face = None          # who this session started with
        self._session_vectors: list = []   # confident matches, averaged in
        self._same_face_at: float | None = None
        # T15.10: every recheck is logged -- a verdict change at once,
        # otherwise a 10 s summary -- so a weak reference face shows up
        # as a run of "unsure" instead of silence (2026-09-04 20:28).
        self._check_log = {"verdict": None, "since": 0.0, "n": 0,
                           "lo": 1.0, "hi": -1.0}
        self._other_since: float | None = None
        self._last_recheck: float = -1e9
        self._force_recheck = False
        self._face_absent_since: float | None = None
        # T15.6: ``speaking()`` -> is the robot talking; a goodbye gets
        # to finish before the next greeting, a greeting waits for the
        # previous visitor's last words.
        self.speaking = speaking
        # T17.3: where the session's face was last seen, so the
        # detect-only passes between recognitions follow that box.
        self._session_bbox = None
        # T17.8: the sightseeing budget, reset per visitor.
        self.budget = budget

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

    async def _wait_quiet(self, max_secs: float,
                          start_grace: float = 0.0) -> None:
        """Let the robot finish what it is saying (T15.6): on 2026-09-04
        a forced goodbye and the next greeting came out in one breath.
        ``start_grace``: also wait this long for speech to *begin* --
        after the notes save the goodbye is still being generated, so
        "not speaking yet" is not "done"."""
        if self.speaking is None:
            return
        t0 = time.monotonic()
        while (start_grace and not self.speaking()
               and time.monotonic() - t0 < start_grace):
            await asyncio.sleep(0.1)
        deadline = t0 + max_secs
        waited = False
        while time.monotonic() < deadline and self.speaking():
            waited = True
            await asyncio.sleep(0.2)
        if waited:
            logger.info("session: waited %.1fs for the robot to finish "
                        "speaking", time.monotonic() - t0)

    # -- choreography ------------------------------------------------------

    def idle_prompt(self) -> str:
        return self.base_prompt + IDLE_NOTE

    def _briefing_for(self, face_vector) -> str:
        """Identity resolution at session start (the once-per-session
        recognition pass) → the right system-prompt addendum."""
        from face import recognize
        from tutor_mode import (
            BRIEFING_SESSIONS, STRANGER_BRIEFING, build_briefing,
            build_unsure_briefing,
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
        return build_unsure_briefing(learner)

    def _prime_stt(self, learner) -> None:
        if self.stt is None:
            return
        from multilingual import bilingual_priming
        from tutor_mode import language_name, native_language_of
        native = native_language_of(learner)
        prompt = bilingual_priming(learner.target_language, native)
        if prompt and hasattr(self.stt, "initial_prompt"):
            self.stt.initial_prompt = prompt
            logger.info("session: whisper priming %s + %s",
                        language_name(native), learner.target_language)

    async def start_session(self, now: float) -> None:
        from face import recognize
        face = recognize.enroll_from_vectors(self._recent_vectors)
        await self._wait_quiet(4.0)
        self.holder.reset()
        if self.voice_identity is not None:
            self.voice_identity.reset()
        if self.budget is not None:
            self.budget.reset()
        self._session_face = face
        self._session_bbox = None
        self._session_vectors = list(self._recent_vectors)
        self._check_log = {"verdict": None, "since": now, "n": 0,
                           "lo": 1.0, "hi": -1.0}
        self._same_face_at = now
        self._other_since = None
        self._last_recheck = now
        self._force_recheck = False
        self._face_absent_since = None
        briefing = self._briefing_for(face)
        await self._set_system_prompt(self.base_prompt + briefing)
        self.machine.session_started(now)
        logger.info("session: started%s", self._snapshot())
        await self._queue_user_turn(WALKUP_CUE_KNOWN if self.holder.learner
                                    is not None else WALKUP_CUE)

    def schedule_rebrief(self, why: str) -> None:
        """A learner was enrolled or confirmed mid-session: once the
        robot has said its one line, make their own briefing the standing
        instruction (2026-09-23: the stranger briefing's "greet them in
        English" outlived a Russian-explained Italian enrollment)."""
        self._rebrief_task = asyncio.create_task(self._rebrief(why))

    async def _rebrief(self, why: str) -> None:
        from tutor_mode import (BRIEFING_SESSIONS, build_briefing,
                                language_name, native_language_of)
        try:
            await asyncio.sleep(0.3)
            await self._wait_quiet(15.0, start_grace=4.0)
            learner = self.holder.learner
            if learner is None or self.machine.state != ACTIVE:
                return
            notes = self.store.read_notes(learner.id,
                                          max_sessions=BRIEFING_SESSIONS)
            prompt = self.base_prompt + build_briefing(
                learner, notes, plan=getattr(self.holder, "plan", None))
            self._prime_stt(learner)
            if self.brain is None:
                # Local brain: swap the system message, keep the history.
                messages = list(self.context.get_messages())
                if messages and messages[0].get("role") == "system":
                    messages[0] = {"role": "system", "content": prompt}
                else:
                    messages.insert(0, {"role": "system", "content": prompt})
                self.context.set_messages(messages)
                logger.info("session: %s: %s's own briefing from here on",
                            why, learner.id)
                return
            self.context.set_messages([{"role": "system", "content": prompt}])
            await self.brain.reset(prompt)
            logger.info("session: %s: fresh Gemini session with %s's own "
                        "briefing", why, learner.id)
            await self._queue_user_turn(REBRIEF_CUE.format(
                name=learner.name,
                native=language_name(native_language_of(learner))))
        except Exception as exc:                               # noqa: BLE001
            logger.warning("session: rebrief failed: %s", exc)

    def _snapshot(self) -> str:
        """Save the frame the session started on; returns ' (file)' for
        the log line, or '' when there is nothing to save."""
        frame, faces = self._last_frame, self._snapshot_faces
        if not self.snapshot_dir or frame is None:
            return ""
        try:
            import cv2
            out = frame.copy()
            for f in faces or []:
                x1, y1, x2, y2 = (int(v) for v in f.bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(out, "%.2f" % f.score, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            folder = os.path.join(self.snapshot_dir, time.strftime("%Y-%m-%d"))
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder,
                                time.strftime("session_%H%M%S.jpg"))
            cv2.imwrite(path, out)
            return " (%s)" % path
        except Exception as exc:                            # noqa: BLE001
            logger.warning("session: snapshot failed: %s", exc)
            return ""

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
        self.holder.walkaway = True      # no wish question to an empty chair
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
            await self._wait_quiet(10.0, start_grace=3.0)  # the goodbye, in full
        self._write_back_face(learner)
        self.holder.reset()
        if self.voice_identity is not None:
            self.voice_identity.reset()
        self._session_face = None
        self._session_bbox = None
        self._session_vectors = []
        self._same_face_at = self._other_since = None
        self._recent_vectors = []        # the next face starts clean
        if self.stt is not None and hasattr(self.stt, "initial_prompt"):
            self.stt.initial_prompt = None
        await self._set_system_prompt(self.idle_prompt(), fresh=False)
        await self._robot_neutral()
        self.machine.session_ended()
        logger.info("session: ended and reset; watching again")

    def face_vouches(self, now: float | None = None) -> bool:
        """Was the session's face seen within ``face_vouch_secs``? (T17.2:
        the voice print asks this before it says anything.)"""
        if self.machine.state != ACTIVE or self._same_face_at is None:
            return False
        now = time.monotonic() if now is None else now
        return now - self._same_face_at <= self.face_vouch_secs

    def _write_back_face(self, learner, min_vectors: int = 6,
                         weight: float = 0.3) -> bool:
        """T17.2: a session that kept matching the learner leaves the
        profile a better reference face (``weight`` on this session's
        averaged face). Only for a recognized/confirmed learner with a
        stored face and enough confident matches."""
        if learner is None or self._session_face is None \
                or not learner.embedding \
                or len(self._session_vectors) < min_vectors:
            return False
        from face import recognize
        import numpy as np
        stored = np.asarray(learner.embedding, dtype=np.float32)
        if stored.shape != self._session_face.shape:
            return False
        merged = (1.0 - weight) * stored + weight * self._session_face
        merged = merged / (np.linalg.norm(merged) or 1.0)
        current = self.store.load(learner.id)
        if current is None:
            return False
        current.embedding = [float(x) for x in merged]
        self.store.save(current)
        learner.embedding = current.embedding
        logger.info("session: %s's stored face updated from %d matches this "
                    "session (score vs old %.3f)", learner.id,
                    len(self._session_vectors),
                    recognize.similarity(stored, merged))
        return True

    def _pick_face(self, faces: list, now: float):
        """T17.3: which of the frame's faces is *the visitor's*. With
        embeddings: the best match to the session's face when it clears
        the accept line, else the largest. Without (a detect-only pass):
        the box nearest to where the session's face was last seen."""
        if not faces:
            return None
        from face import recognize
        reference = self._session_face
        if reference is None and self.holder.learner is not None \
                and self.holder.learner.embedding:
            reference = self.holder.learner.embedding
        with_vectors = [f for f in faces if f.embedding is not None]
        if with_vectors and reference is not None:
            best = max(with_vectors,
                       key=lambda f: recognize.similarity(f.embedding, reference))
            if recognize.similarity(best.embedding, reference) \
                    >= recognize.ACCEPT_THRESHOLD:
                if best is not faces[0]:
                    logger.debug("session: the visitor is not the largest "
                                 "face; following theirs")
                return best
            return faces[0]
        if self._session_bbox is not None and len(faces) > 1:
            cx, cy = ((self._session_bbox[0] + self._session_bbox[2]) / 2.0,
                      (self._session_bbox[1] + self._session_bbox[3]) / 2.0)
            width = max(1.0, abs(self._session_bbox[2] - self._session_bbox[0]))

            def dist(f):
                fx, fy = f.centre
                return ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
            nearest = min(faces, key=dist)
            if dist(nearest) <= 1.5 * width:
                return nearest
        return faces[0]

    async def speaker_changed(self, now: float | None = None) -> None:
        """The voice print says somebody else is talking now (T14.3).

        T15.1: the face is the arbiter. If the session's face was seen
        within ``face_vouch_secs`` the voice is overruled (on 2026-09-04
        a child's own French cost him his session twice) and the voice
        check is re-armed. If a face is in frame but not yet judged, the
        face check decides within ``swap_secs``. Only with nobody in
        frame does the voice end the session (goodbye + notes for the
        person who left); the face loop starts the newcomer's session by
        itself. Nothing to do while watching."""
        if self.machine.state != ACTIVE:
            return
        now = time.monotonic() if now is None else now
        who = self.holder.learner.id if self.holder.learner else "the visitor"
        if self._same_face_at is not None \
                and now - self._same_face_at <= self.face_vouch_secs:
            logger.info("session: voice says someone else, but %s's face was "
                        "seen %.1fs ago; keeping the session", who,
                        now - self._same_face_at)
            if self.voice_identity is not None:
                self.voice_identity.rearm()
            return
        if self._face_absent_since is None and self._presence == "face":
            logger.info("session: voice says someone else; a face is there, "
                        "leaving the verdict to the face check")
            self._force_recheck = True
            if self.voice_identity is not None:
                self.voice_identity.rearm()
            return
        logger.info("session: speaker changed with nobody in frame, ending "
                    "%s's session", who)
        await self.end_session()

    def _check_identity(self, embedding, now: float) -> str:
        """Mid-session recognition pass (T15.1): the largest face against
        the face that started the session. Returns "same", "other",
        "unsure" or "none"."""
        from face import recognize
        self._last_recheck = now
        self._force_recheck = False
        reference = self._session_face
        if reference is None and self.holder.learner is not None:
            reference = self.holder.learner.embedding
        if reference is None or embedding is None:
            return "none"
        score = recognize.similarity(embedding, reference)
        if score >= recognize.ACCEPT_THRESHOLD:
            self._same_face_at = now
            if self.voice_identity is not None \
                    and self.voice_identity.face_vouches is None:
                self.voice_identity.face_vouches = self.face_vouches
            if self._other_since is not None:
                logger.info("session: the session's face is back (score %.3f)",
                            score)
            self._other_since = None
            # T15.10: a confident match strengthens the reference. The
            # three walk-up frames can be a poor likeness (moving, far,
            # half-turned); averaging in what the session keeps seeing
            # makes the swap test sharper as the minutes pass.
            if self._session_face is not None and len(self._session_vectors) < 12:
                self._session_vectors.append(embedding)
                self._session_face = recognize.enroll_from_vectors(
                    self._session_vectors)
            verdict = "same"
        elif score < recognize.REJECT_THRESHOLD:
            if self._other_since is None:
                self._other_since = now
                logger.info("session: a different face is in front of the "
                            "robot (score %.3f vs the session's face)", score)
            verdict = "other"
        else:
            verdict = "unsure"
        self._log_check(verdict, score, now)
        return verdict

    def _log_check(self, verdict: str, score: float, now: float) -> None:
        c = self._check_log
        c["n"] += 1
        c["lo"], c["hi"] = min(c["lo"], score), max(c["hi"], score)
        if verdict != c["verdict"] or now - c["since"] >= 10.0:
            if c["n"] == 1:
                logger.info("session: face check: %s (score %.3f)", verdict, score)
            else:
                logger.info("session: face check: %s (score %.3f; %d checks in "
                            "%.0fs, %.2f-%.2f)", verdict, score, c["n"],
                            now - c["since"], c["lo"], c["hi"])
            c.update(verdict=verdict, since=now, n=0, lo=1.0, hi=-1.0)

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
                now = time.monotonic()
                if frame is None:
                    await asyncio.sleep(1.0 / self.fps)
                    face = None
                elif self.machine.state == WATCHING or self._recheck_due(now):
                    faces = await asyncio.to_thread(recognize.analyze_all, frame)
                else:
                    # Between recognition passes: position and presence
                    # only (the tracker, the walk-away timer).
                    faces = await asyncio.to_thread(recognize.detect_all, frame)
                if frame is not None:
                    # Background look-alikes and people across the room
                    # are not visitors: no greeting, no presence, no gaze.
                    kept = recognize.visitors(faces, frame.shape[1])
                    self._log_ignored(faces, kept, frame.shape[1], now)
                    faces = kept
                    face = faces[0] if faces else None
                    self._last_frame, self._snapshot_faces = frame, faces
                await self.observe(face, now,
                                   None if frame is None else frame.shape[:2],
                                   faces=faces if frame is not None else None)
        finally:
            if self.hub is None:
                source.close()

    def _log_ignored(self, faces, kept, width, now: float) -> None:
        """At most every 30 s: the faces the visitor gate turned away, so
        a log shows what the camera saw that was not a visitor."""
        from face import recognize
        ignored = [f for f in faces if not any(f is k for k in kept)]
        if not ignored or now - self._ignored_logged_at < 30.0:
            return
        self._ignored_logged_at = now
        logger.info("session: not a visitor: %s (need score >= %.2f, "
                    "width >= %.1f%%)", ", ".join(
                        "score %.2f width %.1f%%" % (
                            f.score, 100.0 * (f.bbox[2] - f.bbox[0]) / width)
                        for f in ignored),
                    recognize.VISITOR_MIN_SCORE,
                    100.0 * recognize.VISITOR_MIN_WIDTH)

    def _recheck_due(self, now: float) -> bool:
        """Mid-session: is it time to embed the face for identity again?
        (T15.1: on the cadence, or at once after a gap or a voice doubt.)"""
        return (self._force_recheck
                or now - self._last_recheck >= self.face_recheck_secs)

    async def observe(self, face, now: float, frame_shape=None,
                      faces=None) -> None:
        """One observation: ``face`` is the largest face (with an
        embedding when a recognition pass ran) or None; ``faces`` (T17.3)
        every face in the frame, largest first, when the loop has them.
        Everything the frame loop decides happens here, so tests can
        drive it without a camera."""
        import random
        from moves import ATTRACT_MOVES, LIBRARY

        if faces and self.machine.state == ACTIVE:
            face = self._pick_face(faces, now)
        present = face is not None
        if present and self.machine.state == WATCHING \
                and face.embedding is not None:
            self._recent_vectors.append(face.embedding)
            self._recent_vectors = self._recent_vectors[-self.samples:]

        # T15.1: identity for the whole session.
        if self.machine.state == ACTIVE:
            if present:
                if self._face_absent_since is not None:
                    gone = now - self._face_absent_since
                    self._face_absent_since = None
                    if gone >= 1.0 and face.embedding is None:
                        # Somebody is back after a real gap: look at who,
                        # on the very next frame.
                        logger.info("session: a face is back after %.1fs "
                                    "away, checking who", gone)
                        self._force_recheck = True
                if face.embedding is not None:
                    verdict = self._check_identity(face.embedding, now)
                    if verdict == "same":
                        self._session_bbox = face.bbox
                    if verdict == "other" and self._other_since is not None \
                            and now - self._other_since >= self.swap_secs:
                        who = (self.holder.learner.id if self.holder.learner
                               else "the visitor")
                        logger.info("session: face swap -- someone else has "
                                    "been in front of the robot for %.1fs, "
                                    "ending %s's session", now -
                                    self._other_since, who)
                        await self.end_session()
                        # the newcomer starts from these frames on
                        self._face_absent_since = None
                        present = False       # no walk-away bookkeeping
            else:
                if self._face_absent_since is None:
                    self._face_absent_since = now
                self._other_since = None
            # T17.4: the agreed time is nearly up -- say so, once.
            plan = getattr(self.holder, "plan", None)
            if plan is not None and self.machine.state == ACTIVE \
                    and plan.due_warning(now):
                from intake import time_cue
                logger.info("session: %.0f%% of the agreed %d minutes used; "
                            "wrap-up cue", plan.warn_fraction * 100,
                            plan.minutes)
                await self._queue_user_turn(time_cue(plan, now))

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
            if present and frame_shape is not None:
                h, w = frame_shape
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


# 2026-09-23: the lesson's first turn after a mid-session rebrief.
REBRIEF_CUE = (
    "(Stage direction, not the visitor speaking: {name} is enrolled and "
    "you now have their briefing. You have already greeted them and told "
    "them you will remember them -- do not greet, introduce yourself or "
    "ask anything from the interview again. Speak {native} for every "
    "explanation from this sentence on, as the briefing says. Say the "
    "short plan and start the first exercise now.)")


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
