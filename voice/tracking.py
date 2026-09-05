"""Face tracking (T13.3): keep the robot's face on the person.

"It should follow the current speaker with head and body: up, down, left,
right, wherever you walk." The camera sits in the head, so every
observation is an *offset from where the head already points*: a face
left of the image centre means "turn a bit further left", not "turn to
angle X". The tracker therefore keeps its own estimate of the commanded
head/body yaw and pitch, nudges it by a fraction of the measured offset
(a proportional controller), and hands the yaw to the body when the head
is far enough round that the neck would look strained.

Rules that keep it from twitching or fighting the other motion sources:

* **Dead-band and rate limit.** Offsets under ``dead_band`` (6 deg) are
  ignored and commands are at least ``min_interval`` (0.8 s) apart, with
  a step cap (8 deg) and moves that take most of the interval, so a
  talker who sways a little gets a still robot and a walker gets a head
  that glides after them (retuned after the 2026-09-03 session).
* **DOF ownership** (see ``embodiment.py``): the tracker owns ``head_yaw``
  and ``body_yaw``. Embodiment owns pitch and the antennas; the tracker
  only *biases* pitch through ``Embodiment.pitch_bias`` and writes it
  directly when embodiment is idle. The talking sway drops its yaw
  component when a tracker exists (``Embodiment(own_yaw=False)``).
* **Deliberate moves win.** A tool call that turns the head or body, a
  recorded move, or a reset suspends tracking for the move's duration and
  marks the estimate stale; on resume the tracker re-reads the measured
  pose once instead of jumping back to a stale number.

Geometry: a pinhole model with the Reachy Mini Lite's focal length
expressed as a fraction of image width (vendor intrinsics: fx ≈ 2002 px
at 3840 px wide, i.e. ~0.52), so it holds at any capture resolution. The
lens is quite distorted at the edges; that only changes the effective
gain out there, which the controller tolerates.

Pure Python at import time (no OpenCV, no models): the controller is
unit-tested in the light test venv. ``TrackingLoop`` and the CLI pull in
the face modules lazily.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time

logger = logging.getLogger("tracking")

# Reachy Mini Lite camera, fx / image_width from the vendor's intrinsics.
LITE_FOCAL_PER_WIDTH = 0.52
# Comfort limits mirrored from reachy_target.LIMITS (not imported: that
# module pulls in scipy, which the voice venv may lack).
HEAD_YAW_LIMIT = 0.90
BODY_YAW_LIMIT = 2.79
PITCH_LIMIT = 0.35            # tracker's own cap, inside the 0.45 envelope
HANDOFF_YAW = math.radians(35)


def pixel_to_angles(u: float, v: float, width: float, height: float,
                    focal_per_width: float = LITE_FOCAL_PER_WIDTH
                    ) -> tuple[float, float]:
    """Angular offset of pixel (u, v) from the optical axis.

    Returns (yaw, pitch) in radians in the robot's conventions: yaw
    positive = the robot's left (a face on the image's left, u < cx);
    pitch positive = chin down (a face below centre, v > cy).
    """
    fx = focal_per_width * width
    cx, cy = width / 2.0, height / 2.0
    return math.atan2(cx - u, fx), math.atan2(v - cy, fx)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class FaceTracker:
    """Proportional head/body tracker over ``robot.posture``.

    ``robot`` needs ``posture(duration=..., **dofs)`` (fire-and-forget)
    and, for resync, an awaitable ``status()`` returning
    ``{"measured": {...}}``. Tests pass a fake with just ``posture``.
    """

    def __init__(self, robot, *, embodiment=None, gain: float = 0.5,
                 dead_band: float = math.radians(6.0),
                 min_interval: float = 0.8,
                 max_step: float = math.radians(8.0),
                 handoff: float = HANDOFF_YAW,
                 focal_per_width: float = LITE_FOCAL_PER_WIDTH,
                 relax_after: float = 8.0,
                 move_secs: float = 0.7,
                 clock=time.monotonic) -> None:
        # Defaults retuned after the 2026-09-03 family session (T14.2):
        # the first cut (gain 0.7, 3 deg dead-band, 0.4 s, 20 deg steps,
        # 0.35 s moves) plus the talking sway put ~135 posture commands a
        # minute on the robot and read as twitching. Half the gain, twice
        # the dead-band and interval, steps under 10 deg, moves that take
        # most of the interval: the head glides and settles.
        self.robot = robot
        self.embodiment = embodiment
        self.gain = gain
        self.dead_band = dead_band
        self.min_interval = min_interval
        self.max_step = max_step
        self.handoff = handoff
        self.focal_per_width = focal_per_width
        self.relax_after = relax_after
        self.move_secs = move_secs
        self._clock = clock
        self._minute_start = clock()
        self._minute_moves = 0
        self.enabled = True
        self._yaw = 0.0
        self._body = 0.0
        self._pitch = 0.0
        self._last_cmd = -math.inf
        self._last_seen: float | None = None
        self._suspend_until = -math.inf
        self._stale = False
        self.commands: list[dict] = []     # every posture sent, for tests

    # -- estimates and coordination ----------------------------------------

    @property
    def estimate(self) -> dict:
        return {"head_yaw": self._yaw, "body_yaw": self._body,
                "head_pitch": self._pitch}

    def set_estimate(self, **dofs: float) -> None:
        """Tell the tracker where a deliberate move put the robot."""
        if "head_yaw" in dofs:
            self._yaw = float(dofs["head_yaw"])
        if "body_yaw" in dofs:
            self._body = float(dofs["body_yaw"])
        if "head_pitch" in dofs:
            self._pitch = float(dofs["head_pitch"])
            if self.embodiment is not None:
                self.embodiment.pitch_bias = self._pitch

    def suspend(self, seconds: float, *, stale: bool = True) -> None:
        """Hold off for ``seconds`` (a tool move, a dance). With ``stale``
        the estimate is re-read from the robot before tracking resumes."""
        self._suspend_until = max(self._suspend_until,
                                  self._clock() + max(0.0, seconds))
        self._stale = self._stale or stale

    def reset(self) -> None:
        """The robot went home: estimates back to zero, nothing stale."""
        self._yaw = self._body = self._pitch = 0.0
        self._stale = False
        if self.embodiment is not None:
            self.embodiment.pitch_bias = 0.0

    @property
    def suspended(self) -> bool:
        return self._clock() < self._suspend_until

    async def maybe_resync(self) -> None:
        """After a suspension that moved the robot, read the measured pose
        once so the next nudge starts from where the head really is."""
        if not self._stale or self.suspended:
            return
        self._stale = False
        status = getattr(self.robot, "status", None)
        if status is None:
            return
        try:
            measured = (await status()).get("measured") or {}
        except Exception as exc:                                # noqa: BLE001
            logger.warning("tracker resync failed: %s", exc)
            return
        self._yaw = float(measured.get("head_yaw", self._yaw))
        self._body = float(measured.get("body_yaw", self._body))
        self._pitch = _clamp(float(measured.get("head_pitch", self._pitch)),
                             PITCH_LIMIT)
        if self.embodiment is not None:
            self.embodiment.pitch_bias = self._pitch
        logger.info("tracker: resynced (head %.2f, body %.2f, pitch %.2f)",
                    self._yaw, self._body, self._pitch)

    # -- the controller -------------------------------------------------------

    def observe(self, bbox, width: float, height: float,
                now: float | None = None) -> dict | None:
        """One detection: the largest face's (x1, y1, x2, y2) in a frame of
        ``width`` x ``height``. Returns the posture sent, or None."""
        now = self._clock() if now is None else now
        self._last_seen = now
        if not self.enabled or now < self._suspend_until or self._stale:
            return None
        if now - self._last_cmd < self.min_interval:
            return None
        x1, y1, x2, y2 = bbox
        yaw_off, pitch_off = pixel_to_angles((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                                             width, height,
                                             self.focal_per_width)
        move_yaw = abs(yaw_off) >= self.dead_band
        move_pitch = abs(pitch_off) >= self.dead_band
        if not (move_yaw or move_pitch):
            return None

        cmd: dict = {}
        duration = self.move_secs
        if move_yaw:
            step = _clamp(self.gain * yaw_off, self.max_step)
            target = self._yaw + step
            if abs(target) > self.handoff:
                # The neck has done its share: give the whole angle to
                # the body and recentre the head, slowly enough to read
                # as a turn rather than a flinch.
                self._body = _clamp(self._body + target, BODY_YAW_LIMIT)
                target = 0.0
                duration = 1.4
                cmd["body_yaw"] = self._body
                self.suspend(duration, stale=False)
            self._yaw = _clamp(target, HEAD_YAW_LIMIT)
            cmd["head_yaw"] = self._yaw
        if move_pitch:
            step = _clamp(self.gain * pitch_off, self.max_step)
            self._pitch = _clamp(self._pitch + step, PITCH_LIMIT)
            if self.embodiment is not None:
                self.embodiment.pitch_bias = self._pitch
            if self.embodiment is None or not self.embodiment.busy:
                cmd["head_pitch"] = self._pitch
        if not cmd:
            return None
        self.robot.posture(duration=duration, **cmd)
        self.commands.append(dict(cmd, duration=duration, t=now))
        self._last_cmd = now
        self._count_move(now)
        return cmd

    def _count_move(self, now: float) -> None:
        """One summary line a minute, so a booth log shows how busy the
        head was (the 2026-09-03 log had to be counted by hand)."""
        self._minute_moves += 1
        if now - self._minute_start >= 60.0:
            logger.info("tracker: %d moves in the last minute (head %.0f deg, "
                        "body %.0f deg)", self._minute_moves,
                        math.degrees(self._yaw), math.degrees(self._body))
            self._minute_start, self._minute_moves = now, 0

    def relax(self, now: float | None = None) -> dict | None:
        """No face for a while: drift back to centre, once."""
        now = self._clock() if now is None else now
        if (self._last_seen is None or now - self._last_seen < self.relax_after
                or self.suspended):
            return None
        if abs(self._yaw) < 1e-3 and abs(self._body) < 1e-3 \
                and abs(self._pitch) < 1e-3:
            return None
        self.reset()
        cmd = {"head_yaw": 0.0, "body_yaw": 0.0}
        if self.embodiment is None or not self.embodiment.busy:
            cmd["head_pitch"] = 0.0
        self.robot.posture(duration=1.2, **cmd)
        self.commands.append(dict(cmd, duration=1.2, t=now))
        self._last_cmd = now
        self._last_seen = None
        return cmd


class TrackingLoop:
    """Frames → detector → tracker, for runs without the session runner
    (the T13.3 one-shot cloud booth). With ``--session`` the runner feeds
    the tracker from its own loop instead; never run both."""

    def __init__(self, hub, tracker: FaceTracker, fps: float = 2.0) -> None:
        self.hub = hub
        self.tracker = tracker
        self.fps = fps

    async def run(self) -> None:
        from face import recognize
        frames = self.hub.frames()
        while True:
            frame = await asyncio.to_thread(next, frames, None)
            if frame is None:
                if getattr(self.hub, "exhausted", False):
                    logger.info("tracker: frame source ended")
                    return
                await asyncio.sleep(1.0 / self.fps)
                continue
            face = await asyncio.to_thread(recognize.detect, frame)
            await self.tracker.maybe_resync()
            if face is not None:
                h, w = frame.shape[:2]
                self.tracker.observe(face.bbox, w, h)
            else:
                self.tracker.relax()


# ---------------------------------------------------------------------------
# CLI: run the tracker over a source with a fake robot, print what it would
# command (JSON per line). Lets the light test venv exercise the whole
# detect → track path through voice/.venv.
# ---------------------------------------------------------------------------

class _PrintingRobot:
    def __init__(self):
        self.sent = []

    def posture(self, duration=0.5, **dofs):
        self.sent.append({"duration": duration, **dofs})


def main() -> int:
    import argparse
    import contextlib
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from face.camera import Camera
    from face import recognize
    from face_id import parse_source

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="dry-run the face tracker")
    ap.add_argument("--source", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=60)
    args = ap.parse_args()

    robot = _PrintingRobot()
    # A fake clock that advances one frame interval per observation makes
    # the run deterministic regardless of how fast detection is.
    t = [0.0]
    tracker = FaceTracker(robot, clock=lambda: t[0])
    detections = []
    with contextlib.redirect_stdout(sys.stderr):
        for i, frame in enumerate(Camera(parse_source(args.source),
                                         fps=args.fps).frames()):
            if i >= args.max_frames:
                break
            t[0] += 1.0 / args.fps
            face = recognize.detect(frame)
            h, w = frame.shape[:2]
            if face is None:
                detections.append({"t": t[0], "bbox": None, "cmd": None})
                tracker.relax()
                continue
            cmd = tracker.observe(face.bbox, w, h)
            detections.append({"t": t[0], "bbox": list(face.bbox),
                               "width": w, "height": h, "cmd": cmd})
    print(json.dumps({"frames": detections, "commands": tracker.commands},
                     indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
