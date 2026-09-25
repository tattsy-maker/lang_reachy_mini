#!/usr/bin/env python3
"""StubTarget -- an in-memory Reachy Mini for testing the whole path without hardware.

Same surface as :class:`reachy_target.ReachyMiniTarget`, with the robot replaced
by a dict. It inherits the real clamping and pose maths, so bounds behave
exactly as they do on hardware; only the serial link is missing. Moves complete
instantly rather than sleeping, so a stubbed `demo` finishes in milliseconds.

Used by ``controller.py --stub``. Nothing here talks to a daemon, so it is safe
to run alongside a real ``serve``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from reachy_target import DOF_NAMES, ReachyMiniTarget


class StubTarget(ReachyMiniTarget):
    """A Reachy Mini that exists only in memory."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pushes = 0          # how many postures were "sent" to the robot

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> "StubTarget":
        self._mini = "stub"
        self._motors_on = True
        return self

    def close(self) -> None:
        with self._lock:
            self._cmd = {n: 0.0 for n in DOF_NAMES}
        self._mini = None
        self._motors_on = False

    # -- wire (there is none) -----------------------------------------------

    def _push(self) -> None:
        self._require()
        self.pushes += 1

    def refresh(self) -> Dict[str, float]:
        """The stub robot always reaches exactly what it was told."""
        self._require()
        with self._lock:
            self._measured.update(self._cmd)
            self._last_error = ""
            return dict(self._measured)

    # -- motion -------------------------------------------------------------

    def goto(self, duration: float = 0.5, **dofs: float) -> Dict[str, float]:
        return self.set_posture(**dofs)

    def goto_neutral(self, duration: float = 1.0, _mini=None
                     ) -> Dict[str, float]:
        return self.set_posture(**{n: 0.0 for n in DOF_NAMES})

    def look_at(self, x: float, y: float, z: float,
                duration: float = 1.0) -> Dict[str, float]:
        """A crude stand-in for the daemon's IK: aim yaw/pitch at the point."""
        import math
        yaw = math.atan2(float(y), max(1e-6, float(x)))
        pitch = -math.atan2(float(z), max(1e-6, float(x)))
        return self.set_posture(head_yaw=yaw, head_pitch=pitch)

    # -- motors and emotes --------------------------------------------------

    def enable_motors(self, ids: Optional[List[str]] = None) -> None:
        self._require()
        self._motors_on = True

    def disable_motors(self, ids: Optional[List[str]] = None) -> None:
        self._require()
        self._motors_on = False

    def wake_up(self) -> None:
        self.set_posture(**{n: 0.0 for n in DOF_NAMES})

    def goto_sleep(self) -> None:
        self.set_posture(head_pitch=0.4, head_z=-0.015)

    # -- recorded moves (T13.4): a timed no-op ------------------------------
    #
    # The stub has no daemon to stream a trajectory to, and tests must not
    # need the HuggingFace cache. It validates the name against the same
    # library, "plays" for a token 0.2 s so cancellation paths are real,
    # and records what it was asked to play.

    def play_move(self, name: str) -> Dict[str, float]:
        import time
        from moves import LIBRARY
        spec = LIBRARY.get(name)
        if spec is None or spec.dataset is None:
            raise ValueError("no recorded move %r; recorded ones are %s"
                             % (name, ", ".join(
                                 m.name for m in LIBRARY.values()
                                 if m.dataset)))
        self._require()
        self._move_cancelled = False
        self.moves_played = getattr(self, "moves_played", []) + [name]
        for _ in range(4):
            if self._move_cancelled:
                break
            time.sleep(0.05)
        return self.set_posture(head_yaw=0.0)

    def cancel_move(self) -> None:
        self._move_cancelled = True

    # -- media ownership ----------------------------------------------------
    #
    # The base class delegates these to the vendor SDK object in self._mini,
    # but the stub's _mini is just the sentinel "stub" -- without these
    # overrides, set_media_released over the wire dies with
    # "'str' object has no attribute 'release_media'" (found the first time
    # the voice agent was pointed at a served stub).

    def release_media(self) -> None:
        self._require()
        self._media_released = True

    def acquire_media(self) -> None:
        self._require()
        self._media_released = False

    @property
    def media_released(self) -> bool:
        return getattr(self, "_media_released", False)
