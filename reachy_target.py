#!/usr/bin/env python3
"""Hardware target for the reachy_mini_dc rig -- the layer that touches the robot.

This module knows about Reachy Mini and nothing about Device Connect. It wraps
Pollen's ``reachy_mini.ReachyMini`` client, owns the daemon lifecycle, and
exposes the robot as nine independent scalar degrees of freedom, each one
individually addressable and clamped:

    head:     x, y, z (metres), roll, pitch, yaw (radians)
    body:     yaw (radians)
    antennas: left, right (radians)

The head is a 6-DOF Stewart platform, so its pose is a 4x4 homogeneous matrix on
the wire. Holding the six scalars here and recomposing the matrix on every write
is what lets the driver above expose bounded, individually addressable slots.

Topology on this machine (Reachy Mini Lite, USB-C tethered to the Mac):

    ReachyMiniDriver -> ReachyMini (WebSocket) -> reachy_mini daemon -> serial bus
                                                                        (/dev/cu.usbmodem*)

The daemon is a separate process that owns the serial bus. Only one daemon may
hold the bus at a time; ``spawn_daemon=True`` starts one if none is running.

Imports stdlib + numpy + scipy + reachy_mini only. No Device Connect.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def _ensure_daemon_on_path() -> None:
    """Put this interpreter's bin/ directory on PATH.

    ``spawn_daemon=True`` shells out to the ``reachy-mini-daemon`` console
    script by bare name, so it is only found if the venv's bin/ is on PATH.
    Running ``.venv/bin/python controller.py`` (rather than activating the venv
    first) leaves it off, and the failure surfaces far from its cause as
    ``FileNotFoundError: 'reachy-mini-daemon'``. Prepending it here makes the
    driver work the same whether or not the venv was activated.
    """
    bindir = os.path.dirname(os.path.abspath(sys.executable))
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bindir not in parts:
        os.environ["PATH"] = os.pathsep.join([bindir] + parts)


# ---------------------------------------------------------------------------
# Mechanical envelope
#
# Rotation/body limits are read off the shipped URDF
# (reachy_mini/descriptions/reachy_mini/urdf/robot.urdf):
#   yaw_body       lower=-2.79253 upper=2.79253
#   left_antenna   lower=-3.14159 upper=3.14159
#   right_antenna  lower=-3.14159 upper=3.14159
#
# The head is a parallel (Stewart) mechanism: its six leg joints have limits but
# the reachable *task-space* box is an implicit, pose-coupled subset of them --
# there is no single number in the URDF to quote. The head figures below were
# measured on this robot instead, by commanding each axis outward and reading
# back the pose the mechanism actually reached (2026-07-30):
#
#   z    tracks to -0.050 and beyond; saturates upward at about +0.020.
#        The asymmetry is real -- the head drops far but rises little -- and
#        the robot's own powered-off rest pose sits at z = -0.043, so a
#        symmetric box would have excluded the pose it sits in at rest.
#   x    saturates at about +0.020 measured; commanding +0.020 reaches
#        +0.013, so the limit cannot overdrive the mechanism.
#   pitch  reached +0.70 before saturating; kept at 0.45 with headroom.
#   yaw    still tracking at +1.45; kept at 0.90 with headroom.
#
# The IK refused nothing across that sweep, so these are comfort bounds and not
# a cliff edge. Rotations stay conservative deliberately; widen them here if a
# use case needs the measured headroom. The daemon's IK remains the backstop --
# these bounds exist so a bad write is clamped here rather than argued about
# down at the motors.
# ---------------------------------------------------------------------------

# name -> (lo, hi)
LIMITS: Dict[str, Tuple[float, float]] = {
    "head_x": (-0.02, 0.02),          # metres, +x forward
    "head_y": (-0.02, 0.02),          # metres, +y left
    "head_z": (-0.05, 0.02),          # metres, +z up; asymmetric, see above
    "head_roll": (-0.45, 0.45),       # radians, about +x
    "head_pitch": (-0.45, 0.45),      # radians, about +y
    "head_yaw": (-0.90, 0.90),        # radians, about +z
    "body_yaw": (-2.79, 2.79),        # radians, URDF yaw_body
    "antenna_left": (-3.14, 3.14),    # radians, URDF left_antenna
    "antenna_right": (-3.14, 3.14),   # radians, URDF right_antenna
}

# The neutral posture: every DOF at zero. Zero is defined to be a genuinely
# safe pose (head level and centred, body facing forward, antennas neutral)
# rather than an arbitrary origin, which is what makes "go to zero" the right
# thing to do on shutdown, on estop release, and on any loss of the caller.
DOF_NAMES: Tuple[str, ...] = tuple(LIMITS.keys())


# The vendor's antenna array is ordered [RIGHT, LEFT], not [left, right]:
# io/protocol.py::SetAntennasCmd documents "[right, left]", and
# assets/config/hardware_config.yaml lists right_antenna before left_antenna.
# Getting this backwards silently swaps the two antennas, which is invisible in
# any single-antenna test, so the ordering is isolated in these two helpers
# rather than spelled out at each call site.
ANTENNA_WIRE_ORDER = ("antenna_right", "antenna_left")

# Index 0 of the 7-long head-joint array is body rotation; 1..6 are the Stewart
# legs (daemon/backend/robot/backend.py builds it as
# ``[motor_pos.body_yaw] + motor_pos.stewart``).
BODY_YAW_JOINT_INDEX = 0


def _antennas_wire(cmd: Dict[str, float]) -> List[float]:
    """The antenna pair in the order the vendor SDK expects."""
    return [cmd[name] for name in ANTENNA_WIRE_ORDER]


def clamp(name: str, value: float) -> float:
    """Clamp *value* into the mechanical envelope of DOF *name*."""
    lo, hi = LIMITS[name]
    v = float(value)
    if math.isnan(v):
        return 0.0
    return max(lo, min(hi, v))


def pose_to_matrix(x: float, y: float, z: float,
                   roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Compose a 4x4 head pose from six scalars.

    Rotation is intrinsic xyz Euler (roll about +x, pitch about +y, yaw about
    +z), matching the frame ``look_at_world`` documents: x forward, y left,
    z up, origin at the neutral head position.
    """
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    m[:3, 3] = [x, y, z]
    return m


def matrix_to_pose(m: np.ndarray) -> Dict[str, float]:
    """Decompose a 4x4 head pose back into the six scalars.

    Inverse of :func:`pose_to_matrix`; used to publish measured head pose as
    SENSOR slots.
    """
    roll, pitch, yaw = Rotation.from_matrix(np.asarray(m)[:3, :3]).as_euler("xyz")
    tx, ty, tz = np.asarray(m)[:3, 3]
    return {
        "head_x": float(tx), "head_y": float(ty), "head_z": float(tz),
        "head_roll": float(roll), "head_pitch": float(pitch),
        "head_yaw": float(yaw),
    }


class ReachyMiniTarget:
    """A Reachy Mini behind nine clamped scalar DOFs.

    Commanded values are held here and pushed to the robot as a whole posture on
    every change, because ``set_target`` takes head/antennas/body_yaw together.
    Measured values come from a snapshot that :meth:`refresh` updates, so that
    many slot reads cost one round-trip rather than one each.
    """

    def __init__(self, *, spawn_daemon: bool = True, use_sim: bool = False,
                 host: str = "localhost", port: int = 8000,
                 connection_mode: str = "auto") -> None:
        self._spawn_daemon = spawn_daemon
        self._use_sim = use_sim
        self._host = host
        self._port = port
        self._connection_mode = connection_mode

        self._mini = None                       # reachy_mini.ReachyMini
        self._lock = threading.RLock()          # serialises pushes to the robot
        self._motors_on = False

        # Commanded posture (the CONTROL slot values).
        self._cmd: Dict[str, float] = {n: 0.0 for n in DOF_NAMES}
        # Measured posture (the SENSOR slot values), refreshed by refresh().
        self._measured: Dict[str, float] = {n: 0.0 for n in DOF_NAMES}
        self._last_error: str = ""

    # --- lifecycle ---------------------------------------------------------

    def connect(self) -> "ReachyMiniTarget":
        """Attach to the daemon (spawning one if asked) and enable the motors.

        Leaves the robot holding its present pose: ``enable_motors()`` pins every
        target to the measured pose before flipping torque on, and the commanded
        posture is seeded from that same measurement, so connecting never makes
        the robot jump.
        """
        from reachy_mini import ReachyMini

        if self._spawn_daemon:
            _ensure_daemon_on_path()

        mini = ReachyMini(
            host=self._host,
            port=self._port,
            connection_mode=self._connection_mode,
            spawn_daemon=self._spawn_daemon,
            use_sim=self._use_sim,
            media_backend="no_media",   # this driver controls motion only
        )
        self._mini = mini

        # Torque on first. The SDK pins all targets to the present pose inside
        # enable_motors(), so any set_target() must come *after* it or it is
        # silently overwritten.
        mini.enable_motors()
        self._motors_on = True

        self.refresh()
        with self._lock:
            self._cmd.update(self._measured)
        return self

    def close(self) -> None:
        """Return to neutral, drop torque, and release the daemon connection."""
        mini, self._mini = self._mini, None
        if mini is None:
            return
        try:
            self.goto_neutral(duration=1.0, _mini=mini)
        except Exception as exc:                        # noqa: BLE001
            logger.warning("neutral-on-close failed: %s", exc)
        try:
            mini.disable_motors()
            self._motors_on = False
        except Exception as exc:                        # noqa: BLE001
            logger.warning("disable_motors on close failed: %s", exc)
        # Tear the client down explicitly. ReachyMini starts non-daemon
        # background threads for the WebSocket link and the media manager; if
        # they are not stopped the interpreter will not exit, and a CLI that
        # has already printed its result appears to hang. __exit__ is the
        # vendor's own teardown (media close + client disconnect) -- the same
        # thing __del__ calls, which is documented there as existing "to avoid
        # a thread pending issue".
        try:
            mini.__exit__(None, None, None)
        except Exception as exc:                        # noqa: BLE001
            logger.warning("client teardown on close failed: %s", exc)

    @property
    def connected(self) -> bool:
        return self._mini is not None

    def _require(self):
        if self._mini is None:
            raise RuntimeError(
                "Reachy Mini is not connected: call connect() first "
                "(or the driver's connect() lifecycle hook).")
        return self._mini

    # --- commanded posture (CONTROL slots) ---------------------------------

    def get_cmd(self, name: str) -> float:
        """Read back the last commanded value of DOF *name*."""
        with self._lock:
            return self._cmd[name]

    def set_cmd(self, name: str, value: float) -> None:
        """Clamp, store, and push DOF *name*.

        Pushes the whole posture, because head pose, antennas and body yaw
        travel to the robot as one ``set_target`` call.
        """
        with self._lock:
            self._cmd[name] = clamp(name, value)
            self._push()

    def set_posture(self, **dofs: float) -> Dict[str, float]:
        """Set several DOFs and push once. Returns the clamped posture."""
        with self._lock:
            for name, value in dofs.items():
                if value is None:
                    continue
                if name not in LIMITS:
                    raise ValueError(
                        "unknown DOF %r; expected one of %s"
                        % (name, ", ".join(DOF_NAMES)))
                self._cmd[name] = clamp(name, value)
            self._push()
            return dict(self._cmd)

    def _push(self) -> None:
        """Send the held posture to the robot (caller holds the lock)."""
        mini = self._require()
        c = self._cmd
        head = pose_to_matrix(c["head_x"], c["head_y"], c["head_z"],
                              c["head_roll"], c["head_pitch"], c["head_yaw"])
        mini.set_target(
            head=head,
            antennas=_antennas_wire(c),
            body_yaw=c["body_yaw"],
        )

    # --- interpolated motion -----------------------------------------------

    def goto(self, duration: float = 0.5, **dofs: float) -> Dict[str, float]:
        """Move smoothly to a posture over *duration* seconds (min-jerk).

        Unspecified DOFs hold their current commanded value.
        """
        mini = self._require()
        with self._lock:
            for name, value in dofs.items():
                if value is None:
                    continue
                if name not in LIMITS:
                    raise ValueError(
                        "unknown DOF %r; expected one of %s"
                        % (name, ", ".join(DOF_NAMES)))
                self._cmd[name] = clamp(name, value)
            c = dict(self._cmd)

        # Interpolate only the groups the caller named (T15.9). The vendor's
        # goto re-plans every group it is given from the *present* pose, so
        # passing the held body_yaw with every head nudge re-drove the base
        # servo from wherever it sat to the held target about twice a
        # second -- the 2026-09-04 "base oscillating at ~2 Hz". None means
        # "keep the current value" on the vendor side.
        asked = set(dofs)
        head_asked = bool(asked & {"head_x", "head_y", "head_z", "head_roll",
                                   "head_pitch", "head_yaw"})
        antennas_asked = bool(asked & {"antenna_left", "antenna_right"})
        head = (pose_to_matrix(c["head_x"], c["head_y"], c["head_z"],
                               c["head_roll"], c["head_pitch"], c["head_yaw"])
                if head_asked else None)
        mini.goto_target(
            head=head,
            antennas=_antennas_wire(c) if antennas_asked else None,
            body_yaw=c["body_yaw"] if "body_yaw" in asked else None,
            duration=max(0.01, float(duration)),
        )
        return c

    def goto_neutral(self, duration: float = 1.0, _mini=None) -> Dict[str, float]:
        """Move every DOF to zero: head level and centred, body forward."""
        if _mini is not None and self._mini is None:
            # close() path: the handle is already detached from self.
            with self._lock:
                self._cmd = {n: 0.0 for n in DOF_NAMES}
                c = dict(self._cmd)
            _mini.goto_target(
                head=pose_to_matrix(0, 0, 0, 0, 0, 0),
                antennas=[0.0, 0.0], body_yaw=0.0,
                duration=max(0.01, float(duration)),
            )
            return c
        return self.goto(duration=duration, **{n: 0.0 for n in DOF_NAMES})

    def look_at(self, x: float, y: float, z: float,
                duration: float = 1.0) -> Dict[str, float]:
        """Point the head at a world point (metres, x forward / y left / z up).

        Delegates to the daemon's IK, then syncs the held posture from the pose
        it produced so subsequent slot writes start from where the robot is.
        """
        mini = self._require()
        pose = mini.look_at_world(float(x), float(y), float(z),
                                  duration=max(0.01, float(duration)),
                                  perform_movement=True)
        with self._lock:
            for name, value in matrix_to_pose(pose).items():
                self._cmd[name] = clamp(name, value)
            return dict(self._cmd)

    # --- measured posture (SENSOR slots) -----------------------------------

    def refresh(self) -> Dict[str, float]:
        """Refresh the measured snapshot with one round-trip per source.

        Failures are recorded in ``last_error`` and leave the previous snapshot
        in place: a flaky read must not crash the polling task that calls this.
        """
        mini = self._require()
        snap: Dict[str, float] = {}
        try:
            snap.update(matrix_to_pose(mini.get_current_head_pose()))
            head_joints, antennas = mini.get_current_joint_positions()
            if len(antennas) >= 2:
                for i, name in enumerate(ANTENNA_WIRE_ORDER):
                    snap[name] = float(antennas[i])
            # head_joints is length 7: body rotation first, then the six
            # Stewart legs.
            if len(head_joints) > BODY_YAW_JOINT_INDEX:
                snap["body_yaw"] = float(head_joints[BODY_YAW_JOINT_INDEX])
            self._last_error = ""
        except Exception as exc:                        # noqa: BLE001
            self._last_error = "%s: %s" % (type(exc).__name__, exc)
            logger.warning("refresh failed: %s", self._last_error)
            with self._lock:
                return dict(self._measured)

        with self._lock:
            self._measured.update(snap)
            return dict(self._measured)

    def get_measured(self, name: str) -> float:
        """Read DOF *name* from the last snapshot (no round-trip)."""
        with self._lock:
            return self._measured.get(name, 0.0)

    @property
    def last_error(self) -> str:
        return self._last_error

    # --- motors and canned behaviours --------------------------------------

    @property
    def motors_enabled(self) -> bool:
        return self._motors_on

    def enable_motors(self, ids: Optional[List[str]] = None) -> None:
        self._require().enable_motors(ids)
        self._motors_on = True
        # Targets were just pinned to the present pose; resync so the held
        # posture matches what the robot is actually holding.
        self.refresh()
        with self._lock:
            self._cmd.update(self._measured)

    def disable_motors(self, ids: Optional[List[str]] = None) -> None:
        self._require().disable_motors(ids)
        self._motors_on = False

    def wake_up(self) -> None:
        """Play Pollen's wake-up emote, then resync the held posture."""
        self._require().wake_up()
        self.refresh()
        with self._lock:
            self._cmd.update(self._measured)

    def goto_sleep(self) -> None:
        """Play Pollen's sleep posture, then resync the held posture."""
        self._require().goto_sleep()
        self.refresh()
        with self._lock:
            self._cmd.update(self._measured)

    # --- recorded moves (T13.4) ---------------------------------------------

    def play_move(self, name: str) -> Dict[str, float]:
        """Play one curated recorded move (moves.LIBRARY) to the end, blocking.

        The vendor's ``play_move`` streams targets straight to the daemon,
        bypassing the held posture, so the posture is resynced from the
        measured pose afterwards. Sound is off: the voice agent owns the
        speaker while a conversation runs. ``cancel_move`` (from another
        thread) stops playback at the next tick.
        """
        from moves import LIBRARY
        spec = LIBRARY.get(name)
        if spec is None or spec.dataset is None:
            raise ValueError("no recorded move %r; recorded ones are %s"
                             % (name, ", ".join(
                                 m.name for m in LIBRARY.values()
                                 if m.dataset)))
        mini = self._require()
        move = self._recorded_move(spec.dataset, spec.move)
        with self._lock:
            body_before = self._cmd["body_yaw"]
        try:
            # 1.0 s to the move's first frame (was 0.4): a dance called
            # with the head turned 30 deg by the tracker used to snap.
            mini.play_move(move, sound=False, initial_goto_duration=1.0)
        finally:
            self.refresh()
            with self._lock:
                self._cmd.update(self._measured)
            # T15.11: settle the base. A recorded move streams its last
            # frame and stops; measured on 2026-09-04 (20:40), the base
            # servo was then left limit-cycling +-0.5 deg at 3.5 Hz around
            # a target 1 deg away that it could not reach, for minutes,
            # with the head at rest -- and one ordinary goto to a clean
            # body_yaw ended it at once. So every move ends with a short
            # interpolated goto back to the body yaw held before it.
            try:
                mini.goto_target(body_yaw=float(body_before), duration=0.6)
                with self._lock:
                    self._cmd["body_yaw"] = float(body_before)
            except Exception as exc:                    # noqa: BLE001
                logger.warning("post-move body settle failed: %s", exc)
        with self._lock:
            return dict(self._cmd)

    def cancel_move(self) -> None:
        mini = self._mini
        if mini is not None:
            try:
                mini.cancel_move()
            except Exception as exc:                        # noqa: BLE001
                logger.warning("cancel_move failed: %s", exc)

    _move_libraries: Dict[str, object] = {}

    @classmethod
    def _recorded_move(cls, dataset: str, move: str):
        """A RecordedMove from the (cached) dataset; libraries are loaded
        once per process. Raises if the dataset is not on disk and cannot
        be fetched -- moves.py --preload is the booth's preflight."""
        from reachy_mini.motion.recorded_move import RecordedMoves
        lib = cls._move_libraries.get(dataset)
        if lib is None:
            lib = RecordedMoves(dataset)
            cls._move_libraries[dataset] = lib
        return lib.get(move)

    # --- media ownership ---------------------------------------------------
    #
    # The daemon grabs the robot's camera, mic and speaker at startup. A voice
    # pipeline that wants to open the mic and speaker directly (sounddevice,
    # PyAudio, OpenCV) has to ask the daemon to let go first. Exposing this on
    # the target means the release can be driven over Device Connect, so a
    # separate voice process never needs the vendor SDK.

    def release_media(self) -> None:
        """Hand the camera, mic and speaker to whoever wants them directly."""
        self._require().release_media()

    def acquire_media(self) -> None:
        """Take the camera, mic and speaker back."""
        self._require().acquire_media()

    @property
    def media_released(self) -> bool:
        mini = self._mini
        return bool(getattr(mini, "media_released", False)) if mini else False

    def status(self) -> Dict[str, object]:
        """A single dict describing the whole target."""
        with self._lock:
            return {
                "connected": self.connected,
                "motors_enabled": self._motors_on,
                "media_released": self.media_released,
                "commanded": dict(self._cmd),
                "measured": dict(self._measured),
                "limits": {k: list(v) for k, v in LIMITS.items()},
                "last_error": self._last_error,
            }


def build_target(**kwargs) -> ReachyMiniTarget:
    """Return a connected, READY target. Raises if the robot is absent."""
    return ReachyMiniTarget(**kwargs).connect()
