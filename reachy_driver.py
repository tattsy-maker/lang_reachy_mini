#!/usr/bin/env python3
"""ReachyMiniDriver -- a Reachy Mini Lite as an Arm Device Connect device.

The robot's nine degrees of freedom are exposed as Device Connect RPCs, its
state changes as events, and its telemetry as a periodic snapshot:

    reads     get_posture, get_limits, report_status, get_motion
    writes    set_dof, goto_posture, home, look_at, nod, shake, wake_up, sleep
    control   set_motors, stop, clear_estop, cancel_motion, set_media_released
    events    motion_started, motion_progress, motion_completed,
              motors_changed, estop_changed, link_health_changed, telemetry

    head:     x, y, z (metres), roll, pitch, yaw (radians)
    body:     yaw (radians)
    antennas: left, right (radians)

Bounds live in ``reachy_target.LIMITS`` and every write is clamped into them.
They are comfort bounds, not a cliff edge: every pose inside the mechanical
envelope is a legal pose, so an out-of-envelope write is *clamped into* the
envelope rather than refused. ``get_limits()`` publishes them so a caller can
see what it is working with, and ``set_dof`` reports what it actually applied.

Motion never blocks the command channel
---------------------------------------
Every move returns a ``motion_id`` immediately and runs in a background task.
Progress arrives as ``motion_progress`` events and the outcome as
``motion_completed``. This is not a stylistic choice: Device Connect dispatches
a device's incoming RPCs in order on one subscription, so a move that blocked
its handler for two seconds would also block the ``stop()`` arriving behind it.
An emergency stop that cannot be heard during a move is not an emergency stop.

A caller that wants to wait subscribes to ``motion_completed`` (see
controller.py) or polls ``get_motion()``. A caller that does not want to wait --
the voice agent -- simply ignores the id, which is why a spoken reply never
waits on a 1.5-second head turn.

Only one motion runs at a time. Starting a new one supersedes the running one,
which is what makes rapid conversational gestures compose rather than queue up
behind each other.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from device_connect_edge.drivers import DeviceDriver, emit, periodic, rpc
from device_connect_edge.types import DeviceIdentity, DeviceStatus

from reachy_target import DOF_NAMES, LIMITS, ReachyMiniTarget

logger = logging.getLogger(__name__)

# Procedures that start a background motion. Clients (controller.py,
# voice/robot_link.py) use this to decide whether to wait on a
# motion_completed event or just take the RPC result at face value.
MOTION_PROCEDURES = ("goto_posture", "home", "look_at", "nod", "shake",
                     "wake_up", "sleep")


class EstopEngaged(RuntimeError):
    """Raised when a motion is requested while the estop latch is set."""


class MotionCancelled(RuntimeError):
    """Raised inside a motion task when it is superseded or cancelled."""


class ReachyMiniDriver(DeviceDriver):
    """Pollen Reachy Mini Lite, 9 DOF, over USB."""

    device_type = "reachy_mini"

    def __init__(self, target: Optional[ReachyMiniTarget] = None, *,
                 spawn_daemon: bool = True, use_sim: bool = False,
                 poll_hz: float = 10.0, telemetry_hz: float = 1.0) -> None:
        super().__init__()
        self._target = target or ReachyMiniTarget(
            spawn_daemon=spawn_daemon, use_sim=use_sim)
        self._poll_interval = 1.0 / max(0.1, float(poll_hz))
        self._telemetry_interval = (1.0 / telemetry_hz) if telemetry_hz > 0 else 0.0
        self._estopped = False
        self._last_ok = True

        # The single in-flight motion, if any.
        self._motion: Optional[Dict[str, Any]] = None
        self._motion_task: Optional[asyncio.Task] = None
        # Serialises the actual vendor calls, so a superseding motion never
        # has two goto_target()s in flight against the same serial bus.
        self._motion_lock = asyncio.Lock()

    # -- identity -----------------------------------------------------------

    @property
    def identity(self) -> DeviceIdentity:
        return DeviceIdentity(
            device_type=self.device_type,
            manufacturer="Pollen Robotics",
            model="Reachy Mini Lite",
            description="Desk robot: 6-DOF Stewart-platform head, rotating "
                        "body, two antennas. Nine clamped scalar DOFs.",
        )

    @property
    def status(self) -> DeviceStatus:
        return DeviceStatus()

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """Attach to the robot. Runs off the event loop: the SDK is blocking."""
        await asyncio.to_thread(self._target.connect)
        logger.info("reachy mini connected; motors enabled")

    async def disconnect(self) -> None:
        """Cancel any motion, return to neutral, and drop torque."""
        await self._abort_motion("shutting down")
        await asyncio.to_thread(self._target.close)
        logger.info("reachy mini disconnected")

    # -- hosted (in-process) use -------------------------------------------
    #
    # controller.py's one-shot commands own the robot directly, with no
    # messaging in the picture at all. These two wrap the runtime's own
    # routine lifecycle so a hosted command still gets fresh telemetry.

    async def start_local(self) -> None:
        """connect() plus the periodic routines DeviceRuntime would start."""
        await self.connect()
        await self._start_routines()

    async def stop_local(self) -> None:
        """The inverse of start_local()."""
        await self._stop_routines()
        await self.disconnect()

    # -- guard --------------------------------------------------------------

    def _guard(self) -> None:
        if self._estopped:
            raise EstopEngaged(
                "estop is engaged; motion is refused until clear_estop() is "
                "called")

    # =======================================================================
    # Events
    # =======================================================================

    @emit(labels={"safety": "informational"})
    async def motion_started(self, motion_id: str, kind: str,
                             detail: Dict[str, Any]) -> None:
        """A background motion was accepted and has begun.

        Args:
            motion_id: Identifies this motion across all three motion events.
            kind: The procedure that started it ('nod', 'home', ...).
            detail: The arguments it was started with.
        """

    @emit(labels={"safety": "informational"})
    async def motion_progress(self, motion_id: str, kind: str,
                              progress: Dict[str, Any]) -> None:
        """A running motion reached a checkpoint.

        Args:
            motion_id: The motion this update belongs to.
            kind: The procedure that is running.
            progress: Free-form phase detail, e.g. {'phase': 'nodding',
                'beat': 1, 'of': 2}.
        """

    @emit(labels={"safety": "informational"})
    async def motion_completed(self, motion_id: str, kind: str, status: str,
                               posture: Optional[Dict[str, float]] = None,
                               error: Optional[str] = None) -> None:
        """A motion reached a terminal state.

        Args:
            motion_id: The motion that finished.
            kind: The procedure that ran.
            status: 'succeeded', 'cancelled' or 'failed'.
            posture: The commanded posture it ended on, when it succeeded.
            error: The failure message, when it failed.
        """

    @emit(labels={"safety": "critical"})
    async def motors_changed(self, enabled: bool) -> None:
        """Motor torque was enabled or disabled.

        Args:
            enabled: True if torque is now on.
        """

    @emit(labels={"safety": "critical"})
    async def estop_changed(self, engaged: bool) -> None:
        """The emergency stop latch changed state.

        Args:
            engaged: True if motion is now refused.
        """

    @emit(labels={"safety": "critical"})
    async def link_health_changed(self, healthy: bool, detail: str) -> None:
        """Telemetry polling started or stopped succeeding.

        Args:
            healthy: True if the last poll read the robot successfully.
            detail: The error text when unhealthy, 'ok' otherwise.
        """

    @emit(labels={"safety": "informational", "direction": "read"})
    async def telemetry(self, commanded: Dict[str, float],
                        measured: Dict[str, float], motors_enabled: bool,
                        estopped: bool) -> None:
        """Periodic snapshot of the whole robot.

        Args:
            commanded: The posture the driver is asking for, per DOF.
            measured: The posture the motors report, per DOF.
            motors_enabled: True if torque is on.
            estopped: True if the estop latch is set.
        """

    # =======================================================================
    # Periodic routines
    #
    # Two of them, deliberately. poll_state refreshes the local snapshot fast
    # (10 Hz) so get_posture() and report_status() answer without a round trip
    # to the robot; telemetry publishes it slowly (1 Hz) so a fleet of these
    # does not drown the broker in pose updates nobody asked for.
    # =======================================================================

    @periodic(interval=0.1, start_on_connect=True)
    async def poll_state(self) -> None:
        """Refresh the measured snapshot that every read answers from."""
        await asyncio.to_thread(self._target.refresh)
        ok = not self._target.last_error
        if ok != self._last_ok:
            self._last_ok = ok
            await self.link_health_changed(
                healthy=ok, detail=self._target.last_error or "ok")

    @periodic(interval=1.0, start_on_connect=True)
    async def publish_telemetry(self) -> None:
        """Publish the snapshot poll_state keeps fresh."""
        if self._telemetry_interval <= 0:
            return
        st = self._target.status()
        await self.telemetry(
            commanded=st["commanded"], measured=st["measured"],
            motors_enabled=bool(st["motors_enabled"]), estopped=self._estopped)

    # =======================================================================
    # Reads
    # =======================================================================

    @rpc(labels={"direction": "read", "safety": "informational"})
    async def get_posture(self) -> Dict[str, Any]:
        """Read the commanded and measured posture of all nine DOFs.

        Both dicts are keyed by DOF name. 'commanded' is what the driver is
        asking for; 'measured' is what the motors report, refreshed at 10 Hz.
        """
        st = self._target.status()
        return {"commanded": st["commanded"], "measured": st["measured"],
                "last_error": st["last_error"]}

    @rpc(labels={"direction": "read", "safety": "informational"})
    async def get_limits(self) -> Dict[str, Any]:
        """Read the mechanical envelope: the [lo, hi] bounds of each DOF.

        Writes outside these are clamped into them rather than refused.
        """
        return {"limits": {k: list(v) for k, v in LIMITS.items()},
                "dofs": list(DOF_NAMES)}

    @rpc(labels={"direction": "read", "safety": "informational"})
    async def report_status(self) -> Dict[str, Any]:
        """Full driver and robot status.

        Commanded posture, measured posture, limits, motor state, media
        ownership, estop state, and the motion currently running if any.
        """
        st = dict(self._target.status())
        st["estopped"] = self._estopped
        st["device_type"] = self.device_type
        st["motion"] = self._motion_view()
        return st

    @rpc(labels={"direction": "read", "safety": "informational"})
    async def get_motion(self, motion_id: Optional[str] = None
                         ) -> Dict[str, Any]:
        """Read the state of a motion without blocking.

        Args:
            motion_id: The motion to report on. Omit for the most recent one.
        """
        view = self._motion_view()
        if view is None:
            return {"motion": None}
        if motion_id and view.get("motion_id") != motion_id:
            return {"motion": None, "reason": "unknown motion_id"}
        return {"motion": view}

    def _motion_view(self) -> Optional[Dict[str, Any]]:
        """The public, JSON-safe shape of the current motion record."""
        m = self._motion
        if m is None:
            return None
        return {k: m[k] for k in
                ("motion_id", "kind", "status", "detail", "posture", "error")}

    # =======================================================================
    # Direct writes
    # =======================================================================

    @rpc(labels={"direction": "write", "safety": "critical"})
    async def set_dof(self, name: str, value: float) -> Dict[str, Any]:
        """Set one degree of freedom immediately, with no interpolation.

        The value is clamped into the DOF's mechanical envelope, and the
        clamped value is what comes back. For a smooth move use goto_posture.

        Args:
            name: One of head_x, head_y, head_z, head_roll, head_pitch,
                head_yaw, body_yaw, antenna_left, antenna_right.
            value: Target value, in metres for head_x/y/z and radians for
                every rotation.
        """
        self._guard()
        if name not in LIMITS:
            raise ValueError("unknown DOF %r; expected one of %s"
                             % (name, ", ".join(DOF_NAMES)))
        await asyncio.to_thread(self._target.set_cmd, name, float(value))
        applied = self._target.get_cmd(name)
        return {"name": name, "requested": float(value), "applied": applied,
                "clamped": applied != float(value)}

    @rpc(labels={"direction": "write", "safety": "critical"})
    async def set_motors(self, enabled: bool,
                         ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enable or disable motor torque. Disabling makes the robot go limp.

        Args:
            enabled: True to energise, False to release.
            ids: Optional subset of motor ids; omit for all of them.
        """
        if enabled:
            self._guard()
            await asyncio.to_thread(self._target.enable_motors, ids)
        else:
            await asyncio.to_thread(self._target.disable_motors, ids)
        await self.motors_changed(enabled=bool(enabled))
        return {"motors_enabled": self._target.motors_enabled}

    @rpc(labels={"direction": "write", "safety": "informational"})
    async def set_media_released(self, released: bool) -> Dict[str, Any]:
        """Hand the robot's camera, mic and speaker to another process.

        The vendor daemon grabs all three at startup. A voice pipeline that
        wants to open the mic and speaker directly has to ask it to let go
        first, which is what this does.

        Args:
            released: True to hand them over, False to take them back.
        """
        if released:
            await asyncio.to_thread(self._target.release_media)
        else:
            await asyncio.to_thread(self._target.acquire_media)
        return {"media_released": self._target.media_released}

    # =======================================================================
    # Emergency stop
    #
    # This is the reason motion runs in the background. stop() has to be
    # answerable *while* the robot is moving, which means no motion handler
    # may sit on the command channel waiting for the motors.
    # =======================================================================

    @rpc(labels={"direction": "write", "safety": "critical"})
    async def stop(self) -> Dict[str, Any]:
        """EMERGENCY STOP. Freeze at the present pose, refuse further motion.

        Freezes rather than going limp: cutting torque on a Stewart-platform
        head would let it droop under its own weight, so the commanded posture
        is pinned to the measured one and latched. Motion stays refused until
        clear_estop().
        """
        self._estopped = True
        await self._abort_motion("estop engaged")
        measured = await asyncio.to_thread(self._target.refresh)
        try:
            await asyncio.to_thread(self._target.set_posture, **measured)
        except Exception as exc:                        # noqa: BLE001
            logger.error("estop freeze failed: %s", exc)
        # Cancelling the motion task unblocks *us*, but the vendor call it was
        # waiting on runs on a worker thread that cannot be interrupted, and
        # the daemon behind it keeps interpolating toward the target it was
        # already given. So one freeze is a request, not a guarantee. Re-assert
        # the same pose for a second to outlive any in-flight interpolation --
        # an estop should win the argument, not merely enter it.
        asyncio.ensure_future(self._hold_frozen(measured))
        await self.estop_changed(engaged=True)
        return {"estopped": True, "frozen_at": measured}

    async def _hold_frozen(self, pose: Dict[str, float],
                           seconds: float = 1.0, hz: float = 10.0) -> None:
        """Keep re-commanding the freeze pose, and stop as soon as it is cleared."""
        deadline = asyncio.get_running_loop().time() + seconds
        while self._estopped and asyncio.get_running_loop().time() < deadline:
            try:
                await asyncio.to_thread(self._target.set_posture, **pose)
            except Exception as exc:                    # noqa: BLE001
                logger.warning("estop hold failed: %s", exc)
                return
            await asyncio.sleep(1.0 / hz)

    @rpc(labels={"direction": "write", "safety": "critical"})
    async def clear_estop(self) -> Dict[str, Any]:
        """Release the emergency stop latch set by stop()."""
        self._estopped = False
        await self.estop_changed(engaged=False)
        return {"estopped": False}

    @rpc(labels={"direction": "write", "safety": "critical"})
    async def cancel_motion(self, motion_id: Optional[str] = None
                            ) -> Dict[str, Any]:
        """Cancel the running motion. Leaves the robot where it got to.

        Cancellation is cooperative and lands at the next checkpoint, because
        the vendor SDK call underneath is blocking and cannot be interrupted
        mid-flight. For a multi-beat gesture that is the next beat; for a
        single interpolated move it is when that move returns.

        Args:
            motion_id: Cancel only if this is the motion that is running.
                Omit to cancel whatever is running.
        """
        m = self._motion
        if m is None or m["status"] != "running":
            return {"cancelled": False, "reason": "nothing running"}
        if motion_id and m["motion_id"] != motion_id:
            return {"cancelled": False, "reason": "a different motion is running",
                    "running": m["motion_id"]}
        cancelled = m["motion_id"]
        await self._abort_motion("cancelled by request")
        return {"cancelled": True, "motion_id": cancelled}

    # =======================================================================
    # Motion plumbing
    # =======================================================================

    async def _abort_motion(self, reason: str) -> None:
        """Ask the running motion to stop and wait for it to unwind."""
        m, task = self._motion, self._motion_task
        if m is not None and m["status"] == "running":
            m["cancel_requested"] = reason
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):     # noqa: BLE001
                pass

    async def _begin(self, kind: str, detail: Dict[str, Any], runner
                     ) -> Dict[str, Any]:
        """Accept a motion, start it in the background, and return its id.

        Supersedes whatever was running: a new gesture beats a stale one, which
        is what makes conversational motion compose instead of queueing.
        """
        self._guard()
        await self._abort_motion("superseded by %s" % kind)

        motion_id = "m-%s" % uuid.uuid4().hex[:10]
        self._motion = {
            "motion_id": motion_id, "kind": kind, "status": "running",
            "detail": detail, "posture": None, "error": None,
            "cancel_requested": None,
        }
        self._motion_task = asyncio.create_task(
            self._run_motion(motion_id, kind, runner))
        await self.motion_started(motion_id=motion_id, kind=kind, detail=detail)
        return {"accepted": True, "motion_id": motion_id, "kind": kind,
                "detail": detail}

    async def _run_motion(self, motion_id: str, kind: str, runner) -> None:
        """Drive one motion to a terminal state and announce the outcome."""
        m = self._motion
        try:
            async with self._motion_lock:
                posture = await runner(motion_id)
            m["status"], m["posture"] = "succeeded", posture
            await self.motion_completed(motion_id=motion_id, kind=kind,
                                        status="succeeded", posture=posture)
        except (asyncio.CancelledError, MotionCancelled) as exc:
            m["status"] = "cancelled"
            m["error"] = m["cancel_requested"] or str(exc) or "cancelled"
            # The emit has to outlive this task: awaiting inside a cancelled
            # task raises again immediately, so hand it to the loop instead.
            asyncio.ensure_future(self.motion_completed(
                motion_id=motion_id, kind=kind, status="cancelled",
                posture=self._target.status()["commanded"], error=m["error"]))
            if isinstance(exc, asyncio.CancelledError):
                raise
        except Exception as exc:                            # noqa: BLE001
            m["status"], m["error"] = "failed", "%s: %s" % (type(exc).__name__, exc)
            logger.warning("motion %s (%s) failed: %s", motion_id, kind, exc)
            await self.motion_completed(motion_id=motion_id, kind=kind,
                                        status="failed", error=m["error"])

    def _checkpoint(self, motion_id: str) -> None:
        """Raise if this motion has been asked to stop. Call between beats."""
        m = self._motion
        if m is None or m["motion_id"] != motion_id or m["cancel_requested"]:
            raise MotionCancelled(
                (m or {}).get("cancel_requested") or "superseded")

    async def _progress(self, motion_id: str, kind: str, **progress) -> None:
        await self.motion_progress(motion_id=motion_id, kind=kind,
                                   progress=progress)

    # =======================================================================
    # Motions
    #
    # Each returns immediately with a motion_id; the work happens in the
    # nested runner. The nine DOFs are spelled out rather than collected with
    # **kwargs so the capability manifest advertises each one by name, with
    # its own type and default. An agent reading the manifest can then see
    # what it may pass; a **kwargs signature would teach it nothing.
    # =======================================================================

    @rpc(labels={"direction": "write", "safety": "critical", "motion": "true"})
    async def goto_posture(self, duration: float = 0.5,
                           head_x: Optional[float] = None,
                           head_y: Optional[float] = None,
                           head_z: Optional[float] = None,
                           head_roll: Optional[float] = None,
                           head_pitch: Optional[float] = None,
                           head_yaw: Optional[float] = None,
                           body_yaw: Optional[float] = None,
                           antenna_left: Optional[float] = None,
                           antenna_right: Optional[float] = None,
                           ) -> Dict[str, Any]:
        """Move smoothly to a posture. Returns a motion_id, does not wait.

        Any DOF left out holds its current commanded value, which is what lets
        two callers drive different DOFs without overwriting each other.

        Args:
            duration: Seconds the interpolated move should take.
            head_x: Head translation forward(+)/back(-), metres.
            head_y: Head translation left(+)/right(-), metres.
            head_z: Head translation up(+)/down(-), metres.
            head_roll: Head roll, ear to shoulder, radians.
            head_pitch: Head pitch, nod up(-)/down(+), radians.
            head_yaw: Head yaw, turn left(+)/right(-), radians.
            body_yaw: Whole-body rotation on the base, radians.
            antenna_left: Left antenna angle, radians.
            antenna_right: Right antenna angle, radians.
        """
        dofs = {name: value for name, value in (
            ("head_x", head_x), ("head_y", head_y), ("head_z", head_z),
            ("head_roll", head_roll), ("head_pitch", head_pitch),
            ("head_yaw", head_yaw), ("body_yaw", body_yaw),
            ("antenna_left", antenna_left), ("antenna_right", antenna_right),
        ) if value is not None}
        if not dofs:
            raise ValueError(
                "goto_posture needs at least one DOF; expected any of %s"
                % ", ".join(DOF_NAMES))

        async def run(motion_id):
            await self._progress(motion_id, "goto_posture", phase="moving",
                                 duration=float(duration), dofs=sorted(dofs))
            return await asyncio.to_thread(
                self._target.goto, float(duration), **dofs)

        return await self._begin("goto_posture",
                                 {"duration": float(duration), **dofs}, run)

    @rpc(labels={"direction": "write", "safety": "critical", "motion": "true"})
    async def home(self, duration: float = 1.0) -> Dict[str, Any]:
        """Return every DOF to neutral. Returns a motion_id, does not wait.

        Neutral is head level and centred, body forward, antennas neutral --
        a genuinely safe rest pose, not an arbitrary origin.

        Args:
            duration: Seconds the move should take.
        """
        async def run(motion_id):
            await self._progress(motion_id, "home", phase="homing",
                                 duration=float(duration))
            return await asyncio.to_thread(
                self._target.goto_neutral, float(duration))

        return await self._begin("home", {"duration": float(duration)}, run)

    @rpc(labels={"direction": "write", "safety": "critical", "motion": "true"})
    async def look_at(self, x: float, y: float, z: float,
                      duration: float = 1.0) -> Dict[str, Any]:
        """Point the head at a world point. Returns a motion_id, does not wait.

        Uses the daemon's own inverse kinematics.

        Args:
            x: Metres forward of the neutral head position.
            y: Metres to the robot's left.
            z: Metres above the neutral head position.
            duration: Seconds the move should take.
        """
        async def run(motion_id):
            await self._progress(motion_id, "look_at", phase="looking",
                                 target=[float(x), float(y), float(z)])
            return await asyncio.to_thread(
                self._target.look_at, float(x), float(y), float(z),
                float(duration))

        return await self._begin(
            "look_at", {"x": float(x), "y": float(y), "z": float(z),
                        "duration": float(duration)}, run)

    @rpc(labels={"direction": "write", "safety": "critical", "motion": "true"})
    async def nod(self, times: int = 2, amplitude: float = 0.25,
                  period: float = 0.5) -> Dict[str, Any]:
        """Nod 'yes': pitch the head down and up n times.

        Returns a motion_id and emits a motion_progress event per beat.

        Args:
            times: How many nods.
            amplitude: Peak pitch excursion in radians, clamped to the head's
                pitch limit.
            period: Seconds per nod.
        """
        n, amp, half = self._gesture_args(times, amplitude, period, "head_pitch")

        async def run(motion_id):
            base = self._target.get_cmd("head_pitch")
            for i in range(n):
                self._checkpoint(motion_id)
                await self._progress(motion_id, "nod", phase="nodding",
                                     beat=i + 1, of=n)
                await asyncio.to_thread(self._target.goto, half,
                                        head_pitch=base + amp)
                await asyncio.to_thread(self._target.goto, half,
                                        head_pitch=base - amp * 0.5)
            return await asyncio.to_thread(self._target.goto, half,
                                           head_pitch=base)

        return await self._begin(
            "nod", {"times": n, "amplitude": amp, "period": float(period)}, run)

    @rpc(labels={"direction": "write", "safety": "critical", "motion": "true"})
    async def shake(self, times: int = 2, amplitude: float = 0.4,
                    period: float = 0.5) -> Dict[str, Any]:
        """Shake 'no': yaw the head left and right n times.

        Returns a motion_id and emits a motion_progress event per beat.

        Args:
            times: How many shakes.
            amplitude: Peak yaw excursion in radians, clamped to the head's
                yaw limit.
            period: Seconds per shake.
        """
        n, amp, half = self._gesture_args(times, amplitude, period, "head_yaw")

        async def run(motion_id):
            base = self._target.get_cmd("head_yaw")
            for i in range(n):
                self._checkpoint(motion_id)
                await self._progress(motion_id, "shake", phase="shaking",
                                     beat=i + 1, of=n)
                await asyncio.to_thread(self._target.goto, half,
                                        head_yaw=base + amp)
                await asyncio.to_thread(self._target.goto, half,
                                        head_yaw=base - amp)
            return await asyncio.to_thread(self._target.goto, half,
                                           head_yaw=base)

        return await self._begin(
            "shake", {"times": n, "amplitude": amp, "period": float(period)},
            run)

    @staticmethod
    def _gesture_args(times, amplitude, period, dof):
        """Normalise the three arguments nod and shake share."""
        return (max(1, int(times)),
                min(abs(float(amplitude)), LIMITS[dof][1]),
                max(0.05, float(period) / 2.0))

    @rpc(labels={"direction": "write", "safety": "critical", "motion": "true"})
    async def wake_up(self) -> Dict[str, Any]:
        """Play the vendor wake-up emote. Returns a motion_id, does not wait."""
        async def run(motion_id):
            await self._progress(motion_id, "wake_up", phase="waking")
            await asyncio.to_thread(self._target.wake_up)
            return self._target.status()["commanded"]

        return await self._begin("wake_up", {}, run)

    @rpc(labels={"direction": "write", "safety": "critical", "motion": "true"})
    async def sleep(self) -> Dict[str, Any]:
        """Move to the vendor sleep posture. Returns a motion_id, does not wait."""
        async def run(motion_id):
            await self._progress(motion_id, "sleep", phase="sleeping")
            await asyncio.to_thread(self._target.goto_sleep)
            return self._target.status()["commanded"]

        return await self._begin("sleep", {}, run)
