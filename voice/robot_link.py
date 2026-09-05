#!/usr/bin/env python3
"""robot_link -- the voice agent's connection to the Reachy Mini, over Device Connect.

This module is the whole reason the driver exists. The voice pipeline never
imports the vendor SDK, never opens the serial port, and does not even need to
run on the same machine as the robot: it joins the fleet as a Device Connect
device of its own, discovers the robot, and invokes its functions. Swap in any
other device that offers the same functions and the agent works unchanged.

    voice agent --invoke over zenoh/NATS--> controller.py serve --> robot

Motion is fire-and-forget, and the driver makes that nearly free: every move
returns a `motion_id` immediately and runs in the robot's own background task,
so the RPC round trip is milliseconds regardless of how long the move takes. A
spoken reply never waits on a 1.5-second head turn, and the robot moving *while*
Claude talks is what makes the thing feel alive rather than call-and-response.

`fire()` still wraps the call in a task, because even a millisecond round trip
should not sit in the path of a frame handler, and because failures must be
logged rather than raised into the conversation: a failed antenna twitch must
never break a turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from typing import Any, Dict, Optional

# Dev-tier rig: no TLS, no device authentication, no registry -- devices find
# each other by presence announcement on the local network.
os.environ.setdefault("DEVICE_CONNECT_ALLOW_INSECURE", "true")
os.environ.setdefault("DEVICE_CONNECT_DISCOVERY_MODE", "d2d")

from device_connect_edge import DeviceRuntime                    # noqa: E402
from device_connect_edge.drivers import DeviceDriver             # noqa: E402

logger = logging.getLogger(__name__)


def parse_broker(broker: str) -> tuple[str, list[str]]:
    """Turn a broker URL into the (backend, urls) Device Connect wants.

        zenoh://                brokerless peer mesh, multicast discovery
        zenoh://host:7447       brokerless peer mesh, direct connect
        nats://host:4222        NATS broker
        mqtt://host:1883        MQTT broker
    """
    scheme, _, rest = broker.partition("://")
    scheme = scheme.lower()
    if scheme == "zenoh":
        return "zenoh", ([f"tcp/{rest}"] if rest else [])
    if scheme in ("nats", "tls"):
        return "nats", [broker]
    if scheme in ("mqtt", "mqtts"):
        return "mqtt", [broker]
    raise ValueError(
        "unknown broker scheme %r; expected zenoh://, nats:// or mqtt://"
        % scheme)


class _VoiceDriver(DeviceDriver):
    """The agent's own presence on the fleet. Exposes no functions.

    Device Connect gives a client the same shape as a device, which is how the
    agent gets discovery and remote invocation without a server in the picture.
    """

    device_type = "voice_agent"


class RobotLink:
    """A Device Connect device, reachable as a set of awaitable calls."""

    def __init__(self, broker: str, device_id: str, tenant: str,
                 zenoh_listen: str | None = None) -> None:
        self.broker = broker
        self.zenoh_listen = zenoh_listen
        self.device_id = device_id
        self.tenant = tenant
        self._driver = _VoiceDriver()
        self._runtime: Optional[DeviceRuntime] = None
        self._run_task: Optional[asyncio.Task] = None
        self._background: set = set()

    # -- lifecycle ----------------------------------------------------------

    async def connect(self, timeout: float = 20.0) -> "RobotLink":
        backend, urls = parse_broker(self.broker)
        # ZENOH_LISTEN is read inside the zenoh adapter's config builder and is
        # the supported way to pin listen endpoints. Useful when the robot is on
        # another host and multicast scouting is not dependable.
        if self.zenoh_listen and backend == "zenoh":
            os.environ["ZENOH_LISTEN"] = self.zenoh_listen

        self._runtime = DeviceRuntime(
            driver=self._driver,
            device_id="voice-agent-%s" % uuid.uuid4().hex[:8],
            tenant=self.tenant,
            messaging_backend=backend,
            messaging_urls=urls or None,
        )
        self._run_task = asyncio.create_task(self._runtime.run())

        # run() connects, subscribes, and only then hands the driver its
        # registry. Asking for a peer before that raises "registry not
        # configured", which looks exactly like "robot missing" if you let it.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._driver.registry is None:
            if self._run_task.done():                        # run() failed
                await self._run_task
            if loop.time() > deadline:
                raise RuntimeError("could not connect to %s within %.0fs"
                                   % (self.broker, timeout))
            await asyncio.sleep(0.05)

        try:
            await self._driver.wait_for_device(
                device_id=self.device_id,
                timeout=max(1.0, deadline - loop.time()))
        except Exception as exc:                             # noqa: BLE001
            raise RuntimeError(
                "robot %r did not appear on %s within %.0fs -- is "
                "'controller.py serve --broker %s' running? (%s)"
                % (self.device_id, self.broker, timeout, self.broker, exc))
        logger.info("robot %r found on %s", self.device_id, self.broker)
        return self

    async def close(self) -> None:
        for task in list(self._background):
            task.cancel()
        if self._runtime is not None:
            with contextlib.suppress(Exception):
                await self._runtime.stop()
        if self._run_task is not None:
            self._run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._run_task

    # -- invocation ---------------------------------------------------------

    async def call(self, name: str, **args) -> Any:
        """Invoke a function and return its result.

        For a move this returns the driver's acknowledgement -- a `motion_id`
        and the arguments it accepted -- not the finished posture. The move is
        still running when this returns, which is the point.
        """
        reply = await self._driver.invoke_remote(self.device_id, name, **args)
        if "error" in reply:
            err = reply["error"]
            raise RuntimeError(
                "%s failed: %s"
                % (name, err.get("message") if isinstance(err, dict) else err))
        return reply.get("result")

    def fire(self, name: str, **args) -> None:
        """Start a call and return immediately.

        The conversation continues while the robot moves. Errors are logged
        rather than raised: a failed antenna twitch must never break a turn.
        """
        task = asyncio.create_task(self._fire(name, args))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _fire(self, name: str, args: Dict[str, Any]) -> None:
        try:
            result = await self.call(name, **args)
            if isinstance(result, dict) and result.get("accepted") is False:
                logger.info("robot: %s refused: %s", name,
                            result.get("reason", "no reason given"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:                             # noqa: BLE001
            logger.warning("robot call %s(%s) failed: %s", name, args, exc)

    # -- the functions the agent actually uses ------------------------------

    async def release_media(self, released: bool = True) -> Any:
        """Ask the robot to hand over (or take back) its mic and speaker."""
        return await self.call("set_media_released", released=released)

    async def status(self) -> Dict[str, Any]:
        return await self.call("report_status")

    def look_at(self, x: float, y: float, z: float, duration: float = 0.8):
        self.fire("look_at", x=x, y=y, z=z, duration=duration)

    def posture(self, duration: float = 0.5, **dofs: float):
        self.fire("goto_posture", duration=duration, **dofs)

    def nod(self, times: int = 1, period: float = 0.5):
        self.fire("nod", times=times, period=period)

    def shake(self, times: int = 2, period: float = 0.5):
        self.fire("shake", times=times, period=period)

    def home(self, duration: float = 1.0):
        self.fire("home", duration=duration)

    def perform(self, name: str, repeat: int = 1):
        """Play a curated recorded move (dances, emotions, spin)."""
        self.fire("play_move", move=name, repeat=repeat)
