#!/usr/bin/env python3
"""controller.py -- drive the Reachy Mini through its Device Connect driver.

Two roles, one command set.

HOSTED (default). The controller builds the driver inside its own process and
calls it directly, so it owns the USB link for the life of the command, runs the
command, and lets go. No broker, no discovery, no messaging at all:

    ./controller.py status
    ./controller.py nod --times 3
    ./controller.py pose --head-yaw 0.4 --duration 1.0

SERVE + ATTACH. One process owns the robot and stays up; any number of
controllers (or agents, or other Device Connect clients) drive it over the
messaging layer:

    ./controller.py serve                              # terminal 1
    ./controller.py --attach zenoh:// nod              # terminal 2

Only one process may hold the robot's serial bus at a time, so `serve` and a
hosted command cannot run together. That is the same constraint the vendor
daemon has, surfaced one layer up.

Moves do not block the device. `goto_posture` and the gestures return a
`motion_id` straight away and run in the background, emitting `motion_progress`
and then `motion_completed`. This controller subscribes to those events, so a
command still reads as "run it and print the result" -- and Ctrl-C during a move
sends `cancel_motion` rather than killing the process mid-motion.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import uuid

# Dev-tier rig: skip TLS and device authentication so this runs without
# commissioned credentials. For a real deployment drop this line and commission
# the device against a Device Connect server or portal.
os.environ.setdefault("DEVICE_CONNECT_ALLOW_INSECURE", "true")
# There is no registry here -- devices find each other by presence
# announcement. Set DEVICE_CONNECT_DISCOVERY_MODE=registry before running to
# use a Device Connect server or portal instead.
os.environ.setdefault("DEVICE_CONNECT_DISCOVERY_MODE", "d2d")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from device_connect_edge import DeviceRuntime                    # noqa: E402
from device_connect_edge.drivers import DeviceDriver, on         # noqa: E402

from reachy_driver import ReachyMiniDriver                       # noqa: E402
from reachy_target import DOF_NAMES, LIMITS                      # noqa: E402

DEFAULT_BROKER = "zenoh://"
DEFAULT_DEVICE_ID = "reachy-mini-1"
DEFAULT_TENANT = "lab"


def parse_broker(broker: str) -> tuple[str, list[str]]:
    """Turn a broker URL into the (backend, urls) Device Connect wants.

    Device Connect names its backend and its endpoints separately rather than
    packing both into one URL. This keeps a single familiar `--broker` string
    on the command line and does the translation here.

        zenoh://                brokerless peer mesh, multicast discovery
        zenoh://host:7447       brokerless peer mesh, direct connect
        nats://host:4222        NATS broker
        mqtt://host:1883        MQTT broker
    """
    scheme, _, rest = broker.partition("://")
    scheme = scheme.lower()
    if scheme == "zenoh":
        # No endpoint means pure D2D: zenoh scouts the local network and there
        # is nothing to deploy or keep alive. An endpoint pins an address for
        # the same brokerless mesh, which is what you want on macOS where
        # multicast scouting is not dependable.
        return "zenoh", ([f"tcp/{rest}"] if rest else [])
    if scheme in ("nats", "tls"):
        return "nats", [broker]
    if scheme in ("mqtt", "mqtts"):
        return "mqtt", [broker]
    raise SystemExit(
        "unknown broker scheme %r; expected zenoh://, nats:// or mqtt://"
        % scheme)


def jprint(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def build_driver(args) -> ReachyMiniDriver:
    """Construct the driver for the requested hardware backing."""
    if getattr(args, "stub", False):
        from stub_target import StubTarget
        return ReachyMiniDriver(target=StubTarget())
    return ReachyMiniDriver(spawn_daemon=not args.no_spawn_daemon,
                            use_sim=args.sim)


# ---------------------------------------------------------------------------
# Sessions
#
# Two ways to reach the robot behind one interface: call() invokes a procedure,
# add_listener() taps the event stream. Everything above this line is written
# once and works either way.
# ---------------------------------------------------------------------------

class _Session:
    """Common behaviour: event fan-out, and waiting on a motion."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self._listeners: list = []

    def add_listener(self, cb) -> None:
        """Register cb(event_name, payload). Called for every device event."""
        self._listeners.append(cb)

    def remove_listener(self, cb) -> None:
        with contextlib.suppress(ValueError):
            self._listeners.remove(cb)

    async def _dispatch(self, event_name: str, payload: dict) -> None:
        for cb in list(self._listeners):
            try:
                await cb(event_name, payload)
            except Exception as exc:                            # noqa: BLE001
                print("event handler failed: %s" % exc, file=sys.stderr)

    async def call(self, name: str, **args):
        raise NotImplementedError

    # -- motion -------------------------------------------------------------

    async def run_motion(self, name: str, **args):
        """Start a motion, stream its progress, and wait for it to finish.

        The wait is done by listening for the driver's own motion_completed
        event rather than by holding the RPC open, because the driver
        deliberately does not hold it open -- see reachy_driver.py.
        """
        done: asyncio.Future = asyncio.get_running_loop().create_future()
        state = {"motion_id": None}

        async def listen(event_name: str, payload: dict) -> None:
            if payload.get("motion_id") != state["motion_id"]:
                return
            if event_name == "motion_progress":
                print("  .. %s" % json.dumps(payload.get("progress") or {},
                                             sort_keys=True), file=sys.stderr)
            elif event_name == "motion_completed" and not done.done():
                done.set_result(payload)

        self.add_listener(listen)
        try:
            ack = await self.call(name, **args)
            state["motion_id"] = ack.get("motion_id")
            if not state["motion_id"]:
                return ack
            terminal = await done
            if terminal.get("status") != "succeeded":
                raise SystemExit("%s %s: %s"
                                 % (name, terminal.get("status"),
                                    terminal.get("error") or ""))
            return {"posture": terminal.get("posture"),
                    "motion_id": terminal.get("motion_id")}
        except asyncio.CancelledError:
            if state["motion_id"]:
                print("cancelling %s ..." % name, file=sys.stderr)
                with contextlib.suppress(Exception):
                    await self.call("cancel_motion",
                                    motion_id=state["motion_id"])
            raise
        finally:
            self.remove_listener(listen)


class HostedSession(_Session):
    """The driver lives in this process. No messaging, no discovery."""

    def __init__(self, device_id: str, driver: ReachyMiniDriver) -> None:
        super().__init__(device_id)
        self.driver = driver
        # Route the driver's emits straight back to our listeners. Without
        # this the driver would raise on its first emit: with no DeviceRuntime
        # attached it has nowhere to send events.
        driver.set_event_callback(self._on_driver_event)

    async def _on_driver_event(self, event_name: str, payload: dict) -> None:
        await self._dispatch(event_name, payload)

    async def open(self) -> "HostedSession":
        await self.driver.start_local()
        return self

    async def close(self) -> None:
        await self.driver.stop_local()

    async def call(self, name: str, **args):
        return await self.driver.invoke(name, **args)

    async def manifest(self) -> dict:
        return self.driver.capabilities.model_dump()


class _ControllerDriver(DeviceDriver):
    """A device that exists only to talk to other devices.

    Device Connect gives a client the same shape as a device: it joins the
    fleet, so it can announce itself, discover peers, invoke their functions
    and subscribe to their events. This one exposes no functions of its own.
    """

    device_type = "reachy_controller"

    def __init__(self) -> None:
        super().__init__()
        self.on_event = None

    # No device_id / event_name filter: subscribe to every event in the tenant
    # and filter in the handler. The filter we want (one known device, all of
    # its events) is not expressible as a broker subscription anyway.
    @on()
    async def _any_event(self, device_id: str, event_name: str,
                         payload: dict) -> None:
        if self.on_event is not None:
            await self.on_event(device_id, event_name, payload)


class RemoteSession(_Session):
    """The robot is served by another process; reach it over the broker."""

    def __init__(self, broker: str, device_id: str, tenant: str) -> None:
        super().__init__(device_id)
        self.broker = broker
        self.tenant = tenant
        self._driver = _ControllerDriver()
        self._driver.on_event = self._on_remote_event
        self._runtime = None
        self._task = None

    async def _on_remote_event(self, device_id: str, event_name: str,
                               payload: dict) -> None:
        if device_id == self.device_id:
            await self._dispatch(event_name, payload)

    async def open(self, timeout: float = 20.0) -> "RemoteSession":
        backend, urls = parse_broker(self.broker)
        self._runtime = DeviceRuntime(
            driver=self._driver,
            device_id="reachy-controller-%s" % uuid.uuid4().hex[:8],
            tenant=self.tenant,
            messaging_backend=backend,
            messaging_urls=urls or None,
        )
        self._task = asyncio.create_task(self._runtime.run())

        # run() connects, subscribes, and only then hands the driver its
        # registry. Asking for a peer before that raises "registry not
        # configured", which looks exactly like "robot missing" if you let it.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._driver.registry is None:
            if self._task.done():                       # run() failed outright
                await self._task
            if loop.time() > deadline:
                raise SystemExit("could not connect to %s within %.0fs"
                                 % (self.broker, timeout))
            await asyncio.sleep(0.05)

        try:
            await self._driver.wait_for_device(
                device_id=self.device_id,
                timeout=max(1.0, deadline - loop.time()))
        except Exception:
            raise SystemExit(
                "device %r did not appear on %s within %.0fs.\n"
                "  is a 'controller.py serve' running against this broker?\n"
                "  on macOS, multicast scouting is unreliable: pin an address\n"
                "  with --zenoh-listen on the serving host and attach to it."
                % (self.device_id, self.broker, timeout))
        return self

    async def close(self) -> None:
        if self._runtime is not None:
            with contextlib.suppress(Exception):
                await self._runtime.stop()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def call(self, name: str, **args):
        reply = await self._driver.invoke_remote(self.device_id, name, **args)
        if "error" in reply:
            err = reply["error"]
            raise SystemExit("%s failed: %s" % (
                name, err.get("message") if isinstance(err, dict) else err))
        return reply.get("result")

    async def manifest(self) -> dict:
        rec = await self._driver.get_device(self.device_id)
        return (rec or {}).get("capabilities") or rec or {}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_status(s: _Session, a) -> None:
    jprint(await s.call("report_status"))


async def cmd_manifest(s: _Session, a) -> None:
    jprint(await s.manifest())


async def cmd_slots(s: _Session, a) -> None:
    st = await s.call("report_status")
    print("%-16s %10s %10s   %s" % ("DOF", "COMMANDED", "MEASURED", "LIMITS"))
    for name in DOF_NAMES:
        lo, hi = LIMITS[name]
        print("%-16s %10.4f %10.4f   [%.3f, %.3f]"
              % (name, st["commanded"][name], st["measured"][name], lo, hi))
    print("\nmotors_enabled=%s  estopped=%s  last_error=%r"
          % (st["motors_enabled"], st["estopped"], st["last_error"]))


async def cmd_pose(s: _Session, a) -> None:
    dofs = {n: getattr(a, n) for n in DOF_NAMES if getattr(a, n) is not None}
    if not dofs:
        raise SystemExit(
            "pose needs at least one DOF, e.g. --head-yaw 0.4\n  available: %s"
            % ", ".join("--" + n.replace("_", "-") for n in DOF_NAMES))
    jprint(await s.run_motion("goto_posture", duration=a.duration, **dofs))


async def cmd_home(s: _Session, a) -> None:
    jprint(await s.run_motion("home", duration=a.duration))


async def cmd_look(s: _Session, a) -> None:
    jprint(await s.run_motion("look_at", x=a.x, y=a.y, z=a.z,
                              duration=a.duration))


async def cmd_nod(s: _Session, a) -> None:
    jprint(await s.run_motion("nod", times=a.times, amplitude=a.amplitude,
                              period=a.period))


async def cmd_shake(s: _Session, a) -> None:
    jprint(await s.run_motion("shake", times=a.times, amplitude=a.amplitude,
                              period=a.period))


async def cmd_perform(s: _Session, a) -> None:
    jprint(await s.run_motion("play_move", move=a.name, repeat=a.repeat))


async def cmd_moves(s: _Session, a) -> None:
    jprint(await s.call("list_moves"))


async def cmd_wake(s: _Session, a) -> None:
    jprint(await s.run_motion("wake_up"))


async def cmd_sleep(s: _Session, a) -> None:
    jprint(await s.run_motion("sleep"))


async def cmd_motors(s: _Session, a) -> None:
    jprint(await s.call("set_motors", enabled=(a.state == "on")))


async def cmd_stop(s: _Session, a) -> None:
    jprint(await s.call("stop"))


async def cmd_clear_estop(s: _Session, a) -> None:
    jprint(await s.call("clear_estop"))


async def cmd_watch(s: _Session, a) -> None:
    """Print the driver's events until interrupted."""
    async def on_event(event_name: str, payload: dict) -> None:
        # flush: watch is a stream, and its output is routinely piped or
        # redirected, where Python would otherwise block-buffer and show
        # nothing until the process ends.
        print("%-22s %s" % (event_name, json.dumps(payload, default=str)[:160]),
              flush=True)

    s.add_listener(on_event)
    print("watching events from %s; Ctrl-C to stop" % s.device_id,
          file=sys.stderr)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        s.remove_listener(on_event)


async def cmd_demo(s: _Session, a) -> None:
    """A short scripted sequence that exercises the whole surface."""
    steps = [
        ("home",         {"duration": 1.0}),
        ("goto_posture", {"duration": 0.8, "head_yaw": 0.5}),
        ("goto_posture", {"duration": 0.8, "head_yaw": -0.5}),
        ("goto_posture", {"duration": 0.6, "head_yaw": 0.0,
                          "head_pitch": -0.25}),
        ("nod",          {"times": 2, "period": 0.6}),
        ("shake",        {"times": 2, "period": 0.6}),
        ("goto_posture", {"duration": 0.5, "antenna_left": 1.2,
                          "antenna_right": -1.2}),
        ("goto_posture", {"duration": 0.5, "antenna_left": 0.0,
                          "antenna_right": 0.0}),
        ("goto_posture", {"duration": 1.2, "body_yaw": 0.8}),
        ("home",         {"duration": 1.5}),
    ]
    for i, (name, args) in enumerate(steps, 1):
        print("[%d/%d] %s %s" % (i, len(steps), name,
                                 json.dumps(args, sort_keys=True)),
              file=sys.stderr)
        await s.run_motion(name, **args)
    print("demo complete", file=sys.stderr)


COMMANDS = {
    "status": cmd_status, "manifest": cmd_manifest, "slots": cmd_slots,
    "pose": cmd_pose, "home": cmd_home, "look": cmd_look,
    "nod": cmd_nod, "shake": cmd_shake, "wake": cmd_wake, "sleep": cmd_sleep,
    "perform": cmd_perform, "moves": cmd_moves,
    "motors": cmd_motors, "stop": cmd_stop, "clear-estop": cmd_clear_estop,
    "watch": cmd_watch, "demo": cmd_demo,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="controller.py",
        description="Drive a Reachy Mini through its Device Connect driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("--attach", metavar="BROKER", default=None,
                   help="drive a device already served on BROKER instead of "
                        "hosting one here (e.g. zenoh:// or "
                        "nats://localhost:4222)")
    p.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    p.add_argument("--tenant", default=DEFAULT_TENANT)
    p.add_argument("--sim", action="store_true",
                   help="drive a simulated robot instead of the USB hardware")
    p.add_argument("--no-spawn-daemon", action="store_true",
                   help="require an already-running reachy_mini daemon rather "
                        "than starting one")
    p.add_argument("--zenoh-listen", default=None, metavar="EP",
                   help="zenoh only: TCP endpoints to listen on, comma "
                        "separated (e.g. tcp/0.0.0.0:7447). Set this on the "
                        "serving host so clients can reach it by address when "
                        "multicast discovery is unavailable.")
    p.add_argument("--stub", action="store_true",
                   help="run against an in-memory stub robot; exercises the "
                        "whole path with no hardware and no daemon")

    sub = p.add_subparsers(dest="command", required=True)

    srv = sub.add_parser("serve", help="host the device and stay up")
    srv.add_argument("--zenoh-listen", default=None, metavar="EP",
                     dest="zenoh_listen_serve",
                     help="zenoh only: TCP endpoints to listen on (e.g. "
                          "tcp/0.0.0.0:7447), so clients can reach this host "
                          "by address when multicast is unavailable")
    srv.add_argument("--broker", default=DEFAULT_BROKER,
                     help="where to serve. zenoh:// is brokerless "
                          "peer-to-peer (no server to run, the default); "
                          "nats://host:4222 or mqtt://host:1883 need a broker "
                          "process")

    sub.add_parser("status", help="full driver and robot status (JSON)")
    sub.add_parser("slots", help="commanded vs measured posture, as a table")
    sub.add_parser("manifest", help="the device's capability manifest")
    sub.add_parser("wake", help="play the vendor wake-up emote")
    sub.add_parser("sleep", help="move to the vendor sleep posture")
    sub.add_parser("stop", help="EMERGENCY STOP: freeze and refuse motion")
    sub.add_parser("clear-estop", help="release the emergency stop latch")
    sub.add_parser("watch", help="stream the driver's events")
    sub.add_parser("demo", help="a scripted sequence over the whole surface")

    pose = sub.add_parser("pose", help="move to a posture")
    pose.add_argument("--duration", type=float, default=0.8)
    for name in DOF_NAMES:
        lo, hi = LIMITS[name]
        pose.add_argument("--" + name.replace("_", "-"), dest=name,
                          type=float, default=None,
                          help="[%.3f, %.3f]" % (lo, hi))

    home = sub.add_parser("home", help="return every DOF to neutral")
    home.add_argument("--duration", type=float, default=1.5)

    look = sub.add_parser("look", help="point the head at a world point")
    look.add_argument("x", type=float)
    look.add_argument("y", type=float)
    look.add_argument("z", type=float)
    look.add_argument("--duration", type=float, default=1.0)

    perform = sub.add_parser("perform", help="play a curated recorded move "
                                             "(see 'moves')")
    perform.add_argument("name")
    perform.add_argument("--repeat", type=int, default=1)
    sub.add_parser("moves", help="list the curated recorded moves")
    for name, amp in (("nod", 0.25), ("shake", 0.4)):
        g = sub.add_parser(name, help="%s the head" % name)
        g.add_argument("--times", type=int, default=2)
        g.add_argument("--amplitude", type=float, default=amp)
        g.add_argument("--period", type=float, default=0.6)

    mot = sub.add_parser("motors", help="enable or disable motor torque")
    mot.add_argument("state", choices=["on", "off"])

    return p


async def serve_async(args) -> None:
    """Host the device on a broker and stay up until interrupted."""
    backend, urls = parse_broker(args.broker)
    driver = build_driver(args)
    runtime = DeviceRuntime(
        driver=driver,
        device_id=args.device_id,
        tenant=args.tenant,
        messaging_backend=backend,
        messaging_urls=urls or None,
    )

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    task = asyncio.create_task(runtime.run())
    print("[reachy] serving %r on %s (tenant %s); Ctrl-C to stop"
          % (args.device_id, args.broker, args.tenant), file=sys.stderr)
    await stop.wait()
    print("[reachy] stopping ...", file=sys.stderr)
    await runtime.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def main_async(args) -> None:
    if args.attach:
        session = await RemoteSession(args.attach, args.device_id,
                                      args.tenant).open()
    else:
        session = await HostedSession(args.device_id,
                                      build_driver(args)).open()
    try:
        await COMMANDS[args.command](session, args)
    finally:
        await session.close()


def main() -> None:
    args = build_parser().parse_args()

    # ZENOH_LISTEN is read inside the zenoh adapter's config builder and is the
    # supported way to pin listen endpoints, on either side of the link.
    listen = getattr(args, "zenoh_listen_serve", None) or args.zenoh_listen
    broker = getattr(args, "broker", None) or args.attach or ""
    if listen and broker.startswith("zenoh"):
        os.environ["ZENOH_LISTEN"] = listen
        print("[reachy] zenoh listening on %s" % listen, file=sys.stderr)

    try:
        if args.command == "serve":
            asyncio.run(serve_async(args))
        else:
            asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
