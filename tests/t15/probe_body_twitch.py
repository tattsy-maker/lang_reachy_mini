"""T15.9 metal probe: does a stream of head-only gotos twitch the base?

Attach to a running ``controller.py serve`` and sample the measured
body_yaw at ~15 Hz for a few seconds under three regimes:

  idle          nothing commanded
  old-style     a head/antenna goto every 0.9 s that ALSO carries the
                held body_yaw (what reachy_target.goto sent before T15.9)
  new-style     the same gotos without body_yaw (T15.9)

Prints peak-to-peak and RMS of the measured body_yaw (millirad) and the
dominant frequency of its deviation, per regime. Run with the robot on a
table and nobody touching it:

    .venv/bin/python tests/t15/probe_body_twitch.py --attach zenoh://127.0.0.1:7447
"""

import argparse
import asyncio
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from controller import RemoteSession  # noqa: E402


async def sample(s, seconds: float, hz: float = 15.0):
    t0 = time.monotonic()
    ts, ys = [], []
    while time.monotonic() - t0 < seconds:
        st = await s.call("report_status")
        ys.append(float((st.get("measured") or {}).get("body_yaw", 0.0)))
        ts.append(time.monotonic() - t0)
        await asyncio.sleep(1.0 / hz)
    return np.array(ts), np.array(ys)


def report(name, ts, ys):
    dev = ys - ys.mean()
    p2p = (ys.max() - ys.min()) * 1000
    rms = math.sqrt(float((dev ** 2).mean())) * 1000
    freq = float("nan")
    if len(ys) > 8:
        dt = float(np.median(np.diff(ts)))
        spec = np.abs(np.fft.rfft(dev))
        freqs = np.fft.rfftfreq(len(dev), d=dt)
        spec[0] = 0
        freq = float(freqs[int(np.argmax(spec))])
    print(f"{name:10s} p2p {p2p:6.2f} mrad  rms {rms:5.2f} mrad  "
          f"dominant {freq:4.1f} Hz  ({len(ys)} samples)")


async def regime(s, name, seconds, nudge=None, period=0.9):
    stop = asyncio.Event()

    async def nudger():
        i = 0
        while not stop.is_set():
            pitch = -0.15 + 0.03 * math.sin(i)
            await s.call("goto_posture", duration=0.8, head_pitch=pitch,
                         head_roll=0.01 * (-1) ** i, antenna_left=0.4,
                         antenna_right=-0.4, **(nudge or {}))
            i += 1
            await asyncio.sleep(period)

    task = asyncio.create_task(nudger()) if nudge is not None else None
    ts, ys = await sample(s, seconds)
    if task:
        stop.set()
        await task
    report(name, ts, ys)


async def main(args):
    s = await RemoteSession(args.attach, args.device_id, "lab").open()
    try:
        await s.call("goto_posture", duration=1.0, head_yaw=0.0, body_yaw=0.0,
                     head_pitch=0.0)
        await asyncio.sleep(1.5)
        held = float((await s.call("report_status"))["commanded"]["body_yaw"])
        await regime(s, "idle", args.seconds)
        await regime(s, "old-style", args.seconds, nudge={"body_yaw": held})
        await regime(s, "new-style", args.seconds, nudge={})
        if args.turned:
            await s.call("goto_posture", duration=1.0, head_yaw=math.radians(30))
            await asyncio.sleep(1.5)
            await regime(s, "turned-old", args.seconds, nudge={"body_yaw": held})
            await regime(s, "turned-new", args.seconds, nudge={})
            await s.call("goto_posture", duration=1.0, head_yaw=0.0)
    finally:
        await s.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attach", default="zenoh://127.0.0.1:7447")
    ap.add_argument("--device-id", default="reachy-mini-1")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--turned", action="store_true",
                    help="repeat with the head yawed 30 degrees")
    asyncio.run(main(ap.parse_args()))
