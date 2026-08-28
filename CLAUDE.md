# CLAUDE.md

Guidance for Claude Code when starting and stopping this app. `README.md`
explains what the app *is* (the control surface, the functions, the design);
this file is just the runbook for getting it up and back down.

## What runs, and in what order

Two of our processes, plus one vendor process we do not start ourselves:

| Process | Owns | Started by |
|---|---|---|
| `reachy-mini-daemon` (vendor) | the USB serial bus, the mic, the camera | spawned automatically by `serve` |
| `controller.py serve` | the device `reachy-mini-1` | you |
| `voice/agent.py` | the conversation, borrows the mic and speaker | you, after `serve` is up |

Order matters: the daemon must be answering before `serve` connects, and `serve`
must be up before the voice agent can discover the robot.

**Only one process may own the robot.** Do not run `serve` and a hosted one-shot
command (`controller.py nod` with no `--attach`) at the same time.

## Start it

Robot only, no conversation:

```bash
cd reachy_mini_dc
.venv/bin/python controller.py serve            # stays up, Ctrl-C to stop
```

Zenoh is the default and is brokerless, so there is no server to run first.
Then drive it from any other terminal:

```bash
.venv/bin/python controller.py --attach zenoh:// slots      # commanded vs measured
.venv/bin/python controller.py --attach zenoh:// nod
.venv/bin/python controller.py --attach zenoh:// watch      # stream events
```

Add the talking app on top (needs `serve` already running):

```bash
cd voice
.venv/bin/python agent.py
```

Wait for `agent: ready -- say something` in its output. Startup is about 40s:
roughly 15s of robot connect and discovery, then 10s warming Whisper and Kokoro.
Talking before "ready" is wasted breath.

No hardware attached? `.venv/bin/python controller.py --stub demo` runs the whole
path against an in-memory robot.

## Stop it

Always SIGINT, never `kill`. SIGINT runs the cleanup that returns the robot to
neutral and hands the mic and speaker back; SIGTERM skips it and leaves the robot
holding its pose with its media released.

```bash
pkill -INT -f "agent.py"                # voice agent first
pkill -INT -f "controller.py serve"     # then the device
```

A clean `serve` shutdown ends with `Driver disconnected` in its log.

The vendor daemon deliberately outlives both. Leaving it up is the fast path
(the next `serve` reuses it and skips the motor configuration pass). Kill it only
if you need the USB bus, mic or camera free for something else:

```bash
pkill -f reachy-mini-daemon
```

## Traps we have actually hit

**A cold `serve` can lose a race with its own daemon.** On the first start after
the daemon is gone, the vendor daemon spends about 15s checking the configuration
of all 9 servos. `serve` gives up on `ws://localhost:8000` before that finishes
and exits with `ConnectionRefusedError: [Errno 61]`. The daemon survives, so the
fix is simply to run the same `serve` command again; the second one connects at
once. This is only ever a first-start problem.

**Discovery is not instant, and an early question gets a misleading answer.**
A freshly started `--attach` takes about 4 seconds to see the served device.
Worse, a client that asks before its own runtime has finished starting gets
"Registry not configured", which reads exactly like "the robot is not there".
Both clients here already wait for `driver.registry` before asking; if you write
a new one, do the same.

**Multicast scouting is unreliable on macOS.** If two hosts cannot see each
other over plain `zenoh://`, pin an address rather than debugging multicast:
`--zenoh-listen tcp/0.0.0.0:7447` on the serving host, and
`--attach zenoh://<host>:7447` on the client. It stays brokerless either way.

**The serial port in `README.md` is not stable.** It has already changed once.
Nothing reads that number, the daemon discovers the bus itself, so do not go
editing ports when something fails. Confirm the robot is plugged in with
`ls /dev/cu.usbmodem*` and look elsewhere for the cause.

**The voice agent reads `voice/.env`, not the repo root `.env`.** Both paths are
gitignored, so a fresh clone has neither and needs a key put in place before the
voice agent will run.

**Exit code 1 from a backgrounded `serve` or `agent.py` is normal** when you have
just sent it SIGINT. Read the log tail before treating it as a failure: a clean
agent shutdown ends with `returning the robot to neutral and taking media back`.

**A move's RPC returns before the move does.** That is the design, not a bug --
see "How motion works" in `README.md`. If you are scripting against the device,
either wait for the `motion_completed` event carrying your `motion_id` (as
`controller.py::run_motion` does) or poll `get_motion()`. Do not assume the robot
has stopped moving because the call came back.

**A "where predicates are unavailable" warning at startup is harmless.** It just
means `cel-python` is not installed. Nothing here uses broadcast `where` clauses.

**Where the logs go.** `serve.log` and `voice/run.log` in this tree, both
gitignored and both appended to rather than truncated, so check timestamps when
reading them. If the agent stops answering mid-conversation, the first thing to
check is `grep "dropping unconvertible thought" voice/agent.log` (see the voice
README for why).
