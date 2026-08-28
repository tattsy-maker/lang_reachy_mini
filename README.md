# reachy_mini_dc

A Pollen Robotics **Reachy Mini Lite** exposed as an **[Arm Device Connect](https://github.com/arm/device-connect) device**, driver-first.

The robot's 9 degrees of freedom become bounded, individually addressable
functions with a real emergency stop; moves and emotes are Device Connect RPCs;
state changes are Device Connect events. Anything that speaks Device Connect (an
agent, another rig, the MCP bridge) can then drive the robot without knowing
anything about Pollen's SDK.

## The robot on this machine

Detected over USB-C, VID:PID `1a86:55d3` (the Lite's serial bridge):

| Interface | Where |
|---|---|
| Motor bus (9 servos) | `/dev/cu.usbmodem*` |
| Camera | `Reachy Mini Camera` (SunplusIT), not used by this driver |
| Audio | `Reachy Mini Audio` (Pollen Robotics), not used by this driver |

The Lite is tethered: the Mac runs everything. (The Wireless version runs the
same daemon on an on-board Pi instead.)

## Architecture

```
  controller.py  --invoke/events-->  ReachyMiniDriver  (reachy_driver.py)
  (or any DC client / agent)                |
                                            |  9 scalar DOFs
                                            v
                                    ReachyMiniTarget  (reachy_target.py)
                                            |  reachy_mini.ReachyMini
                                            v
                                      reachy_mini daemon  (separate process)
                                            |  serial
                                            v
                                      9x servos on the USB bus
```

Three layers, deliberately: `reachy_target.py` knows the robot and nothing about
Device Connect; `reachy_driver.py` knows Device Connect and nothing about serial
ports; `controller.py` is a client and could be deleted without the device
ceasing to work.

**The daemon owns the serial bus, and only one process may hold it.** So exactly
one device process may own the robot at a time. `serve` and a hosted one-shot
command cannot run together.

## Files

| File | What it is |
|---|---|
| `reachy_driver.py` | **The driver.** 17 functions, 7 events, 2 periodic routines. The deliverable. |
| `reachy_target.py` | Hardware layer: wraps `reachy_mini.ReachyMini`, owns daemon lifecycle, composes/decomposes the head's 4x4 pose, clamps to the mechanical envelope. |
| `controller.py` | The CLI. Hosts the device itself, or attaches to one already served. |
| `stub_target.py` | In-memory robot for `--stub`: exercises the whole path with no hardware. |
| `voice/` | **Spoken conversation** with the robot, in six languages: local STT/TTS/turn-detection, Claude for language, robot reached over Device Connect. Its own venv. See [voice/README.md](voice/README.md). |
| `.venv/` | `reachy-mini` + `device-connect-edge` (which brings zenoh and NATS). |

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install device-connect-edge reachy-mini
```

`device-connect-edge` pulls in `eclipse-zenoh` and `nats-py`, so there is
nothing else to add for either transport.

## Use

### Hosted (simplest): the controller owns the robot for one command

No broker, no discovery, no messaging at all. The controller builds the driver
in its own process and calls it directly.

```bash
.venv/bin/python controller.py slots            # commanded vs measured, as a table
.venv/bin/python controller.py home
.venv/bin/python controller.py pose --head-yaw 0.4 --head-pitch -0.2 --duration 1.0
.venv/bin/python controller.py look 0.5 0.2 0.1
.venv/bin/python controller.py nod --times 3
.venv/bin/python controller.py demo             # scripted tour of the surface
```

### Served: one owner, many clients

One process owns the USB link and stays up; any number of clients drive it.

**Zenoh, brokerless (the default -- no server to run):**

```bash
.venv/bin/python controller.py serve                            # terminal 1
.venv/bin/python controller.py --attach zenoh:// nod            # terminal 2
```

**NATS (needs a broker process):**

```bash
nats-server -p 4222 &
.venv/bin/python controller.py serve --broker nats://localhost:4222
.venv/bin/python controller.py --attach nats://localhost:4222 nod
```

### Choosing a transport

| | `zenoh://` | `nats://host:4222` | hosted |
|---|---|---|---|
| Broker process | **none** | `nats-server` | none |
| Across machines | yes | yes | no |
| Across processes | yes | yes | no |
| Discovery | multicast, or by address | via broker | n/a |

Zenoh runs as a true peer mesh, so there is nothing to deploy or keep alive.
That is the reason it is worth preferring once the robot and its clients are on
different machines. Both transports run **without a Device Connect server**:
devices announce their own presence rather than registering with a registry,
which is what `DEVICE_CONNECT_DISCOVERY_MODE=d2d` selects. `controller.py` sets
that for you.

**Cross-host on a LAN.** Multicast discovery is not dependable on macOS, so pin
an address on the serving host and have clients connect to it. This is still
brokerless -- a peer mesh reached by address rather than by multicast.

```bash
# on the robot's host
.venv/bin/python controller.py serve --zenoh-listen tcp/0.0.0.0:7447

# on any other machine
.venv/bin/python controller.py --attach zenoh://192.168.1.50:7447 nod
```

### No hardware

```bash
.venv/bin/python controller.py --stub demo      # full path, in-memory robot
```

## The control surface

Nine degrees of freedom, each clamped into the mechanical envelope on every
write. `get_posture()` reads them all in one call; `set_dof(name, value)` writes
one immediately; `goto_posture(...)` moves smoothly to any subset of them.

| DOF | Unit | Range (clamped) | Neutral |
|---|---|---|---|
| `head_x` / `head_y` | metre | -0.02 .. 0.02 | centred |
| `head_z` | metre | **-0.05 .. 0.02** | centred |
| `head_roll` / `head_pitch` | radian | -0.45 .. 0.45 | level |
| `head_yaw` | radian | -0.90 .. 0.90 | forward |
| `body_yaw` | radian | -2.79 .. 2.79 | forward |
| `antenna_left` / `antenna_right` | radian | -3.14 .. 3.14 | neutral |

Every DOF is mirrored by a measured value carrying what the motors actually
report, refreshed at 10 Hz by the `poll_state` routine and published at 1 Hz as
a `telemetry` event. `get_limits()` publishes the table above so a client can
discover it rather than hard-code it.

**Why the neutral pose is zero.** Zero is not an arbitrary origin: it is the
neutral posture (head level and centred, body forward, antennas neutral). That
is what makes "go to zero" the right thing to do on shutdown and after an estop
release, rather than holding a strained pose.

**Why writes clamp rather than refuse.** Every pose inside the mechanical
envelope is a legal pose, so an out-of-envelope write is clamped *into* the
envelope. Verified: `pose --head-yaw 99` lands as `0.9`, and `set_dof` reports
`{"requested": -0.09, "applied": -0.05, "clamped": true}` so a caller can tell.

Body and antenna limits are read off the shipped URDF. The head is a parallel
(Stewart) mechanism whose reachable task-space box is pose-coupled and not a
single number in the URDF, so the head figures were **measured on this robot**
by commanding each axis outward and reading back what the mechanism reached:

| Axis | Measured | Set to |
|---|---|---|
| `head_z` down | tracks past -0.050 | -0.05 |
| `head_z` up | saturates near +0.020 | +0.02 |
| `head_x` | saturates near +0.020 (commanding +0.020 reaches +0.013) | 0.02 |
| `head_pitch` | reached +0.70 | 0.45, headroom kept |
| `head_yaw` | still tracking at +1.45 | 0.90, headroom kept |

`head_z` is asymmetric because the mechanism is: the head drops far but rises
little. A symmetric box would have excluded the pose the robot sits in at rest
(z = -0.043 powered off). The IK refused nothing across that sweep, so these are
comfort bounds rather than a cliff edge; widen the rotations in
`reachy_target.py` if you need the measured headroom.

## Functions

| Function | Notes |
|---|---|
| `get_posture()` / `get_limits()` / `report_status()` | Reads. `report_status` is the whole picture: both postures, limits, motor, media and estop state, and the running motion. |
| `set_dof(name, value)` | One DOF, immediately, no interpolation. Returns what it actually applied. |
| `goto_posture(duration, head_x..antenna_right)` | Motion. Every DOF is a named, optional parameter, so the manifest is self-describing. Unnamed DOFs hold their current value. |
| `home(duration)` | Motion. Every DOF to neutral. |
| `look_at(x, y, z, duration)` | Motion. Daemon-side IK; x forward, y left, z up. |
| `nod(times, amplitude, period)` / `shake(...)` | Motion, cancellable, emits per-beat progress. |
| `wake_up()` / `sleep()` | Motion. Vendor emotes. |
| `set_motors(enabled, ids)` | Torque on/off. |
| `set_media_released(released)` | Hand the robot's camera, mic and speaker to another process, or take them back. This is how `voice/` claims the microphone. |
| `stop()` | **Estop.** Freezes at the present pose. |
| `clear_estop()` | Releases the latch. |
| `cancel_motion(motion_id)` | Cancel the running move without latching. |
| `get_motion(motion_id)` | Poll a motion's state without blocking. |

Every function carries discovery `labels` -- `direction: read|write`,
`safety: critical|informational`, and `motion: true` on the seven that move the
robot -- so a client can filter the surface before it reads any schemas.

## How motion works

**Moves never block the device.** `goto_posture`, `home`, `look_at`, `nod`,
`shake`, `wake_up` and `sleep` return a `motion_id` immediately and run in a
background task on the robot. Progress arrives as `motion_progress` events, the
outcome as `motion_completed`.

This is not a stylistic choice. Device Connect dispatches a device's incoming
RPCs in order on one subscription, so a move that held its handler open for two
seconds would also hold up the `stop()` arriving behind it. An emergency stop
that cannot be heard during a move is not an emergency stop. Measured on the
stub: `stop()` answered in **2 ms** while a four-second nod was running, and
cancelled it.

Two consequences worth knowing:

- **A caller that wants to wait subscribes to `motion_completed`.** That is what
  `controller.py` does, which is why its commands still read as "run it and
  print the result", and why Ctrl-C mid-move sends `cancel_motion` rather than
  killing the process mid-motion.
- **A caller that does not want to wait just ignores the id.** That is what
  `voice/` does, and it is why a spoken reply never waits on a 1.5-second head
  turn.

**Starting a new motion supersedes the running one.** A fresh gesture beats a
stale one, which is what makes rapid conversational motion compose instead of
queueing up behind itself.

**The estop freezes rather than going limp.** Cutting torque on a
Stewart-platform head would let it droop under its own weight, so `stop()` pins
the commanded posture to the measured one and latches. Motion stays refused
until `clear_estop()`.

**The estop asserts its freeze, it does not merely request it.** Cancelling the
motion task unblocks the driver, but the vendor call it was waiting on runs on a
worker thread that cannot be interrupted, and the daemon behind that keeps
interpolating toward the target it was already given. So `stop()` re-commands
the freeze pose at 10 Hz for a second afterwards, to outlive any interpolation
already in flight. Cancellation for the same reason is cooperative: it lands at
the next beat of a gesture, or when the current interpolated move returns.

## Events

| Event | Fires when |
|---|---|
| `motion_started` / `motion_progress` / `motion_completed` | A move is accepted, reaches a checkpoint, and finishes (`succeeded` / `cancelled` / `failed`). |
| `motors_changed` | Torque was enabled or disabled. |
| `estop_changed` | The latch was set or released. |
| `link_health_changed` | Telemetry polling started or stopped succeeding. |
| `telemetry` | 1 Hz snapshot of both postures plus motor and estop state. |

Stream them with `controller.py watch`.

Telemetry is published at 1 Hz while the local snapshot refreshes at 10 Hz.
Reads answer from the fast snapshot without a round trip to the robot; the slow
publication is what keeps a fleet of these from drowning the broker in pose
updates nobody asked for.

## What has been verified

**On the physical robot** (2026-07-30), through `reachy_target.py`, which is the
layer that touches the hardware and is unchanged since:

- daemon spawned, serial bus attached, real telemetry read back from all 9 DOFs
- `home` moved the robot to neutral
- the full 10-step `demo` ran start to finish: yaw sweep, pitch, nod x2,
  shake x2, antennas, an 0.8 rad body rotation, home
- clamping on metal: `head_z = -0.09` landed as `-0.05`
- clean process exit after every command

**On the stub**, hosted (single process, no messaging):

- the manifest builds: 17 functions, 7 events, 2 periodic routines, with
  `direction` / `safety` / `motion` labels on every function
- clamping at the owner: `head_pitch=99` landed as `0.45`, `head_yaw=99` as
  `0.9`, `set_dof(head_z, -0.09)` reported `applied=-0.05, clamped=true`
- motions: ack with a `motion_id`, streaming progress (`beat 1/2`, `2/2`),
  terminal `motion_completed`
- **estop during a move**: `stop()` answered in 2 ms while a four-second nod was
  running, the nod ended `cancelled`, a subsequent `home` was refused with the
  estop message, and `clear_estop()` released it
- supersede: a `shake` started mid-`nod` cancelled the nod and succeeded itself

**On the stub, cross-process over a real zenoh peer mesh** (`serve` in one
process, `--attach` in another, no broker and no server anywhere):

- discovery by presence announcement, then `slots`, `nod` and the full 10-step
  `demo` driven end to end
- per-beat progress delivered across the transport
- the whole estop cycle: `stop`, a refused `nod` carrying the estop message,
  `clear-estop`, and a working `home` afterwards
- a third process running `watch` saw all of it: `motion_started`,
  `motion_progress`, `motion_completed` and the 1 Hz `telemetry`
- clean SIGINT shutdown, ending in `Driver disconnected`

**Not verified:**

- **this driver against the physical robot.** The hardware layer under it is
  unchanged and was proven on metal, but the Device Connect driver on top of it
  has only been run against the stub. The estop's assert-the-freeze behaviour in
  particular is the one piece that cannot be judged without the motors: whether
  the vendor daemon lets a fresh `set_target` override an interpolation it is
  already running is not documented, which is exactly why the driver re-commands
  rather than trusting one write.
- NATS and MQTT backends, and cross-host zenoh. Only same-host zenoh was run.
- that `antenna_left` physically drives the left antenna. The ordering is taken
  from two agreeing vendor sources (below) but was never confirmed by eye. Watch
  one antenna move and check.

## Gotchas worth knowing

### The robot

- **The vendor's antenna array is `[right, left]`, not `[left, right]`.**
  `io/protocol.py::SetAntennasCmd` documents "[right, left]" and
  `hardware_config.yaml` lists `right_antenna` first. Getting this backwards
  silently swaps the antennas and is invisible in any single-antenna test, so
  the ordering is isolated in `ANTENNA_WIRE_ORDER` rather than repeated at each
  call site.
- **Head joint index 0 is body rotation, not index 6.** The 7-long array from
  `get_current_joint_positions()` is built as `[body_yaw] + stewart[6]`.
- **`spawn_daemon=True` shells out to `reachy-mini-daemon` by bare name**, so it
  is only found when the venv's `bin/` is on PATH. Running
  `.venv/bin/python controller.py` without activating the venv leaves it off;
  `reachy_target._ensure_daemon_on_path()` fixes that.
- **`ReachyMini` starts non-daemon threads.** If they are not stopped the
  interpreter never exits and a CLI that has already printed its result looks
  like it hung. `ReachyMiniTarget.close()` calls the vendor's `__exit__`
  (media close + client disconnect) to avoid it.
- **`set_target` after `enable_motors`, never before.** `enable_motors()` pins
  every target to the present pose before flipping torque on, so a target set
  first is silently overwritten. `ReachyMiniTarget.connect()` orders it
  correctly and seeds the commanded posture from the measurement, so connecting
  never makes the robot jump.
- **Each hosted command ends with the head going limp.** `close()` returns the
  robot to neutral and then drops torque, so the head droops to its rest pose
  (z about -0.043) between one-shot commands. That is the safe default for an
  unsupervised robot; use `serve` if you want it to stay energised across many
  commands.
- **The vendor daemon is a separate process** that outlives the controller and
  keeps the serial bus. That is deliberate (the next command reuses it and
  starts faster), but it means one is probably still running now. Stop it with
  `pkill -f reachy-mini-daemon`. If a connect hangs or reports the port busy,
  look for a stray daemon before suspecting the cable.

### Device Connect

- **Discovery takes a few seconds, and asking too early lies to you.** A client
  runtime hands its driver a registry only part-way through `run()`, so calling
  `wait_for_device()` immediately raises "Registry not configured" -- which
  reads exactly like "the robot is missing". Both clients here wait for
  `driver.registry` to appear first. Peers then show up in about 4 seconds.
- **A driver with no `DeviceRuntime` raises on its first emit.** That is the
  whole reason `HostedSession` calls `set_event_callback()`: with no runtime
  attached there is nowhere for an event to go, and the driver says so rather
  than dropping it.
- **`@on()` with no arguments is how you subscribe to everything.** The filter
  this rig actually wants -- one known device, all of its events -- is not
  expressible as a broker subscription, so the client subscribes broadly and
  filters in the handler.
- **Backend and endpoints are named separately.** Device Connect takes
  `messaging_backend="zenoh"` plus `messaging_urls=["tcp/host:7447"]` rather
  than one URL. `controller.py::parse_broker` keeps a familiar `--broker` string
  on the command line and does the translation in one place.
- **Zenoh with no endpoint is the brokerless case.** Passing an endpoint pins an
  address for the same peer mesh; it does not turn it into a client/router
  session. Set listen endpoints with `--zenoh-listen`, which sets `ZENOH_LISTEN`
  for the adapter's config builder.
- **`asyncio.to_thread` cancellation frees the waiter, not the thread.** Which
  is the whole reason the estop re-commands its freeze pose. Do not assume a
  cancelled motion has stopped the motors.
- **A "where predicates are unavailable" warning at startup is harmless.** It
  means `cel-python` is not installed and this device will not evaluate
  broadcast `where` clauses. Nothing here uses them.
- Security is `DEVICE_CONNECT_ALLOW_INSECURE=true` (dev tier): no TLS, no device
  authentication. For a real deployment remove that default in `controller.py`
  and commission the device against a Device Connect server or portal.
