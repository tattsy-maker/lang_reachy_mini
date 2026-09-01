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

## Language-tutor additions (2026-08-31)

The tutor work (see `LANGUAGE_TUTOR_SPEC.md`, tracker in `TASKS.md`,
per-task notes in `progress/`) added to the voice agent:

- `--learner NAME --learners-root DIR` — tutor mode for one learner;
  `--face-source SRC` — identify by face / conversational enrollment;
  `--session --stable-secs S --absent-secs S` — the full booth loop;
  `--deaf` — never open the mic (**always use with `--say` scripted runs**,
  room noise becomes phantom user turns otherwise);
  `--language ru|zh` — spoken by Piper (models in `voice/piper_voices/`,
  download command in `voice/piper_tts.py`'s docstring).
- Tests: `tests/run.sh [t0..t11]` (own venv, markers skip cleanly when
  hardware/keys/models are absent). Measurement reports live in
  `tests/reports/`.
- **The `video` group trap** (third of its kind after `dialout`/`audio`):
  the Reachy camera at `/dev/video0` needs `sudo usermod -aG video altha`
  once, else every open fails. `/dev/video1` is the camera's metadata
  node — never read frames from it.
- Real learner data lives in `learners/` (gitignored, private by design);
  end-of-day guest wipe: `python tutor/wipe_guests.py`.

## First-time setup on a fresh clone

Neither `.venv/` exists until you build it -- both are gitignored, and so is
`voice/.env`. The commands below are what actually got a from-scratch clone
running on a headless Linux box (Ubuntu 24.04, aarch64, inside a Docker
container with `network_mode: host`); a Mac following the READMEs' own install
sections may not hit any of this.

```bash
python3 -m venv .venv
.venv/bin/pip install device-connect-edge reachy-mini

cd voice
python3 -m venv .venv
.venv/bin/pip install "pipecat-ai[anthropic,mlx-whisper,whisper,kokoro,local-smart-turn,silero,local]==1.6.0"
.venv/bin/pip install device-connect-edge
```

Pin `pipecat-ai` to the version the README actually names (1.6.0 at the time
of writing). Installing it unpinned pulls latest, which has already renamed an
import (`assert_given` -> `is_given` in `pipecat.services.settings`) that this
repo's `multilingual.py` depends on directly; the symptom is `ImportError:
cannot import name 'assert_given'` the moment `agent.py` starts.

**Linux build deps `reachy-mini` and `pyaudio` need but don't declare.**
Neither wheel exists for aarch64, so pip builds them from source, and each
needs headers the Mac instructions never mention:

```bash
sudo apt-get install -y libcairo2-dev libgirepository1.0-dev pkg-config \
    python3-dev portaudio19-dev
```

Without `libcairo2-dev` (+ `python3-dev` for the second failure it uncovers),
`reachy-mini`'s pull of PyGObject fails in meson with `Dependency "cairo" not
found`, then `Python dependency not found`. Without `portaudio19-dev`, pyaudio
fails with `fatal error: portaudio.h: No such file or directory`. `brew install
portaudio` in `voice/README.md` is the same requirement, just for the other OS.

**The user needs `dialout` and `audio` group membership, and won't have
either on a fresh account.** Without `dialout`, `serve` can't open
`/dev/ttyACM0`. Without `audio`, ALSA can't see any card at all --
`aplay -l` / `arecord -l` report "no soundcards found" even though
`/proc/asound/cards` lists them, which reads like a missing driver rather than
a permissions problem.

```bash
sudo usermod -aG dialout,audio "$USER"
```

`usermod` doesn't touch the current shell's group list. Either start a fresh
login shell, or wrap the one command that needs the new membership in
`sg <group> -c "..."` (stacks for two groups: `sg dialout -c "sg audio -c '...'"`,
or just run `serve` under `sg dialout` and the voice agent under `sg audio`
since each only needs the one).

**Plain `zenoh://` multicast discovery does not work in this container.**
`controller.py --attach zenoh://` (and the voice agent's default
`--broker zenoh://`) time out after 20s with "did not appear on zenoh://",
even with `serve` demonstrably up and healthy. This is the same failure mode
the macOS multicast note above describes, just on a different OS -- the fix is
identical: pin an address instead of debugging multicast.

```bash
.venv/bin/python controller.py serve --zenoh-listen tcp/0.0.0.0:7447
.venv/bin/python controller.py --attach zenoh://127.0.0.1:7447 slots
cd voice && .venv/bin/python agent.py --broker zenoh://127.0.0.1:7447
```

**`voice/.env` needs `ANTHROPIC_API_KEY` and there is no default.** Both
`agent.py --auth api-key` (the default) and `--auth oauth` are real options --
oauth needs the separate `ant` CLI (`ant auth login`), which is not installed
here. Pick one before trying to start the agent; there's no working default
that needs nothing from you.

**The MLX wheel on Linux ships without its own runtime library.** `pip install
mlx` succeeds and `import mlx_whisper` inside `agent.py` fails at first use
with `libmlx.so: cannot open shared object file`, not at import time -- the
failure only shows up once the voice agent tries to transcribe, which makes it
easy to mistake for an audio problem. The fix is installing the separate
backend package `mlx`'s own metadata names for your platform: `mlx-cuda-13`
(or `-12`, matching your CUDA toolkit major version) on a Linux box with an
NVIDIA GPU, `mlx-cpu` with none. Check with `nvcc --version` first, then:

```bash
.venv/bin/pip install "mlx-cuda-13==<same version as mlx>"
```

**Do not swap `MultilingualWhisperMLX` for `faster-whisper`/CTranslate2 on
aarch64 -- it silently runs on the CPU, not the GPU.** This looks like the
obviously-more-correct choice for an NVIDIA box (CTranslate2 *is* the
CUDA-native inference engine `faster-whisper` wraps), and pipecat's own
`WhisperSTTService` base class already uses it, so the swap looks like it's
removing an Apple-only dependency in favor of the vendor-native one. It is not:
CTranslate2's PyPI wheels ship CUDA support for `linux_x86_64` only.
`ctranslate2.get_cuda_device_count()` returns `0` on `linux_aarch64`, and
`WhisperModel(..., device="cuda")` raises `This CTranslate2 package was not
compiled with CUDA support`. Measured on this box: `large-v3-turbo-ct2` at
`compute_type="int8"` on the CPU took **8.8s to transcribe 3s of audio** --
unusable for a live conversation. MLX's CUDA backend (the `mlx-cuda-13`
package above) is, counterintuitively, the only one of the two that is
actually GPU-accelerated here, despite MLX's Apple-Silicon origins and
faster-whisper's CUDA-native reputation. Verify before trusting either
framework's reputation for a given platform: `mx.default_device()` should say
`gpu`, and `ctranslate2.get_cuda_device_count()` should be nonzero, before
building anything on top.

**The Reachy Mini speaker's ALSA volume defaults to a moderate, not full,
level.** `amixer -c <n> sget PCM,0` (where `<n>` is whatever
`/proc/asound/cards` lists "Reachy Mini Audio" as) came up around 62-67%
(-23dB/-20dB) on this unit, which sounds noticeably quiet next to a normal
speaking voice. This is a hardware/driver-level mixer setting independent of
Kokoro or pipecat -- raising it doesn't need the agent restarted, it takes
effect immediately:

```bash
sg audio -c "amixer -c <n> sset PCM,0 90%"
sg audio -c "amixer -c <n> sset PCM,1 90%"
```
