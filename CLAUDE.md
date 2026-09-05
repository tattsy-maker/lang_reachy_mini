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

## Language tutor (state as of 2026-09-04)

Product spec: `LANGUAGE_TUTOR_SPEC.md`. Tracker with per-task status and
dated logs: `TASKS.md`. Details and learnings per task: `progress/T*.md`.
T0–T10 are done; T11 (Faire hardening) is mid-rehearsal; T13, T14 and
T15 (the three family sessions' feedback: goals and presence; sight,
calm tracking and one visitor at a time over the cloud voice; identity
for the whole session, one greeting, the lag line) are built on the
simulated path. Read `progress/T15.md` first for what the last
session's log showed and the protocol to run at the next one.

### Start the booth

```bash
./start_booth.sh                       # cloud voice, hands-free visitor loop
```

Knobs: `BOOTH_SPEECH` (`cloud` default since T14.3, `local` for
Claude), `BOOTH_ABSENT_SECS` (walk-away timer, "still there?" at two
thirds), `BOOTH_ATTRACT_SECS` (idle dance; 0 = off), `BOOTH_PERSONA`
(`booth` default, `plain` for none), `BOOTH_MIC_DEVICE` (preferred USB
mic, default `USB Composite Device`; when it is not plugged in the
robot's own mic is used -- see the two-mics trap below). Face tracking,
voice prints and
the `look` tool are on by default with a camera. Visitors swap without
anyone touching the keyboard: a walk-away or a changed voice ends the
session and the next face starts a fresh one (fresh Gemini history).

The family chose the **Gemini Live voice** at rehearsal. Before T14.3,
cloud mode was one visitor per launch; the hand-started form is still
useful for a quick check without the session loop:

```bash
.venv/bin/python controller.py serve --zenoh-listen tcp/0.0.0.0:7447 >> serve.log 2>&1 &
voice/.venv/bin/python voice/agent.py --broker zenoh://127.0.0.1:7447 \
    --speech cloud --face-source 0 --audio-device "Reachy Mini Audio" >> voice/run.log 2>&1 &
```

Needs `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `voice/.env`. Stand in
front of the camera during the first ~5 s after launch or it starts in
stranger mode. The robot greets first. Stop with `pkill -INT -f
voice/agent.py`, then serve.

### Agent flags added by the tutor work

`--learner NAME --learners-root DIR` (tutor one learner) ·
`--face-source SRC` (camera index / video / image dir) ·
`--session --stable-secs S --absent-secs S` (booth loop, local mode) ·
`--speech local|cloud`, `--gemini-model`, `--gemini-voice`,
`--no-web-search` (cloud mode gives Gemini its native Google Search
grounding by default; a search shows in the log as `web search: N
sources`) ·
`--language ru|zh` (Piper; models in `voice/piper_voices/`, fetch command
in `voice/piper_tts.py`) · `--native-language CODE` (T16, local speech:
the language the student is taught in and the voice for untagged text;
default the `--learner` profile's `native_language`, else `en`; the booth
script passes `BOOTH_NATIVE_LANGUAGE`) · `--deaf` (never open the mic — **always** with
`--say` scripted runs) · `--persona booth|plain` (T13.5 quips + the
wishlist question; the booth script passes booth) · `--no-track` (face
tracking is on whenever there is a camera and a robot, T13.3) ·
`--attract-secs S` (session mode: idle dance after S s with nobody in
frame, T13.4) · `--wishes-file PATH` · `--no-voice-id` (voice prints are
on in tutor mode, T13.9) · `--mic-device NAME[,NAME]` (preferred mic
substrings tried in order, default the USB desk mic, falling back to the
`--audio-device` mic; 2026-09-04) · `--voice-source WAV` (testing: hear this file
as the visitor at each `--say`). Tools the model can call besides
motion: `save_session_notes`, `update_learner_level`, `set_learner_goal`,
`set_target_language`, `set_native_language` (T16: the language
explanations are given in), `forget_me`, `enroll_new_learner`,
`confirm_identity`, `set_volume`, `perform` (dances/emotions/spin from
`moves.py`), `record_wish`.

### Tests and data

`tests/run.sh [t0..t12]` (own venv; markers skip with a printed reason
when hardware/keys/models are absent). `tests/reports/` holds the
measured gates. Real learner data is `learners/` (gitignored); the booth
script wipes guests on shutdown, or `python tutor/wipe_guests.py`.

### Traps the tutor work hit (in addition to the list above)

- **Groups:** `video` is needed for the camera (`sudo usermod -aG video
  $USER`, done 2026-09-02). A shell that predates the usermod must wrap
  camera commands in `sg video -c "..."`. `/dev/video1` is the camera's
  metadata node; never read frames from it.
- **Gemini Live drops mic audio until it has an initial context.** Every
  `--say` test provided one; a live session did not, and the robot sat
  silent. The agent now kicks the conversation off itself in cloud mode.
  Related: after the first turn, text injected through the aggregator is
  ignored by Gemini; later `--say` turns use the service's own injection
  path. And Gemini needs an explicit "actually call the tool" line that
  Claude never did.
- **In tutor mode the voice must not follow Whisper's language guess.**
  Room noise detected as Russian once switched the voice to Piper's
  Russian and English came out as gibberish. The model now chooses voices
  via `[es]...[/es]` span tags; untagged text is English.
- **Append-only logs lie to `grep`.** `serve.log` and `voice/run.log`
  contain every previous run's "ready" line; read from this run's offset
  (the booth script does).
- **A `&` job from a tool shell dies with the shell.** Launch long-lived
  processes detached (the harness's background mode, or `nohup`/`setsid`).
- **`pkill -f "controller.py serve"` from a tool shell kills the shell
  too** — the pattern matches the `bash -c` wrapper's own command line,
  so the command dies silently with exit 1 after the pkill. Use a
  bracket pattern that cannot match itself: `pkill -INT -f
  "[c]ontroller.py serve"`, `pkill -INT -f "[v]oice/agent.py"`.
- **Keep every booth run's logs.** After a session, copy this run's
  slice of `voice/run.log` and `serve.log` into `booth/logs/<date>_<name>/`
  (gitignored). The append-only logs are the only record of what
  happened; the 2026-09-03 session lives there.
- **Two mics, two sample rates.** The USB desk mic (`USB Composite
  Device`, Jieli, card 2) only opens at 48 kHz; the robot's own mic only
  at 16 kHz -- PortAudio answers `Invalid sample rate` to the other. The
  agent picks the USB mic when it is plugged in, else the robot's, opens
  it at its own rate and resamples to the pipeline's 16 kHz
  (`ResamplingAudioInput` in `voice/agent.py`). The speaker is always
  the robot's. `grep 'audio: mic' voice/run.log` says which mic a run
  got; `--input-device N` forces one. Both paths measured 2026-09-04.
- **Speaker volume:** the mixer default is quiet; the booth script and
  `set_volume` use `amixer -c <card> sset PCM,0 N%`.
- **One camera, one opener.** `/dev/video0` cannot be streamed by two
  `VideoCapture`s. Anything that runs for the whole session (the session
  watcher, the face tracker) reads from the shared `FrameHub`
  (`face/camera.py`), and enrollment takes its snapshots from the hub
  too. Do not open `Camera(0)` yourself while the agent is up.
- **Identity is the face's call for the whole session (T15).** The
  runner re-embeds the largest face every 2 s while a session is on and
  compares it with the face that started it. The same face keeps the
  session whatever the voice print says (`voice says someone else, but
  X's face was seen`); a different face for 3 s is a `face swap`
  (goodbye + notes, newcomer greeted); the voice ends a session only
  with nobody in frame. `confirm_identity` checks the current face
  before accepting a "yes". If a session ends "for no reason", read the
  `session:` lines before touching thresholds.
- **Gemini answers the seed unless told not to.** pipecat sends each new
  connection's context (the system prompt as a user turn) with
  `turn_complete=True` by default; that was the greeting to an empty
  chair and the extra greetings at every walk-up. Session mode passes
  `inference_on_context_initialization=False`; the walk-up cue is the
  only greeting. A hand-started cloud run (no `--session`) keeps the
  default so the robot still greets first.
- **A recorded move owns the body (T15.9).** While `play_move` runs the
  driver refuses every nudge (`accepted=False`, "a recorded move is
  playing"); only another move, `home`, `sleep`, `wake_up` or
  `cancel_motion` interrupts it. The agent holds embodiment and the
  motion tools for the move's length. Before this, the talking sway cut
  every dance to about a second and a head turn 8 ms after a cheer was
  a jerk that nearly toppled the robot (2026-09-04).
- **`goto` moves only the joints it names.** `reachy_target.goto`
  passes None for any group (head, antennas, body_yaw) the caller did
  not mention; the vendor re-plans every group it is given from the
  present pose, so passing the held body_yaw with each head nudge
  twitched the base at ~2 Hz. `tests/t15/probe_body_twitch.py` measures
  it on the metal.
- **The base twitching by itself after a dance** is the body servo
  limit-cycling (±0.5°, 3.5 Hz) around a target it cannot reach once a
  recorded move stops streaming; commanded values stay flat, so the
  agent log shows nothing. `play_move` now ends with a settling goto
  (T15.11). To check live without disturbing a session, poll
  `report_status` over zenoh and compare measured vs commanded
  body_yaw; if measured wobbles and commanded is flat, it is the servo.
- **What did it see?** Every `look` frame is saved under
  `booth/logs/looks/<date>/` (the `look:` log line names the file), and
  every mid-session face check logs `session: face check: same|unsure|
  other (score …)`. Open the jpg before debating whether the model
  hallucinated (2026-09-04, "someone wearing glasses").
- **Lag has a number now.** `grep "turn: first sound" voice/run.log`
  gives visitor-stop → first sound per reply (cloud: from the voice
  collector's energy gate, so ±0.8 s). Measure before tuning.
- **Who owns which joint.** The face tracker owns `head_yaw`/`body_yaw`;
  embodiment owns pitch and antennas (the tracker only biases pitch).
  A tool call that turns the head/body, `perform`, or `reset_pose`
  suspends the tracker and it re-reads the measured pose on resume.
  Keep that split if you add motion (docstrings in `voice/tracking.py`
  and `voice/embodiment.py`).
- **Google Search grounding on Gemini Live needs a billing-enabled key.**
  With `{"google_search": {}}` in the Live setup, the current key gets
  close code 1011 "You exceeded your current quota" on connect (the
  same key converses fine without the tool), and pipecat turns that
  into a silent, dead session. The agent probes the key at startup and
  logs `web search: unavailable on this key (...)`, then runs without
  it; the T8 grounding test skips with that reason. Enable billing on
  the AI Studio project to get it (2026-09-03).
- **Voice prints run on the GPU, unlike faces.** SpeechBrain ECAPA in
  `voice/.venv` (torch is the cu130 build): 5 ms per 3 s clip on CUDA,
  40 ms on CPU, 5 s model load (paid in warmup). `voice/verify_voiceid.py`
  is the gate; thresholds 0.60/0.45 were measured on synthetic Kokoro
  voices — re-run it on the family's real recordings before trusting a
  challenge on a human. Gemini Live emits no user-speaking frames, so
  the collector gates on audio energy, not turn events.
- **English is a target language too (T16).** A profile has
  `native_language` (default `en`) next to `target_language`; the
  briefing explains in the native one and practises the target. A
  Russian speaker learning English is `ru`/`en`. Cloud mode needs no
  flag; local mode's untagged voice is one language per launch
  (`--native-language`). `grep "taught in" voice/run.log` shows what a
  session got.
- **Recorded moves need their datasets on disk.** `moves.py --cached`
  says whether the two Pollen HuggingFace libraries are present;
  `--preload` fetches them. The booth preflight does this with a 60 s
  cap; without them `perform` only has `spin` and `wiggle`.

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
