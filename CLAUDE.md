# CLAUDE.md

Guidance for Claude Code when starting and stopping this app. `README.md`
and `docs/driver.md` explain what the app *is* (the control surface, the
functions, the design);
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
*Update 2026-09-23:* it no longer exits. The connect error is swallowed and
`serve` stays up with no robot: `[reachy] serving` printed, never `Driver
connected`, and clients time out after 20s. `start_booth.sh` now starts
`reachy-mini-daemon` itself, waits for `/api/daemon/status` to say the
backend is ready, and only then starts `serve`. By hand: start the daemon
first, or run `serve` a second time. `[reachy] serving` is printed before the
driver connects, so it is not a readiness signal. Wait for `Subscribed to
commands on device-connect.lab.reachy-mini-1.cmd`.

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

**The serial port in `docs/driver.md` is not stable.** It has already changed once.
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
see "How motion works" in `docs/driver.md`. If you are scripting against the device,
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

## Language tutor (state as of 2026-09-05)

Product spec: `LANGUAGE_TUTOR_SPEC.md`. Tracker with per-task status and
dated logs: `TASKS.md`. Details and learnings per task: `progress/T*.md`.
T0–T10 are done; T11 (Faire hardening) is mid-rehearsal; T13, T14 and
T15 (the three family sessions' feedback: goals and presence; sight,
calm tracking and one visitor at a time over the cloud voice; identity
for the whole session, one greeting, the lag line) and T16 (any native
language) are built on the simulated path. T17 is the fourth session's
feedback (2026-09-05: let them finish, trust a confirmed identity,
bystanders, one language policy, a real intake and closing), built on
the simulated path the same day. Read `progress/T17.md` first for what
the last session's log showed and the protocol to run at the next one.

### Start the booth

```bash
./start_booth.sh                       # cloud voice, hands-free visitor loop
```

Knobs: `BOOTH_SPEECH` (`cloud` default since T14.3, `local` for
Claude), `BOOTH_ABSENT_SECS` (walk-away timer, "still there?" at two
thirds), `BOOTH_ATTRACT_SECS` (idle dance; 0 = off), `BOOTH_PERSONA`
(`booth` default, `plain` for none), `BOOTH_MIC_DEVICE` (preferred USB
mic, default `USB Composite Device`; when it is not plugged in the
robot's own mic is used -- see the two-mics trap below),
`BOOTH_TURN_PATIENCE_MS` (T17.1: silence before Gemini takes the turn,
booth default 1200 since T19.8, agent default 1800; 0 = Gemini's
default), `BOOTH_TURN_ONSET_MS` (T19.8: speech before a turn starts,
default 100), `BOOTH_CALL_OUT_SECS` (T19.9: invite an onlooker over,
at most this often, default 40; 0 = off), `BOOTH_SPEECH_RATE` (T17.10: play
replies slower, pitch kept; default 1.0 = off), `BOOTH_LOUDNESS_DB`
(T18.1: soft-clip drive; default 0 = off since 2026-09-25 evening, 12
= about 8 dB louder on the robot's own speaker), `BOOTH_SPEAKER_DEVICE`
(`auto` default: a USB-to-jack adapter for a bigger speaker when one is
plugged in, else the robot's; `''` = always the robot's),
`BOOTH_ONBOARDING` (T18.3: `quick` default, `full` = the T17.4
interview), `BOOTH_ATTRACT_EVERY` (T18.4: seconds from one idle move's
start to the next, default 12; `BOOTH_ATTRACT_SECS` now defaults to 5),
`BOOTH_IDLE_GLANCE_SECS` (T18.5: head glances between dances, default
4). `BOOTH_ABSENT_SECS` defaults to 20 since 2026-09-25 (was 60). Face tracking,
voice prints and
the `look` tool are on by default with a camera. Visitors swap without
anyone touching the keyboard: a walk-away or a changed voice ends the
session and the next face starts a fresh one (fresh Gemini history).

**Unattended (since 2026-09-23): the booth is a systemd user service**,
`reachy-booth`, enabled at boot (user lingering is on, so it runs with
nobody logged in). The unit is `booth/reachy-booth.service`, copied to
`~/.config/systemd/user/`. It runs `start_booth.sh` with `BOOTH_SERVICE=1`,
which waits for the robot's USB (serial by-id + its sound card), the
camera (followed by by-id, so a replug that renumbers it is fine) and
Gemini's host before starting. Once up it exits on the first fault, and
systemd restarts the whole stack cold after 5 s, with no restart limit.
Faults: robot USB gone, serve or daemon not ready, agent exited, agent
heartbeat (`voice/.heartbeat`, 5 s) over 60 s old, and the agent's own
watchdog: Gemini not connected for 45 s → exit 75. Fault restarts keep
guest profiles; a real stop (`systemctl --user stop`, a reboot) wipes them.

```bash
systemctl --user status reachy-booth         # is it up
journalctl --user -u reachy-booth -f          # why it restarted ([FAIL] lines)
systemctl --user stop reachy-booth            # clean stop, robot to neutral
systemctl --user restart reachy-booth         # after changing code
systemctl --user disable --now reachy-booth   # back to hand-started runs
```

**Do not hand-start `start_booth.sh` or `serve` while the service is up**
(one owner of the robot). Measured 2026-09-23: agent killed → live again in
~35 s; agent frozen → caught at 62 s; Gemini unreachable → watchdog at 45 s;
clean stop in 12 s. A USB unplug/replug and a reboot were not yet tried on
the metal.

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
as the visitor at each `--say`) · `--turn-patience-ms N` (T17.1, cloud:
Gemini's end-of-turn silence, default 1800) · `--speech-rate R` (T17.10:
WSOLA time stretch of the reply audio, 1.0 = off) · `--loudness-db D`
(T18.1, 0 = off) · `--onboarding quick|full` (T18.3, default quick) ·
`--attract-every S` (T18.4). Tools the model can
call besides motion: `start_lesson` (T18.3: a guest lesson for a
newcomer, nothing stored), `save_session_notes`, `update_learner_level`,
`set_learner_goal`, `set_target_language`, `set_native_language` (T16:
the language explanations are given in), `forget_me`, `intake_answer`
and `enroll_new_learner` (T17.4: the interview, one field per call;
enrollment refuses until all nine are in), `set_session_plan` (minutes
and today's focus; the runner injects a wrap-up cue at 80 %), `quiz`
(T17.6: a spoken gap-fill / multiple choice script), `confirm_identity`,
`set_volume`, `perform` (dances/emotions/spin from `moves.py`),
`record_wish` (kind `improve` or `wish`; answers land in
`booth/feedback.md`, and the runner catches the answer from the
transcript even when the tool is not called).

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
- **The voice print asks the face before it asks the visitor (T17.2).**
  A mismatch while the session's face was seen in the last 5 s is
  logged as `voice: X does not sound like themselves ... but their face
  is in front of the robot; not asking`; two of those, or a
  `confirm_identity` the face accepted, log `voice: trusting this
  session's identity from here on` and the voice is silent for the rest
  of the session. On 2026-09-05 one visitor was challenged five times
  with her face at 0.6-0.95. A confirmed visit also folds what was heard
  and seen into the stored prints (`stored print updated`, `stored face
  updated`).
- **A bystander is not a swap (T17.3).** The runner embeds every face
  in the frame (`recognize.analyze_all`) and the session's face anywhere
  in it is `same`, whichever is largest; only a frame with no matching
  face counts toward `swap_secs`. Before this a second face leaning in
  was "the largest" for 2-5 s about once a minute (34 near-zero `other`
  verdicts on 2026-09-05).
- **`enroll_new_learner` refuses an unfinished interview (T17.4).**
  `tutor: enrollment refused, N answers missing (...)` in the log means
  the model tried to enroll from one sentence (the 2026-09-05 mother:
  target, level, goal and native language all invented). The interview
  goes through `intake_answer`; `grep "tutor: intake"` shows each
  answer. Scripted tests must volunteer all nine answers or script the
  seven questions in order (`tests/t13/test_goals.py::INTERVIEW`).
- **Turn patience costs lag, on purpose (T17.1).** `--turn-patience-ms`
  (default 1800) is added to every reply's `turn: first sound` number;
  the 2026-09-05 median was 2.2 s with Gemini's default. Do not "fix"
  the lag by lowering it without a session's worth of `heard:` lines
  showing nobody was cut off. The booth went to 1200 on 2026-09-25
  (T19.8) on the Faire's own evidence: median 3.1 s, p90 5.8 s, and
  visitors left waiting after saying a word back.
- **`prefix_padding_ms` is not a pre-roll (T19.8).** In Gemini Live it is
  how much speech must be detected before a start is committed. It was
  300 under a comment saying the opposite, and a quick "hola" went
  unheard ("they had to repeat two or three times"). Now 100
  (`--turn-onset-ms`); raise it only if hall noise starts turns.
- **Read `heard:` before theorising (T17.7).** The visitor's words are
  now INFO lines next to `said:`; older logs have them only as pipecat
  DEBUG `[Transcription:user]` lines. `tests/t17/judge_corrections.py
  LOG` counts mistakes vs corrections per session (Claude Haiku as the
  judge); the 2026-09-05 baseline was 0-20 % corrected.
- **`pace:` has a number (T17.10).** One `pace: N words in S s (W wpm)`
  line per reply in cloud mode. Try the prompt first; if beginners still
  hear it too fast, `BOOTH_SPEECH_RATE=0.85` stretches the audio
  (pitch kept) and the `pace:` line shows the result.
- **Not every face is a visitor (2026-09-23).** A bag of clutter behind
  the table was a steady face to insightface (score 0.72, 4.4 % of the
  frame wide): it started sessions in an empty room, then held the
  robot's gaze. `recognize.visitors()` now drops faces under 6 % of the
  frame width (~1.3 m) or score 0.6 before presence, sessions and
  tracking see them. Size is the gate, not score: the fixture clip's real
  face scores 0.71. `grep "not a visitor" voice/run.log` shows what was
  turned away (every 30 s at most), and each session start saves
  `booth/logs/looks/<date>/session_HHMMSS.jpg` with the boxes drawn in.
  The tracker was retuned the same day to keep the visitor centred (gain
  0.65, 4° dead band, 12° steps; the 0.8 s interval stays).
- **`head_yaw` is the gaze in the world frame (2026-09-23).** The vendor
  solves the head pose in the world and turns the base underneath it
  (keeping them within 65° of each other), so `body_yaw` alone never
  moves the camera. The tracker's old "hand the angle to the body and
  recentre the head to 0" swung the gaze back every ~4.5 s for a whole
  lesson (50 moves a minute, body estimate at its 160° clamp, blurred
  `look` frames). The handoff now brings the base under the head and
  leaves the gaze alone; `tests/t13/test_tracking.py` simulates the real
  geometry. `turn_body` still turns only the base, so it does not change
  what the camera sees.
- **A learner met mid-session gets their own briefing (2026-09-23).** The
  standing instruction used to stay the stranger's ("greet them warmly in
  English") for the whole visit after enrollment, and a Russian-explained
  Italian lesson came out in English. Now `enroll_new_learner` and
  `confirm_identity` say one line, then the runner reconnects Gemini with
  `build_briefing(...)` and cues the lesson (`session: enrolled: fresh
  Gemini session with X's own briefing`). This needs the embodiment's
  speaking state, which is now tracked even with `--no-robot`; before
  that, `_wait_quiet` never waited in the live tests.
- **During a lesson `perform` plays once**, a few seconds (a 27 s and a
  32 s dance inside a 5-minute lesson, 2026-09-23). `look` results now
  say to describe only the picture and never to announce a new person
  (it said "I see someone else now!" of a frame showing only the
  visitor, and "the Maker Faire booth" of a room at home).
- **Stopping must never drop the head (2026-09-24).** `serve`'s `close()`
  used to go to neutral and cut torque, so the head fell; the daemon's own
  shutdown (`reset_to_sleep`) then lifted it back up and lowered it. Now
  `close()` plays the vendor sleep move, cuts torque at rest and logs
  `[reachy] asleep at rest, torque off`; seeing that line, the service
  stops the daemon with `POST /api/daemon/stop?goto_sleep=false`, so there
  is one descent. Without it (serve crashed) the daemon's sleep still runs.
- **A beginner is taught in their own language, whatever `explain_in`
  says (2026-09-24).** The model recorded `explain_in=both` from a Hindi
  "very much beginner" who had only answered "English", and every line
  came in Hindi first. `explain_policy()` in `voice/tutor_mode.py` forces
  `native` for beginners; the beginner briefing speaks the native language
  for everything and teaches one word or phrase at a time.
- **Recorded moves need their datasets on disk.** `moves.py --cached`
  says whether the two Pollen HuggingFace libraries are present;
  `--preload` fetches them. The booth preflight does this with a 60 s
  cap; without them `perform` only has `spin` and `wiggle`.
- **The mixer is already at the top (2026-09-25).** Both `PCM` controls
  are 60/60 = 0 dB, and Gemini's audio peaks at 0 dBFS, so amixer and
  plain gain have nothing left. Louder comes from `--loudness-db` (a
  soft clip, `voice/loudness.py` has the measurements); if the voice
  sounds harsh, lower `BOOTH_LOUDNESS_DB` (9 is about +6.5 dB, 6 about
  +4.5, on Gemini 3.1's audio). `grep "loudness:" voice/run.log` shows
  what a run used. Since the evening of 2026-09-25 the booth plays with
  no software gain and gets its volume from a bigger speaker instead.
- **Another speaker costs barge-in (2026-09-25).** `--speaker-device`
  (booth: `auto`) plays through any other USB sound card, e.g. a
  USB-to-jack adapter; `grep "audio: speaker"` says which, and flags one
  that is not the robot's. The robot's mic hears the robot at -55 to -68
  dBFS only because its XVF3800 board cancels the echo of what *it*
  plays; a separate speaker's sound reaches the mic at full level, the
  echo gate raises its threshold to match, and visitors can rarely talk
  over the robot. Set volume and `set_volume` act on whichever card
  plays (PCM/Speaker/Headphone/Master, whichever it has). The AB13X
  jack adapter plays only stereo at 8 or 48 kHz (`cat
  /proc/asound/cardN/stream0`): `ResamplingAudioOutput` opens it at 48
  kHz in stereo with the voice on both channels (opened in mono,
  PortAudio silences the second channel). Measured 2026-09-25 with the
  desk mic: the robot's voice from the big speaker reads -3 dBFS. The
  echo gate then opened on the robot's own voice half a second into
  every reply (its echo estimate collapsed as replies got cut short),
  so on any speaker that is not the robot's the agent now turns
  barge-in off (`barge-in: off -- the speaker is not the robot's own`)
  and keeps the mic muted 0.25 s after each reply
  (`barge_in.make_tail_strategy`).
- **A language missing from the prompt gets refused (2026-09-25).** The
  base prompt says to refuse any language it does not list; cloud mode
  listed eight plus "most other languages", and Gemini told a visitor
  "I do not speak Arabic yet" while the tool had accepted `ar`. Cloud
  mode now lists Gemini Live's 99 by name (`cloud_language_names()`).
- **Quick start stores nothing (T18.3).** `tutor: quick lesson: ar,
  beginner, taught in en (a guest, nothing stored)` is a newcomer's
  lesson; there is no profile, so `save_session_notes` answers "a
  guest" and a walk-away saves nothing. Only "remember me" enrolls,
  after `intake_answer` name + goal. `--onboarding full` brings back
  the T17.4 interview.
- **Idle is mostly dancing (T18.4/T18.5).** With nobody in frame the
  robot dances ~85 % of the time: first move 5 s after the frame
  empties, the next as soon as it ends (12 s start to start), random
  head glances (`--idle-glance-secs`) in the gaps. A recorded pass
  costs ~1.7 s beyond its clip (1 s to the first frame, 0.6 s base
  settle, `moves.PASS_OVERHEAD_SECS`); a glance sent during a move is
  refused (`accepted=False, a recorded move is playing`), harmless. A
  visitor who walks up mid-move gets `attractor: a visitor arrived
  mid-X; stopping it` and the robot goes home before the greeting.
  After a visitor leaves, the 20 s walk-away timer ("still there?" at
  13 s) is the only still time. Only smooth moves idle: the sharp clips
  (electric, stumble, chicken, grid snap) and 1 s `home` drops scared a
  girl on 2026-09-25; `home` is 2 s now.
- **A guest lesson survives a new face (2026-09-25).** At the Faire the
  T15 face swap ended a quick-start session five times in three minutes
  with one group of kids (a passer-by starts it, the real visitor is
  "someone else" 4 s later): each end was a head drop and a new "Hello,
  I am Reachy". With nobody enrolled or being confirmed, a different
  face now just becomes the session's face (`session: someone new in
  front of the robot; a guest lesson, so carrying on`). An enrolled
  learner still gets the swap.
- **Voice presence is bounded in the booth (2026-09-25).** Hall chatter
  kept empty sessions alive and re-asked "¿Sigues ahí?" three times in
  70 s. `--voice-hold-secs 45`: voice (the energy gate, and since the
  same day every `heard:` transcript) keeps a session alive only 45 s
  past the last face, and only a face re-arms "still there?". Without
  the flag, T13.2's home behaviour stands.
- **Barge-in (`--barge-in`, booth on) needs a mic that can tell the
  robot from the visitor.** The robot's own mic can (its board cancels
  the robot's echo: -55 to -68 dBFS, barge-ins seen working end to end
  on 2026-09-25, 17:43-18:08); the desk mic cannot. Cloud mode swaps `AlwaysUserMuteStrategy`
  for `barge_in.EchoGatedUserMuteStrategy`: mid-reply the mic opens when
  it hears a voice `--barge-in-margin-db` (10) above the robot's own
  echo, and Gemini interrupts itself. With the desk mic's Auto Gain
  Control on, the echo read -4 to -8 dBFS -- as loud as anyone -- so it
  never triggered; AGC was turned off by hand at 13:47 on 2026-09-25
  (`sg audio -c "amixer -c 1 sset 'Auto Gain Control' off"`; the card
  number is the `USB Composite Device` one; this does not survive a
  replug). `grep "barge-in:" voice/run.log` gives the echo level every
  ten replies and each interruption.
- **The vendor daemon's wake-up is off (T19.1).** Started with
  `--no-wake-up-on-start`: its wake ends in a 20-degree head snap in
  0.4 s. The agent's `wake_gently` rises over 3.5 s instead (`robot:
  awake` in the log). A daemon started by hand without the flag brings
  the snap back.
- **Recorded moves play at `MoveSpec.speed` (T19.7).** Fast clips are
  time-stretched to stay inside `moves.CALM`; `moves.py --measure` shows
  every clip's peak speeds as played and the speed CALM would give it.
  Adding a clip: measure it and set its speed; `tests/t19` checks every
  cached clip against CALM. `pass_secs`, not `seconds`, is how long a
  pass plays.
- **`look` is for when the visitor asks (T19.4).** Not to check who is
  there (face recognition does that) and never from an earlier frame. A
  `look:` line after "Hello" or "Close the door" means the prompt
  drifted again.
- **Call-outs (T19.9).** `attractor: someone looking from a few steps
  away ... calling out` is a face 1.3-3 m off while nobody is being
  served; Gemini invites them over. A still face (a poster) is called to
  once, and a bystander who watched a lesson is not new when it ends.

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
