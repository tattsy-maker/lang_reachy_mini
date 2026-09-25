# Reachy, the language tutor

A [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) desk robot
that tutors languages out loud. Walk up and it turns to look at you,
recognizes your face, remembers what you practised last time and picks up
where you left off, in the language you are learning, explained in the one
you already speak. New people get a short spoken interview first: name,
target language, level, goal and how long they have.

Built by Tatiana, Yaroslav and Andrey Tsyplikhin for
[Maker Faire Bay Area 2026](https://makerfaire.com/maker/entry/reachy-mini-the-family-language-tutor-a-robot-that-knows-who-78898/),
and tested over several family sessions at home before that.

- **Languages.** Spanish, French, Italian, Portuguese, Russian, Mandarin and
  English as targets; explanations in any language the speech model speaks.
  Mixed-language sentences work in both directions ("how do you say *la
  biblioteca* in French?").
- **Two speech modes.** *Cloud* (the booth default): one Gemini Live
  speech-to-speech stream. *Local*: Silero VAD, smart-turn, Whisper (MLX),
  Claude for the tutor, Kokoro or Piper for the voice.
- **It knows who is there.** Face recognition (InsightFace) starts and ends
  sessions; a voice print (SpeechBrain ECAPA) notices when someone else
  starts talking. A bystander leaning in does not end a lesson.
- **It has a body.** It tracks the visitor's face, nods and sways while
  talking, can look at something on request (`look` sends a camera frame to
  the model), and dances for an idle crowd.
- **It remembers.** Per-learner profile and lesson notes on local disk,
  quizzes, a session plan with a wrap-up cue, and "forget me" on request.
- **It runs unattended.** One script starts the whole booth; a systemd user
  service restarts it cold on any fault.

## How it fits together

```
  Reachy Mini (USB)                       this computer
  ─────────────────   ┌───────────────────────────────────────────────────────┐
  9 servos  ◄──serial─┤ reachy-mini-daemon (vendor)                           │
                      │        ▲                                              │
                      │ controller.py serve  ── the robot as a Device Connect │
                      │        ▲               device (zenoh, brokerless)     │
                      │        │ RPCs + events                                │
  camera ────────────►│ voice/agent.py ── face/ (recognize, track)            │
  mic, speaker ◄─────►│        │          tutor/ (learner store, wipe)        │
                      └────────┼──────────────────────────────────────────────┘
                               ▼
                   Gemini Live (cloud mode)  or  Claude (local mode)
```

The agent never touches the serial port: it drives the robot over
[Arm Device Connect](https://github.com/arm/device-connect), so the robot
side and the speech side live in separate virtualenvs and could run on
separate machines. The driver layer is documented in
[docs/driver.md](docs/driver.md), the speech side in
[voice/README.md](voice/README.md).

## What you need

- A **Reachy Mini Lite** (the USB-tethered one) with its camera, mic and
  speaker. A USB desk mic is optional and preferred when plugged in.
- A computer to drive it. The booth runs on Ubuntu 24.04 (aarch64, NVIDIA
  GPU); the driver and the local voice were first built on a Mac, and the
  Linux-specific setup below is what differs.
- Python 3.12.
- An API key: **Google AI Studio** (`GOOGLE_API_KEY`) for cloud mode,
  **Anthropic** (`ANTHROPIC_API_KEY`) for local mode.

## Setup

```bash
git clone https://github.com/tattsy-maker/lang_reachy_mini
cd lang_reachy_mini

# the robot side
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# the voice agent, faces and voice prints
cd voice
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then put your key(s) in voice/.env
cd ..
```

On **Linux**, before the installs (`reachy-mini` and `pyaudio` build from
source on aarch64):

```bash
sudo apt-get install -y libcairo2-dev libgirepository1.0-dev pkg-config \
    python3-dev portaudio19-dev
sudo usermod -aG dialout,audio,video "$USER"   # serial port, sound cards, camera; log in again
```

With an NVIDIA GPU, local mode's Whisper also needs MLX's CUDA backend
(see the comment at the bottom of `voice/requirements.txt`). The face and
recorded-move models download on first use: InsightFace `buffalo_l`
(~280 MB, into `~/.insightface`) and Pollen's move libraries from Hugging
Face (`.venv/bin/python moves.py --preload`).

## Run it

**No robot yet?** The whole driver path against an in-memory robot:

```bash
.venv/bin/python controller.py --stub demo
```

**The booth** (cloud voice, hands-free visitor loop, clean shutdown with
Ctrl-C):

```bash
./start_booth.sh
```

It starts the vendor daemon, `serve` and the agent in order, runs a
preflight checklist, and prints `ready` after about 40 s. Knobs are
environment variables documented at the top of the script (`BOOTH_SPEECH`,
`BOOTH_ABSENT_SECS`, `BOOTH_TURN_PATIENCE_MS`, ...). To run it unattended at
boot, install [booth/reachy-booth.service](booth/reachy-booth.service) as a
systemd user unit (instructions in the file).

**By hand**, one piece at a time:

```bash
.venv/bin/python controller.py serve --zenoh-listen tcp/0.0.0.0:7447     # terminal 1
.venv/bin/python controller.py --attach zenoh://127.0.0.1:7447 nod       # terminal 2: poke it
voice/.venv/bin/python voice/agent.py --broker zenoh://127.0.0.1:7447 \
    --speech cloud --face-source 0 --audio-device "Reachy Mini Audio"    # terminal 2: talk
```

Stop with Ctrl-C (SIGINT), never `kill`: SIGINT is what puts the robot back
to rest and releases the mic. [CLAUDE.md](CLAUDE.md) is the full runbook,
including every trap the project hit, and is worth reading before
debugging anything.

## Privacy

The robot remembers people, so be deliberate about what it keeps.

- **What is stored:** first name, languages, level, goal, lesson notes, a
  face embedding and a voice embedding (numbers, not photos or recordings),
  in `learners/<id>/` on the local disk. Nothing in `learners/` is ever
  committed.
- **What leaves the machine:** in cloud mode, conversation audio and any
  `look` frame go to Google's Gemini Live API; in local mode, the text of
  the conversation and any `look` frame go to Anthropic, and no audio
  leaves the machine. Face and voice matching always run locally.
- **Guests are temporary.** Learners enrolled at the booth are `guest` tier
  and wiped when the booth stops (`tutor/wipe_guests.py`); anyone can say
  "forget me". [booth/SIGNAGE.md](booth/SIGNAGE.md) is the disclosure the
  booth displays.
- **Logs are personal data.** `serve.log`, `voice/run.log`, `booth/logs/`
  (transcripts and every `look` frame) and `booth/feedback.md` are all
  gitignored. Keep it that way.

## Tests

```bash
tests/run.sh          # everything
tests/run.sh t17      # one task's tests
```

Tests that need the robot, a sound card, big models or an API key skip with
a printed reason, so the suite is green on any machine. See
[tests/README.md](tests/README.md). The fixture faces are public-domain NASA
portraits and the fixture voices are synthetic
([tests/fixtures/README.md](tests/fixtures/README.md)).

## Repository map

| Path | What it is |
|---|---|
| `controller.py`, `reachy_driver.py`, `reachy_target.py`, `stub_target.py` | The robot as a Device Connect device, its CLI, and an in-memory stand-in. [docs/driver.md](docs/driver.md) |
| `moves.py` | Dances and emotions from Pollen's recorded-move libraries |
| `voice/` | The conversational agent: pipeline, tutor prompts, tools, sessions, tracking, voice prints |
| `face/` | Camera sharing, face detection and recognition |
| `tutor/` | The learner store and the guest wipe |
| `booth/` | Booth service unit, signage, the desk slide deck generator |
| `start_booth.sh` | One-command booth startup and supervisor |
| `tests/` | Per-task test suites, fixtures, measured reports |
| `LANGUAGE_TUTOR_SPEC.md` | The product spec, written for non-experts |
| `TASKS.md`, `progress/` | The build log: every task, what the family sessions showed, and what was changed because of it |

## Third-party models and licenses

This repository's code is MIT-licensed (see [LICENSE](LICENSE)). The models
it downloads at runtime have licenses of their own, so check them before
building anything commercial on top:

- **InsightFace `buffalo_l`** (face recognition): the pretrained models are
  for **non-commercial research use only**.
- **Kokoro** (local English/European voices): Apache-2.0.
- **Piper voices** (local Russian and Mandarin): licensed per voice; see each
  voice's model card.
- **SpeechBrain ECAPA** (voice prints) and **Silero VAD**: see their model
  cards.
- **Pollen's recorded-move libraries** on Hugging Face: see the dataset cards.
- **Gemini Live** and **Claude** are cloud APIs under Google's and
  Anthropic's terms.

## License

[MIT](LICENSE), © 2026 the Tsyplikhin family.
