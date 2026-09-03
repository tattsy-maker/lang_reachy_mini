# voice -- talk to the Reachy Mini

A low-latency spoken conversation with the robot. Everything except the language
model runs on this laptop, and the robot is reached over **Device Connect**
rather than driven directly.

```
  Reachy mic --> Silero VAD --> smart-turn v3 --> MLX Whisper --> Claude --> Kokoro --> Reachy speaker
                  (local)         (local)        (detects lang)   (cloud)   (matching
                                                                             voice)
                  (local)         (local)           (local)      (cloud)    (local)

                                                       | tool calls
                                                       v
                             Device Connect invoke --> controller.py serve --> motors
```

Only one hop leaves the machine: the language model. Speech recognition,
speech synthesis, voice activity detection and end-of-turn prediction are all
local, which is what keeps the loop tight.

## Why it talks to the robot over Device Connect

The voice agent never imports the vendor SDK and never opens the serial port. It
joins the fleet as a Device Connect device of its own, discovers the robot, and
invokes its functions. That buys three things:

- **No dependency collision.** pipecat pulls in torch, MLX, ONNX and a large
  transitive tree; the robot side pulls in the Pollen SDK. They live in separate
  venvs and never meet.
- **The robot can be elsewhere.** Point `--broker` at any reachable peer or
  broker and the agent drives a robot on another machine with no code change.
- **Any device with these functions works.** Nothing here is Reachy-specific
  below the tool layer.

## Files

| File | What it is |
|---|---|
| `agent.py` | The pipeline, the model's tools, and the CLI |
| `robot_link.py` | Device Connect client: discovery, invoke, fire-and-forget motion |
| `embodiment.py` | Maps conversation state to body language (see below) |
| `multilingual.py` | Language detection and voice switching (see below) |
| `.venv/` | pipecat 1.6 + MLX Whisper + Kokoro + smart-turn + `device-connect-edge` |

## Install

Already done in `voice/.venv`. To rebuild:

```bash
brew install portaudio                 # PyAudio has no wheel; it needs this first
python3 -m venv .venv
CFLAGS="-I$(brew --prefix portaudio)/include" LDFLAGS="-L$(brew --prefix portaudio)/lib" \
  .venv/bin/pip install "pipecat-ai[anthropic,mlx-whisper,whisper,kokoro,local-smart-turn,silero,local]"
.venv/bin/pip install device-connect-edge
```

The Whisper, Kokoro and smart-turn models download themselves on first run
(a few hundred MB, cached under `~/.cache`).

## Run

Two terminals. Zenoh is the default and needs no broker at all:

```bash
# 1. the robot (owns the USB link, stays up)
cd .. && .venv/bin/python controller.py serve

# 2. the voice agent
cp .env.example .env        # paste an API key in, then:
.venv/bin/python agent.py
```

Or over NATS, which needs a broker process:

```bash
nats-server -p 4222 &
cd .. && .venv/bin/python controller.py serve --broker nats://localhost:4222
.venv/bin/python agent.py --broker nats://localhost:4222
```

Cross-host, pin an address on the serving side
(`--zenoh-listen tcp/0.0.0.0:7447`) and point the agent at it
(`--broker zenoh://<host>:7447`); it stays brokerless either way. See the
transport table in [../README.md](../README.md).

### Credentials

The model is the one part of this that is not local, so it needs credentials.
Two ways, and `agent.py` loads `voice/.env` at startup (a real environment
variable always wins over the file):

```bash
# A. an API key on disk
cp .env.example .env && $EDITOR .env        # ANTHROPIC_API_KEY=sk-ant-...

# B. OAuth, with no key stored anywhere
ant auth login
.venv/bin/python agent.py --auth oauth
```

`--auth oauth` builds a bare `AsyncAnthropic()`, which resolves the profile
`ant auth login` writes, and hands it to pipecat as `client=` (pipecat does
`self._client = client or AsyncAnthropic(api_key=api_key)`, so no key is
consulted). `.env` is gitignored.

> **A Claude Code or Claude.ai subscription cannot be used here.** That
> credential authenticates Claude Code itself; API traffic from your own
> applications is a separate product, billed separately. There is no supported
> way to point this agent at a subscription, and lifting a token out of another
> tool's credential store is not one either.

Then talk to it. Try "look to your left", "nod if you can hear me", "turn around",
"how are your motors doing". **Stop it with Ctrl-C** so it returns the robot to
neutral and hands the mic and speaker back.

Useful flags:

```bash
.venv/bin/python agent.py --say "nod and say hello"   # test without a microphone
.venv/bin/python agent.py --say "hello" --say "what did I just say?"  # multi-turn
.venv/bin/python agent.py --list-devices        # audio devices and their indices
.venv/bin/python agent.py --no-robot            # voice only, robot untouched
.venv/bin/python agent.py --audio-device "MacBook Pro"   # use laptop audio instead
.venv/bin/python agent.py --fast                # Claude fast mode (see below)
.venv/bin/python agent.py --language fr         # pin to one language, no detection
```

## What the model can do with its body

Seven tools, all reaching the robot over Device Connect. Angles are exposed in
**degrees**, not radians, because models are markedly more reliable reasoning
about "turn thirty degrees left" than about `0.52` -- and the driver clamps
whatever arrives.

| Tool | What it does |
|---|---|
| `move_head` | Point the head (yaw / pitch / roll) |
| `turn_body` | Rotate the whole robot on its base |
| `nod` | Agree, greet, confirm |
| `shake_head` | Disagree, decline |
| `wiggle_antennas` | Delight |
| `perform` | A recorded dance/emotion, `spin`, or `wiggle` (T13.4; names in `moves.py`) |
| `reset_pose` | Back to neutral |
| `get_robot_status` | Read its own joints, motor and estop state |

Motion is **fire-and-forget**, and the driver makes that nearly free: every move
returns a `motion_id` immediately and runs in the robot's own background task, so
the round trip is milliseconds no matter how long the move takes. The robot moves
*while* Claude talks, rather than the reply waiting on a 1.5-second head turn.
Failures are logged, never raised into the conversation.

## Embodiment: the motion the model does not control

A voice agent with a speaker attached is a speaker. What makes the robot read as
*listening* is that it reacts before it answers -- and that is not the model's
job, because waiting for a model round trip to look attentive is exactly the
latency you are trying to hide.

`embodiment.py` is a pipecat processor that watches the four speaking frames:

| Event | The robot |
|---|---|
| you start speaking | leans in, antennas up |
| you stop speaking | small dip, antennas settle |
| it starts speaking | loose irregular sway while talking |
| it stops speaking | settles back to attentive |

**Why this does not fight the model's tool calls.** Embodiment only writes the
DOFs it names, and `goto_posture` leaves every unnamed DOF holding its current
commanded value. So embodiment owns pitch and the antennas while a `move_head`
call owns yaw, and the two compose instead of overwriting each other. Keep that
split if you add gestures. Turn the sway off with `--no-sway`.

## Speaking other languages

Reachy answers in the language you spoke to it. Say something in French and the
reply comes back in French, in a French voice, with no flag and no restart.

Six languages work: **English, Spanish, French, Italian, Portuguese, Hindi.**

None of this needed a new model or a new dependency. Kokoro was always
multilingual (the voices file carries 54 voices) and Whisper always detected
language; pipecat pins both to English by default, and the detected language was
being thrown away before anything could act on it. What was missing was the
wiring, which is `multilingual.py`.

```bash
.venv/bin/python agent.py                       # detect per turn (default)
.venv/bin/python agent.py --language fr         # pin to French, skip detection
.venv/bin/python agent.py --voice bm_george     # a specific voice to start in
```

### How the language gets chosen

`MultilingualWhisperMLX` asks Whisper to detect rather than telling it what to
expect, and keeps the answer. `LanguageRouter` sits right after it and pushes a
settings update when the language changes, on the same path as the transcript,
so the switch lands before the reply it belongs to. Voice and language move
together: Kokoro voices are language-specific, and reading French text with an
English voice produces confident nonsense rather than an error.

Two guards keep it from flapping, and they matter more than the switching:

- **A switch needs about twelve characters.** Whisper will detect a language
  from a cough. "Yes" and "hmm" detect as something, often not English.
- **Only languages with a voice count.** Anything else is ignored and the robot
  stays where it is.

Without those, one noisy moment flips the robot into another language and it
cannot be talked back out, because everything it hears next is transcribed
under the wrong assumption.

### Japanese and Chinese are deliberately excluded

Kokoro ships eight CJK voices and they are not usable here. kokoro-onnx
phonemizes through espeak, and espeak is not good enough at either script to
feed this model, which was trained on misaki phonemes carrying pitch accent and
real word segmentation.

Measured by speaking a sentence and transcribing it back, the same round trip
the six shipping languages pass at 96-100%:

| | Match | Note |
|---|---|---|
| Japanese | 19% | 12.2s of audio for a 2.5s sentence: it loops |
| Chinese | 41% | |

"Hello, I am a small robot on your desk" comes back in Japanese as *"I read my
information. I read my information."* espeak collapses the word for "small"
into a single consonant. It sounds fluent and means nothing, which is the worst
way for this to fail: only a Japanese speaker would notice.

If you want them, the fix is misaki rather than a different engine.
`create_stream` takes phonemes directly (`is_phonemes=True`), so `misaki[ja]`
and `misaki[zh]` can do the g2p espeak cannot. Both pull in heavier
dependencies, which is why this stops here.

German, Korean, Arabic and Russian are not in Kokoro at all. Those need a
different engine.

### A trap worth knowing

**`--say` cannot exercise language detection.** It injects text straight into
the model's context, which is the point of it, but that path never touches
Whisper, so there is nothing to detect and the voice does not switch. Pair it
with `--language` to test a language without a microphone:

```bash
.venv/bin/python agent.py --language fr --say "Dis bonjour."
```

## Latency: what it actually measures

Time from end-of-utterance to first sound, measured on this M4 Pro:

| Turn | Claude Opus 5 | Claude Haiku 4.5 |
|---|---|---|
| Plain question | 4.8s | **2.7s** |
| With one movement | **3.2s** | not measured |

Where the time goes on an Opus 5 turn: model 1.5-3.5s (the dominant term, and
run-to-run variance is large), speech synthesis 0.31-0.40s, robot motion 0.0s
(fire-and-forget). Local speech recognition and turn detection are not on this
path at all -- they finish before the model is called.

Three things that were worth fixing, each of which is now in the code:

**Models are warmed at startup.** Whisper's first call took **13.6 seconds**
cold, and Kokoro's 2.8s. Without warmup that lands on the first thing you say.
A throwaway inference at boot costs 3s once and takes synthesis to ~0.31s
steady-state. Disable with `--no-warmup`.

**The model is told to speak in the same turn as it moves.** Left alone it
chained three separate round trips (wiggle, then move, then talk) before saying
anything: 4.8s. Instructing it to move once and speak in the same response cut
that to 3.2s.

**Short sentences are a latency instruction, not a style one.** Synthesis cannot
start until the first sentence is complete, so one long opening sentence delays
all audio. The prompt asks for sentences under about fifteen words for exactly
this reason.

| Knob | Default | Why |
|---|---|---|
| `--model` | `claude-opus-5` | `claude-haiku-4-5` is roughly 2s faster per turn if you want speed over depth |
| `--effort` | `low` | The fast end of the ladder. Sent only to models that accept it (Haiku 4.5 rejects it with a 400; the agent checks the Models API and omits it) |
| thinking | **on** | See below -- do not turn this off |
| `--whisper-model` | `whisper-large-v3-turbo-q4` | Quantised turbo; smaller is faster still |
| `--stop-secs` | `0.35` | VAD silence before end-of-speech is *considered* |
| `--no-warmup` | off | Skip the startup warmup and pay it on your first sentence instead |
| `--fast` | off | Claude fast mode: same model, up to 2.5x faster output, at premium pricing ($10/$50 per MTok). Off by default because it doubles cost. |

**Thinking stays on deliberately.** Disabling it looks like the obvious latency
win, but with thinking off Claude Opus 5 can emit a tool call as *plain text* --
the turn completes, no error is raised, and the call silently never runs. In a
robot that means "nod" gets spoken instead of performed. Low effort is the right
lever; disabling thinking is not.

**The system prompt does the brevity work, not `effort`.** Opus 5 writes longer
answers than earlier models by default and effort does not reliably shorten
visible output. For voice that matters twice over, so `SYSTEM_PROMPT` is explicit
about one-to-two-sentence replies and no markdown.

## What has been verified

**The speech half of this, on the real robot with the real audio hardware.**
All of it below the robot link is unchanged, and was proven end to end:

- `set_media_released` hands the mic and speaker from the daemon to the agent
- the robot's mic captures and its speaker plays from the voice process
- fire-and-forget motion returns in **0.0 ms**, so speech never waits on it
- the full pipeline assembles: 7 tools registered, all models loaded
- agent startup reaches "ready", and **Ctrl-C** returns the robot to neutral and
  hands the media back

**Not verified: this agent against the Device Connect driver.** `robot_link.py`
is the one file here that changed, and it has not been run against a live
`controller.py serve` -- the voice venv is a separate, heavy install and was not
rebuilt for this. What it does has been proven from the other side: the same
discover-then-invoke path, the same functions, and fire-and-forget motion all
work cross-process over zenoh from `controller.py --attach` (see
[../README.md](../README.md)). Expect the robot link to want a few seconds
longer to find the robot than it used to, and read the two new gotchas at the
bottom before debugging anything.

**The loop is closed.** With a real key, full turns run end to end. Asked to
"nod twice, then tell me in one short sentence what you can do", Claude called
`nod(times=2)`, the robot nodded, and it said through the robot's own speaker:

> *"I can turn my body, tilt and point my head, and wiggle my antennas while we chat."*

Asked to "look to your left, then say hi in three words", it called
`move_head(yaw_degrees=40)` and said *"Hi there, friend!"* as it turned.

**Multilingual, verified by round trip** (2026-08-28): each language spoken by
Kokoro, fed back through the real MLX Whisper service, and compared to the
input. English and French 100%, Italian 99%, Spanish 98%, Portuguese and Hindi
96%. Every one was detected as the right language. The router switches on a
French transcript, ignores a four-character one, and refuses to switch into
Japanese.

**Not verified: a live spoken conversation that changes language mid-way.** The
detection, the switch and the synthesis were each verified, but the room was too
noisy to complete a microphone-driven turn: ambient speech interrupted every
reply before it started, and with `--language fr` pinned, Whisper transcribed
that noise as French. That is the sensitive-mic gotcha below, not a fault in the
language handling. Try it with a headset.

**Speech recognition works on a real voice.** Spoken into the robot's own
microphone and transcribed correctly: "How much is 3 plus 2?", "Hi, how are
you?", "How are you doing?". MLX Whisper turned these round in about 0.8s.

**Not verified: a long live conversation with both fixes below in place.** The
two bugs they address were each reproduced and each fix verified in isolation,
and four consecutive injected turns now run clean. But the context guard did not
fire during that run, so it has not yet proven itself against a live wedge.

## Two bugs worth knowing about

Both were found by actually talking to it, and both fail in the same nasty way:
nothing crashes, the robot keeps listening and moving, and it simply stops
answering.

### The conversation wedges permanently (`KeyError: 'role'`)

When the model thinks, pipecat appends a `{"type": "thought", ...}` message to
the context. Its Anthropic adapter converts that back into a real message *only*
when the thought has non-empty text **and** a signature; otherwise it falls
through to "assume it is already in Anthropic format" and returns the raw dict,
which has no `role` key. The next request dies on `KeyError: 'role'` inside the
adapter, and since the bad message is now in history, so does every turn after
it. The error frame is non-fatal, so the agent just goes mute.

Claude Opus 5 walks straight into this: it never returns raw chain of thought,
so `thinking.display` defaults to `"omitted"` and thinking blocks arrive with
empty text. Reproduced directly against the adapter:

```
text='' (Opus 5 display=omitted)       -> FAILS: KeyError: 'role'
text=None                              -> FAILS: KeyError: 'role'
no signature                           -> FAILS: KeyError: 'role'
text + signature (display=summarized)  -> CONVERTS OK
```

`SafeLLMContext` in `agent.py` drops thought messages that cannot be converted.
An empty thought carries no information, so nothing is lost. Setting
`thinking: {"display": "summarized"}` also fixes it, but pays for summary tokens
on every turn, which is the wrong trade in a voice loop.

If it ever goes mute again, this is the first thing to check:

```bash
grep "dropping unconvertible thought" agent.log
```

### The robot hears itself

The speaker and microphone are the same USB device, centimetres apart, with no
echo cancellation. The robot hears its own voice, Whisper hallucinates whole
sentences out of the feedback, and those arrive as phantom user turns. Real
transcripts captured while the robot was talking and nobody was in the room:

> *"80% of boys are not just..."* &middot; *"Oh, my boy, how old are you?"* &middot; *"He has been painted."*

The robot then answers things nobody asked and drifts further off with each
exchange. Fixed with `AlwaysUserMuteStrategy` + `FunctionCallUserMuteStrategy`,
which mute the mic while the robot speaks and while a motor command runs. Pass
`--no-mute` only with a headset or a mic well away from the speaker, or use
`--audio-device "MacBook Pro"` to sidestep the coupling entirely.

## Gotchas

- **PyAudio needs portaudio first** (`brew install portaudio`), and the build
  needs `CFLAGS`/`LDFLAGS` pointing at it. Without them the wheel fails to
  compile and the whole install aborts.
- **The daemon owns the mic and speaker until asked to let go.** The agent calls
  `set_media_released(True)` at startup. If audio input is silent, check that
  call succeeded -- the symptom is a working pipeline that simply never hears
  anything.
- **Ctrl-C, not `kill`.** SIGINT runs the cleanup (robot home, media handed back);
  SIGTERM does not, and leaves the robot holding its pose with media released.
- **VAD and turn detection moved in pipecat 1.6.** They are no longer transport
  params -- they hang off `LLMUserAggregatorParams` (`vad_analyzer=`,
  `user_turn_strategies=`). Older examples that pass `vad_analyzer` to the
  transport silently do nothing.
- **`temperature` / `top_p` / `top_k` must stay unset.** Claude Opus 5 rejects
  them with a 400. pipecat leaves them `NOT_GIVEN` by default, so just do not
  set them.
- **`--effort` is not universal.** Haiku 4.5 rejects it with a 400 on every
  turn. The agent asks the Models API whether the chosen model accepts it and
  omits it when not, so `--model claude-haiku-4-5` just works.
- **The robot's mic is sensitive.** It picks up room noise and will occasionally
  answer nothing at all. Raise `--stop-secs` or use `--audio-device "MacBook Pro"`
  if that gets annoying.
- **Both venvs must exist.** `voice/.venv` runs the agent; `../.venv` runs the
  robot. They are separate on purpose.
- **Discovery takes a few seconds.** `RobotLink.connect()` waits for its own
  runtime to hand it a registry before asking for the robot, because asking too
  early raises "Registry not configured" -- which reads exactly like "the robot
  is not there". Budget about 4 seconds for the peer to appear.
- **A move's call returns before the move does.** `robot.nod()` comes back once
  the robot has accepted the gesture, not once it has finished nodding. That is
  the point, but do not write code that assumes otherwise.
