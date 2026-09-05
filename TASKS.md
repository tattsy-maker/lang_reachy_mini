# Language Tutor — Task Breakdown & Progress Tracker

Implementation tasks for [LANGUAGE_TUTOR_SPEC.md](LANGUAGE_TUTOR_SPEC.md).
Read the spec first; this document says *what to build in what order*, the
spec says *why*.

## How to work a task (instructions to the implementing agent)

- **One task per session.** Each task is scoped to be completable and
  verifiable in a single coding-agent session. Do not start a task whose
  dependencies are not `done` in the tracker below.
- **Update the tracker.** When you start, set status to `in progress`. When
  the task's Definition of Done is met, set `done` and add a dated line to
  the task's Progress log. Never mark `done` with failing tests.
- **Integration tests are part of the task**, not a follow-up. Every task
  lands its tests under `tests/` and they must pass via
  `tests/run.sh <task-id>` (created in T0). Tests that need hardware, big
  models, or API keys must skip cleanly (with a printed reason) when the
  prerequisite is absent — never fail on a machine that lacks it.
- **Honest logs.** If something is verified only on the stub and not on the
  robot, say so in the Progress log. The spec's history shows the gap
  between "works on the stub" and "works on metal" is where bugs live.
- **Environment facts** (see [CLAUDE.md](CLAUDE.md) for the full runbook):
  two venvs (`./.venv` robot, `voice/.venv` agent); zenoh needs a pinned
  address in this container (`--zenoh-listen tcp/0.0.0.0:7447` /
  `zenoh://127.0.0.1:7447`); run audio under `sg audio`, serial under
  `sg dialout`; `voice/.env` holds `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`
  (cloud mode; `GOOGLE_API_KEY` also accepted). Camera commands need
  `sg video -c` only in shells older than the 2026-09-02 usermod.

Each task keeps dated progress notes and learnings in `progress/<task-id>.md`,
linked from its Progress log below.

## Status tracker

Statuses: `todo` · `in progress` · `done` · `verified-on-metal` (done +
exercised on the physical robot) · `cut` (dropped per spec).

| ID | Task | Depends on | Status | Last update |
|----|------|-----------|--------|-------------|
| T0 | Test harness & fixtures | — | done | 2026-08-31 |
| T1 | Camera capture module | T0 | verified-on-metal | 2026-08-31 |
| T2 | Face recognition core | T0, T1 | done | 2026-08-31 |
| T3 | Learner store | T0 | done | 2026-08-31 |
| T4 | Tutor mode: briefing + memory tools | T0, T3 | done | 2026-08-31 |
| T5 | Piper TTS + per-language engine routing | T0 | done | 2026-08-31 |
| T6 | Mixed-language TTS assembly | T5 | done | 2026-08-31 |
| T7 | Mixed-language STT hardening | T0 | done | 2026-08-31 |
| T8 | Cloud speech mode (Gemini Live) | T4 | done | 2026-08-31 |
| T9 | Conversational enrollment | T2, T3, T4 | done | 2026-08-31 |
| T10 | Session lifecycle | T2, T4, T9 | done | 2026-08-31 |
| T11 | Faire hardening & dress rehearsal | all | in progress | 2026-09-02 |
| T12 | Cloud session lifecycle (watch / walk-away / reset over Gemini) | T8, T10 | done (as T14.3) | 2026-09-03 |
| T13 | Family feedback round 1: goals, presence, tracking, showmanship | T11, T12 | in progress | 2026-09-02 |
| T14 | Family feedback round 2: sight, calm tracking, one visitor at a time, longer dances | T13 | in progress | 2026-09-03 |
| T15 | Family feedback round 3: identity for the whole session, one greeting, honest lag | T14 | in progress | 2026-09-04 |

Parallelizable from the start (after T0): T1, T3, T5, T7 have no
dependencies on each other. The critical path is
T0 → T1 → T2 → T9 → T10 → T11. T12 was added after the first
in-person rehearsal (2026-09-02): the family chose the cloud voice, and
the booth loop must work over it. T13 collects the rest of that
rehearsal's feedback (recorded family debrief, same day) as one task of
nine subtasks; T13.1, T13.4–T13.7 and T13.9 do not touch the session
loop and may start before T12 lands.

---

## T0 — Test harness & fixtures

**Goal.** A `tests/` tree every later task plugs into, runnable without
hardware, keys, or big models by default.

**Build:**
- `tests/run.sh [task-id]` — creates/uses a test venv, runs pytest for one
  task's tests or all of them. Markers: `robot` (needs hardware), `models`
  (downloads/loads big local models), `anthropic` / `google` (needs that
  key), `audio` (needs a sound card). Unmarked tests must run anywhere.
- Fixtures under `tests/fixtures/`: 4–6 face photos of at least 3 distinct
  people (family photos or CC0 images — record provenance in a README
  there), a short webcam-style video clip containing one of those faces,
  and a `learners/` sample tree.
- A pytest fixture that launches `controller.py --stub` serving on a pinned
  zenoh address and tears it down with SIGINT (the repo's stub is the
  hardware-free robot; SIGINT is mandatory, see CLAUDE.md).
- A pytest fixture that runs `voice/agent.py --no-robot --say "..."` and
  captures its log output (the repo's established way to drive a full agent
  turn without a microphone).

**E2E check.** `tests/run.sh` green on this machine with no keys exported;
one demo test drives `--say "hello"` through the real agent (marked
`anthropic`) and asserts a reply was synthesized.

**Definition of Done.** Harness + fixtures merged; both demo tests pass;
tracker updated.

**Progress log.**
- 2026-08-31 — done; full suite green with no keys exported (7 passed, 1
  clean skip). Found and fixed a real bug on the way: `agent.py --no-robot`
  crashed under pipecat 1.6 (`tools=None` rejected by `LLMContext`).
  Details and learnings in [progress/T0.md](progress/T0.md).

---

## T1 — Camera capture module

**Goal.** Frames from the Reachy camera on demand, with a hardware-free
substitute for tests.

**Build:**
- `face/camera.py`: `Camera.frames()` yielding OpenCV images at a capped
  rate (~2 fps), sourced from (a) a V4L2 device index (the Reachy camera),
  or (b) a video file / image directory — selected by argument. Handle the
  camera being held by the vendor daemon: document/perform the
  `set_media_released` handoff needed before opening it.
- `face/check_camera.py`: prints device candidates, grabs one frame, saves
  it to disk — the on-hardware smoke test.
- Decide and record (in the task log) where face deps live: try
  `voice/.venv` first; if OpenCV/onnxruntime conflict with pipecat's tree,
  create `face/.venv` and note the interface boundary.

**Integration tests.** Frames from the fixture video: correct shape, rate
cap respected, clean iterator shutdown. A `robot`-marked test opens the
real camera and asserts a non-black frame.

**E2E check.** On this machine: `face/check_camera.py` saves a real frame
from the Reachy camera (run once, note result in log).

**Definition of Done.** File-source path fully tested; real-camera path
exercised once on hardware or explicitly logged as blocked.

**Progress log.**
- 2026-08-31 — done; file-source path fully tested (7 passed, 1 skip).
  Face deps live in `voice/.venv` (OpenCV coexists with pipecat; verified).
  Real-camera grab was blocked on `video` group membership — documented in
  [progress/T1.md](progress/T1.md).
- 2026-08-31 (later) — **verified on metal** after the usermod:
  `check_camera.py` saved a real 1920×1080 frame from `/dev/video0`
  (dark room, mean brightness 24.6 — still clears the non-black check)
  and all 8 tests pass including the robot-marked one. Note: this
  session's shells predate the usermod, so everything camera ran under
  `sg video -c`; fresh logins won't need that.

---

## T2 — Face recognition core

**Goal.** Enroll, match, and reject faces from images. No agent wiring yet.

**Build:**
- `face/recognize.py`: `embed(image) -> vector | None`,
  `enroll(images) -> averaged vector`, `match(vector, known) ->
  (name, score) | None` with the spec's ask-don't-guess band: a hard accept
  threshold, a hard reject threshold, and an "unsure" band in between that
  callers surface as a question. Largest-face-only selection when multiple
  faces are in frame.
- InsightFace `buffalo_l` via ONNX Runtime. **Verify and log which
  execution provider actually runs** (GPU vs CPU) — this box's history
  says never trust the reputation (see CLAUDE.md's CTranslate2 story). CPU
  is acceptable if frame-rate holds at ~2 fps.
- Model download pinned & cached; a `models`-marked test exercises it.

**Integration tests.** With fixture photos: same person across two photos
matches above threshold; different people fall below; a no-face image
returns `None`; multi-face frame picks the largest. Thresholds asserted
against measured fixture scores, not guessed.

**E2E check.** A tiny script: enroll person A from 3 fixture photos, then
identify them in the fixture video via `Camera` (T1) — prints name + score.

**Definition of Done.** All tests green; execution provider + measured
score distributions recorded in the Progress log (these numbers feed T9).

**Progress log.**
- 2026-08-31 — done; 6 tests green. Provider is CPU (no CUDA EP in the
  aarch64 onnxruntime wheel) and it's plenty: 95 ms/embed ≈ 10 fps vs the
  2 fps need. Measured scores: same-person 0.66–0.98, different-person
  ≤ 0.063; thresholds accept ≥ 0.45 / reject < 0.25 with the ask band
  between. Video E2E: 12/12 frames matched at ≥ 0.775. Full numbers in
  [progress/T2.md](progress/T2.md).

---

## T3 — Learner store

**Goal.** The `learners/` folder exactly as specced: profiles, notes,
tiers, expiry. Pure filesystem, no AI, no robot.

**Build:**
- `tutor/store.py`: create/load/save `learners/<name>/profile.json`
  (name, target language, level, embedding, session count, last-seen,
  `tier: family|guest`), append-newest-first to `notes.md`, list learners,
  delete learner ("forget me"), `wipe_guests()` deleting every guest
  profile.
- `tutor/wipe_guests.py` CLI for the end-of-day wipe.
- Name collisions ("Maria" twice at the booth): disambiguate folder names,
  keep display name.

**Integration tests.** Full CRUD; tier semantics (wipe removes guests,
never family); notes ordering; collision handling; corrupt-profile.json
handled without crashing (skip + warn, don't delete).

**E2E check.** Scripted: enroll → 2 sessions of notes → wipe → assert
family survives, guest gone.

**Definition of Done.** Store API stable and documented in the module
docstring (T4/T9/T10 build against it); tests green.

**Progress log.**
- 2026-08-31 — done; 10 tests green including the CLI-driven
  enroll → notes → wipe E2E. API contract in the module docstring;
  decisions (bookkeeping lives in `append_session`, corrupt profiles are
  never wiped, collision slugs) in [progress/T3.md](progress/T3.md).
  `/learners/` is now gitignored — learner data stays off the repo.

---

## T4 — Tutor mode: briefing + memory tools

**Goal.** The agent becomes a tutor: it loads a learner, teaches at their
level, and writes notes back.

**Build (in `voice/agent.py` + new `voice/tutor.py`):**
- `--learner <name>` flag: load profile + recent notes from the store (T3)
  into the system prompt per the spec's briefing (target language from
  profile, level-pitched, short sentences, correct-kindly).
- New tools registered alongside the seven motion tools:
  `save_session_notes`, `update_learner_level`. Notes format per spec
  (Practiced / Struggled with / Wins / Next time).
- Keep the existing latency rules (short sentences, move-and-speak in one
  turn); booth model default noted but not hardcoded (`--model` exists).

**Integration tests** (`anthropic`-marked, stub robot, `--say` driven):
- A scripted 3-turn Spanish mini-lesson with a seeded intermediate profile:
  assert the reply is mostly Spanish and references the seeded "Next time"
  note.
- A goodbye turn: assert `save_session_notes` fired and `notes.md` gained a
  well-formed entry.
- Level respect: beginner profile → reply contains English scaffolding.
  (Assert via cheap checks — language of output, section headers in
  notes — not LLM-judging.)

**E2E check.** Live: `agent.py --learner maria --say "hola" --say "adiós"`
against the stub; inspect the written notes by eye.

**Definition of Done.** Briefing + both tools working end-to-end via
`--say`; note in log whether a real-microphone run happened.

**Progress log.**
- 2026-08-31 — done; 7 tests green (3 unit + 4 live `--say` runs: Spanish
  lesson with correction, goodbye→notes, beginner scaffolding, level
  update). Stub-robot E2E run by hand — which found and fixed a stub bug
  (`set_media_released` was broken against a served stub). New module is
  `voice/tutor_mode.py` (the `tutor` package name was taken by the store).
  No dedicated real-microphone run yet, though the live mic was open
  during all `--say` runs (see learnings). [progress/T4.md](progress/T4.md).

---

## T5 — Piper TTS + per-language engine routing

**Goal.** Russian and Mandarin speak, via Piper, routed automatically.
(Spec's designated cut-if-late task.)

**Build:**
- `voice/piper_tts.py`: a pipecat TTS service wrapping Piper (local,
  streaming into the existing pipeline), with chosen ru + zh voices.
- Extend `multilingual.py`'s router: language → (engine, voice) map;
  Kokoro keeps es/fr/it/pt/en/hi, Piper takes ru/zh. Extend the "only
  languages with a voice count" guard to cover both engines.
- `voice/verify_language.py`: the round-trip gate — synthesize a fixed
  sentence set, transcribe back with the real Whisper service, print a
  match % (the repo's established method; ≥90% ships, below falls back per
  spec section 6).

**Integration tests** (`models`-marked): router picks the right engine per
language; Piper produces nonzero, sane-duration audio for ru and zh;
round-trip gate runs and its report is written to `tests/reports/`.

**E2E check.** `agent.py --language ru --say "..."` speaks Russian through
the pipeline (stub robot, `audio`-marked or written to wav).

**Definition of Done.** Both languages pass the ≥90% gate **or** the
fallback decision (misaki / cut) is made and logged with the numbers.

**Progress log.**
- 2026-08-31 — done, **both languages PASS the gate**: ru 95.9%, zh 92.9%
  (report in `tests/reports/verify_language_2026-08-31.*`). DualEngineTTS
  dispatches Kokoro/Piper per turn; `--language ru` E2E through the real
  agent produced Russian replies spoken by Piper. One trap found: the
  system prompt's language roster had to become dynamic — Claude refused
  Russian while the stack could speak it. [progress/T5.md](progress/T5.md).

---

## T6 — Mixed-language TTS assembly

**Goal.** One spoken reply containing two languages, seamlessly stitched.

**Build:**
- Span-tag convention in the system prompt (e.g. `[es]…[/es]` around
  non-primary-language spans) — added to the tutor briefing (T4).
- `voice/spans.py`: split tagged text into (language, text) spans;
  untagged text uses the turn's primary language; malformed tags degrade to
  plain text, never crash, never get spoken aloud as brackets.
- Synthesis assembly: per-span engine/voice via the T5 router, stitched in
  order into one utterance; timbre-adjacent voice pairs chosen and
  documented.

**Integration tests.** Span parser unit-tested hard (nesting, unclosed,
empty, tag-only). `models`-marked: a tagged EN+ES sentence → audio →
Whisper transcribes both halves correctly (the code-switched round trip);
same for one pair involving Piper (EN+RU) if T5 shipped.

**E2E check.** `--learner` + `--say "How do you say 'the library' in
Spanish?"` produces a reply that audibly switches into Spanish for the
answer (listen to the wav).

**Definition of Done.** Round-trip passes for at least EN+ES and EN+FR;
tag leakage (brackets spoken aloud) impossible by test.

**Progress log.**
- 2026-08-31 — done; 13 tests green. EN+ES and EN+FR round-trip with both
  halves; the live agent tagged `[es]la biblioteca[/es]` unprompted and
  the log shows the voice switching per span. EN+RU speaks correctly but
  plain Whisper keeps only the dominant half of the clip — T7's problem,
  honestly recorded. [progress/T6.md](progress/T6.md).

---

## T7 — Mixed-language STT hardening

**Goal.** Code-switched *input* measured and improved; the gate that
decides whether local mode meets the MUST or cloud mode is required.

**Build:**
- Bilingual priming: feed Whisper an initial prompt containing English +
  the learner's target language each turn (wire into
  `MultilingualWhisperMLX`).
- Briefing addition (with T4): tell Claude transcripts may garble embedded
  foreign words and to repair from context.
- `voice/verify_codeswitch.py`: a test set of ~10 code-switched phrases per
  pair (EN+each target language), synthesized via T6's assembly, fed
  through the real Whisper service, scored on whether the embedded foreign
  span survives recognizably. Report to `tests/reports/`, with and without
  priming, so the priming's effect is measured not assumed.

**Integration tests** (`models`-marked): the gate script runs end-to-end
and writes its report; priming plumbing verified (the prompt actually
reaches Whisper).

**E2E check.** Speak one code-switched sentence at the real mic (or via the
gate script) and read the transcript.

**Definition of Done.** Per-pair scores recorded in the Progress log and in
the spec's terms: which pairs are "seamless locally" vs "cloud mode
required". No fixed pass bar — the deliverable is the honest number.

**Progress log.**
- 2026-08-31 — done; full six-pair gate measured (8 phrases/pair, with and
  without priming). Best-config span survival: es 96%, it 92%, ru 88%
  (priming turns Whisper's translate-the-Russian habit off, +26 points),
  fr 82%, pt 74%, zh 69% — **pt and zh are the cloud-mode pairs**.
  Priming *hurts* fr/pt/it/zh (hallucination loops), so it's now a
  measured per-pair policy (`PRIMING_HELPS`). [progress/T7.md](progress/T7.md).

---

## T8 — Cloud speech mode (Gemini Live)

**Goal.** `--speech cloud` runs the whole session over Gemini 3.1 Flash
Live with the same tools and memory; `--speech local` unchanged.

**Build:**
- Pipeline variant in `agent.py` using pipecat's `GeminiLiveLLMService`
  (speech-to-speech): replaces STT + LLM + TTS stages; VAD/turn handling
  per pipecat's s2s guidance; robot motion tools + T4 memory tools
  re-registered on the Gemini service; tutor briefing passed as system
  instruction; `GOOGLE_API_KEY` in `voice/.env`.
- Mute-while-speaking strategy reviewed for full-duplex (Gemini supports
  barge-in; decide and document whether the robot's self-hearing problem
  allows enabling it with the booth mic).
- `--speech` flag, default `local`.

**Integration tests** (`google`-marked): a `--say`-driven turn in cloud
mode gets an audio reply; a "nod twice" turn fires the motion tool on the
stub robot; a goodbye turn writes notes via `save_session_notes`. Local
mode's tests (T4) still green — the flag must not disturb the default path.

**E2E check.** Live mic conversation in cloud mode including one
code-switched sentence each way; log latency and subjective quality next to
local mode's numbers.

**Definition of Done.** Both modes selectable and passing their tests;
cloud-mode tutoring tone spot-checked against the briefing (spec risk:
"the prompt work doesn't automatically carry over").

**Progress log.**
- 2026-08-31 — wired and key-ready, blocked on a Google key: `--speech
  cloud` builds the Gemini Live pipeline with the same briefing and tools,
  fails fast without the key (tested). Span tags made local-only (a
  speech-to-speech voice would read brackets aloud).
- 2026-08-31 (later) — **done**: key arrived (as `GEMINI_API_KEY`, now
  accepted alongside `GOOGLE_API_KEY`; model
  `models/gemini-3.1-flash-live-preview`). All 4 tests pass live: audio
  reply, goodbye→well-formed notes, nod on the stub robot. Two-turn
  tutor session verified: TTFB **0.6–1.7 s** (vs ~2.7–5 s local), opening
  picked up the seeded notes, walk-away-quality notes written. Two real
  findings fixed on the way — multi-turn text injection and Gemini's
  tool-call discipline — details in [progress/T8.md](progress/T8.md).
  Live-mic cloud conversation deferred to the T11 rehearsal.
- 2026-09-02 — live-mic cloud conversation done in person and it found a
  real bug: Gemini drops microphone audio until it has an initial
  context (every `--say` test supplied one). The agent now opens the
  conversation itself in cloud mode. Verdict: the family prefers this
  voice for the booth. See [progress/T11.md](progress/T11.md).

---

## T9 — Conversational enrollment

**Goal.** A stranger becomes a guest learner entirely by voice.

**Build:**
- `enroll_new_learner` tool (spec section 4C): when the face module reports
  "unknown", the briefing tells the tutor to ask for consent + name; the
  tool captures 3–5 frames via T1/T2, averages the embedding, creates a
  guest profile via T3.
- The unsure band from T2 surfaces as "Maria, is that you?" rather than a
  wrong greeting.
- "Forget me" tool wired to the store's delete.

**Integration tests.** With the face pipeline fed the fixture video and a
scripted `--say` dialog (`anthropic`-marked): unknown face → consent
question asked → name given → guest profile exists with an embedding →
same video now matches. "No thanks" path creates nothing. "Forget me"
deletes and is confirmed verbally.

**E2E check.** A real person not in the store walks up to the robot and
enrolls by voice; comes back and is greeted by name.

**Definition of Done.** Scripted path fully tested; live path exercised
once and logged.

**Progress log.**
- 2026-08-31 — scripted path fully tested (5 green: enroll with consent →
  re-recognized in the same video; no-thanks stores nothing; manufactured
  unsure-band → "Maria, is that you?" → confirm; forget-me deletes with
  spoken confirmation). Live in-person run **blocked on `video` group**
  (same one-liner as T1). Added `--deaf` (agent ignores the mic) after
  room noise made scripted runs flaky. [progress/T9.md](progress/T9.md).

---

## T10 — Session lifecycle

**Goal.** The booth loop with nobody touching a keyboard: watch → greet →
tutor → save on walk-away → reset → watch.

**Build:**
- A session controller in the agent: idle state polls the camera (~2 fps)
  via T1/T2; a stable face (≈2s, largest-face rule) starts a session with
  that learner's briefing; face absent ~60s → force `save_session_notes`,
  reset conversation context, robot to neutral, back to idle. Face
  recognition paused during active conversation (spec: GPU is busy).
- Clean interaction with SIGINT shutdown (robot home, media back — the
  existing contract must survive).

**Integration tests.** Drive the state machine with a synthetic
frame/transcript timeline (no models needed): correct transitions,
walk-away save fires exactly once, context is empty at next session start,
two-visitors-in-a-row don't leak notes into each other's files.
`anthropic`-marked: one full simulated visitor (fixture video + `--say`
turns + disappearance) ends with a well-formed notes entry.

**E2E check.** Live: two family members take turns at the robot without
touching the keyboard; each gets their own greeting and their own notes.

**Definition of Done.** State machine fully unit-tested; one live
two-visitor run logged.

**Progress log.**
- 2026-08-31 — done on the simulated path: machine fully unit-tested
  (fake clock), runner choreography tested with fakes (including
  two-visitor no-leak), and one live simulated visitor end-to-end
  (recognize → greet → converse → walk-away save → well-formed notes →
  reset, 29 s). The live on-hardware two-visitor run is **blocked on the
  `video` group** (see T1). [progress/T10.md](progress/T10.md).

---

## T11 — Faire hardening & dress rehearsal

**Goal.** The booth checklist, executed. Mostly ops; small code.

**Build / do:**
- Booth mic: select the handheld/headset device via `--audio-device`,
  verify mute-while-speaking behavior with it, document the exact device
  name and startup line in CLAUDE.md.
- One-command startup script (serve + agent, pinned zenoh, correct groups,
  booth flags) and a printed startup checklist including the ~40s warmup
  and the known cold-start `serve` retry.
- Local-vs-cloud bake-off on venue-like conditions (hotspot): run T7/T8's
  measured comparisons, record the booth-default decision.
- Signage text (face recognition disclosure, mode disclosure per spec §8),
  end-of-day wipe in the shutdown path, pre-enrolled family profiles with
  real session history.
- Full dress rehearsal: 5 consecutive visitors (mix of family, guest
  enrollments, one "forget me", one language switch, one code-switched
  exchange), all off the startup script.

**Integration tests.** The startup script gets a `robot`-marked smoke test
(comes up, agent reaches "ready", SIGINT cleans up). Everything else is the
rehearsal checklist, checked off in the Progress log.

**Definition of Done.** Dress rehearsal completed with every checklist item
either passing or written up as a known issue with a booth workaround.

**Progress log.**
- 2026-08-31 — code and copy done: `start_booth.sh` (preflight → serve
  with cold-start retry → session-mode agent on Haiku → SIGINT shutdown +
  guest wipe), `booth/SIGNAGE.md` (all §8 disclosures, per-mode signs),
  and tests (static anywhere; live smoke robot-marked, camera-gated).
  Remaining items all need the user or the venue — the list with exact
  unblock commands is in [progress/T11.md](progress/T11.md).
- 2026-09-02 — **first in-person rehearsal**, cloud mode: two visitors
  enrolled by voice from the live camera, lessons, language switch,
  volume by voice, notes on goodbye. Six bugs fixed on the spot (voice
  following Whisper's guess, "I can't see you", cloud mic gating,
  Russian refusal, transcripts hidden, volume). Not yet done: the
  comeback (camera missed the face at startup), "forget me", family
  pre-enrollment, booth mic, bake-off. Cloud is one visitor per launch
  until T12. [progress/T11.md](progress/T11.md).
- 2026-09-04 — **booth mic done in code**: the USB desk mic ("USB
  Composite Device") when plugged in, the robot's own mic otherwise,
  speaker always the robot's (`--mic-device`, `BOOTH_MIC_DEVICE`). Needed
  a resampling input transport: the USB mic only opens at 48 kHz, the
  robot's only at 16 kHz. Both paths measured on the real devices;
  mute-while-speaking with the desk mic still to be checked in person.
  [progress/T11.md](progress/T11.md).

---

## T12 — Cloud session lifecycle

**Goal.** The booth loop (T10: watch → greet → tutor → save on walk-away →
reset → watch) working over Gemini Live, since the family chose the
cloud voice at the 2026-09-02 rehearsal. Today cloud mode is one visitor
per launch and looks for a face only once at startup.

**Why it does not just work.** Gemini Live keeps conversation state
server-side: pipecat's service ignores later local context swaps
(`set_messages`), its system instruction is fixed at connect, and text
injected through the user aggregator after the first turn vanishes (see
progress/T8.md). The session runner relies on all three.

**Build:**
- Make `SessionRunner` cloud-aware: on session start, deliver the
  visitor's briefing and the walk-up cue through the service's own
  injection path (`_create_single_response`, the route `--say` already
  uses after turn 1), not `set_messages`; on session end, cue the notes
  save the same way, then reconnect the Gemini session (the service has
  `_reconnect()`) so the next visitor starts with a clean server-side
  history — no leakage between visitors.
- Keep the camera watch running in cloud mode (today `--session` is
  effectively local-only); presence detection is CPU, unaffected.
- Notes must follow the *current* profile language after
  `set_target_language` (rehearsal: "next time: French" after switching
  to Russian) — a one-line briefing/tool-result fix, do it here.
- `start_booth.sh`: a `BOOTH_SPEECH=cloud` knob, cloud as the default if
  the bake-off confirms.

**Integration tests** (`google`-marked): a simulated two-visitor sequence
(fixture video + `--say`) over cloud mode ends with two separate,
well-formed notes files and no cross-talk; walk-away save fires once;
the second visitor's greeting does not mention the first. Existing local
T10 tests untouched.

**E2E check.** In person: two people take turns at the robot in cloud
mode without touching the keyboard; the comeback greeting by name
happens for both.

**Definition of Done.** Two-visitor cloud run logged in person; tests
green; `start_booth.sh` can launch cloud mode.

**Progress log.**
- 2026-09-03 — absorbed into T14.3 after the second family session made
  the one-visitor-per-launch gap the evening's main failure. Built as
  `session.CloudBrain` (per-visitor reset of the Gemini session) plus a
  voice-print speaker-change trigger. See T14.

---

## T13 — Family feedback round 1: goals, presence, tracking, showmanship

**Goal.** Everything the family asked for in the recorded debrief after the
2026-09-02 rehearsal, turned into shippable pieces. The full transcript
digest and the item-by-item mapping live in
[progress/T13.md](progress/T13.md); the two items already owned elsewhere
(the booth mic → T11 item 1; the cloud walk-away loop → T12) are *not*
repeated here.

**Subtasks.** Ordered by booth value; each has its own tests under
`tests/t13/` and its own line in the Progress log.

- **T13.1 — Learner goals and an explicit level at enrollment.** The
  debrief: Italian "started strangely" while French asked the level at
  once; the tutor must *ask* the level rather than default to beginner,
  and must know *why* the person is learning ("just conversation" /
  "exam prep" / "job interview" — the three examples given). Build: a
  `goal` field in `profile.json` (a short enum — `conversation`, `exam`,
  `work`, `travel`, `other` — plus a free-text `goal_note`), defaulting
  so existing profiles load unchanged; `enroll_new_learner` takes both;
  a `set_learner_goal` tool for changing it later; the stranger briefing
  becomes a fixed four-question script (name → language → level → goal,
  one question each, in that order, no lesson before all four); and the
  learner briefing gets per-goal guidance (exam/work: push richer
  vocabulary, offer synonyms and register; conversation: keep it flowing,
  correct less). Tests: store round-trip with and without the new keys;
  `anthropic`- and `google`-marked `--say` enrollments assert the level
  question is asked before the first lesson and the goal lands in the
  profile.
- **T13.2 — Presence policy the family can predict.** "Nobody could say
  how long it takes to forget you." Build on T12's loop: speech from the
  visitor counts as presence (a face-less but talking visitor never
  times out — the "went for tea, kept talking" case); at ~two-thirds of
  the walk-away timer the robot asks once, out loud, "Still there?";
  when the timer expires it says a one-line goodbye *before* saving, so
  the save is never silent for someone who is actually still there;
  `BOOTH_ABSENT_SECS` knob in `start_booth.sh`; the timer's state
  (`presence: face`, `presence: voice`, `presence: asking`,
  `presence: gone`) in the INFO log. Tests: fake-clock machine tests for
  the voice-extends-presence and ask-once transitions; the T10/T12
  no-leak tests still green.
- **T13.3 — Follow the speaker with head and body.** The robot should
  keep its face on the person: up, down, left, right, and turn the body
  when the face leaves the head's comfortable yaw range. Build: a
  `detect(image) -> bbox | None` in `face/recognize.py` (detection only,
  largest face, no embedding — cheaper than `embed`); a tracker in the
  session runner that maps bbox centre → head yaw/pitch (camera FOV
  from the vendor's `kinematics_data.json` or a measured constant) and
  streams `goto_posture` at ~2 fps with dead-band and rate limits so it
  never jitters; body yaw takes over past ±35° of head yaw. **DOF
  ownership must be written down** (`voice/embodiment.py` docstring
  already sets the rule): the tracker owns `head_yaw` and `body_yaw`;
  embodiment keeps antennas and its pitch gestures, so vertical tracking
  is a slow pitch bias applied only between the embodiment's nods.
  Tracking pauses during any recorded move (T13.4) and during
  `reset_pose`. Tests: bbox→angle mapping unit-tested at the frame's
  centre, edges and corners; a fixture-video run asserts the commanded
  yaw follows the face's horizontal drift and stays inside limits; a
  `robot`-marked test moves a printed face across the real camera.
- **T13.4 — Tricks: dances, emotions, a spin, and an idle attractor.**
  "It must do the standard Reachy Mini tricks — dance, spin, and so
  on." The vendor SDK ships recorded-move libraries
  (`pollen-robotics/reachy-mini-dances-library`,
  `pollen-robotics/reachy-mini-emotions-library`, HuggingFace datasets,
  played by the daemon via `play_move`). Build: a `play_move` motion
  procedure in `reachy_driver.py` / `reachy_target.py` (cancelable,
  emits `motion_completed`, returns to the previous pose); a `perform`
  tool listing a curated subset by name (two or three dances, `spin` =
  a full body-yaw sweep and back, a handful of emotions) so the model
  can answer "can you dance?"; the datasets pre-fetched in the booth
  preflight (fail soft with a printed warning — venue internet is a
  known risk); and an **idle attractor**: with nobody in frame for
  `BOOTH_ATTRACT_SECS` the robot plays a short dance or antenna flourish
  every few minutes — the family's "people watch from a distance before
  they dare to come over" — silenced by any face. Stub robot accepts
  `play_move` as a timed no-op. Tests: stub-side procedure and event
  contract; `robot`-marked playback of one dance; attractor timing on
  the fake clock.
- **T13.5 — Booth persona: a few lines of character.** The family wants
  the robot to have a couple of jokes with a mild edge: "I'll remember
  you" at enrollment success, a mock-dramatic farewell, and "you don't
  sound like yourself today — suspicious, say something else" once
  T13.9 exists. **Decision (2026-09-02): no allusions to Skynet,
  Terminator, or robots-taking-over; keep the edge gentle.** Build: a
  short `BOOTH_QUIPS`
  block in `tutor_mode.py` appended to the booth briefing (local and
  cloud), each quip tied to one moment and used at most once per
  visitor, in the target language when the learner is intermediate or
  above. Tests: static (the briefing carries the quips only in booth
  mode); one `--say` run per speech mode logs an enrollment-success line
  for a human to read; no LLM-judging.
- **T13.6 — The wishlist tool.** "If this were a product you bought,
  what would you want it to do?" A `record_wish` tool that appends the
  visitor's wish (with date and, if enrolled, first name) to
  `booth/wishes.md`, gitignored, plus a briefing line inviting the
  question once per visitor. **Decision (2026-09-02): the poster itself
  is a physical deliverable and out of scope for the code tasks.**
  Tests: `record_wish` round-trip; wishes survive the guest wipe.
*T13.7–T13.9 are nice-to-have (decision 2026-09-02): do them after
T13.1–T13.6 and T12, in this order, if time allows.*

- **T13.7 — Booth resilience: "the computer froze".** The debrief ended
  with the machine hanging and nobody knowing why. Build: an agent
  heartbeat line every 30 s (`alive: state=<idle|active> turns=<n>`); a
  stuck detector (face present, no bot turn for 3 min → WARNING plus a
  self-kick of the conversation); `start_booth.sh` restarts the agent
  if it dies while serve is healthy (bounded retries, logged); and a
  `booth/postmortem.sh` that snapshots the last 200 lines of each log,
  `dmesg` tail, memory and GPU state into `booth/postmortems/<ts>/` for
  the next time it happens. Tests: heartbeat and stuck detection on the
  fake clock; the restart path with a deliberately killed stub agent.
- **T13.8 — Code-switched *accent*: measure, mitigate, or document.**
  "English words inside a Russian sentence come out with a Russian
  accent." In cloud mode the voice is Gemini's; in local mode span tags
  already switch engines per span (T6). Build: a `google`-marked probe
  that synthesizes a fixed set of mixed sentences (EN inside ru/es/fr,
  ru/es inside EN) over Gemini Live, transcribes with Whisper, and
  scores span survival like T7; try one prompt-level mitigation
  ("pronounce embedded English as a native English speaker") and record
  whether it moves the number. Deliverable is the number plus a demo
  note in `booth/SIGNAGE.md`'s operator section: which pairs to
  showcase and which to avoid. No fixed pass bar.
- **T13.9 — Voice print as a second identity signal (stretch,
  cut-if-late).** Store a speaker embedding alongside the face
  embedding; on a later visit, face-known but voice-mismatched →
  playful challenge (T13.5's line) then fall into the ask band; face
  unsure but voice-matched → confirm without asking. Measurement-first
  like T2: pick a speaker-verification model that runs on aarch64
  (ECAPA/WeSpeaker via ONNX; **verify the execution provider, don't
  trust the reputation**), record same/different-speaker score
  distributions on CC0 clips added to `tests/fixtures/` with provenance,
  and choose thresholds from the numbers. Audio comes from pipecat's
  input frames in both speech modes; cloud mode still sends audio to
  Gemini, so the voice embedding is computed locally and never leaves
  the booth (add that line to Sign 1). Fully cuttable: nothing in
  T13.1–T13.8 depends on it. **Known-good starting point** (from the
  user's other project): SpeechBrain ECAPA,
  `speechbrain/spkrec-ecapa-voxceleb` via `EncoderClassifier` (older
  builds import from `speechbrain.pretrained`), torch/torchaudio, cosine
  match threshold 0.65 measured there; prints stay on the machine.

**E2E check.** In person, off `./start_booth.sh` in cloud mode: a new
visitor is asked level and goal before the first lesson; the head follows
them as they step left and back; "can you dance?" gets a dance; they
walk away, hear "still there?", then a goodbye, and their notes mention
the goal; a second visitor gets an "I'll remember you" on enrollment;
the attractor fires while nobody is in frame.

**Definition of Done.** T13.1–T13.7 done with tests green and the E2E
check logged in person; T13.8's numbers recorded; T13.9 either done or
explicitly cut in the Progress log. Second family test session held
with the booth mic (T11 item 1) and its debrief filed as
`progress/T13.md`'s next section.

**Progress log.**
- 2026-09-02 — task written from the family debrief transcript; digest
  and mapping in [progress/T13.md](progress/T13.md).
- 2026-09-02 (later) — **T13.1–T13.6 built and tested on the simulated
  path** (`tests/run.sh t13`: 33 unmarked + 1 models + 2 live-agent
  tests green; whole unmarked suite green). T13.1: goal/goal_note in
  profiles (legacy profiles load), four-question enrollment script,
  `set_learner_goal`; the interview verified live in both speech modes
  (level asked before enrolling, goal stored as `work`). T13.2: speech
  counts as presence, "still there?" at two thirds, spoken goodbye
  before the save, `presence:` log lines, `BOOTH_ABSENT_SECS`. T13.3:
  `detect()`/`analyze()`, `FaceTracker` (dead-band, rate cap, ±35°
  body handoff, pitch via embodiment bias, suspend/resync around
  deliberate moves), the shared `FrameHub` so the watcher, tracker and
  enrollment share one camera (a latent T10-on-metal bug), on by default
  with a camera and robot. T13.4: `moves.py` library, `play_move` on
  target/stub/driver with cancel, `perform` tool, `spin`/`wiggle` built
  in, idle `Attractor`, preflight dataset fetch. T13.5: gentle-edge
  quips only (`--persona booth`). T13.6: `record_wish` → `booth/wishes.md`.
  Stub-robot E2E: "can you dance for me?" → `perform(dance)` → the
  robot accepted `play_move` while the tracker ran over the fixture
  clip. **Not yet on metal**: the in-person E2E check (tracking a real
  visitor, a dance on the real robot, the still-there/goodbye timing)
  and the second family session with the mic remain. T13.7–T13.8 not
  started (nice-to-have). [progress/T13.md](progress/T13.md).
- 2026-09-02 (evening) — **T13.9 done on the simulated path.**
  SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) in
  `voice/voiceid.py`, on CUDA here (5 ms/clip). Measured gate
  (`voice/verify_voiceid.py`, 20 synthetic Kokoro clips, 5 voices):
  same-voice 0.717–0.898, different-voice −0.028–0.391 → accept ≥ 0.60,
  reject < 0.45, band clean; report in `tests/reports/`. Prints live in
  `profile.json` (`voice_embedding`), are computed locally in both
  speech modes, and die with the profile. Policy: face-sure + mismatch
  → one playful challenge, then "is that you?"; face-unsure + match →
  confirmed without asking; family profiles without a print learn one.
  Enrollment stores the interview's voice. Live tests: enrollment stored
  a print that matches a second clip of the same voice; a seeded face
  with the wrong voice was challenged in Spanish ("hoy no suenas del
  todo como tú misma"), then downgraded to the ask band. Sign 1 now
  discloses the voice signature. **Open**: thresholds are calibrated on
  synthetic voices only; measure the family's real voices at the next
  session (the gate takes `--dir`).

---

## T14 — Family feedback round 2: sight, calm tracking, one visitor at a time

**Goal.** What the second family session (2026-09-03, cloud voice, all of
T13 on) asked for, with the log evidence in
[progress/T14.md](progress/T14.md). Logs of every booth run are now kept
under `booth/logs/<date>_<name>/` (gitignored: names and transcripts).

**What the log showed.** One launch, 28 minutes, three people in turn
without anyone restarting the agent. John enrolled and said goodbye;
Rock walked up and tried to enroll and got "already tutoring John";
when John came back and said "I am John Maritime, I was before Rock",
the robot had one long Gemini history with everyone in it. Gemini's
own 10-minute session limit forced three reconnects, each replaying the
whole history. The voice check never fired for Rock because the running
print averaged Rock's samples into John's. The tracker and embodiment
together sent ~135 posture commands a minute. And the robot never
logged what it said (no output transcript in cloud mode).

**Subtasks.**

- **T14.1 — Sight on demand.** A `look` tool: grab the newest frame
  from the shared camera hub and show it to the model — cloud: push it
  as an input image to Gemini Live (its native video input); local:
  append an image message to Claude's context. Used when asked ("can
  you describe what you see?" was asked twice tonight), or when the
  robot wants to check who is in front of it. One frame per call, never
  a stream (privacy: the prompt says it sees faces only for recognition
  otherwise). Tests: tool round-trip on the fixture clip in both modes.
- **T14.2 — Calm tracking.** Gentler controller defaults (lower gain,
  wider dead-band, longer minimum interval, smaller steps, slower
  moves) and a quieter embodiment sway, so the head *settles* on the
  person instead of twitching; a `tracker:` summary line per minute so
  the log shows how much it moved. Tests: the controller tests updated
  to the new numbers; a fixture run asserts command rate ≤ 1/s.
- **T14.3 — One visitor at a time over the cloud voice (absorbs T12).**
  The session runner runs in cloud mode: on session start it sets the
  visitor's briefing as the context's system message and reconnects
  Gemini (fresh server-side history, no replay); on session end it
  cues the goodbye + notes, then resets the context and reconnects
  again. Two triggers end a session: the face walk-away timer (T13.2)
  and a **speaker change** from the voice print — compared per recent
  sample window, not the session mean, with two consecutive mismatches
  after a verified match meaning a new person is talking. A changed
  speaker gets the stranger flow (enroll or "is that you?") instead of
  "already tutoring John". Session-level `record_wish`/notes stay with
  the right person. `start_booth.sh` gets `BOOTH_SPEECH=cloud` as the
  default. Tests: simulated two-visitor sequence over cloud mode with
  `--voice-source` switching between speakers; existing local T10 tests
  untouched.
- **T14.4 — The wishlist question, asked like a person.** Only once
  the visitor says goodbye: "before you go, one question…", then wait
  for the answer, then `record_wish`, then the goodbye and the notes.
  Never announced mid-lesson. Tests: persona text; a scripted goodbye
  turn ends with `record_wish` fired *after* an answer.
- **T14.5 — Longer dances.** `play_move` takes `repeat`; the `perform`
  tool takes `seconds` (up to 60) and loops the move to fill it; the
  default dance runs ~30 s. "Funny music alongside" is noted, not built
  (the speaker belongs to the voice). Tests: stub playback with repeat,
  duration reported.
- **T14.6 — Spoken transcript in the log.** Log Gemini's output
  transcription as `said: …` next to `heard: …`, so a booth log is a
  readable two-sided transcript. Local mode already logs TTS lines.
- **T14.7 — Web search: enable it.** Guide in
  [progress/T14.md](progress/T14.md): billing on the AI Studio project
  behind `GEMINI_API_KEY`; the probe in the agent then switches it on
  by itself. No code.

**Definition of Done.** T14.1–T14.6 green on the simulated path; the
next family session run off one launch with people swapping in front
of the robot and each getting their own greeting, notes and wish.

**Progress log.**
- 2026-09-03 — task written from the second family session's logs.
- 2026-09-03 (late) — **T14.1–T14.6 built and green on the simulated
  path** (`tests/run.sh t14`: 12 unmarked, 1 models, 3 live). Live:
  the `look` tool described the fixture face in both speech modes; one
  cloud launch served two visits of the same clip — stranger enrolled
  with a voice print, said goodbye (notes), left (mic muted, idle),
  came back, was recognized at 0.992 and greeted by name in Spanish
  from a fresh Gemini session, left again (second notes entry) — and the
  log carried `said:` lines for every reply. The wrong-voice challenge
  → downgrade flow still passes with the per-sample decision. T14.7 is
  the guide in progress/T14.md (billing on the AI Studio project);
  nothing to build. **Open**: in person — does the retuned tracker
  read as calm, does a family member swapping in mid-session trigger
  the speaker change within two sentences, do the 30 s dances please.

## T15 — Family feedback round 3: identity for the whole session

**Goal.** What the third family session (2026-09-04, cloud voice,
everything from T14 on) asked for, with the log evidence in
[progress/T15.md](progress/T15.md) and the debrief the family recorded
afterwards. Logs in `booth/logs/2026-09-04_family/`.

**What the log showed.** Yaroslav's lesson was ended twice by the voice
print (his own French scored 0.20–0.45 against the print enrolled
minutes earlier; the face said the same person throughout and
re-recognized him five seconds later, so the goodbye ran straight into
a "welcome back"). Tanya enrolled, a voice challenge downgraded her to
"is that you?", somebody said yes, and from then on the voice check was
off; when Yaroslav's father took her seat the robot tutored him under
her name for five minutes, because the face is never looked at again
after the greeting. Three greetings in fifteen seconds at every
walk-up. The wish question never fired all day. "Comment dit-on
'novel' en français" asked in French sounds silly. The prompt said
the robot could not see while the `look` tool said it could.

**Subtasks.**

- **T15.1 — The face is the arbiter, for the whole session.**
  `SessionRunner` embeds the largest face every `face_recheck_secs`
  (2 s) during a session and compares it with the face that started
  it; a voice mismatch while that face is present is ignored (and the
  voice check re-armed); a different face held for `swap_secs` (3 s)
  ends the session (goodbye + notes) and the newcomer starts theirs;
  a face back after a gap is checked on the next frame. The frame loop
  is now `observe(face, now)`, testable without a camera. Spec §A
  updated. Tests: same face overrules the voice; swap after 3 s, not
  after a glimpse; gap forces a recheck; voice with nobody in frame
  still ends; live seat-swap clip over the cloud voice.
- **T15.2 — The voice asks, never ends.** `change_after` 2 → 4;
  samples under `decision_min_secs` (3 s) feed the print but are not
  judged; `VoiceIdentity.rearm()`; the real-voice scores from the booth
  logs recorded next to the thresholds. Calibration on the family's
  recordings is still a to-do (needs them present).
- **T15.3 — A yes is checked against the face.** `confirm_identity`
  captures the current face and refuses when it is below the
  recognizer's reject line (the model is told to ask their name and
  treat them as new); on success it re-arms the voice check.
  `face_confirms()` is the pure decision. Seam: `current_face`.
- **T15.4 — Tell the robot where it is and what it can see.**
  `VISION_FACE_LOOK` (the look tool exists, never claim blindness);
  `BOOTH_NOTE` on every prompt the runner builds ("people swap seats");
  the briefing's "Setting a task" rule (task in English below advanced,
  never a one-word "how do you say" whose answer is in the question,
  short role-play).
- **T15.5 — One greeting.** `inference_on_context_initialization=False`
  in session mode: pipecat seeds each connection with the prompt as a
  user turn and, by default, has the model answer it -- that was the
  greeting to an empty chair at startup and one of the three at every
  walk-up. The runner's walk-up cue is the only greeting.
- **T15.6 — Clean endings, the wish question on a real goodbye.**
  The runner waits for the robot to finish speaking before the next
  greeting (`speaking` seam, `Embodiment.bot_speaking`);
  `save_session_notes` for a visitor still present answers with the
  wish question to ask next (`wish_followup`, `holder.walkaway` /
  `wish_recorded`).
- **T15.7 — Measure the lag.** `turn: first sound Ns after the visitor
  stopped speaking` per reply (local: pipecat's user-stopped frame;
  cloud: the voice collector's energy gate). Tuning waits for numbers.
- **T15.8 — Identity protocol for the next family session.** In the
  booth checklist (`start_booth.sh`) and progress/T15.md; each row has
  a simulated twin in `tests/t15/`.
- **T15.9 — Recorded moves own the body; gotos touch only what they
  name (from the 2026-09-04 evening run).** The driver refuses any
  nudge (`goto_posture`, `nod`, `shake`, `look_at`) while a `play_move`
  runs -- only another move, `home`, `sleep`, `wake_up` or
  `cancel_motion` may cut it; before this the embodiment's talking
  sway superseded every dance within ~1 s, and a `move_head` 8 ms
  after a cheer was the head jerk. The embodiment is held quiet and
  the other motion tools answer "skipped" for the move's duration; the
  vendor's initial goto to a move's first frame is 1.0 s (was 0.4).
  `reachy_target.goto` passes only the joint groups the caller named,
  so a head nudge no longer re-drives the base servo from its present
  position to the held target (the ~2 Hz base twitch; probe in
  `tests/t15/probe_body_twitch.py`). Also the false-goodbye chain:
  `save_session_notes` takes `farewell`, the wish follow-up needs it,
  and the briefing/persona say that teaching the word goodbye (or a
  dance request) is not the student leaving. Tests: driver refusal on
  the stub, goto groups on a fake vendor object, embodiment hold and
  tool skips in the voice venv, farewell gating and prompt text.
- **T15.10 — Answerable sight and identity (from the 20:27 run).** The
  robot described "someone with glasses and a blue shirt" twice, then
  the actual visitor; nothing could say whether a bystander, a face on
  the open laptop, or a hallucination. Now every `look` frame is saved
  under `booth/logs/looks/<date>/` (`--look-dir`, '' to disable) and
  the log line names the file; every mid-session face check is logged
  (a verdict change at once, else a 10 s summary with the score
  range); and confident matches are averaged into the session's
  reference face, so a poor walk-up likeness does not leave a visitor
  in the unsure band for minutes. Tests: frame saver, check logging,
  reference strengthening.
- **T15.11 — The base hunts after a recorded move.** Measured live
  (progress/T15.md): after a dance the base servo limit-cycled ±0.5°
  at 3.5 Hz around a target 1° away, for minutes, with nothing
  commanding it; one interpolated goto to a clean body_yaw stopped it.
  `reachy_target.play_move` now ends with a 0.6 s goto back to the
  body yaw held before the move. Test: fake vendor object sees
  play then goto(body_yaw=before).

**Definition of Done.** `tests/run.sh t15` green (unit + live seat
swap); the next family session run with people swapping seats
mid-lesson and each getting their own greeting and notes, one greeting
per walk-up, and `turn:` lines in the log.

**Progress log.**
- 2026-09-04 — task written from the third family session's logs and
  the recorded debrief; T15.1–T15.7 built the same evening.
- 2026-09-04 (late) — **T15.1–T15.7 green on the simulated path**
  (`tests/run.sh t15`: 17 unit + 2 live; whole unmarked suite 127
  passed). The live seat-swap clip over the cloud voice: swap caught in
  4.1 s, goodbye + notes for the person who left, the newcomer greeted
  once from a fresh Gemini session, nothing said before the first
  walk-up. Its first run found two more holes (enrollment stored the
  face present at call time; the goodbye was not yet playing when the
  runner checked), both fixed. **Open**: the in-person protocol
  (T15.8), real-voice calibration, `turn:` numbers on a real mic.
- 2026-09-04 (evening, on metal) — first booth run on T15: one
  greeting, the face recheck and the `turn:` line (1.3–2.7 s) all
  showed; three new faults seen live and fixed as **T15.9** the same
  night (see progress/T15.md "Seen live"): head jerks and one-second
  dances (both: anything superseded a recorded move), a base twitch at
  ~2 Hz (every goto re-drove the body servo), and a goodbye nobody said
  that then triggered the wish question twice.
- 2026-09-04 (night) — **T15.10** after the 20:27 run's "someone with
  glasses": look frames kept on disk, every face check logged, the
  reference face strengthened by confident matches. `tests/run.sh t15`
  26 passed.
- 2026-09-04 (night) — **T15.11**: the base oscillation was measured
  live with the booth up: a servo limit cycle left behind by the dance,
  not a command; fixed by settling the base after every recorded move.

