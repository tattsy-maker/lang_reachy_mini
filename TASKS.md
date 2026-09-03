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
| T12 | Cloud session lifecycle (watch / walk-away / reset over Gemini) | T8, T10 | todo | — |

Parallelizable from the start (after T0): T1, T3, T5, T7 have no
dependencies on each other. The critical path is
T0 → T1 → T2 → T9 → T10 → T11. T12 was added after the first
in-person rehearsal (2026-09-02): the family chose the cloud voice, and
the booth loop must work over it.

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
- —
