# Reachy the Language Tutor — Product Specification

*A family language tutor built into a Reachy Mini robot, for Maker Faire Bay Area 2026.*

This document says what we are building, how it works, and what is left to
build. It is written for a non-expert; every technical term is explained the
first time it appears, and there is a glossary at the end.

---

## 1. What we are building

A desk robot that acts as a personal language tutor for a whole family. Walk
up to it and it looks at you, recognizes your face, remembers what *you* have
been practicing — your vocabulary, your recurring mistakes, how far you have
come — and starts a spoken conversation in your target language, pitched to
your level. When you leave, it writes down what you worked on. Next time, it
picks up where you left off.

At Maker Faire, visitors can introduce themselves, have a short tutoring
conversation, walk away, and come back later to see whether the robot
remembers them and what they practiced. It does.

**Tutoring languages:** Spanish, French, Italian, Portuguese, Russian, and
Mandarin, with English as the language of explanations. The first four work
today; Russian and Mandarin need one new component (see section 6).

**Priority:** the Maker Faire demo comes first. Every trade-off in this
document — session length, model choice, which features get hardened —
is decided in favor of a booth that works flawlessly in a loud, crowded
hall. Long-term family use is the second audience, not the first.

**Hard requirement — mixed-language phrases.** Real learner speech mixes
languages mid-sentence (*"How do you say 'la biblioteca' in French?"*), and
the tutor's replies do too. Both directions MUST work seamlessly: hearing a
single phrase that contains two languages, and speaking one. Section 7
defines how, including a switchable cloud speech mode that handles this
natively.

---

## 2. What a session looks like

1. You step in front of the robot. It sees you, perks up (head tilts, antennas
   rise), and greets you **by name** — or, if it has never seen you, asks who
   you are and which language you would like to practice.
2. It opens with something personal: *"¡Hola Maria! Last time the past tense
   of 'ir' kept tripping you up — shall we warm up with that?"*
3. You talk. It listens, replies out loud in the target language, gently
   corrects mistakes, and drops in an occasional English explanation when
   you are stuck. It nods when you get things right, shakes its head kindly
   when you do not, and sways while it talks.
4. You say goodbye. The robot summarizes the session to itself — new words
   covered, mistakes to revisit, an updated sense of your level — and saves
   that to your personal file.
5. Next session starts at step 1, but now it knows more about you.

---

## 3. What already exists (the foundation)

This project is not starting from zero. This repository already contains a
working two-part system, verified on the physical robot:

**The robot driver** (`controller.py serve` + `reachy_driver.py`) — a program
that owns the robot's motors over USB and exposes them as safe, named
commands: *nod*, *shake your head*, *look here*, *wiggle antennas*, plus an
emergency stop. Anything on the machine (or the network) can send it those
commands using **Device Connect**, a messaging protocol — think of it as a
common language that lets separate programs talk to devices without knowing
how the hardware works inside.

**The voice agent** (`voice/agent.py`) — a program that has a real-time
spoken conversation through the robot's own microphone and speaker. Almost
everything runs locally on this computer:

- **Speech-to-text** (turning your voice into written words): Whisper, a
  speech-recognition model, running on this machine's GPU. It also detects
  *which language* you spoke.
- **Text-to-speech** (turning the reply into a voice): Kokoro, a local voice
  synthesizer with per-language voices.
- **Turn detection** (knowing when you have finished talking and it is the
  robot's turn): a small local model, so the robot does not interrupt you.
- **The brain**: Claude, Anthropic's large language model (**LLM** — an AI
  that understands and generates language), reached over the internet. This
  is the only part that leaves the machine.

The voice agent already speaks six languages, switches language and voice
automatically based on what it hears, and moves the robot's body while it
talks. A conversation turn takes roughly 3–5 seconds from you finishing a
sentence to the robot starting its answer.

---

## 4. What we are adding (the tutor)

Three new capabilities turn "a robot you can chat with" into "a tutor that
knows you":

| # | Capability | One-line description |
|---|---|---|
| A | **Face recognition** | The robot's camera identifies who is standing in front of it. |
| B | **Per-learner memory** | A file on disk per person: level, vocabulary, mistakes, session history. |
| C | **Tutor dialog mode** | Claude is briefed with your file and instructed to *teach*, not just chat — and to update your file when you leave. |

### A. Face recognition

**What it is.** The robot has a built-in USB camera (currently unused). We
grab an image from it, find the face in the image, and convert that face into
an **embedding** — a list of a few hundred numbers that acts like a
fingerprint of the face. Two photos of the same person produce nearly the
same numbers; two different people do not. Recognition is just comparing the
new fingerprint against the saved ones and picking the closest match (if it
is close enough — otherwise the person is a stranger).

**How we build it.**

- Library: **InsightFace** (a well-established open-source face-analysis
  toolkit) with its standard `buffalo_l` model pack, running through ONNX
  Runtime. Chosen because it is accurate, runs fine on this machine's ARM +
  NVIDIA hardware, and needs no cloud service.
- Camera access: the driver already has a `set_media_released` command that
  hands the robot's camera, mic and speaker to the voice agent; the face
  module reads frames from that released camera using OpenCV (a standard
  image library).
- **Enrollment** (meeting someone new): when no saved face matches, the robot
  asks *"I don't think we've met — what's your name?"*. The spoken answer
  becomes the learner's name; the robot captures 3–5 face snapshots over a
  few seconds, averages their embeddings, and saves the result. No dialog
  boxes, no keyboard — enrollment happens in conversation.
- **Two tiers of learner.** Family members (the adults and teens pre-enrolled
  before the event) are *permanent* profiles. Everyone enrolled at the booth
  is a *guest*: a full, working profile — they can leave and come back an
  hour later and be remembered — that is automatically deleted when the
  booth closes for the day. The tier is a single flag in the profile; the
  experience is identical while it lasts.
- **When it runs**: continuously between sessions (about twice per second,
  watching for a face to appear), once more at session start to confirm,
  and — since T15 (2026-09-04) — every couple of seconds *during* the
  conversation as well, comparing the face in front of the robot with the
  face that started the session. The original design paused recognition
  mid-session; at the third family session one visitor took another's seat
  and was tutored under her name for five minutes. The face is now the
  arbiter of identity for the whole session: the same face keeps it alive
  whatever the voice print says, a different face held for a few seconds
  ends it (notes for the person who left) and starts the newcomer's own.
  Recognition runs on the CPU, so it does not compete with speech.
- **Match threshold**: we compare embeddings with cosine similarity and
  require a comfortable margin before claiming a match. When unsure, the
  robot asks — *"Maria, is that you?"* — rather than guessing. A wrong
  greeting is the worst failure mode this feature has, so we bias toward
  asking.

### B. Per-learner memory

**What it is.** A folder per learner on this computer's disk. Plain files,
readable by a human — no database server, nothing in the cloud.

```
learners/
  maria/
    profile.json      # name, target language, level, face fingerprint
    notes.md          # the tutor's running notes, newest session first
  dad/
    profile.json
    notes.md
```

`profile.json` holds the structured facts:

- name (as spoken at enrollment)
- target language and self-declared or estimated level
  (beginner / intermediate / advanced)
- the face embedding (the number-fingerprint — **not** a photo)
- session count and last-seen date
- tier: `family` (permanent) or `guest` (auto-deleted at end of day)

`notes.md` is written *by the tutor itself* at the end of each session, in a
fixed short format:

- **Practiced:** topics and vocabulary covered this session
- **Struggled with:** specific mistakes, with the correction
- **Wins:** things that clicked
- **Next time:** what to open with

Keeping the notes as prose that Claude writes and later reads is deliberate:
it is exactly the form the LLM uses best, it is trivially inspectable and
editable by us, and there is nothing to migrate or administer.

### C. Tutor dialog mode

**What it is.** The existing voice agent already lets Claude converse and
move the robot. Tutor mode changes what Claude is *told to do* and what it
is *able to save*.

**Briefing (at session start).** When a face is recognized, the agent loads
that learner's `profile.json` and the most recent sessions from `notes.md`
and injects them into Claude's **system prompt** — the standing instructions
an LLM is given before a conversation starts. The prompt says, in essence:

> You are a friendly language tutor in a small robot body. Your student is
> Maria, intermediate Spanish, 7th session. Her recent notes are below.
> Speak mostly Spanish at her level; explain in English only when she is
> stuck. Correct errors briefly and kindly — never let one slide, never
> lecture. Keep replies to one or two short sentences (this is a spoken
> conversation). Nod for right answers, shake your head gently for wrong
> ones. Open by picking up where the notes leave off.

**New tools (at session end and during it).** A "tool" here is an action the
LLM is allowed to take besides talking — the agent already gives Claude seven
motion tools (nod, look, etc.). We add:

| Tool | What Claude uses it for |
|---|---|
| `save_session_notes` | Write the end-of-session summary into the learner's `notes.md`. Called when the learner says goodbye, and also automatically if they simply walk away (face gone for ~60 seconds). |
| `update_learner_level` | Promote/demote the stored level when the evidence is clear. |
| `enroll_new_learner` | Create the folder and capture the face during the "what's your name?" exchange. |
| `confirm_identity` | When the face match is uncertain and the person says "yes, it's me". |
| `set_target_language` | Switch the language a learner practices when they ask to, and remember it. |
| `forget_me` | Delete the learner's folder on the spot (section 8). |
| `set_volume` | The robot's own speaker level, by request ("speak up", "quieter"). |

**Level adaptation** is prompt-driven, not code-driven: beginner sessions get
slower, simpler target-language sentences and more English; advanced sessions
get target-language-only with idioms. The stored level just selects the
briefing; Claude handles the moment-to-moment pitch.

---

## 5. How it all fits together

```
                       ┌───────────────────────────────────────────────┐
                       │            this computer (DGX Spark)          │
                       │                                               │
  Reachy camera ───────┼─▶ Face recognition ──▶ learner profile        │
                       │      (InsightFace)        (learners/ folder)  │
                       │                                │ briefing     │
                       │                                ▼              │
  Reachy mic ──────────┼─▶ Speech-to-text ──▶  Claude (cloud) ◀──────┐ │
                       │      (Whisper, local)      │       │ notes  │ │
                       │                            │       ▼        │ │
  Reachy speaker ◀─────┼── Text-to-speech ◀─────────┘   learners/ ───┘ │
                       │   (Kokoro + Piper, local)                     │
                       │                            │ motion commands  │
                       │                            ▼                  │
  Reachy motors ◀──────┼── robot driver (controller.py serve)          │
                       │        via Device Connect                     │
                       └───────────────────────────────────────────────┘
```

Everything except Claude runs on this machine. Faces, voices, and learner
files never leave it; the only thing sent to the cloud is the conversation
text plus the learner's notes in the briefing.

(This diagram shows **local speech mode**, the default. In **cloud speech
mode** — section 7 — the three speech boxes are replaced by one streaming
connection to Gemini Live; face recognition, learner files, and the robot
driver are unchanged.)

**Hardware:** an NVIDIA DGX Spark (a small desk computer with a powerful
GPU — the chip that runs the AI models fast), connected to a Reachy Mini
robot over a single USB-C cable that carries the motors, camera, microphone
and speaker.

---

## 6. Languages

The family's six tutoring languages split into two groups:

**Working today — Spanish, French, Italian, Portuguese** (plus English for
explanations, and Hindi comes along for free). Each was verified end-to-end
at 96–100% round-trip accuracy with its own natural voice from Kokoro, our
existing text-to-speech engine.

**Needing new work — Russian and Mandarin.** The listening side is already
fine: Whisper recognizes both well. The gap is the *speaking* side — Kokoro
has no Russian voice at all, and its Mandarin was measured at 41%
intelligibility (it sounds fluent and means nothing; see
[voice/README.md](voice/README.md)). The plan:

- Add **Piper**, a second local text-to-speech engine, used *only* for
  Russian and Mandarin. Piper is small, fast, runs fully on this machine,
  and ships well-regarded voices for both languages.
- The agent routes each reply to the right engine by language — Kokoro for
  the four verified languages, Piper for these two. The existing language
  router already switches voices per turn; this extends it to switch
  *engines* per turn.
- Before either language is offered at the booth, it must pass the same
  round-trip test the other four did (speak a sentence, transcribe it back,
  compare): we require ≥90% before it goes in front of visitors. If Piper's
  Mandarin falls short, the documented fallback is proper Chinese
  phonemization (misaki) into Kokoro's existing Chinese voices.

One tutoring-specific wrinkle: the existing agent auto-switches its language
to whatever it hears. In tutor mode that stays on — a learner mixing English
and Spanish is normal — but the *target* language comes from the learner's
profile, not from detection, so one English question does not derail a
Spanish lesson.

### Mixed-language phrases (code-switching) — a MUST

**Code-switching** means mixing languages inside a single phrase, and it is
the natural register of language learning: an English sentence with a
Spanish word being asked about, a Spanish sentence that falls back to
English halfway. This must work in both directions.

**Hearing it (speech-to-text).** This is the local stack's weak spot:
Whisper decides on *one* language per utterance and transcribes everything
under that assumption, so an embedded foreign phrase can come back
translated, misspelled, or forced into the wrong alphabet. Local
mitigations, applied in order:

1. **Bilingual priming.** Whisper accepts a short priming text before each
   transcription; we feed it a bilingual one (English + the learner's target
   language) so both languages stay "in mind" mid-utterance.
2. **The brain repairs the ears.** Claude receives the transcript *with* the
   learner's profile and lesson context, and is instructed that transcripts
   may garble embedded foreign words — it can usually reconstruct what a
   Spanish learner must have said. Good enough for tutoring flow; not for
   pronunciation judging.
3. **Measured, not assumed.** A scripted test set of code-switched phrases
   per language pair (synthesized, transcribed back, compared — the same
   method that validated the six languages) gates the feature. If a pair
   fails locally, cloud mode (section 7) is the guaranteed path: a native
   audio model has no per-utterance language commitment at all.

**Speaking it (text-to-speech).** Local voices are per-language, so a mixed
reply is *assembled*: Claude tags each language span in its own reply with a
lightweight marker (it always knows which language it is writing), and the
speech layer splits the reply at those markers, synthesizes each span with
the matching voice and engine (Kokoro or Piper), and stitches the audio
together seamlessly. One honest side-effect: the voice's timbre shifts
slightly at a language boundary, because each language has its own voice.
We choose the most similar-sounding voice pairs to minimize it; cloud mode
has no seam at all (one model, one voice, any mix).

---

## 7. Two speech modes: local and cloud

The whole hear-think-speak loop comes in two switchable modes, selected by
one flag at startup (`--speech local` / `--speech cloud`). Everything else —
face recognition, learner memory, robot motion, the tutor briefing — is
identical in both.

**Local mode (the default).** The stack described so far: Whisper listens,
Claude thinks, Kokoro/Piper speak, all speech processing on this machine.
Best privacy (only conversation *text* ever leaves the house), no dependence
on streaming bandwidth, and Claude as the tutor brain. Mixed-language
support is engineered and test-gated as in section 6.

**Cloud mode.** The three speech stages are replaced by a single
**speech-to-speech model** — one AI that listens to raw audio and answers
with raw audio, with no separate transcription or synthesis steps. We use
Google's **Gemini 3.1 Flash Live** over its streaming Live API. What it buys:

- **Native mixed-language handling** — no per-utterance language
  commitment, 90+ languages, one consistent voice across any mix. This is
  the guaranteed path for the code-switching MUST.
- **Speed** — sub-second responses, and the visitor can interrupt the robot
  mid-sentence (full-duplex audio), which local mode cannot do.
- **Tool calls still work** — the robot's motion tools and the memory tools
  (save notes, enroll) ride along; the model calls them mid-conversation.
- **Web search built in** — the tutor can look things up live, a nice booth
  moment ("ask it anything about Spain").

What it costs: raw audio streams to Google (a privacy step down from
text-only), it needs solid venue internet continuously, per-minute cloud
pricing, and **the tutor brain is Gemini, not Claude** — a speech-to-speech
model bundles the brain with the voice, so the Haiku/Opus choice in section
9 applies to local mode only.

| | Local (default) | Cloud (Gemini Live) |
|---|---|---|
| Mixed-language phrases | engineered + test-gated | native |
| Tutor brain | Claude | Gemini |
| What leaves the machine | conversation text only | live audio stream |
| Response time | ~2.7–5s | under a second |
| Interruptible mid-reply | no | yes |
| Voice across languages | shifts at boundaries | one voice throughout |
| Needs internet | a little (text) | a lot (streaming audio) |

This is a mode switch rather than a second application because pipecat, the
pipeline framework the agent is already built on, ships Gemini Live as a
drop-in speech-to-speech service — same microphone, same robot tools, same
learner files, different middle.

**Which mode runs at the booth** is decided at dress rehearsal by testing
both against real noise and real venue internet. Working assumption: local
is the baseline (it cannot be taken down by Wi-Fi), cloud is switched on for
the mixed-language showpiece when the connection proves solid.

**Rehearsal update (2026-09-02).** The first in-person session ran both
modes. The family judged the local voice poor and the Gemini voice good,
and cloud replies came in well under a second (measured 0.6–1.7 s to
first audio versus 2.7–5 s locally), so **cloud is now the intended
booth default** and local is the fallback if venue internet fails. The
one thing cloud mode does not yet do is the hands-free visitor loop
(section 9): today it is one visitor per launch, because the
speech-to-speech service keeps the conversation on Google's side and
cannot simply be handed a new learner mid-run. Closing that gap is
build-plan item 12 below.

## 8. Privacy — especially for the Faire

Face data is sensitive, and Maker Faire visitors are the public, so:

- **Consent first.** The robot only enrolls someone who has verbally agreed
  ("Would you like me to remember you for the rest of the day?"). A sign at
  the booth says the demo recognizes faces and stores them locally.
- **Guests expire automatically.** Every booth enrollment is a guest profile
  that lives until close of day, then a wipe script deletes all of them —
  no manual step to forget. Only the family's own profiles (adults and
  teens) survive the event.
- **No photos stored.** Only the numeric embedding is kept — it cannot be
  turned back into a picture of you.
- **Nothing leaves the machine — in local speech mode.** No cloud face
  services; recognition and speech synthesis are fully local, and only
  conversation text goes to Claude's API. In cloud speech mode (section 7),
  live audio streams to Google's API instead; the booth sign says which mode
  is running, and face data stays local in both.
- **Easy forgetting.** "Forget me" spoken to the robot deletes the learner's
  folder on the spot, without waiting for the end-of-day wipe.
- Children only with a parent's okay, same as any booth photo. The family's
  own demo profiles are adults and teens only.

---

## 9. Running it at Maker Faire

Practical realities, from what we have already measured:

- **Noise is the enemy.** The robot's built-in mic is sensitive and the hall
  will be loud. Decision: a directional handheld/headset mic handed to each
  visitor, selected with the agent's existing `--audio-device` flag. The
  robot's built-in mic is the at-home path, not the booth path. The agent's
  self-hearing mutes (it deafens itself while speaking) stay on.
- **The booth runs Haiku.** Claude Haiku turns are about 2 seconds faster
  than Opus (~2.7s vs 3–5s from end of speech to first sound), and in a
  3-minute demo, snappiness beats depth. Opus remains the default for real
  tutoring sessions at home, where correction quality matters more than
  pace. This is one command-line flag (`--model`), already supported.
- **Startup takes ~40 seconds** (robot connect ~15s, AI models warming
  ~10s). We start everything before the doors open and leave it up; the
  runbook is [CLAUDE.md](CLAUDE.md).
- **Sessions are short.** Tutor sessions at the booth target 2–4 minutes;
  the walk-away timeout (face gone 60s → save notes, reset to neutral,
  return to watching) keeps the line moving without anyone pressing a
  button.
- **Demo insurance.** The agent's `--say` flag injects a typed line as if
  spoken — if the hall gets too loud for any mic, we can still demonstrate
  the full recognize → remember → converse loop.
- **The robot takes requests about itself.** Visitors ask it to speak up,
  turn around, or switch languages mid-lesson; all three are tools it can
  call, so the booth crew never touches a mixer or a keyboard for them.
- **The comeback moment is the show.** The family's own profiles (adults and
  teens, with real session history) are pre-enrolled, so any visitor can
  immediately *watch* the robot recognize someone and resume a lesson, even
  before they enroll themselves as a guest.

---

## 10. Build plan

> The implementation-level breakdown of this plan — one task per coding
> session, each with integration tests and a progress tracker — lives in
> [TASKS.md](TASKS.md).

Ordered so that every milestone is demoable on its own:

| Milestone | Deliverable | Builds on |
|---|---|---|
| 1. Camera online | Grab frames from the Reachy camera on this machine; show a face box on screen. | `set_media_released`, OpenCV |
| 2. Recognition | Enroll two family members from snapshots; robot greets each by name (voice line only). | InsightFace |
| 3. Memory | `learners/` folder format with family/guest tiers; agent loads a profile into the system prompt; `save_session_notes` tool writes back. | existing agent tools |
| 4. Tutor prompt | Full tutor briefing; level adaptation; correction style; verified in `--say` scripted runs per language. Booth runs Haiku, home runs Opus. | milestone 3 |
| 5. Russian + Mandarin | Piper engine integrated and routed by language; both pass the ≥90% round-trip test or fall back per section 6. | existing language router |
| 6. Mixed-language | Span-tagged multilingual TTS assembly; bilingual Whisper priming; the code-switched test set passes per language pair. | 4 + 5 |
| 7. Cloud speech mode | `--speech cloud` runs the session over Gemini Live with the same motion + memory tools and tutor briefing; verified against the mixed-language test set. | pipecat's Gemini Live service |
| 8. Conversational enrollment | "What's your name?" flow creates a guest profile and captures the face, all by voice, in both speech modes. | 2 + 3 |
| 9. Session lifecycle | Face-triggered start, walk-away save, reset between visitors, end-of-day guest wipe script. | all above |
| 10. Faire hardening | Handheld/headset mic path (`--audio-device`), booth signage, family profiles pre-enrolled with real history, local-vs-cloud bake-off on venue internet, full dress rehearsal. | all above |
| 11. Cloud visitor loop | The hands-free watch → greet → tutor → save → reset loop working over Gemini Live, since cloud is the chosen booth voice. | 7 + 9 |

Milestone 5 is the one that can be cut without touching anything else: if
Piper disappoints or time runs out, the booth ships with four languages and
Russian/Mandarin become post-Faire work. Milestones 6 and 7 are the two
routes to the mixed-language MUST — at least one of them has to land, which
is exactly why there are two.

---

## 11. Risks and open questions

- **Face recognition of strangers in a crowd.** A booth has many faces in
  frame at once; we track only the largest (closest) face and require it to
  be stable for ~2 seconds before greeting. Needs testing with real crowds.
- **InsightFace on this machine's ARM architecture.** Expected to work via
  ONNX Runtime, but this box has already taught us (see CLAUDE.md) that
  library GPU support on ARM often differs from its reputation — we verify
  GPU use explicitly at milestone 1 and fall back to CPU (face recognition
  is cheap enough) if needed.
- **Piper quality is unproven here.** Its Russian and Mandarin voices are
  well-regarded, but so was the engine that produced 41% Mandarin — nothing
  ships without passing our own round-trip measurement. Milestone 5 is
  explicitly cuttable for this reason.
- **Local code-switched *hearing* may top out below "seamless."** Whisper's
  one-language-per-utterance design is a real ceiling; priming and
  Claude-side repair narrow the gap but may not close it for every pair.
  That is why cloud mode exists as the guaranteed path — the risk is not
  "the MUST fails" but "the MUST requires internet."
- **Cloud mode swaps the brain.** In cloud mode the tutor is Gemini, not
  Claude — tutoring tone, correction style, and tool-calling reliability all
  need their own verification pass, not an assumption that the local mode's
  prompt work carries over. It also runs on venue internet and per-minute
  pricing; the dress-rehearsal bake-off decides whether it is the booth
  default or the showpiece switch.
- **Latency budget.** Booth turns on Haiku should run ~2.7s. Adding the
  learner briefing makes the prompt longer but does not add a network round
  trip; we re-measure at milestone 4. One known wrinkle: Haiku rejects the
  `--effort` speed setting, which the agent already handles automatically.
- **Two people talking at once** confuses turn detection. Booth layout
  (one chair, one mic) is the mitigation, not software.
- **Claude API dependence + venue Wi-Fi.** The one cloud hop needs internet.
  Mitigation: phone hotspot as backup uplink; there is no offline mode.

---

## Glossary

- **Code-switching** — mixing two or more languages within a single phrase
  or conversation; the natural way learners actually speak.
- **Device Connect** — a messaging protocol that lets separate programs
  discover and control devices (here, the robot) without sharing code.
- **Embedding** — a list of numbers an AI model produces to summarize
  something (a face, a sentence) so that similar things get similar numbers.
- **GPU** — the processor that runs AI models quickly; this computer's is
  made by NVIDIA.
- **LLM (large language model)** — an AI system, like Claude, that
  understands and generates human language.
- **Speech-to-speech model** — a single AI that listens to raw audio and
  replies with raw audio, with no separate transcription or synthesis steps
  (here: Gemini 3.1 Flash Live, the cloud speech mode).
- **Speech-to-text (STT)** — converting spoken audio into written words
  (done here by Whisper, locally).
- **Text-to-speech (TTS)** — converting written words into a spoken voice
  (done here locally: Kokoro for Spanish/French/Italian/Portuguese, Piper
  for Russian/Mandarin).
- **System prompt** — the standing instructions given to an LLM before the
  conversation starts; how we turn a chatbot into *this learner's* tutor.
- **Tool (for an LLM)** — a named action the model is allowed to take
  besides replying with text: nod, save notes, enroll a face.
- **Turn detection** — deciding that a speaker has finished and it is the
  other party's turn to talk.
- **VAD (voice activity detection)** — noticing that someone is speaking at
  all, as opposed to background noise.
