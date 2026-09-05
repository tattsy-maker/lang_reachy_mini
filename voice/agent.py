#!/usr/bin/env python3
"""agent.py -- talk to the Reachy Mini.

A low-latency voice loop where everything except the language model runs on this
laptop, and the robot is reached over Device Connect rather than driven directly:

    Reachy mic -> Silero VAD + smart-turn -> MLX Whisper -> Claude -> Kokoro -> Reachy speaker
                        (local)               (local)      (cloud)   (local)
                                                  |
                                            tool calls
                                                  |
                                                  v
                              Device Connect invoke -> controller.py serve -> motors

Run it:

    # terminal 1 -- the robot. zenoh is brokerless, so there is nothing to start
    cd .. && .venv/bin/python controller.py serve

    # terminal 2 -- the voice agent
    export ANTHROPIC_API_KEY=sk-ant-...
    .venv/bin/python agent.py

Then talk to it. "Look to your left." "Nod if you can hear me." "What can you
see?" ... and it answers while it moves.

The head/antenna motion under the conversation is not scripted by the model --
see embodiment.py. The model only drives the deliberate gestures, through the
tools below.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import contextlib
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger as loguru_logger                              # noqa: E402
from pipecat.adapters.schemas.function_schema import FunctionSchema     # noqa: E402
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema  # noqa: E402
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (         # noqa: E402
    LocalSmartTurnAnalyzerV3,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer                  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams                    # noqa: E402
from pipecat.frames.frames import (                                     # noqa: E402
    BotStoppedSpeakingFrame,
    InputImageRawFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.pipeline.pipeline import Pipeline                          # noqa: E402
from pipecat.pipeline.runner import PipelineRunner                      # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask          # noqa: E402
from pipecat.processors.aggregators.llm_context import (                # noqa: E402
    LLMContext,
    LLMSpecificMessage,
)
from pipecat.processors.frame_processor import FrameProcessor           # noqa: E402
from pipecat.services.google.frames import LLMSearchResponseFrame       # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (     # noqa: E402
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import (                                   # noqa: E402
    AlwaysUserMuteStrategy,
    FunctionCallUserMuteStrategy,
)
from pipecat.turns.user_stop import (                                   # noqa: E402
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies       # noqa: E402
from pipecat.services.anthropic.llm import AnthropicLLMService          # noqa: E402
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX  # noqa: E402
from pipecat.transports.base_input import BaseInputTransport          # noqa: E402
from pipecat.transports.local.audio import (                            # noqa: E402
    LocalAudioInputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from embodiment import Embodiment                                       # noqa: E402
from robot_link import RobotLink                                        # noqa: E402
from multilingual import (                                              # noqa: E402
    LANGUAGES,
    SPOKEN_LANGUAGES,
    LanguageRouter,
    MultilingualKokoro,
    MultilingualWhisperMLX,
    bilingual_priming,
)
from piper_tts import (                                                 # noqa: E402
    DEFAULT_VOICES_DIR as PIPER_VOICES_DIR,
    PIPER_LANGUAGES,
    DualEngineTTS,
    piper_available,
)
from tutor_mode import (                                                # noqa: E402
    BRIEFING_SESSIONS,
    DEFAULT_LEARNERS_ROOT,
    PERSONAS,
    STRANGER_BRIEFING,
    CurrentLearner,
    LearnerStore,
    build_briefing,
    build_enrollment_tools,
    build_persona,
    build_tutor_tools,
    build_unsure_briefing,
    language_name,
    load_learner,
    native_language_of,
    voice_cue,
)
from tracking import FaceTracker, TrackingLoop                          # noqa: E402
from audio_devices import (                                             # noqa: E402
    MIC_DEVICE_NAME,
    SPEAKER_DEVICE_NAME,
    choose_audio_devices,
    input_rate_for,
    list_audio_devices,
    parse_mic_prefs,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("agent")

DEFAULT_BROKER = "zenoh://"
DEFAULT_DEVICE_ID = "reachy-mini-1"
DEFAULT_TENANT = "lab"
AUDIO_DEVICE_NAME = SPEAKER_DEVICE_NAME
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOOK_DIR = os.path.join(os.path.dirname(_HERE), "booth", "logs", "looks")
ENV_FILE = os.path.join(_HERE, ".env")

# Where a .env might reasonably live: next to this script, at the project root
# one level up, or in whatever directory the agent was launched from. Checked in
# that order; the first file to define a given key wins.
ENV_CANDIDATES = (
    ENV_FILE,
    os.path.join(os.path.dirname(_HERE), ".env"),
    os.path.join(os.getcwd(), ".env"),
)


def load_env_file(paths=ENV_CANDIDATES) -> list:
    """Load KEY=VALUE lines from any .env found, without overriding the real
    environment.

    Deliberately stdlib-only and deliberately non-overriding: an env var you
    exported for one run should always beat a stale value in a file.

    Returns the list of files actually read, so startup can say where the
    credentials came from.
    """
    if isinstance(paths, str):
        paths = (paths,)
    loaded = []
    seen = set()
    for path in paths:
        real = os.path.realpath(path)
        if real in seen or not os.path.exists(real):
            continue
        seen.add(real)
        with open(real, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                # Tolerate `export KEY=value`, which is what people paste when
                # they are used to shell rc files.
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        loaded.append(real)
    return loaded

# Claude Opus 5 writes longer answers than earlier models by default, and effort
# does not reliably shorten visible output -- prompting is the lever that works.
# For voice that matters twice over: every extra clause is dead air.
SYSTEM_PROMPT = """\
You are Reachy Mini, a small desk robot with a moving head, a rotating body and \
two antennas. You are having a spoken conversation, and your replies are read \
aloud, so:

- Keep replies to one or two sentences. Answer first, elaborate only if asked.
- **Keep individual sentences short, under about fifteen words.** Speech \
synthesis cannot start until your first sentence is complete, so one long \
opening sentence makes you slow to answer. Two short sentences beat one long \
one every time.
- Write plain spoken prose. No markdown, no bullet points, no headings, no \
emoji, no stage directions.
- Use only characters that can be spoken. No dashes, semicolons, parentheses, \
slashes or symbols; write out what you mean instead.
- Say numbers as words ("about twenty degrees", not "~20deg").
- If you did not catch something, say so briefly and ask again.
- **Reply in the language you were spoken to in.** You can speak {languages}. \
If you are addressed in any other language, answer in English and say plainly \
that you cannot speak that one yet.

You have a body, so use it. Call your movement tools naturally as part of \
talking: nod when you agree, shake your head when you disagree or cannot do \
something, look toward whatever you are talking about, and wiggle your antennas \
when you are pleased. If someone asks you to dance, spin, or show a trick, \
call perform with a move from its list, and say something short while it \
plays. Do not narrate your own movements -- just move, and let \
the person watching see it. Do not announce that you are about to use a tool. If someone asks you to \
speak louder or quieter, call set_volume.

Two rules about moving, because they decide how quickly you can answer:

- **Say your reply in the same response as the movement**, never in a later \
one. Your movements run while you speak, so the person sees and hears you at \
once. Holding your reply until after a movement finishes just makes you slow.
- **At most one movement per reply**, unless you are explicitly asked for \
several. Chaining movements one after another adds a noticeable pause before \
you say anything.

{vision}"""

VISION_NONE = """\
You cannot see -- you have no camera feed in this conversation. If you are asked \
what you can see, say so plainly rather than inventing something."""

VISION_FACE = """\
You have a camera, but only for recognizing faces: you know when someone is in \
front of you, and who they are if you have met them. You see nothing else, so \
do not describe surroundings or invent details."""

# With the look tool (T14.1). The old wording made the robot say "I cannot
# describe you, I only see you to recognize you" and then, ten seconds
# later, describe the visitor's hair (2026-09-04) -- the prompt and the
# tool contradicted each other. T15.4.
VISION_FACE_LOOK = """\
You have a camera. On your own you use it only to recognize faces: you know \
when someone is in front of you, and who they are if you have met them; you \
do not watch the room. When someone asks what you see, or you want to check \
who is in front of you, call look and describe that one picture in a \
sentence or two. Never say you cannot see; never describe anything you did \
not just look at."""


def vision_text(args) -> str:
    if not args.face_source:
        return VISION_NONE
    return VISION_FACE if args.no_look else VISION_FACE_LOOK

# {languages} is filled in at startup with the languages that can really
# be spoken on this machine: Kokoro's verified set plus whatever Piper
# voices are on disk (T5). A static list here would make Claude refuse
# languages the speech stack can in fact speak.

# Local speech mode only: the span-tag convention (T6) that routes each
# phrase to the right per-language voice. NEVER give this to a
# speech-to-speech model (cloud mode) -- it has one voice for every
# language and would read the brackets aloud.
SPAN_TAG_RULE = """
- Your voice is {main} unless you say otherwise with span tags. Wrap EVERY \
phrase or sentence in another language in tags with its two letter code, \
including whole sentences: 'library' is [es]la biblioteca[/es]; \
[es]¿Qué quieres practicar hoy?[/es]; thank you is [ru]спасибо[/ru]. The tags \
pick the voice for that text and are never read aloud. Never tag {main}, \
and never mention the tags.
"""


# ---------------------------------------------------------------------------
# Tools -- the model's deliberate control over the body
#
# Angles are exposed in DEGREES. The driver works in radians, but models are
# markedly more reliable reasoning about "turn thirty degrees left" than about
# 0.52, and the driver clamps whatever arrives anyway.
# ---------------------------------------------------------------------------

def reachy_audio_card() -> str | None:
    """ALSA card index of the Reachy Mini speaker, from /proc/asound/cards."""
    try:
        for line in open("/proc/asound/cards"):
            if "Reachy Mini Audio" in line and line.strip()[0].isdigit():
                return line.strip().split()[0]
    except OSError:
        pass
    return None


def build_audio_tools() -> list:
    """set_volume: the robot's own speaker level, by voice request.

    Drives the ALSA mixer directly (amixer), the same control CLAUDE.md
    says to raise by hand -- it defaults to a quiet ~65% on this unit.
    Needs a machine with the Reachy speaker; elsewhere the tool reports
    that it cannot.
    """
    import subprocess

    async def set_volume(params):
        card = reachy_audio_card()
        if card is None:
            await params.result_callback(
                {"error": "no Reachy speaker on this machine"})
            return
        try:
            percent = int(float(params.arguments.get("percent", 100)))
        except (TypeError, ValueError):
            await params.result_callback({"error": "percent must be a number"})
            return
        percent = max(10, min(100, percent))  # never fully mute yourself

        def apply():
            for control in ("PCM,0", "PCM,1"):
                subprocess.run(["amixer", "-c", card, "sset", control,
                                f"{percent}%"], capture_output=True)
        await asyncio.to_thread(apply)
        logger.info("volume set to %d%%", percent)
        await params.result_callback({"volume_percent": percent})

    return [FunctionSchema(
        name="set_volume",
        description="Set your own speaker volume, as a percentage from 10 "
                    "to 100. Use when asked to speak louder or quieter; "
                    "louder means about 20 points up, quieter about 20 "
                    "down. It starts at 100.",
        properties={"percent": {"type": "integer",
                                "description": "10 to 100"}},
        required=["percent"],
        handler=set_volume,
    )]


def build_tools(robot: RobotLink, tracker: FaceTracker | None = None,
                embodiment=None) -> list:
    """The motion tools. With a face tracker (T13.3) a deliberate head or
    body move suspends tracking for a few seconds and tells the tracker
    where it put the robot, so the two never fight over yaw. While a
    recorded move plays (T15.9) the other motion tools answer "skipped"
    instead of sending anything, and the embodiment is held quiet: the
    move owns the whole body, and the driver would refuse the nudge
    anyway."""
    from moves import LIBRARY, describe as describe_moves

    moving = {"until": -math.inf, "name": None}

    def dance_in_progress() -> dict | None:
        if time.monotonic() < moving["until"]:
            return {"skipped": True,
                    "reason": f"the {moving['name']} move is still playing and "
                              "moves the whole body; no other motion until "
                              "it ends. Just keep talking."}
        return None

    def deliberate(seconds: float, **dofs: float) -> None:
        if tracker is not None:
            tracker.suspend(seconds, stale=not dofs)
            if dofs:
                tracker.set_estimate(**dofs)

    async def move_head(params):
        if (busy := dance_in_progress()):
            await params.result_callback(busy)
            return
        a = params.arguments
        dofs = {}
        if a.get("yaw_degrees") is not None:
            dofs["head_yaw"] = math.radians(float(a["yaw_degrees"]))
        if a.get("pitch_degrees") is not None:
            dofs["head_pitch"] = math.radians(float(a["pitch_degrees"]))
        if a.get("roll_degrees") is not None:
            dofs["head_roll"] = math.radians(float(a["roll_degrees"]))
        if not dofs:
            await params.result_callback({"error": "give at least one angle"})
            return
        duration = float(a.get("duration", 0.6))
        robot.posture(duration=duration, **dofs)
        deliberate(duration + 3.0, **{k: v for k, v in dofs.items()
                                      if k in ("head_yaw", "head_pitch")})
        await params.result_callback({"moving": True, **dofs})

    async def turn_body(params):
        if (busy := dance_in_progress()):
            await params.result_callback(busy)
            return
        a = params.arguments
        duration = float(a.get("duration", 1.0))
        body_yaw = math.radians(float(a["degrees"]))
        robot.posture(duration=duration, body_yaw=body_yaw)
        deliberate(duration + 3.0, body_yaw=body_yaw)
        await params.result_callback({"turning": True})

    async def perform(params):
        name = str(params.arguments.get("move", "dance")).strip().lower()
        spec = LIBRARY.get(name)
        if spec is None:
            await params.result_callback(
                {"error": "unknown move; choose one of "
                          + ", ".join(LIBRARY)})
            return
        try:
            seconds = float(params.arguments.get("seconds") or 0)
        except (TypeError, ValueError):
            seconds = 0.0
        passes = spec.passes_for(seconds)
        total = spec.seconds * passes
        robot.perform(spec.name, repeat=passes)
        # 1 s for the vendor's initial goto to the first frame, 1 s slack.
        moving["until"] = time.monotonic() + total + 2.0
        moving["name"] = spec.name
        if tracker is not None:
            tracker.suspend(total + 2.0, stale=True)
        if embodiment is not None:
            embodiment.hold(total + 2.0)
        logger.info("perform: %s x%d (%.0fs)", spec.name, passes, total)
        await params.result_callback(
            {"performing": spec.name, "seconds": total,
             "note": "keep talking while it plays; no other movement "
                     "until it ends"})

    async def nod(params):
        if (busy := dance_in_progress()):
            await params.result_callback(busy)
            return
        robot.nod(times=int(params.arguments.get("times", 2)))
        await params.result_callback({"nodding": True})

    async def shake_head(params):
        if (busy := dance_in_progress()):
            await params.result_callback(busy)
            return
        robot.shake(times=int(params.arguments.get("times", 2)))
        await params.result_callback({"shaking": True})

    async def wiggle_antennas(params):
        if (busy := dance_in_progress()):
            await params.result_callback(busy)
            return
        # A quick flick out and back. Fire-and-forget so speech continues.
        robot.posture(duration=0.25, antenna_left=1.3, antenna_right=-1.3)
        await asyncio.sleep(0.3)
        robot.posture(duration=0.25, antenna_left=0.15, antenna_right=-0.15)
        await params.result_callback({"wiggled": True})

    async def reset_pose(params):
        robot.home(duration=1.0)
        if tracker is not None:
            tracker.suspend(1.5, stale=False)
            tracker.reset()
        await params.result_callback({"homing": True})

    async def get_robot_status(params):
        try:
            st = await robot.status()
            await params.result_callback({
                "motors_enabled": st.get("motors_enabled"),
                "estopped": st.get("estopped"),
                "measured": st.get("measured"),
            })
        except Exception as exc:                                # noqa: BLE001
            await params.result_callback({"error": str(exc)})

    deg = {"type": "number", "description": "degrees"}
    return [
        FunctionSchema(
            name="move_head",
            description="Point the head. Positive yaw turns the head to its "
                        "own left; positive pitch tips the chin down; roll "
                        "tilts ear-to-shoulder. Range is about 50 degrees of "
                        "yaw and 25 of pitch or roll.",
            properties={
                "yaw_degrees": deg, "pitch_degrees": deg, "roll_degrees": deg,
                "duration": {"type": "number",
                             "description": "seconds, default 0.6"},
            },
            required=[],
            handler=move_head,
        ),
        FunctionSchema(
            name="turn_body",
            description="Rotate the whole robot on its base. Positive is to "
                        "the robot's left. Up to about 160 degrees either way.",
            properties={"degrees": deg,
                        "duration": {"type": "number",
                                     "description": "seconds, default 1.0"}},
            required=["degrees"],
            handler=turn_body,
        ),
        FunctionSchema(
            name="nod",
            description="Nod the head up and down. Use it to agree, greet, or "
                        "confirm you heard something.",
            properties={"times": {"type": "integer",
                                  "description": "default 2"}},
            required=[],
            handler=nod,
        ),
        FunctionSchema(
            name="shake_head",
            description="Shake the head side to side. Use it to disagree or to "
                        "say you cannot do something.",
            properties={"times": {"type": "integer",
                                  "description": "default 2"}},
            required=[],
            handler=shake_head,
        ),
        FunctionSchema(
            name="wiggle_antennas",
            description="Flick both antennas. Use it to show delight, "
                        "excitement, or playfulness.",
            properties={}, required=[], handler=wiggle_antennas,
        ),
        FunctionSchema(
            name="perform",
            description="Play a recorded move: a dance, a spin, or an "
                        "emotion. Use when asked to dance, spin, or show "
                        "what you can do, or to celebrate. Moves: "
                        + describe_moves() + ". Say something short "
                        "while it plays; it takes a few seconds.",
            properties={"move": {"type": "string",
                                 "enum": list(LIBRARY),
                                 "description": "which move"},
                        "seconds": {"type": "number",
                                    "description": "how long to keep it "
                                                   "going, up to 60; "
                                                   "dances default to "
                                                   "about 30"}},
            required=["move"],
            handler=perform,
        ),
        FunctionSchema(
            name="reset_pose",
            description="Return head, body and antennas to the neutral "
                        "resting posture.",
            properties={}, required=[], handler=reset_pose,
        ),
        FunctionSchema(
            name="get_robot_status",
            description="Read the robot's own current joint positions, motor "
                        "state and emergency-stop state.",
            properties={}, required=[], handler=get_robot_status,
        ),
    ]


# ---------------------------------------------------------------------------
# Sight on demand (T14.1)
# ---------------------------------------------------------------------------

def build_look_tool(hub, speech: str, task_ref: dict, gemini_ref: dict,
                    save_dir=None) -> list:
    """``look``: one frame from the camera, shown to the model on request.

    The family asked for it twice ("can you describe what you see?"). One
    frame per call, never a stream -- the prompt still says the camera
    is for recognizing faces, and this is the single deliberate
    exception, taken when the visitor asks or when the robot wants to
    check who is there.

    Cloud: the frame goes to Gemini Live as an input image (its native
    video path), then the tool result tells it to describe what it just
    received. Local: pipecat's function-call image pattern -- a
    UserImageRawFrame tagged with this tool call's id lands in the user
    aggregator, which attaches it to the function result, so Claude
    sees the picture as the tool's answer (the message order Anthropic
    requires stays intact).
    """
    import cv2
    from face.camera import save_frame

    def grab():
        seq, frame = hub.latest()
        if frame is None:
            return None
        saved = save_frame(frame, save_dir) if save_dir else None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return rgb.tobytes(), (w, h), saved

    async def look(params):
        got = grab()
        if got is None:
            await params.result_callback(
                {"error": "no camera frame available right now"})
            return
        image, size, saved = got
        logger.info("look: one %dx%d frame shown to the model%s", *size,
                    (" (saved %s)" % saved) if saved else "")
        if speech == "cloud":
            gemini = gemini_ref.get("service")
            if gemini is None:
                await params.result_callback({"error": "no vision channel"})
                return
            gemini._last_sent_time = 0.0     # bypass the 1 fps video throttle
            await gemini._send_user_video(
                InputImageRawFrame(image=image, size=size, format="RGB"))
            await params.result_callback(
                {"looked": True,
                 "note": "an image of what your camera sees right now was "
                         "just sent to you; describe it from that image "
                         "in one or two sentences, in the conversation's "
                         "language"})
            return
        task = task_ref.get("task")
        request = UserImageRequestFrame(
            user_id="visitor", function_name=params.function_name,
            tool_call_id=params.tool_call_id, append_to_context=True)
        await task.queue_frames([UserImageRawFrame(
            user_id="visitor", image=image, size=size, format="RGB",
            text="What your camera sees right now.", request=request,
            append_to_context=True)])
        await params.result_callback(
            {"looked": True,
             "note": "describe the attached camera image in one or two "
                     "sentences"})

    return [FunctionSchema(
        name="look",
        description="Take one look through your camera and see what is in "
                    "front of you right now. Use it when asked what you "
                    "see, or to check who or what is there. One still "
                    "image, not a video.",
        properties={}, required=[], handler=look,
    )]


# ---------------------------------------------------------------------------
# Web search (cloud mode)
# ---------------------------------------------------------------------------

async def probe_web_search(api_key: str, model: str) -> str | None:
    """Can this key use Google Search grounding on the Live API?

    Opens (and immediately closes) one Live session that declares the
    ``google_search`` tool. Measured 2026-09-03: a key without billing
    gets close code 1011 "You exceeded your current quota" the moment the
    tool is in the setup, while the same key without the tool converses
    fine -- and pipecat's service swallows that failure into a dead,
    silent session. Probing first turns a mute robot into one clear log
    line and a run without search. Returns None when search is usable,
    else the reason.
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        async with client.aio.live.connect(
                model=model,
                config={"response_modalities": ["AUDIO"],
                        "tools": [{"google_search": {}}]}):
            return None
    except Exception as exc:                                    # noqa: BLE001
        text = str(exc).split("For more information")[0].strip()
        return f"{type(exc).__name__}: {text[:200]}"


class CloudTranscriptLogger(FrameProcessor):
    """Makes a cloud-mode run log readable (T14.6).

    Gemini Live speaks audio, so nothing downstream logs its words; the
    2026-09-03 session log had every ``heard`` line and not one reply.
    Gemini's output transcription arrives as LLMTextFrame chunks, which
    are gathered here and logged as one ``said: ...`` line per turn. A
    Google Search the model made arrives as an LLMSearchResponseFrame
    with its sources; that becomes a ``web search: N sources`` line.
    """

    def __init__(self):
        super().__init__(name="CloudTranscriptLogger")
        self._said: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self._said.append(frame.text)
        elif isinstance(frame, (LLMFullResponseEndFrame,
                                BotStoppedSpeakingFrame)):
            # Gemini does not end every turn with a response-end frame
            # (a cue's reply can run into the next), so the bot going
            # quiet closes a line too.
            text = "".join(self._said).strip()
            self._said = []
            if text:
                logger.info("said: %s", " ".join(text.split()))
        elif isinstance(frame, LLMSearchResponseFrame):
            sites = sorted({(o.site_title or o.site_uri or "?")
                            for o in (frame.origins or [])})
            logger.info("web search: %d sources%s", len(sites),
                        (" (" + ", ".join(sites[:5]) + ")") if sites else "")
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

class SafeLLMContext(LLMContext):
    """An LLMContext that refuses thought messages Anthropic cannot convert.

    Without this the conversation silently dies after a couple of turns, and
    the failure is nasty: the robot keeps listening and moving, so it looks
    alive, but never speaks again.

    Why it happens. When the model thinks, pipecat appends
    ``{"type": "thought", "text": ..., "signature": ...}`` to the context
    (llm_response_universal.py::_handle_thought_end). Its Anthropic adapter
    only converts that back into a real message when the text is non-empty
    *and* a signature is present; otherwise it falls through to "assume it is
    already in Anthropic format" and returns the raw dict, which has no
    ``role`` key. The next request then dies on ``KeyError: 'role'`` inside
    the adapter -- and because the bad message is now in the history,
    every subsequent turn dies the same way. The error frame is non-fatal, so
    nothing crashes; the agent just goes mute forever.

    Claude Opus 5 walks straight into this: it never returns raw chain of
    thought, and ``thinking.display`` defaults to ``"omitted"``, so thinking
    blocks arrive with empty text.

    Dropping them costs nothing -- an empty thought carries no information.
    The alternative, ``thinking: {"display": "summarized"}``, also fixes the
    conversion but pays for summary tokens on every turn, which is the wrong
    trade in a latency-sensitive voice loop.
    """

    def add_message(self, message) -> None:
        if isinstance(message, LLMSpecificMessage):
            m = message.message
            if (isinstance(m, dict) and m.get("type") == "thought"
                    and not (m.get("text") and m.get("signature"))):
                logger.debug("dropping unconvertible thought message "
                             "(text=%r, signature=%s)",
                             m.get("text"), bool(m.get("signature")))
                return
        super().add_message(message)


# ---------------------------------------------------------------------------
# Model capabilities
# ---------------------------------------------------------------------------

async def supports_effort(client, model: str) -> bool:
    """Does this model accept output_config.effort?

    Not every model does: Haiku 4.5 rejects it with
    ``400 This model does not support the effort parameter``, and because the
    model and the effort level are independent flags here, a mismatched pair
    would fail on every single turn.

    Asked live via the Models API rather than kept as a hardcoded list, so this
    stays correct as models come and go. If the lookup itself fails, fall back
    to sending effort only for the families known to take it.
    """
    try:
        info = await client.models.retrieve(model)
        caps = getattr(info, "capabilities", None) or {}
        return bool(caps.get("effort", {}).get("supported", False))
    except Exception as exc:                                    # noqa: BLE001
        logger.debug("capability lookup for %s failed (%s); using fallback",
                     model, exc)
        return model.startswith(("claude-opus-", "claude-sonnet-5",
                                 "claude-sonnet-4-6", "claude-fable-",
                                 "claude-mythos-"))


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

async def warm_up(stt, tts, sample_rate: int) -> None:
    """Run one throwaway inference through Whisper and Kokoro.

    Both models load lazily on first use. Measured cold on an M4 Pro, the first
    Whisper call took **13.6 seconds** and the first Kokoro call 1.2s -- and
    without this, that cost lands on the first thing the user actually says,
    which is the worst possible moment. Paying it here, during startup, takes
    steady-state latency to roughly 0.3s for speech synthesis.

    Output is discarded. Nothing is played and nothing reaches the pipeline.
    """
    t0 = asyncio.get_running_loop().time()

    # Half a second of silence is enough to force the model load and one full
    # decode pass. int16 mono at the transport's rate.
    silence = b"\x00\x00" * (sample_rate // 2)
    try:
        async for _ in stt.run_stt(silence):
            pass
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("speech-recognition warmup failed: %s", exc)

    try:
        async for _ in tts.run_tts("ready", "warmup"):
            pass
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("speech-synthesis warmup failed: %s", exc)

    logger.info("models warm (%.1fs) -- first turn will not pay the load cost",
                asyncio.get_running_loop().time() - t0)


# ---------------------------------------------------------------------------
# Audio device selection
# ---------------------------------------------------------------------------

class ResamplingAudioInput(LocalAudioInputTransport):
    """PyAudio input opened at the device's own rate, resampled to the pipeline's.

    The stock transport opens the mic at the pipeline rate (16 kHz), which
    the USB booth mic refuses (PortAudio: "Invalid sample rate"; it only
    does 48 kHz), while everything downstream -- Silero VAD, Whisper,
    Gemini Live, the voice-print collector -- was built and measured at
    16 kHz. So: open at ``device_rate``, resample each 20 ms callback
    chunk with a streaming soxr resampler (keeps history across chunks,
    so no clicks at the edges), and push the frame at the pipeline rate.
    With equal rates it is the stock transport.
    """

    def __init__(self, py_audio, params, device_rate: int):
        super().__init__(py_audio, params)
        self._device_rate = device_rate
        self._resampler = None

    async def start(self, frame: StartFrame):
        # BaseInputTransport.start on purpose: the direct parent's start()
        # is exactly the open-at-pipeline-rate we are replacing.
        await BaseInputTransport.start(self, frame)
        if self._in_stream:
            return
        self._sample_rate = (self._params.audio_in_sample_rate
                             or frame.audio_in_sample_rate)
        channels = self._params.audio_in_channels
        if self._device_rate != self._sample_rate:
            import soxr
            self._resampler = soxr.ResampleStream(
                self._device_rate, self._sample_rate, channels,
                dtype="int16", quality="HQ")
        num_frames = int(self._device_rate / 100) * 2  # 20 ms of audio
        self._in_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=channels,
            rate=self._device_rate,
            frames_per_buffer=num_frames,
            stream_callback=self._audio_in_callback,
            input=True,
            input_device_index=self._params.input_device_index,
        )
        self._in_stream.start_stream()
        await self.set_transport_ready(frame)

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        if self._resampler is not None:
            import numpy as np
            samples = np.frombuffer(in_data, dtype=np.int16)
            channels = self._params.audio_in_channels
            if channels > 1:
                samples = samples.reshape(-1, channels)
            in_data = self._resampler.resample_chunk(samples).tobytes()
        return super()._audio_in_callback(in_data, frame_count, time_info,
                                          status)


class ResamplingAudioTransport(LocalAudioTransport):
    """LocalAudioTransport whose input side is a ``ResamplingAudioInput``."""

    def __init__(self, params: LocalAudioTransportParams, input_device_rate: int):
        super().__init__(params)
        self._input_device_rate = input_device_rate

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = ResamplingAudioInput(self._pyaudio, self._params,
                                               self._input_device_rate)
        return self._input


def pick_audio_devices(args) -> tuple[int | None, int | None, int]:
    """Resolve (input_index, output_index, input_open_rate) and log the choice.

    Booth rule: the USB mic when one is plugged in (``--mic-device``, tried
    in order), else the robot's own mic; the speaker is always the robot's
    (``--audio-device``). ``--input-device`` / ``--output-device`` force an
    index. The input rate is whatever the chosen mic will actually open at
    (``ResamplingAudioInput`` brings it to the pipeline rate).
    """
    prefs = parse_mic_prefs(args.mic_device)
    devices = list_audio_devices()
    choice = choose_audio_devices(devices, prefs, args.audio_device)
    if args.input_device is not None:
        in_idx = args.input_device
        logger.info("audio: mic index %d forced by --input-device", in_idx)
    else:
        in_idx = choice.input.index if choice.input else None
        if choice.input is None:
            logger.warning("audio: no microphone found for %r or %r; "
                           "PyAudio's default input will be used",
                           args.mic_device, args.audio_device)
        elif choice.input_fallback:
            logger.info("audio: no USB mic matching %r; using the robot's "
                        "built-in mic %r (index %d)", args.mic_device,
                        choice.input.name, in_idx)
        else:
            logger.info("audio: mic %r (index %d)", choice.input.name, in_idx)
    if args.output_device is not None:
        out_idx = args.output_device
    else:
        out_idx = choice.output.index if choice.output else None
        if choice.output is None:
            logger.warning("audio: no speaker matching %r; PyAudio's default "
                           "output will be used", args.audio_device)
        else:
            logger.info("audio: speaker %r (index %d)", choice.output.name,
                        out_idx)
    in_rate = args.sample_rate
    if in_idx is not None and devices:
        in_rate = input_rate_for(in_idx, args.sample_rate,
                                 args.audio_in_channels)
        if in_rate != args.sample_rate:
            logger.info("audio: mic opens at %d Hz, resampling to %d Hz",
                        in_rate, args.sample_rate)
    return in_idx, out_idx, in_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_llm_client(args):
    """Resolve Anthropic credentials. Returns (api_key, prebuilt_client).

    Two supported paths, neither of which involves borrowing credentials from
    some other running tool:

      api-key  ANTHROPIC_API_KEY, from the environment or voice/.env.
      oauth    No key at all. A bare AsyncAnthropic() resolves the profile that
               `ant auth login` wrote, and pipecat uses the client verbatim
               (`self._client = client or AsyncAnthropic(api_key=api_key)`),
               so api_key is never consulted.
    """
    if args.auth == "oauth":
        from anthropic import AsyncAnthropic
        try:
            client = AsyncAnthropic()
        except Exception as exc:                                # noqa: BLE001
            raise SystemExit(
                "--auth oauth could not resolve credentials (%s).\n"
                "Run `ant auth login` first, or use an API key instead."
                % exc)
        logger.info("auth: OAuth profile (no API key in use)")
        return "", client

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "No Anthropic credentials found.\n"
            "\n"
            "Speech recognition, synthesis and turn-taking all run locally, but\n"
            "the language model is Claude and needs credentials of its own.\n"
            "A Claude Code subscription does not cover this: it authenticates\n"
            "Claude Code, not third-party API traffic, which is billed\n"
            "separately. Pick one:\n"
            "\n"
            "  1. An API key from https://console.anthropic.com/settings/keys\n"
            "       cp %s.example %s   # then paste the key in\n"
            "     or:  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "\n"
            "  2. OAuth, so no key is stored anywhere:\n"
            "       ant auth login\n"
            "       %s --auth oauth\n"
            % (ENV_FILE, ENV_FILE, os.path.basename(__file__)))
    return api_key, None


async def run(args) -> None:
    # Credentials fail fast, before any model loads: local mode needs
    # Anthropic, cloud mode needs Google. Neither needs the other's key.
    api_key = llm_client = google_key = None
    if args.speech == "local":
        api_key, llm_client = build_llm_client(args)
    else:
        # Google's SDK accepts either name; so do we.
        google_key = (os.environ.get("GOOGLE_API_KEY")
                      or os.environ.get("GEMINI_API_KEY"))
        if not google_key:
            raise SystemExit(
                "--speech cloud needs GOOGLE_API_KEY (or GEMINI_API_KEY), "
                "in the environment or in voice/.env "
                "(get one at https://aistudio.google.com).")

    # -- robot ------------------------------------------------------------
    robot = None
    if not args.no_robot:
        robot = await RobotLink(args.broker, args.device_id, args.tenant,
                                zenoh_listen=args.zenoh_listen).connect()
        # The daemon holds the mic and speaker until asked to let go.
        logger.info("asking the robot to release its mic and speaker ...")
        logger.info("media: %s", await robot.release_media(True))
        robot.home(duration=1.0)

    # -- audio device ------------------------------------------------------
    # USB mic if present, else the robot's own; speaker always the robot's.
    in_idx, out_idx, in_rate = pick_audio_devices(args)

    # --deaf: never open the microphone. Scripted --say runs otherwise pick
    # up room noise as phantom user turns (Whisper will happily transcribe a
    # hallway), which makes them nondeterministic.
    transport = ResamplingAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=not args.deaf,
        audio_out_enabled=True,
        audio_in_sample_rate=args.sample_rate,
        audio_out_sample_rate=args.sample_rate,
        input_device_index=in_idx,
        output_device_index=out_idx,
    ), input_device_rate=in_rate)

    # Both run on this machine. Whisper goes through MLX, so transcription is
    # on the Apple-Silicon GPU rather than the CPU; Kokoro synthesises through
    # ONNX. Neither leaves the laptop, so neither costs a network round trip --
    # the only remote hop in the whole loop is the language model.
    #
    # Language handling. Kokoro speaks nine languages and Whisper recognises
    # far more; pipecat's defaults pin both to English. With --language auto
    # (the default) Whisper detects what it heard and LanguageRouter points
    # Kokoro at a matching voice. Pin it to one language to skip all that.
    # T16: the voice for untagged text (local mode) is the language the
    # student is taught *in*: --native-language, else the --learner
    # profile's native_language, else English. A Russian speaker learning
    # English hears Russian explanations through Piper and English
    # practice phrases through Kokoro's [en] spans.
    native_lang = args.native_language
    if native_lang is None and args.learner:
        native_lang = load_learner(args.learners_root,
                                   args.learner)[0].native_language
    native_lang = (native_lang or "en").lower()
    auto_language = args.language == "auto"
    start_lang = native_lang if auto_language else args.language

    stt = tts = language_router = None
    if args.speech == "local":
        # Two engines, one voice map: Kokoro's verified languages plus
        # whatever Piper voices are actually on disk (T5: ru/zh, ~63 MB
        # each, fetched once -- see piper_tts.py's docstring).
        args.piper_voices = args.piper_voices or PIPER_VOICES_DIR
        piper_langs = piper_available(args.piper_voices)
        if piper_langs:
            logger.info("piper: %s available (%s)",
                        ", ".join(v.name for v in piper_langs.values()),
                        args.piper_voices)
        speakable = {**LANGUAGES, **piper_langs}
        if start_lang not in speakable:
            raise SystemExit("%s %r has no local voice (have: %s)"
                             % ("--language" if not auto_language
                                else "native language",
                                start_lang, ", ".join(sorted(speakable))))
        start_voice = args.voice or speakable[start_lang].voice
        spoken_names = ", ".join(v.name for v in speakable.values())
        # The span-tag rule is local-only: per-language voices need it,
        # a speech-to-speech model would read the brackets aloud.
        tutor_mode = bool(args.learner or args.face_source or args.session)
        base_prompt = SYSTEM_PROMPT.format(
            languages=spoken_names,
            vision=vision_text(args),
        ) + SPAN_TAG_RULE.format(main=speakable[start_lang].name)

        if auto_language:
            stt = MultilingualWhisperMLX(
                settings=MultilingualWhisperMLX.Settings(
                    model=args.whisper_model))
        else:
            stt = WhisperSTTServiceMLX(
                settings=WhisperSTTServiceMLX.Settings(
                    model=args.whisper_model,
                    language=speakable[start_lang].language))
        tts = DualEngineTTS(
            voices_dir=args.piper_voices,
            settings=DualEngineTTS.Settings(
                voice=start_voice, language=speakable[start_lang].language))
        # In tutor mode the voice must NOT follow Whisper's guess about what
        # it heard (spec 6: the target language comes from the profile, not
        # detection). Rehearsal showed why: room noise detected as Russian
        # switched the voice, and English replies came out through Piper's
        # Russian phonemes. The model picks the voice itself via span tags.
        language_router = LanguageRouter(
            initial=start_lang, voice=args.voice,
            enabled=auto_language and not tutor_mode,
            extra=piper_langs)
        if tutor_mode and auto_language:
            logger.info("tutor mode: voice follows the model's span tags, "
                        "not detection (untagged = %s)",
                        speakable[start_lang].name)
    else:
        # Cloud mode (T8): one speech-to-speech model, one voice, native
        # mixed-language handling -- no tags, no router, no local engines.
        spoken_names = ("English, Spanish, French, Italian, Portuguese, "
                        "Russian, Mandarin, Hindi, and most other languages")
        # Measured (progress/T8.md): Gemini Live sometimes speaks the
        # goodbye but skips the save_session_notes call the briefing
        # demands. Claude does not need this reminder; Gemini does.
        # Filled in below once the probe has run; the prompt must not
        # promise a search the key cannot make.
        web_search_note = ("" if args.no_web_search else
                           "You can search the web (Google Search grounding) "
                           "when a question needs a current fact: today's news, "
                           "weather, prices, an event, a word's usage. Use it "
                           "for real lesson material, keep the answer to one "
                           "or two spoken sentences, and never read out URLs.\n")
        base_prompt = SYSTEM_PROMPT.format(
            languages=spoken_names,
            vision=vision_text(args)) + """
You can teach every language you can speak, Russian and Mandarin included, \
and English to a speaker of any other language. Explain in the student's \
own language, whatever it is. If a student asks to practice a different \
language, switch at once and call set_target_language; if they ask to be \
taught in a different language, call set_native_language.
{web_search}\
Tool discipline: your tools are real actions, not things to mention. \
Whenever your instructions say to call a tool at a moment (for example \
save_session_notes when the student says goodbye), you must actually \
emit the tool call in that same turn, alongside anything you say.""".format(
            web_search=web_search_note)

    # Claude Opus 5. Two deliberate choices for a voice loop:
    #
    #  * effort "low" -- the cheap, fast end of the ladder, which is plenty for
    #    conversational turns and keeps time-to-first-token down.
    #  * thinking left ON (the default on Opus 5). Disabling it is tempting for
    #    latency, but with thinking off the model can emit a tool call as plain
    #    text -- the call silently never runs, with no error. In a robot that
    #    means "nod" gets spoken instead of performed. Low effort is the right
    #    lever; disabling thinking is not.
    llm = None
    if args.speech == "local":
        from anthropic import AsyncAnthropic
        probe = llm_client or AsyncAnthropic(api_key=api_key)
        extra = {}
        if await supports_effort(probe, args.model):
            extra["output_config"] = {"effort": args.effort}
        else:
            logger.info("%s does not accept the effort parameter; "
                        "omitting it", args.model)
        if args.fast:
            extra["speed"] = "fast"
            extra["betas"] = ["fast-mode-2026-02-01"]
        llm = AnthropicLLMService(
            api_key=api_key,
            model=args.model,
            client=llm_client,      # set only for --auth oauth; else None
            params=AnthropicLLMService.InputParams(
                max_tokens=args.max_tokens,
                extra=extra,
            ),
        )

    # Face tracking (T13.3): on when there is a camera and a robot, unless
    # --no-track. The tracker owns head/body yaw; embodiment then leaves
    # yaw out of its talking sway (the DOF split in embodiment.py).
    tracking = bool(robot and args.face_source and not args.no_track)
    embodiment = Embodiment(robot, enabled=robot is not None,
                            sway=not args.no_sway, own_yaw=not tracking)
    tracker = FaceTracker(robot, embodiment=embodiment) if tracking else None
    if tracking:
        logger.info("tracking: head and body follow the largest face")

    # Booth persona (T13.5/T13.6): quips and the wishlist question. Goes on
    # the base prompt so the session runner's rebuilt prompts carry it too.
    base_prompt += build_persona(args.persona)

    # Each FunctionSchema carries its own handler, so pipecat dispatches
    # straight from the schema. Calling register_function() as well is
    # redundant and pipecat warns about it.
    tools = ((build_tools(robot, tracker, embodiment) if robot else [])
             + build_audio_tools())

    # Tutor mode: append the right briefing to the system prompt and
    # register the memory (and, with a face source, enrollment) tools next
    # to the motion tools. Works with or without a robot -- these tools
    # only touch the learner store and the camera.
    system_prompt = base_prompt
    holder = store = hub = None
    voice_identity = voice_collector = None
    session_runner = None
    late = {}          # {"task": PipelineTask, "service": Gemini}, filled below
    if args.session and not args.face_source:
        raise SystemExit("--session needs --face-source (who would it watch?)")
    if args.learner or args.face_source:
        holder = CurrentLearner()
        if args.learner:
            learner, notes, store = load_learner(args.learners_root,
                                                 args.learner)
            holder.learner = learner
            system_prompt += build_briefing(learner, notes)
        else:
            store = LearnerStore(args.learners_root)
        if args.session:
            # The session runner (started after the pipeline exists) owns
            # identity: the agent boots into the idle prompt and waits.
            from session import IDLE_NOTE
            system_prompt = base_prompt + IDLE_NOTE
        elif args.face_source and holder.learner is None:
            from face_id import identify_from_source
            ident = await asyncio.to_thread(
                identify_from_source, args.face_source, store)
            logger.info("face: %s%s", ident.status,
                        (" %s (score %s)" % (ident.learner.id, ident.score))
                        if ident.learner else "")
            if ident.status == "known":
                holder.learner = ident.learner
                notes = store.read_notes(ident.learner.id,
                                         max_sessions=BRIEFING_SESSIONS)
                system_prompt += build_briefing(ident.learner, notes)
            elif ident.status == "unsure":
                holder.candidate = ident.learner
                system_prompt += build_unsure_briefing(holder.candidate)
            else:  # unknown face, or no face at all: same stranger flow
                system_prompt += STRANGER_BRIEFING.format(
                    languages=spoken_names)
        tools = tools + build_tutor_tools(store, holder,
                                          wishes_path=args.wishes_file,
                                          ask_wish=(args.persona == "booth"))

        # Voice print (T13.9): a second identity signal, fused with the
        # face's verdict. Samples come from the raw input audio (or from
        # --voice-source in scripted runs); actions become spoken cues
        # through say_cue, defined once the pipeline exists below.
        if not args.no_voice_id:
            from voiceid import VoiceCollector, VoiceIdentity
            voice_identity = VoiceIdentity(store, holder)

            async def on_voice_sample(vector, secs):
                if session_runner is not None:
                    session_runner.note_voice()   # speech is presence
                action = voice_identity.on_sample(vector, secs)
                if action == "speaker_changed" and session_runner is not None:
                    await session_runner.speaker_changed()
                elif action in ("challenge", "downgrade", "confirmed"):
                    who = holder.learner or holder.candidate
                    cue = voice_cue(action, who, store)
                    if cue:
                        await say_cue(cue)

            voice_collector = VoiceCollector(on_voice_sample)
            logger.info("voice id: on (ECAPA prints; samples from %s)",
                        "--voice-source" if args.voice_source else "the mic")
        elif args.voice_source:
            raise SystemExit("--voice-source needs voice id (drop --no-voice-id)")
        # One camera, many readers (T13.3): the session watcher, the
        # tracker and enrollment all read from a shared hub whenever any
        # loop will hold the camera for the whole run.
        hub = None
        if args.face_source and (args.session or tracking):
            from face.camera import FrameHub
            hub = FrameHub(args.face_source, fps=2.0).start()
        if args.face_source:
            tools = tools + build_enrollment_tools(
                store, holder, args.face_source,
                frames_factory=(hub.frames if hub is not None else None),
                voice_identity=voice_identity,
                # T15.1: the session runner (built below) knows which
                # face started the session; enrollment stores that one
                # when the capture disagrees with it.
                session_face=lambda: getattr(late.get("runner"), "_session_face",
                                             None))
        if hub is None and args.face_source and not args.no_look:
            # Sight needs a live frame source for the whole run.
            from face.camera import FrameHub
            hub = FrameHub(args.face_source, fps=2.0).start()
        if hub is not None and not args.no_look:
            tools = tools + build_look_tool(hub, args.speech, late, late,
                                            save_dir=args.look_dir or None)
        if holder.learner:
            native = native_language_of(holder.learner)
            logger.info("tutor mode: student %s (%s %s taught in %s, %d "
                        "prior sessions, tier %s)", holder.learner.name,
                        holder.learner.level, holder.learner.target_language,
                        native, holder.learner.sessions, holder.learner.tier)
            # Bilingual priming (T7): keep both lesson languages "in mind"
            # so embedded foreign phrases survive transcription better.
            if isinstance(stt, MultilingualWhisperMLX):
                prompt = bilingual_priming(holder.learner.target_language,
                                           native)
                if prompt:
                    stt.initial_prompt = prompt
                    logger.info("whisper priming: %s + %s",
                                language_name(native),
                                holder.learner.target_language)

    # Cloud mode: Gemini Live's native Google Search grounding rides along
    # as a provider-specific tool. pipecat's Gemini adapter appends
    # ``custom_tools[GEMINI]`` verbatim to the function declarations, and
    # the Live setup takes the *context's* tools over the service's, so
    # the same ToolsSchema goes to both.
    web_search = args.speech == "cloud" and not args.no_web_search
    if web_search:
        problem = await probe_web_search(google_key, args.gemini_model)
        if problem:
            web_search = False
            logger.warning("web search: unavailable on this key (%s); "
                           "continuing without Google Search grounding. "
                           "Grounding on the Live API needs a billing-"
                           "enabled Google AI Studio key.", problem)
    if web_search:
        tools = ToolsSchema(
            standard_tools=list(tools),
            custom_tools={AdapterType.GEMINI: [{"google_search": {}}]})
        logger.info("web search: Google Search grounding enabled for Gemini")
    elif args.speech == "cloud" and not args.no_web_search:
        system_prompt = system_prompt.replace(web_search_note, "")
        base_prompt = base_prompt.replace(web_search_note, "")

    # pipecat 1.6's LLMContext accepts a tools list or NOT_GIVEN but not None,
    # so the no-robot path must omit the argument entirely.
    if tools:
        context = SafeLLMContext(
            messages=[{"role": "system", "content": system_prompt}],
            tools=tools,
        )
    else:
        context = SafeLLMContext(
            messages=[{"role": "system", "content": system_prompt}],
        )

    # Turn-taking. In pipecat 1.6 both VAD and end-of-turn detection hang off
    # the user aggregator, not the transport.
    #
    # Silero answers "is there speech right now"; the smart-turn model answers
    # the harder question "has this person actually finished". That second model
    # is why a thinking pause mid-sentence does not get treated as your turn
    # ending -- silence alone is a bad end-of-turn signal for natural speech.
    # Both run locally, so neither costs a network round trip.
    # Cloud mode leaves turn detection to Gemini's own server-side VAD.
    turn_strategies = None
    if args.speech == "local" and not args.no_smart_turn:
        turn_strategies = UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3())],
        )

    # Mute the microphone while the robot is talking.
    #
    # Reachy Mini's speaker and microphone are the same USB device, centimetres
    # apart, with no echo cancellation between them. Without this the robot
    # hears its own voice, Whisper transcribes the feedback (badly -- it
    # hallucinates whole sentences out of speaker noise), and those phantom
    # utterances arrive as user turns. The symptom is a robot that talks to
    # itself, answers things nobody asked, and drifts further off with every
    # exchange.
    #
    # Muting during function calls as well keeps a motor whirr from being
    # picked up as speech.
    mute_strategies = []
    if not args.no_mute:
        mute_strategies = [AlwaysUserMuteStrategy(), FunctionCallUserMuteStrategy()]

    if args.speech == "local":
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(stop_secs=args.stop_secs)),
                user_turn_strategies=turn_strategies,
                user_mute_strategies=mute_strategies,
            ),
        )
    else:
        # No local VAD or smart-turn: the speech-to-speech service hears
        # raw audio and decides turns itself. The self-hearing mutes stay:
        # Gemini supports barge-in, but the robot's speaker and mic are
        # centimetres apart with no echo cancellation, so full-duplex is
        # off until the booth mic proves otherwise (T11 decides).
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_mute_strategies=mute_strategies,
            ),
        )

    if args.speech == "cloud":
        # T8: the three local speech stages collapse into one streaming
        # speech-to-speech service; motion + memory tools ride along.
        from pipecat.services.google.gemini_live.llm import (
            GeminiLiveLLMService,
        )
        gemini_kwargs = dict(
            api_key=google_key,
            system_instruction=system_prompt,
        )
        if args.gemini_model:
            gemini_kwargs["model"] = args.gemini_model
        if args.gemini_voice:
            gemini_kwargs["voice_id"] = args.gemini_voice
        if tools:
            gemini_kwargs["tools"] = tools
        if args.session:
            # Nobody is there at startup: do not stream the room until a
            # face starts a session (CloudBrain.reset resumes the mic).
            gemini_kwargs["start_audio_paused"] = True
            # T15.5: pipecat seeds every fresh connection with the prompt
            # as a user turn and, by default, makes the model answer it.
            # That was the "Hello there, I do not see anyone" to an empty
            # chair at startup and the second of three greetings at each
            # walk-up (2026-09-04). In session mode the runner's walk-up
            # cue is the one and only greeting.
            gemini_kwargs["inference_on_context_initialization"] = False
        gemini = GeminiLiveLLMService(**gemini_kwargs)
        late["service"] = gemini
        logger.info("speech: cloud (Gemini Live, model %s)",
                    args.gemini_model or "pipecat default")
        # The voice tap sits after the user aggregator: its mute strategy
        # drops the audio while the robot speaks, so the tap never hears
        # the robot's own voice.
        stages = [transport.input(), aggregators.user()]
        if voice_collector is not None:
            stages.append(voice_collector.as_processor())
            # T15.7: no user-speaking frames from Gemini Live, so the
            # turn timer takes the visitor's stop from the energy gate.
            voice_collector.on_speech_end = embodiment.note_user_stopped
        pipeline = Pipeline(stages + [
            gemini,
            CloudTranscriptLogger(),  # "said: ..." and "web search: ..."
            embodiment,      # observes speaking state, passes frames through
            transport.output(),
            aggregators.assistant(),
        ])
    else:
        # Locally the STT consumes the audio, so the tap goes before it;
        # it drops the robot's own voice itself, from the bot-speaking
        # frames the output transport pushes upstream.
        stages = [transport.input()]
        if voice_collector is not None:
            stages.append(voice_collector.as_processor())
        pipeline = Pipeline(stages + [
            stt,
            language_router,  # switches the voice to match what was heard
            aggregators.user(),
            llm,
            tts,
            embodiment,      # observes speaking state, passes frames through
            transport.output(),
            aggregators.assistant(),
        ])

    task = PipelineTask(pipeline, params=PipelineParams(
        audio_in_sample_rate=args.sample_rate,
        audio_out_sample_rate=args.sample_rate,
        enable_metrics=True,
        enable_usage_metrics=True,
    ))
    late["task"] = task

    async def say_cue(text: str) -> None:
        """Inject a user-turn cue mid-conversation, whichever brain runs.
        Local: through the aggregator. Cloud: Gemini keeps its history
        server-side and ignores the aggregator after turn one, so use
        the service's own injection path (the one --say uses)."""
        if args.speech == "cloud":
            await gemini._create_single_response(
                [{"role": "user", "content": text}])
        else:
            await task.queue_frames([LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": text}], run_llm=True)])

    if args.speech == "local" and not args.no_warmup:
        await warm_up(stt, tts, args.sample_rate)
    if voice_collector is not None and not args.no_warmup:
        from voiceid import warm_up as warm_up_voiceid
        await asyncio.to_thread(warm_up_voiceid)

    # Session lifecycle (T10): a background task watches the face source
    # and starts/ends tutoring sessions on the live pipeline.
    session_task = tracking_task = None
    if args.session:
        from session import CloudBrain, SessionRunner
        session_runner = SessionRunner(
            source=args.face_source, store=store, holder=holder,
            context=context, task=task, base_prompt=base_prompt,
            languages=spoken_names, robot=robot, stt=stt,
            stable_secs=args.stable_secs, absent_secs=args.absent_secs,
            hub=hub, tracker=tracker, attract_secs=args.attract_secs,
            voice_identity=voice_identity,
            speaking=lambda: embodiment.bot_speaking,     # T15.6
            # Cloud mode (T14.3): cues go through Gemini's own injection
            # path and every visitor gets a fresh server-side session.
            cue=say_cue if args.speech == "cloud" else None,
            brain=(CloudBrain(gemini, context) if args.speech == "cloud"
                   else None))
        # Speech is presence (T13.2): a visitor out of frame but talking
        # is not gone.
        embodiment.on_user_speech = session_runner.note_voice
        late["runner"] = session_runner
        session_task = asyncio.create_task(session_runner.run())
        logger.info("session mode: watching for a face (stable %.1fs, "
                    "still-there at %.0fs, walk-away %.0fs%s)",
                    args.stable_secs, args.absent_secs * 2 / 3,
                    args.absent_secs,
                    (", attractor after %.0fs" % args.attract_secs)
                    if args.attract_secs > 0 else "")
    elif tracker is not None and hub is not None:
        tracking_task = asyncio.create_task(
            TrackingLoop(hub, tracker).run())

    # --say injects an utterance as if it had been transcribed, which exercises
    # the whole loop (model -> tools -> motion -> speech) without a microphone.
    # Useful for demos, and for checking the robot end of things on a machine
    # you are not sitting in front of.
    if args.say:
        async def kickoff():
            for i, utterance in enumerate(args.say):
                await asyncio.sleep(args.say_delay if i == 0 else args.say_gap)
                logger.info("injecting utterance %d/%d: %r",
                            i + 1, len(args.say), utterance)
                if voice_collector is not None and args.voice_source:
                    # Scripted voice: the i-th file is "what was heard"
                    # at the i-th utterance (the last one repeats).
                    wav = args.voice_source[min(i, len(args.voice_source) - 1)]
                    await voice_collector.inject_wav(wav)
                if args.speech == "cloud" and i > 0:
                    # The FIRST turn must go through the aggregator: it
                    # seeds the service's context object (without which
                    # Gemini's tool calls are refused) and triggers the
                    # initial inference. But Gemini Live keeps conversation
                    # state server-side and ignores LATER context updates
                    # (_handle_context in pipecat's gemini_live llm.py), so
                    # a second --say through the aggregator vanishes. Later
                    # turns use the service's own text-injection path,
                    # which sends client content + the Gemini-3 nudge.
                    await gemini._create_single_response(
                        [{"role": "user", "content": utterance}])
                else:
                    await task.queue_frames([LLMMessagesAppendFrame(
                        messages=[{"role": "user", "content": utterance}],
                        run_llm=True)])
        asyncio.create_task(kickoff())

    if args.speech == "cloud" and not args.say:
        # Gemini Live only starts forwarding microphone audio once it has
        # been handed an initial context (pipecat gates realtime input on
        # it). Every --say test supplied one; a live-mic session never did,
        # and the robot sat there unresponsive. Kick the conversation off
        # the way pipecat's own examples do -- the model greets first.
        # In session mode the runner greets when a face is stable; the
        # kick-off here only seeds the context so the mic starts flowing,
        # and inference_on_context_initialization=False (above) keeps
        # the model from answering the seed out loud (T15.5).
        async def cloud_kickoff():
            await asyncio.sleep(1.0)
            await task.queue_frames([LLMRunFrame()])
        asyncio.create_task(cloud_kickoff())

    logger.info("ready -- say something. Ctrl-C to stop.")
    runner = PipelineRunner(handle_sigint=True)
    try:
        await runner.run(task)
    finally:
        for bg in (session_task, tracking_task):
            if bg is not None:
                bg.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await bg
        if hub is not None:
            hub.close()
        if robot is not None:
            logger.info("returning the robot to neutral and taking media back")
            try:
                await robot.call("home", duration=1.0)
                await robot.release_media(False)
            except Exception as exc:                            # noqa: BLE001
                logger.warning("cleanup failed: %s", exc)
            await robot.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent.py",
        description="Voice conversation with the Reachy Mini over Device Connect.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("robot (Device Connect)")
    g.add_argument("--broker", default=DEFAULT_BROKER)
    g.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    g.add_argument("--tenant", default=DEFAULT_TENANT)
    g.add_argument("--zenoh-listen", default=None, metavar="EP",
                   help="zenoh only: TCP endpoints to listen on, comma "
                        "separated (e.g. tcp/0.0.0.0:7448)")
    g.add_argument("--no-robot", action="store_true",
                   help="voice only; do not connect to or move the robot")

    g = p.add_argument_group("audio")
    g.add_argument("--audio-device", default=AUDIO_DEVICE_NAME,
                   help="substring of the speaker device's name; its mic is "
                        "the fallback when no --mic-device is present")
    g.add_argument("--mic-device", default=MIC_DEVICE_NAME,
                   help="preferred microphone(s): comma-separated name "
                        "substrings tried in order, falling back to the "
                        "--audio-device mic; '' means always the fallback")
    g.add_argument("--audio-in-channels", type=int, default=1,
                   help="channels to open the mic with")
    g.add_argument("--input-device", type=int, default=None,
                   help="explicit PyAudio input index (overrides --audio-device)")
    g.add_argument("--output-device", type=int, default=None,
                   help="explicit PyAudio output index")
    g.add_argument("--sample-rate", type=int, default=16000)
    g.add_argument("--list-devices", action="store_true",
                   help="print the audio devices and exit")

    g = p.add_argument_group("speech (local)")
    g.add_argument("--whisper-model", default=MLXModel.LARGE_V3_TURBO_Q4.value,
                   help="MLX Whisper model; smaller is faster")
    g.add_argument("--piper-voices", default=None, metavar="DIR",
                   help="directory of Piper voice models for ru/zh "
                        "(default: voice/piper_voices; missing models just "
                        "disable those languages)")
    g.add_argument("--voice", default=None,
                   help="Kokoro voice id, overriding the default for the "
                        "starting language (default af_heart for English). "
                        "Switching language picks that language's own voice.")
    g.add_argument("--language", default="auto",
                   choices=["auto"] + list(LANGUAGES) + list(PIPER_LANGUAGES),
                   help="auto (default) detects the language you speak each "
                        "turn and answers in it. Naming one pins both "
                        "recognition and speech to it. ru/zh need their "
                        "Piper voices on disk (see --piper-voices).")
    g.add_argument("--native-language", default=None, metavar="CODE",
                   choices=list(LANGUAGES) + list(PIPER_LANGUAGES),
                   help="the language the student is taught in, which is "
                        "the voice for untagged text in local mode "
                        "(default: the --learner profile's native "
                        "language, else en). Cloud mode reads it from the "
                        "profile and needs no flag.")
    g.add_argument("--stop-secs", type=float, default=0.35,
                   help="VAD silence before end-of-speech is considered")
    g.add_argument("--no-warmup", action="store_true",
                   help="skip the startup warmup inference (the first real "
                        "utterance then pays the model-load cost)")
    g.add_argument("--no-mute", action="store_true",
                   help="do not mute the mic while the robot speaks. Only "
                        "safe with a headset or a mic away from the speaker; "
                        "on the robot's own mic it will hear itself.")
    g.add_argument("--no-smart-turn", action="store_true",
                   help="use VAD silence alone instead of the smart-turn model")

    g = p.add_argument_group("speech mode")
    g.add_argument("--speech", default="local", choices=["local", "cloud"],
                   help="local (default): Whisper + Claude + Kokoro/Piper "
                        "on this machine. cloud: one Gemini Live "
                        "speech-to-speech stream (needs GOOGLE_API_KEY; "
                        "raw audio leaves the machine; the tutor brain is "
                        "Gemini, not Claude)")
    g.add_argument("--gemini-model",
                   default="models/gemini-3.1-flash-live-preview",
                   metavar="MODEL",
                   help="Gemini Live model id for --speech cloud")
    g.add_argument("--gemini-voice", default=None, metavar="VOICE",
                   help="Gemini Live voice for --speech cloud "
                        "(default: the service's default)")
    g.add_argument("--no-web-search", action="store_true",
                   help="cloud mode: do not give Gemini its native Google "
                        "Search grounding tool (on by default)")

    g = p.add_argument_group("model (cloud)")
    g.add_argument("--auth", default="api-key", choices=["api-key", "oauth"],
                   help="'api-key' reads ANTHROPIC_API_KEY (environment or "
                        "voice/.env); 'oauth' uses the profile from "
                        "`ant auth login` and stores no key")
    g.add_argument("--model", default="claude-opus-5")
    g.add_argument("--effort", default="low",
                   choices=["low", "medium", "high", "xhigh", "max"])
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--fast", action="store_true",
                   help="Claude fast mode: same model, up to 2.5x faster "
                        "output, at premium pricing ($10/$50 per MTok)")

    g = p.add_argument_group("tutor")
    g.add_argument("--learner", default=None, metavar="NAME",
                   help="run as this learner's language tutor: their profile "
                        "and recent notes go into the briefing, and the "
                        "save_session_notes / update_learner_level tools are "
                        "enabled (accepts a folder id or a display name)")
    g.add_argument("--learners-root", default=DEFAULT_LEARNERS_ROOT,
                   metavar="DIR",
                   help="learner store root")
    g.add_argument("--face-source", default=None, metavar="SRC",
                   help="identify who is present at startup and enable "
                        "conversational enrollment: a V4L2 index (the "
                        "Reachy camera), a video file, or an image "
                        "directory")
    g.add_argument("--session", action="store_true",
                   help="booth loop: watch the face source, start a session "
                        "when a face is stable, save notes and reset when "
                        "it walks away (needs --face-source)")
    g.add_argument("--stable-secs", type=float, default=2.0,
                   help="how long a face must be present before greeting")
    g.add_argument("--absent-secs", type=float, default=60.0,
                   help="how long a face must be gone (and no speech "
                        "heard) before the session ends and notes are "
                        "saved; the robot asks 'still there?' at two "
                        "thirds of this")
    g.add_argument("--attract-secs", type=float, default=0.0,
                   help="session mode: with nobody in frame this long, "
                        "play a short move every few minutes to draw "
                        "people in (0 = off)")
    g.add_argument("--persona", default="plain", choices=list(PERSONAS),
                   help="booth: a few gentle quips and the wishlist "
                        "question; plain: none")
    g.add_argument("--wishes-file", default=None, metavar="PATH",
                   help="where record_wish appends (default booth/wishes.md)")
    g.add_argument("--no-voice-id", action="store_true",
                   help="do not keep or check voice prints (on by default "
                        "in tutor mode; prints are computed locally)")
    g.add_argument("--voice-source", action="append", default=None,
                   metavar="WAV",
                   help="testing: hear this wav as the visitor's voice at "
                        "each --say turn instead of the microphone; repeat "
                        "the flag to change speakers turn by turn")
    g.add_argument("--look-dir", default=DEFAULT_LOOK_DIR, metavar="DIR",
                   help="keep every frame the look tool shows the model "
                        "under DIR/<date>/ (T15.10); '' to keep none")
    g.add_argument("--no-look", action="store_true",
                   help="do not offer the 'look' camera tool")
    g.add_argument("--no-track", action="store_true",
                   help="do not follow the visitor's face with head and "
                        "body (tracking is on whenever there is a camera "
                        "and a robot)")

    g = p.add_argument_group("embodiment")
    g.add_argument("--no-sway", action="store_true",
                   help="hold still while speaking instead of swaying")

    g = p.add_argument_group("testing")
    g.add_argument("--say", action="append", default=None, metavar="TEXT",
                   help="inject TEXT as a user utterance, as if it had been "
                        "transcribed; exercises the full loop with no "
                        "microphone. Repeat for a multi-turn conversation.")
    g.add_argument("--say-delay", type=float, default=2.0,
                   help="seconds to wait before the first --say")
    g.add_argument("--say-gap", type=float, default=14.0,
                   help="seconds between repeated --say utterances")
    g.add_argument("--deaf", action="store_true",
                   help="never open the microphone; with --say this makes a "
                        "run fully scripted (no room noise becoming phantom "
                        "user turns)")

    p.add_argument("--quiet", action="store_true",
                   help="silence pipecat's own logging")
    return p


def main() -> None:
    for path in load_env_file():
        logger.info("loaded credentials from %s", path)
    args = build_parser().parse_args()
    if args.list_devices:
        import pyaudio
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            print("%-3d %-34s in=%d out=%d %.0fHz"
                  % (i, d["name"], d["maxInputChannels"],
                     d["maxOutputChannels"], d["defaultSampleRate"]))
        pa.terminate()
        return
    if args.quiet:
        loguru_logger.remove()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)


if __name__ == "__main__":
    main()
