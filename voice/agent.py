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
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger as loguru_logger                              # noqa: E402
from pipecat.adapters.schemas.function_schema import FunctionSchema     # noqa: E402
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (         # noqa: E402
    LocalSmartTurnAnalyzerV3,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer                  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams                    # noqa: E402
from pipecat.frames.frames import LLMMessagesAppendFrame                # noqa: E402
from pipecat.pipeline.pipeline import Pipeline                          # noqa: E402
from pipecat.pipeline.runner import PipelineRunner                      # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask          # noqa: E402
from pipecat.processors.aggregators.llm_context import (                # noqa: E402
    LLMContext,
    LLMSpecificMessage,
)
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
from pipecat.transports.local.audio import (                            # noqa: E402
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
)
from tutor_mode import (                                                # noqa: E402
    DEFAULT_LEARNERS_ROOT,
    build_briefing,
    build_tutor_tools,
    load_learner,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("agent")

DEFAULT_BROKER = "zenoh://"
DEFAULT_DEVICE_ID = "reachy-mini-1"
DEFAULT_TENANT = "lab"
AUDIO_DEVICE_NAME = "Reachy Mini Audio"
_HERE = os.path.dirname(os.path.abspath(__file__))
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
when you are pleased. Do not narrate your own movements -- just move, and let \
the person watching see it. Do not announce that you are about to use a tool.

Two rules about moving, because they decide how quickly you can answer:

- **Say your reply in the same response as the movement**, never in a later \
one. Your movements run while you speak, so the person sees and hears you at \
once. Holding your reply until after a movement finishes just makes you slow.
- **At most one movement per reply**, unless you are explicitly asked for \
several. Chaining movements one after another adds a noticeable pause before \
you say anything.

You cannot see -- you have no camera feed in this conversation. If you are asked \
what you can see, say so plainly rather than inventing something."""

SYSTEM_PROMPT = SYSTEM_PROMPT.format(languages=SPOKEN_LANGUAGES)


# ---------------------------------------------------------------------------
# Tools -- the model's deliberate control over the body
#
# Angles are exposed in DEGREES. The driver works in radians, but models are
# markedly more reliable reasoning about "turn thirty degrees left" than about
# 0.52, and the driver clamps whatever arrives anyway.
# ---------------------------------------------------------------------------

def build_tools(robot: RobotLink) -> list:
    async def move_head(params):
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
        robot.posture(duration=float(a.get("duration", 0.6)), **dofs)
        await params.result_callback({"moving": True, **dofs})

    async def turn_body(params):
        a = params.arguments
        robot.posture(duration=float(a.get("duration", 1.0)),
                      body_yaw=math.radians(float(a["degrees"])))
        await params.result_callback({"turning": True})

    async def nod(params):
        robot.nod(times=int(params.arguments.get("times", 2)))
        await params.result_callback({"nodding": True})

    async def shake_head(params):
        robot.shake(times=int(params.arguments.get("times", 2)))
        await params.result_callback({"shaking": True})

    async def wiggle_antennas(params):
        # A quick flick out and back. Fire-and-forget so speech continues.
        robot.posture(duration=0.25, antenna_left=1.3, antenna_right=-1.3)
        await asyncio.sleep(0.3)
        robot.posture(duration=0.25, antenna_left=0.15, antenna_right=-0.15)
        await params.result_callback({"wiggled": True})

    async def reset_pose(params):
        robot.home(duration=1.0)
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

def find_audio_device(name_fragment: str) -> tuple[int | None, int | None]:
    """Return (input_index, output_index) for the first device matching a name."""
    try:
        import pyaudio
    except ImportError:
        return None, None
    pa = pyaudio.PyAudio()
    try:
        want = name_fragment.lower()
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if want in str(d["name"]).lower():
                return (i if d["maxInputChannels"] > 0 else None,
                        i if d["maxOutputChannels"] > 0 else None)
        return None, None
    finally:
        pa.terminate()


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
    api_key, llm_client = build_llm_client(args)

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
    in_idx, out_idx = (args.input_device, args.output_device)
    if in_idx is None or out_idx is None:
        found_in, found_out = find_audio_device(args.audio_device)
        in_idx = in_idx if in_idx is not None else found_in
        out_idx = out_idx if out_idx is not None else found_out
    logger.info("audio: input_device=%s output_device=%s (%r)",
                in_idx, out_idx, args.audio_device)

    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=args.sample_rate,
        audio_out_sample_rate=args.sample_rate,
        input_device_index=in_idx,
        output_device_index=out_idx,
    ))

    # Both run on this machine. Whisper goes through MLX, so transcription is
    # on the Apple-Silicon GPU rather than the CPU; Kokoro synthesises through
    # ONNX. Neither leaves the laptop, so neither costs a network round trip --
    # the only remote hop in the whole loop is the language model.
    #
    # Language handling. Kokoro speaks nine languages and Whisper recognises
    # far more; pipecat's defaults pin both to English. With --language auto
    # (the default) Whisper detects what it heard and LanguageRouter points
    # Kokoro at a matching voice. Pin it to one language to skip all that.
    auto_language = args.language == "auto"
    start_lang = "en" if auto_language else args.language
    start_voice = args.voice or LANGUAGES[start_lang].voice

    if auto_language:
        stt = MultilingualWhisperMLX(
            settings=MultilingualWhisperMLX.Settings(model=args.whisper_model))
    else:
        stt = WhisperSTTServiceMLX(settings=WhisperSTTServiceMLX.Settings(
            model=args.whisper_model, language=LANGUAGES[start_lang].language))
    tts = MultilingualKokoro(settings=MultilingualKokoro.Settings(
        voice=start_voice, language=LANGUAGES[start_lang].language))
    language_router = LanguageRouter(
        initial=start_lang, voice=args.voice, enabled=auto_language)

    # Claude Opus 5. Two deliberate choices for a voice loop:
    #
    #  * effort "low" -- the cheap, fast end of the ladder, which is plenty for
    #    conversational turns and keeps time-to-first-token down.
    #  * thinking left ON (the default on Opus 5). Disabling it is tempting for
    #    latency, but with thinking off the model can emit a tool call as plain
    #    text -- the call silently never runs, with no error. In a robot that
    #    means "nod" gets spoken instead of performed. Low effort is the right
    #    lever; disabling thinking is not.
    from anthropic import AsyncAnthropic
    probe = llm_client or AsyncAnthropic(api_key=api_key)
    extra = {}
    if await supports_effort(probe, args.model):
        extra["output_config"] = {"effort": args.effort}
    else:
        logger.info("%s does not accept the effort parameter; omitting it",
                    args.model)
    if args.fast:
        extra["speed"] = "fast"
        extra["betas"] = ["fast-mode-2026-02-01"]
    llm = AnthropicLLMService(
        api_key=api_key,
        model=args.model,
        client=llm_client,          # set only for --auth oauth; else None
        params=AnthropicLLMService.InputParams(
            max_tokens=args.max_tokens,
            extra=extra,
        ),
    )

    # Each FunctionSchema carries its own handler, so pipecat dispatches
    # straight from the schema. Calling register_function() as well is
    # redundant and pipecat warns about it.
    tools = build_tools(robot) if robot else []

    # Tutor mode: append the learner's briefing to the system prompt and
    # register the memory tools next to the motion tools. Works with or
    # without a robot -- the memory tools only touch the learner store.
    system_prompt = SYSTEM_PROMPT
    if args.learner:
        learner, notes, store = load_learner(args.learners_root, args.learner)
        system_prompt += build_briefing(learner, notes)
        tools = tools + build_tutor_tools(store, learner)
        logger.info("tutor mode: student %s (%s %s, %d prior sessions, "
                    "tier %s)", learner.name, learner.level,
                    learner.target_language, learner.sessions, learner.tier)

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
    turn_strategies = None
    if not args.no_smart_turn:
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

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=args.stop_secs)),
            user_turn_strategies=turn_strategies,
            user_mute_strategies=mute_strategies,
        ),
    )

    embodiment = Embodiment(robot, enabled=robot is not None,
                            sway=not args.no_sway)

    pipeline = Pipeline([
        transport.input(),
        stt,
        language_router,     # switches the voice to match what was heard
        aggregators.user(),
        llm,
        tts,
        embodiment,          # observes speaking state, passes frames through
        transport.output(),
        aggregators.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(
        audio_in_sample_rate=args.sample_rate,
        audio_out_sample_rate=args.sample_rate,
        enable_metrics=True,
        enable_usage_metrics=True,
    ))

    if not args.no_warmup:
        await warm_up(stt, tts, args.sample_rate)

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
                await task.queue_frames([LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": utterance}],
                    run_llm=True)])
        asyncio.create_task(kickoff())

    logger.info("ready -- say something. Ctrl-C to stop.")
    runner = PipelineRunner(handle_sigint=True)
    try:
        await runner.run(task)
    finally:
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
                   help="substring of the audio device name to use")
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
    g.add_argument("--voice", default=None,
                   help="Kokoro voice id, overriding the default for the "
                        "starting language (default af_heart for English). "
                        "Switching language picks that language's own voice.")
    g.add_argument("--language", default="auto",
                   choices=["auto"] + list(LANGUAGES),
                   help="auto (default) detects the language you speak each "
                        "turn and answers in it. Naming one pins both "
                        "recognition and speech to it.")
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
