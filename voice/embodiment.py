#!/usr/bin/env python3
"""embodiment -- map conversation state onto robot motion.

A voice agent with a speaker attached is a speaker. What makes Reachy Mini read
as *listening* is that it reacts before it answers: it perks up when you start
talking, dips its head when you finish, and moves while it speaks.

This is a pipecat FrameProcessor that watches the four speaking frames and
drives the robot through the Device Connect link. It sits in the pipeline purely as an
observer -- every frame is passed through untouched.

    UserStartedSpeaking   -> lean in, antennas up          (I'm listening)
    UserStoppedSpeaking   -> small dip                     (got it, thinking)
    BotStartedSpeaking    -> idle sway while talking       (I'm the one talking)
    BotStoppedSpeaking    -> settle back to attentive       (your turn)

**Why this does not fight the LLM's own tool calls.** Embodiment only ever
writes the DOFs it names, and `goto_posture` leaves every unnamed DOF holding
its current commanded value. So embodiment can own pitch and the antennas while
a `look_at` tool call owns yaw, and the two compose instead of overwriting each
other. Keep that split if you add gestures: pick DOFs the LLM tools don't drive.
"""

from __future__ import annotations

import asyncio
import logging
import random

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from robot_link import RobotLink

logger = logging.getLogger(__name__)

# Deliberately small. These play under a conversation, not as the main event --
# big motions read as twitchy and collide with whatever the LLM just asked for.
ATTENTIVE = {"head_pitch": -0.10, "head_z": 0.006,
             "antenna_left": 0.45, "antenna_right": -0.45}
THINKING = {"head_pitch": 0.10, "head_z": -0.004,
            "antenna_left": 0.0, "antenna_right": 0.0}
RESTING = {"head_pitch": 0.0, "head_z": 0.0,
           "antenna_left": 0.15, "antenna_right": -0.15}


class Embodiment(FrameProcessor):
    """Drives the robot from conversation state. Passes all frames through."""

    def __init__(self, robot: RobotLink, *, enabled: bool = True,
                 sway: bool = True) -> None:
        super().__init__()
        self._robot = robot
        self._enabled = enabled
        self._sway_enabled = sway
        self._sway_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._enabled:
            try:
                self._react(frame)
            except Exception as exc:                          # noqa: BLE001
                # Embodiment is decoration. It must never take the call down.
                logger.warning("embodiment reaction failed: %s", exc)

        await self.push_frame(frame, direction)

    # -- reactions ----------------------------------------------------------

    def _react(self, frame: Frame) -> None:
        if isinstance(frame, UserStartedSpeakingFrame):
            self._stop_sway()
            self._robot.posture(duration=0.35, **ATTENTIVE)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._robot.posture(duration=0.3, **THINKING)

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._start_sway()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._stop_sway()
            self._robot.posture(duration=0.5, **RESTING)

    # -- talking sway -------------------------------------------------------

    def _start_sway(self) -> None:
        if not self._sway_enabled or (
                self._sway_task and not self._sway_task.done()):
            return
        self._sway_task = asyncio.create_task(self._sway())

    def _stop_sway(self) -> None:
        if self._sway_task and not self._sway_task.done():
            self._sway_task.cancel()
        self._sway_task = None

    async def _sway(self) -> None:
        """A loose, irregular head/antenna motion for as long as the bot talks.

        Randomised rather than periodic: a fixed cycle reads as a machine
        pulsing, an uneven one reads as someone talking.
        """
        try:
            while True:
                self._robot.posture(
                    duration=0.45,
                    head_yaw=random.uniform(-0.10, 0.10),
                    head_pitch=random.uniform(-0.08, 0.04),
                    head_roll=random.uniform(-0.06, 0.06),
                    antenna_left=random.uniform(0.1, 0.7),
                    antenna_right=random.uniform(-0.7, -0.1),
                )
                await asyncio.sleep(random.uniform(0.5, 0.9))
        except asyncio.CancelledError:
            raise
        except Exception as exc:                              # noqa: BLE001
            logger.warning("sway loop stopped: %s", exc)
