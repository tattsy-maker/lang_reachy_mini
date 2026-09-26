"""2026-09-25, evening: a bigger speaker on a USB-to-jack adapter, and no
software gain by default. Pure logic; no sound card needed."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "voice"))

from audio_devices import (                                  # noqa: E402
    AudioDevice, alsa_card_of, auto_speakers, choose_audio_devices,
    parse_asound_cards,
)

CARDS = """\
 0 [NVIDIA         ]: hda-acpi - NVIDIA
                      NVIDIA HDA Controller at 0x36078000 irq 310
 1 [Audio          ]: USB-Audio - Reachy Mini Audio
                      Pollen Robotics Reachy Mini Audio at usb-1.1, high speed
 2 [Device         ]: USB-Audio - USB Composite Device
                      Jieli Technology USB Composite Device at usb-1.2
 3 [Device_1       ]: USB-Audio - USB Audio Device
                      C-Media Electronics Inc. USB Audio Device at usb-1.3
"""

DEVICES = [
    AudioDevice(2, "NVIDIA: HDMI 0 (hw:0,3)", 0, 8, 44100),
    AudioDevice(4, "Reachy Mini Audio: USB Audio (hw:1,0)", 2, 2, 16000),
    AudioDevice(5, "USB Composite Device: Audio (hw:2,0)", 1, 0, 48000),
    AudioDevice(6, "USB Audio Device: USB Audio (hw:3,0)", 1, 2, 48000),
]


def test_cards_parse():
    cards = parse_asound_cards(CARDS)
    assert cards == [(0, "hda-acpi", "NVIDIA"),
                     (1, "USB-Audio", "Reachy Mini Audio"),
                     (2, "USB-Audio", "USB Composite Device"),
                     (3, "USB-Audio", "USB Audio Device")]


def test_auto_finds_the_adapter_not_the_robot_hdmi_or_the_mic():
    cards = parse_asound_cards(CARDS)
    assert auto_speakers(cards, ["Reachy Mini Audio", "USB Composite Device"]) \
        == ["USB Audio Device"]
    assert auto_speakers(cards, ["Reachy Mini Audio", "USB Composite Device"],
                         can_play=lambda i: i != 3) == []


def test_the_adapter_plays_and_the_robot_mic_still_listens():
    c = choose_audio_devices(DEVICES, ["USB Composite Device"],
                             "Reachy Mini Audio", ["USB Audio Device"])
    assert c.output.index == 6 and not c.output_fallback
    assert c.input.index == 5
    # desk mic unplugged: the robot's mic, even though the adapter has one
    c = choose_audio_devices([d for d in DEVICES if d.index != 5],
                             ["USB Composite Device"], "Reachy Mini Audio",
                             ["USB Audio Device"])
    assert c.input.index == 4 and c.input_fallback and c.output.index == 6
    # adapter unplugged: the robot's speaker
    c = choose_audio_devices(DEVICES[:3], [], "Reachy Mini Audio",
                             ["USB Audio Device"])
    assert c.output.index == 4 and c.output_fallback
    assert alsa_card_of(DEVICES[3].name) == "3"
    assert alsa_card_of("default") is None


def test_the_booth_plays_without_software_gain():
    booth = (REPO / "start_booth.sh").read_text()
    assert 'LOUDNESS_DB="${BOOTH_LOUDNESS_DB:-0}"' in booth
    assert 'SPEAKER_DEVICE="${BOOTH_SPEAKER_DEVICE-auto}"' in booth
    assert '--speaker-device "$SPEAKER_DEVICE"' in booth


def test_the_speaker_output_is_resampled_to_the_devices_rate():
    import asyncio
    import numpy as np
    import pytest
    pytest.importorskip("pipecat")
    pytest.importorskip("soxr")
    import agent
    from pipecat.frames.frames import OutputAudioRawFrame, StartFrame
    from pipecat.transports.local.audio import LocalAudioTransportParams

    class Stream:
        def __init__(self, rate, channels):
            self.rate, self.channels, self.written = rate, channels, []

        def start_stream(self):
            pass

        def write(self, data):
            self.written.append(data)

    class PyAudio:
        def get_format_from_width(self, width):
            return 8

        def open(self, **kw):
            self.stream = Stream(kw["rate"], kw["channels"])
            return self.stream

    params = LocalAudioTransportParams(audio_out_enabled=True,
                                       audio_out_sample_rate=16000)
    pa = PyAudio()
    out = agent.ResamplingAudioOutput(pa, params, device_rate=48000)

    async def main():
        out.set_transport_ready = lambda frame: asyncio.sleep(0)
        out.get_event_loop = asyncio.get_running_loop   # no TaskManager here

        async def base_start(self, frame):
            return None
        orig = agent.BaseOutputTransport.start
        agent.BaseOutputTransport.start = base_start
        try:
            await out.start(StartFrame(audio_out_sample_rate=16000))
        finally:
            agent.BaseOutputTransport.start = orig
        assert pa.stream.rate == 48000 and out._sample_rate == 16000
        tone = (np.sin(np.arange(3200) / 5) * 8000).astype(np.int16)
        for i in range(10):
            await out.write_audio_frame(OutputAudioRawFrame(
                tone[i * 320:(i + 1) * 320].tobytes(), 16000, 1))
        held = sum(len(b) for b in pa.stream.written) // 4
        assert held < 3 * 3200, "HQ holds a few ms back mid-reply"
        await asyncio.sleep(out.FLUSH_AFTER_SECS + 0.1)   # the reply ended

    asyncio.run(main())
    total = sum(len(b) for b in pa.stream.written) // 4    # stereo frames
    assert abs(total - 3 * 3200) <= 3, total  # every sample, tail included
    both = np.frombuffer(b"".join(pa.stream.written), np.int16).reshape(-1, 2)
    assert (both[:, 0] == both[:, 1]).all() and pa.stream.channels == 2


def test_barge_in_is_off_and_the_tail_muted_on_another_speaker():
    import asyncio
    import time
    import pytest
    pytest.importorskip("pipecat")
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame,
    )
    from barge_in import make_tail_strategy

    tail = make_tail_strategy(0.2)
    mic = InputAudioRawFrame(b"\0\0" * 320, 16000, 1)

    async def main():
        assert not await tail.process_frame(mic)
        await tail.process_frame(BotStartedSpeakingFrame())
        assert await tail.process_frame(BotStoppedSpeakingFrame())
        assert await tail.process_frame(mic), "the room still rings"
        await asyncio.sleep(0.25)
        assert not await tail.process_frame(mic)

    asyncio.run(main())
    text = (REPO / "voice" / "agent.py").read_text()
    assert 'if args.barge_in and SPEAKER["external"]:' in text
    assert 'SPEAKER["external"] = not choice.output_fallback' in text
