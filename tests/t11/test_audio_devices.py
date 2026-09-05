"""T11 item 1, the booth mic (2026-09-04): the USB microphone when it is
plugged in, the robot's built-in mic otherwise, the robot's speaker
always. Selection is pure and runs anywhere against this box's device
table; the rate probe and a real open run on metal (audio + models)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from audio_devices import (  # noqa: E402
    MIC_DEVICE_NAME,
    SPEAKER_DEVICE_NAME,
    AudioDevice,
    choose_audio_devices,
    parse_mic_prefs,
)

# agent.py --list-devices on spark-c7bc, 2026-09-04, USB mic plugged in.
HDMI = [AudioDevice(i, f"NVIDIA: HDMI {i} (hw:0,{n})", 0, 8, 44100)
        for i, n in enumerate((3, 7, 8, 9))]
REACHY = AudioDevice(4, "Reachy Mini Audio: USB Audio (hw:1,0)", 2, 2, 16000)
USB_MIC = AudioDevice(5, "USB Composite Device: Audio (hw:2,0)", 1, 0, 48000)
VIRTUAL = [AudioDevice(6, "hdmi", 0, 8, 44100),
           AudioDevice(7, "pipewire", 64, 64, 44100),
           AudioDevice(8, "default", 64, 64, 44100)]
WITH_USB = HDMI + [REACHY, USB_MIC] + VIRTUAL
WITHOUT_USB = HDMI + [REACHY] + VIRTUAL
PREFS = parse_mic_prefs(MIC_DEVICE_NAME)


def test_usb_mic_wins_when_present():
    c = choose_audio_devices(WITH_USB, PREFS, SPEAKER_DEVICE_NAME)
    assert c.input == USB_MIC and not c.input_fallback
    assert c.output == REACHY


def test_robot_mic_when_usb_absent():
    c = choose_audio_devices(WITHOUT_USB, PREFS, SPEAKER_DEVICE_NAME)
    assert c.input == REACHY and c.input_fallback
    assert c.output == REACHY


def test_speaker_is_never_the_usb_mic():
    # The USB device has no outputs; even a mic-first table keeps the
    # robot as the speaker.
    c = choose_audio_devices([USB_MIC, REACHY], PREFS, SPEAKER_DEVICE_NAME)
    assert c.output == REACHY


def test_prefs_are_tried_in_order():
    c = choose_audio_devices(WITH_USB, ["Yeti", MIC_DEVICE_NAME],
                             SPEAKER_DEVICE_NAME)
    assert c.input == USB_MIC
    # Matching is case-insensitive and needs an input-capable device:
    # "hdmi" matches an output-only card and must be skipped.
    c = choose_audio_devices(WITH_USB, ["hdmi", "usb composite"],
                             SPEAKER_DEVICE_NAME)
    assert c.input == USB_MIC


def test_empty_prefs_go_straight_to_the_robot():
    c = choose_audio_devices(WITH_USB, parse_mic_prefs(""), SPEAKER_DEVICE_NAME)
    assert c.input == REACHY and c.input_fallback


def test_nothing_matches():
    c = choose_audio_devices(HDMI, PREFS, SPEAKER_DEVICE_NAME)
    assert c.input is None and c.output is None and c.input_fallback


def test_parse_mic_prefs():
    assert parse_mic_prefs(None) == []
    assert parse_mic_prefs(" , ") == []
    assert parse_mic_prefs("Yeti, USB Composite Device ,") == \
        ["Yeti", "USB Composite Device"]


PROBE = r"""
import json, sys
sys.path.insert(0, "voice")
from audio_devices import *
import pyaudio
devices = list_audio_devices()
c = choose_audio_devices(devices, parse_mic_prefs(MIC_DEVICE_NAME), SPEAKER_DEVICE_NAME)
assert c.input is not None, "no microphone at all"
rate = input_rate_for(c.input.index, 16000)
pa = pyaudio.PyAudio()
s = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
            input_device_index=c.input.index, frames_per_buffer=rate // 50)
data = s.read(rate // 5, exception_on_overflow=False)
s.close(); pa.terminate()
print(json.dumps({"mic": c.input.name, "fallback": c.input_fallback,
                  "rate": rate, "bytes": len(data)}))
"""


@pytest.mark.audio
@pytest.mark.models
def test_chosen_mic_opens_at_the_probed_rate(paths):
    """Whatever mic is plugged in right now opens at the rate the probe
    picked and yields 200 ms of samples. Fails loudly on the exact error
    this work exists to avoid: PortAudio's 'Invalid sample rate'."""
    out = subprocess.run([str(paths.voice_py), "-c", PROBE], cwd=paths.repo,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["bytes"] == 2 * (result["rate"] // 5)
    if not result["fallback"]:
        assert MIC_DEVICE_NAME.lower() in result["mic"].lower()
    else:
        assert SPEAKER_DEVICE_NAME.lower() in result["mic"].lower()
