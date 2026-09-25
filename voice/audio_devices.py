"""Which microphone and speaker the voice agent opens.

The booth rule (2026-09-04): use the USB microphone when one is plugged
in, otherwise the robot's own built-in mic. The speaker is always the
robot's. The two mics disagree on sample rate -- the Jieli USB mic only
opens at 48 kHz, the Reachy Mini Audio device only at 16 kHz (PortAudio
answers "Invalid sample rate" to the other) -- so the choice carries the
rate the device has to be opened at, and the agent resamples to the
pipeline's 16 kHz on the way in (``ResamplingAudioTransport`` in
``agent.py``).

``choose_audio_devices`` is pure so it can be tested on a device table
without PyAudio; ``list_audio_devices`` and ``input_rate_for`` are the
thin hardware wrappers around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# The booth's USB desk mic, as PyAudio and /proc/asound/cards both name it
# ("Jieli Technology USB Composite Device", card 2 on 2026-09-04). Any
# other mic works via --mic-device / BOOTH_MIC_DEVICE.
MIC_DEVICE_NAME = "USB Composite Device"
# The robot: its speaker always, its mic when no USB mic is present.
SPEAKER_DEVICE_NAME = "Reachy Mini Audio"


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    inputs: int
    outputs: int
    default_rate: int

    def matches(self, fragment: str) -> bool:
        return fragment.lower() in self.name.lower()


@dataclass(frozen=True)
class AudioChoice:
    input: AudioDevice | None
    output: AudioDevice | None
    # True when no preferred mic was found and the speaker device's own
    # mic is being used instead.
    input_fallback: bool


def parse_mic_prefs(spec: str | None) -> list[str]:
    """``"Yeti, USB Composite Device"`` -> ``["Yeti", "USB Composite Device"]``.

    Empty or ``None`` means "no preferred mic": go straight to the
    speaker device's own microphone.
    """
    if not spec:
        return []
    return [part.strip() for part in spec.split(",") if part.strip()]


def choose_audio_devices(devices: Sequence[AudioDevice],
                         mic_prefs: Sequence[str],
                         speaker_name: str) -> AudioChoice:
    """Pick the mic and speaker from a device table.

    Speaker: the first device matching ``speaker_name`` that can play.
    Mic: the first device matching each of ``mic_prefs`` in order that can
    record; failing all of them, the speaker device's own mic.
    """
    output = next((d for d in devices
                   if d.matches(speaker_name) and d.outputs > 0), None)
    for pref in mic_prefs:
        mic = next((d for d in devices if d.matches(pref) and d.inputs > 0),
                   None)
        if mic is not None:
            return AudioChoice(mic, output, input_fallback=False)
    fallback = next((d for d in devices
                     if d.matches(speaker_name) and d.inputs > 0), None)
    return AudioChoice(fallback, output, input_fallback=True)


def list_audio_devices() -> list[AudioDevice]:
    """The PyAudio device table (empty when PyAudio is not installed)."""
    try:
        import pyaudio
    except ImportError:
        return []
    pa = pyaudio.PyAudio()
    try:
        out = []
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            out.append(AudioDevice(i, str(d["name"]),
                                   int(d["maxInputChannels"]),
                                   int(d["maxOutputChannels"]),
                                   int(d["defaultSampleRate"])))
        return out
    finally:
        pa.terminate()


def input_rate_for(index: int, want: int, channels: int = 1) -> int:
    """The rate to open input device ``index`` at: ``want`` if PortAudio
    accepts it, else the device's own default rate (which the caller then
    has to resample from)."""
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        try:
            if pa.is_format_supported(want, input_device=index,
                                      input_channels=channels,
                                      input_format=pyaudio.paInt16):
                return want
        except ValueError:
            pass
        return int(pa.get_device_info_by_index(index)["defaultSampleRate"])
    finally:
        pa.terminate()
