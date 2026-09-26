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

2026-09-25: the speaker can be another sound card -- a bigger speaker
on a USB-to-jack adapter (``--speaker-device``: names tried in order,
or ``auto`` for any other USB card that can play). The robot's own mic
stays the fallback mic either way. Note what that costs: the robot's
XVF3800 audio board cancels the echo of what *it* plays, so with the
speaker on another card its mic hears the robot at full level and
barge-in (barge_in.py) mostly stops triggering.
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
    # True when none of the preferred speakers was found and the robot's
    # own speaker is used.
    output_fallback: bool = True


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
                         speaker_name: str,
                         speaker_prefs: Sequence[str] = ()) -> AudioChoice:
    """Pick the mic and speaker from a device table.

    Speaker: the first device matching each of ``speaker_prefs`` in order
    that can play; failing all of them, the first matching
    ``speaker_name`` (the robot). Mic: the first device matching each of
    ``mic_prefs`` in order that can record; failing all of them, the
    robot's own mic (``speaker_name``).
    """
    output, out_fallback = None, True
    for pref in speaker_prefs:
        output = next((d for d in devices
                       if d.matches(pref) and d.outputs > 0), None)
        if output is not None:
            out_fallback = False
            break
    if output is None:
        output = next((d for d in devices
                       if d.matches(speaker_name) and d.outputs > 0), None)
    for pref in mic_prefs:
        mic = next((d for d in devices if d.matches(pref) and d.inputs > 0),
                   None)
        if mic is not None:
            return AudioChoice(mic, output, input_fallback=False,
                               output_fallback=out_fallback)
    fallback = next((d for d in devices
                     if d.matches(speaker_name) and d.inputs > 0), None)
    return AudioChoice(fallback, output, input_fallback=True,
                       output_fallback=out_fallback)


def parse_asound_cards(text: str) -> list[tuple[int, str, str]]:
    """``/proc/asound/cards`` -> [(index, driver, name)], e.g.
    ``(1, "USB-Audio", "Reachy Mini Audio")``."""
    out = []
    for line in text.splitlines():
        head, sep, rest = line.partition("]: ")
        index = head.strip().split(" ")[0]
        if not sep or not index.isdigit():
            continue
        driver, _, name = rest.partition(" - ")
        out.append((int(index), driver.strip(), name.strip()))
    return out


def auto_speakers(cards: Sequence[tuple[int, str, str]],
                  exclude: Sequence[str],
                  can_play=lambda index: True) -> list[str]:
    """``--speaker-device auto``: every USB sound card that can play and
    is neither the robot's nor a preferred mic's (``exclude``, name
    substrings), as names to try in order."""
    return [name for index, driver, name in cards
            if driver == "USB-Audio" and can_play(index)
            and not any(x and x.lower() in name.lower() for x in exclude)]


def card_can_play(index: int) -> bool:
    import glob
    return bool(glob.glob(f"/proc/asound/card{index}/pcm*p"))


def speaker_prefs(spec: str | None, robot: str,
                  mic_prefs: Sequence[str]) -> list[str]:
    """``--speaker-device`` -> the names to try before the robot's own
    speaker: ``""`` none, ``auto`` any other USB card that can play, else
    a comma-separated list."""
    if not spec:
        return []
    if spec.strip().lower() != "auto":
        return parse_mic_prefs(spec)
    try:
        with open("/proc/asound/cards") as fh:
            cards = parse_asound_cards(fh.read())
    except OSError:
        return []
    return auto_speakers(cards, [robot, *mic_prefs], card_can_play)


def alsa_card_of(name: str) -> str | None:
    """The ALSA card index in a PyAudio name ("... (hw:3,0)")."""
    import re
    m = re.search(r"\(hw:(\d+),", name or "")
    return m.group(1) if m else None


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


def output_rate_for(index: int, want: int, channels: int = 1) -> int:
    """The rate to open output device ``index`` at: ``want`` if PortAudio
    accepts it, else the device's own default rate (2026-09-25: the AB13X
    USB speaker adapter plays only 8 or 48 kHz, the robot only 16 kHz)."""
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        try:
            if pa.is_format_supported(want, output_device=index,
                                      output_channels=channels,
                                      output_format=pyaudio.paInt16):
                return want
        except ValueError:
            pass
        return int(pa.get_device_info_by_index(index)["defaultSampleRate"])
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
