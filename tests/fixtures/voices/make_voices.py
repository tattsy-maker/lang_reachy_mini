#!/usr/bin/env python3
"""Synthesize the speaker-verification fixtures (T13.9).

Five Kokoro voices, four sentences each, 16 kHz mono WAV: a "speaker" is
one Kokoro voice, so same-speaker pairs are the same voice saying
different things and different-speaker pairs are different voices. These
are synthetic stand-ins (see README.md): they measure whether ECAPA
separates *these* timbres, not how human day-to-day variation behaves —
that measurement needs the family at the robot and is logged in
progress/T13.md when it happens.

Run from voice/.venv (Kokoro lives there):

    ../../../voice/.venv/bin/python make_voices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
MODEL = Path.home() / ".cache/pipecat/kokoro-onnx/kokoro-v1.0.onnx"
VOICES = Path.home() / ".cache/pipecat/kokoro-onnx/voices-v1.0.bin"

SPEAKERS = {
    "af_heart": "en-us",     # American female
    "am_adam": "en-us",      # American male
    "bf_emma": "en-gb",      # British female
    "bm_george": "en-gb",    # British male
    "ef_dora": "es",         # Spanish female (a different language entirely)
}
SENTENCES = {
    "en-us": [
        "Good morning, I would like to practice a little Spanish today.",
        "Could you say that again, more slowly, so I can repeat it?",
        "Yesterday we went to the market and bought far too many oranges.",
        "I think my level is somewhere between beginner and intermediate.",
    ],
    "es": [
        "Buenos días, hoy me gustaría practicar un poco de inglés.",
        "¿Puedes repetirlo más despacio para que lo pueda decir yo?",
        "Ayer fuimos al mercado y compramos demasiadas naranjas.",
        "Creo que mi nivel está entre principiante e intermedio.",
    ],
}
SENTENCES["en-gb"] = SENTENCES["en-us"]


def main() -> int:
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(str(MODEL), str(VOICES))
    for voice, lang in SPEAKERS.items():
        folder = HERE / voice
        folder.mkdir(exist_ok=True)
        for i, text in enumerate(SENTENCES[lang], 1):
            samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang=lang)
            pcm = resample_poly(samples, 16000, sr) if sr != 16000 else samples
            pcm = np.clip(pcm, -1.0, 1.0)
            path = folder / f"{voice}_{i}.wav"
            wavfile.write(path, 16000, (pcm * 32767).astype(np.int16))
            print(f"{path.relative_to(HERE)}  {len(pcm) / 16000:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
