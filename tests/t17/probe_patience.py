#!/usr/bin/env python3
"""Does Gemini wait through a mid-sentence pause with our VAD settings?
(T17.1, run under voice/.venv; needs GEMINI_API_KEY.)

    voice/.venv/bin/python tests/t17/probe_patience.py --patience-ms 1800
    voice/.venv/bin/python tests/t17/probe_patience.py --patience-ms 0   # Gemini's default

Streams, in real time, one fixture sentence + ``--pause`` seconds of
silence + a second sentence + a few seconds of silence to a Gemini Live
session configured exactly as agent.py configures it (turns.vad_settings),
with input transcription on, and counts how many times the model took
the turn. One turn means the pause was not taken for the end of the
sentence. Prints JSON: {"patience_ms", "turns", "transcripts", "secs"}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "voice"))

VOICES = REPO / "tests" / "fixtures" / "voices" / "af_heart"
RATE = 16000


def build_audio(pause_secs: float, tail_secs: float) -> np.ndarray:
    from scipy.io import wavfile
    parts = []
    for name in ("af_heart_1.wav", "af_heart_2.wav"):
        sr, data = wavfile.read(str(VOICES / name))
        assert sr == RATE, (name, sr)
        parts.append(np.asarray(data, dtype=np.int16).reshape(-1))
        parts.append(np.zeros(int(RATE * pause_secs), dtype=np.int16))
    parts[-1] = np.zeros(int(RATE * tail_secs), dtype=np.int16)
    return np.concatenate(parts)


async def probe(api_key: str, model: str, patience_ms: int, pause_secs: float,
                tail_secs: float, max_secs: float) -> dict:
    from google import genai
    from google.genai import types
    from turns import vad_settings

    vad = vad_settings(patience_ms)
    kwargs = dict(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction="You are a test partner. Whatever you hear, "
                           "answer with one short word.",
    )
    if vad is not None:
        kwargs["realtime_input_config"] = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                end_of_speech_sensitivity=types.EndSensitivity(vad["end_sensitivity"]),
                silence_duration_ms=vad["silence_duration_ms"],
                prefix_padding_ms=vad["prefix_padding_ms"]))
    config = types.LiveConnectConfig(**kwargs)
    audio = build_audio(pause_secs, tail_secs)
    turns, transcripts = 0, []
    t0 = time.monotonic()
    client = genai.Client(api_key=api_key)
    async with client.aio.live.connect(model=model, config=config) as session:

        async def sender():
            chunk = RATE // 50                      # 20 ms
            for i in range(0, len(audio), chunk):
                await session.send_realtime_input(audio=types.Blob(
                    data=audio[i:i + chunk].tobytes(),
                    mime_type=f"audio/pcm;rate={RATE}"))
                await asyncio.sleep(0.02)
            # keep feeding silence so the server keeps talking to us
            # while the last reply comes in
            silence = np.zeros(chunk, dtype=np.int16).tobytes()
            for _ in range(150):                    # 3 s
                await session.send_realtime_input(audio=types.Blob(
                    data=silence, mime_type=f"audio/pcm;rate={RATE}"))
                await asyncio.sleep(0.02)

        async def receiver():
            nonlocal turns
            while True:
                async for msg in session.receive():
                    sc = getattr(msg, "server_content", None)
                    if sc is None:
                        continue
                    it = getattr(sc, "input_transcription", None)
                    if it is not None and it.text:
                        transcripts.append(it.text)
                    if getattr(sc, "turn_complete", False):
                        turns += 1

        rx = asyncio.create_task(receiver())
        try:
            await asyncio.wait_for(sender(), timeout=max_secs)
        finally:
            rx.cancel()
            try:
                await rx
            except (asyncio.CancelledError, Exception):     # noqa: BLE001
                pass
    return {"patience_ms": patience_ms, "turns": turns,
            "transcripts": ["".join(transcripts)],
            "secs": round(time.monotonic() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--patience-ms", type=int, default=1800)
    ap.add_argument("--pause", type=float, default=1.2)
    ap.add_argument("--tail", type=float, default=4.0)
    ap.add_argument("--max-secs", type=float, default=40.0)
    ap.add_argument("--model", default="models/gemini-3.1-flash-live-preview")
    args = ap.parse_args()
    from agent import load_env_file
    load_env_file()
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("no GEMINI_API_KEY", file=sys.stderr)
        return 2
    result = asyncio.run(probe(key, args.model, args.patience_ms, args.pause,
                               args.tail, args.max_secs))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
