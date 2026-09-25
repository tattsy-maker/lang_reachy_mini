"""The round-trip gate (T5): does a language survive its own voice?

The repo's established shipping test: synthesize a fixed sentence set with
the engine that would speak it live, transcribe the audio back with the
same Whisper model the agent listens with, and score the match. ≥90%
average ships; below falls back per spec section 6.

    voice/.venv/bin/python voice/verify_language.py            # ru + zh
    voice/.venv/bin/python voice/verify_language.py --languages ru --limit 3

Reports (JSON + Markdown) land in tests/reports/. No ffmpeg needed —
audio goes to Whisper as a 16 kHz float array, same as the live pipeline.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
REPORTS_DIR = _REPO / "tests" / "reports"

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo-q4"

SENTENCES = {
    "ru": [
        "Привет! Я маленький робот и я помогу тебе учить русский язык.",
        "Как прошли твои выходные? Расскажи мне немного.",
        "Это слово произносится немного иначе, попробуй ещё раз.",
        "Очень хорошо! Ты делаешь большие успехи.",
        "Вчера мы говорили о путешествиях, помнишь?",
        "Не волнуйся, ошибки это часть учёбы.",
        "Давай повторим числа от одного до десяти.",
        "До свидания, увидимся на следующем занятии!",
    ],
    "zh": [
        "你好！我是一个小机器人，我会帮你学习中文。",
        "你周末过得怎么样？跟我说说吧。",
        "这个词的发音有点不一样，再试一次。",
        "非常好！你进步很大。",
        "昨天我们聊了旅行，你还记得吗？",
        "别担心，犯错误是学习的一部分。",
        "我们来复习一到十的数字吧。",
        "再见，下次课见！",
    ],
}


def synth_piper(code: str, text: str, voices_dir=None) -> np.ndarray:
    """Piper synthesis -> mono float32 at 16 kHz (the pipeline's rate)."""
    from piper import PiperVoice
    from piper_tts import DEFAULT_VOICES_DIR, PIPER_LANGUAGES
    voices_dir = Path(voices_dir or DEFAULT_VOICES_DIR)
    name = PIPER_LANGUAGES[code].voice
    cache = synth_piper.__dict__.setdefault("cache", {})
    if name not in cache:
        cache[name] = PiperVoice.load(voices_dir / f"{name}.onnx")
    voice = cache[name]
    pieces, rate = [], 22050
    for chunk in voice.synthesize(text):
        pieces.append(chunk.audio_float_array)
        rate = chunk.sample_rate
    audio = np.concatenate(pieces).astype(np.float32)
    return resample_16k(audio, rate)


def synth_kokoro(code: str, text: str, voice: str | None = None) -> np.ndarray:
    """Kokoro synthesis -> mono float32 at 16 kHz, engine-direct (no
    pipeline), using the same cached model files as the live service."""
    from pipecat.services.kokoro.tts import (
        KOKORO_CACHE_DIR, _ensure_model_files)
    from kokoro_onnx import Kokoro
    from multilingual import LANGUAGES, MultilingualKokoro
    cache = synth_kokoro.__dict__
    if "kokoro" not in cache:
        model = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
        voices = KOKORO_CACHE_DIR / "voices-v1.0.bin"
        _ensure_model_files(model, voices)
        cache["kokoro"] = Kokoro(str(model), str(voices))
    spoken = LANGUAGES[code]
    lang = MultilingualKokoro.ESPEAK_CODES.get(spoken.language)
    if lang is None:
        lang = str(spoken.language.value).lower()
        lang = {"en": "en-us"}.get(lang, lang)
    samples, rate = cache["kokoro"].create(
        text, voice=voice or spoken.voice, speed=1.0, lang=lang)
    return resample_16k(samples.astype(np.float32), rate)


def synth_mixed(text: str, primary: str, voices_dir=None) -> np.ndarray:
    """T6's assembly, engine-direct: split the tagged text into spans,
    synthesize each with its engine (Kokoro or Piper), stitch in order.
    Mirrors DualEngineTTS's per-span dispatch for measurement scripts."""
    from spans import split_spans
    from piper_tts import PIPER_LANGUAGES
    from multilingual import LANGUAGES
    known = set(LANGUAGES) | set(PIPER_LANGUAGES)
    pieces = []
    for span in split_spans(text, primary, known=known):
        if not span.text:
            continue
        if span.language in PIPER_LANGUAGES:
            pieces.append(synth_piper(span.language, span.text, voices_dir))
        else:
            pieces.append(synth_kokoro(span.language, span.text))
        # a hair of silence at the seam reads as natural pacing
        pieces.append(np.zeros(int(0.08 * 16000), dtype=np.float32))
    return (np.concatenate(pieces) if pieces
            else np.zeros(1600, dtype=np.float32))


def resample_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return audio
    n = int(len(audio) * 16000 / rate)
    return np.interp(np.linspace(0, len(audio) - 1, n),
                     np.arange(len(audio)), audio).astype(np.float32)


def transcribe(audio: np.ndarray) -> tuple[str, str]:
    """(text, detected language) via the agent's own Whisper model."""
    import mlx_whisper
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL,
                                    language=None)
    return result["text"].strip(), str(result.get("language", ""))


def normalize(text: str, code: str) -> str:
    """Strip everything that is not meaning: case, punctuation, spacing.
    Russian ё/е are folded (Whisper writes both)."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^\w]+", "", text, flags=re.UNICODE)
    return text


def match_score(said: str, heard: str, code: str) -> float:
    """0-100 similarity between what was said and what came back."""
    a, b = normalize(said, code), normalize(heard, code)
    if not a:
        return 0.0
    return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()


def run_gate(code: str, limit: int | None = None,
             voices_dir=None) -> dict:
    sentences = SENTENCES[code][:limit]
    rows = []
    for said in sentences:
        t = time.monotonic()
        audio = synth_piper(code, said, voices_dir)
        synth_secs = time.monotonic() - t
        heard, detected = transcribe(audio)
        rows.append({
            "said": said, "heard": heard, "detected": detected,
            "score": round(match_score(said, heard, code), 1),
            "audio_secs": round(len(audio) / 16000, 2),
            "synth_secs": round(synth_secs, 2),
        })
    scores = [r["score"] for r in rows]
    return {
        "language": code,
        "engine": "piper",
        "whisper_model": WHISPER_MODEL,
        "average": round(sum(scores) / len(scores), 1),
        "minimum": min(scores),
        "passes_90": sum(scores) / len(scores) >= 90.0,
        "sentences": rows,
    }


def write_report(results: list[dict], tag: str = "") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d") + tag
    path = REPORTS_DIR / f"verify_language_{stamp}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")

    md = [f"# Round-trip gate — {stamp}", ""]
    for r in results:
        md.append(f"## {r['language']} ({r['engine']}) — average "
                  f"{r['average']}%, min {r['minimum']}% — "
                  f"{'PASS' if r['passes_90'] else 'FAIL'} (gate 90%)")
        md.append("")
        for row in r["sentences"]:
            md.append(f"- {row['score']}% [{row['detected']}] "
                      f"“{row['said']}” → “{row['heard']}”")
        md.append("")
    (REPORTS_DIR / f"verify_language_{stamp}.md").write_text(
        "\n".join(md) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--languages", nargs="+", default=["ru", "zh"],
                    choices=sorted(SENTENCES))
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N sentences (smoke mode)")
    ap.add_argument("--voices-dir", default=None)
    ap.add_argument("--tag", default="",
                    help="suffix for the report filename (smoke runs use "
                         "one so they never clobber a recorded full run)")
    args = ap.parse_args()

    results = [run_gate(code, args.limit, args.voices_dir)
               for code in args.languages]
    if args.limit and not args.tag:
        args.tag = "-smoke"
    path = write_report(results, args.tag)
    for r in results:
        print(f"{r['language']}: average {r['average']}% "
              f"(min {r['minimum']}%) -> "
              f"{'PASS' if r['passes_90'] else 'FAIL'}")
    print(f"report: {path}")
    return 0 if all(r["passes_90"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
