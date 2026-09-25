"""The code-switch gate (T7): does an embedded foreign phrase survive?

For every pair (English + each tutoring language) this synthesizes a set
of code-switched phrases with T6's assembly (each span in its own voice
and engine), transcribes the audio with the agent's Whisper **with and
without bilingual priming**, and scores whether the embedded foreign span
came back recognizably. The deliverable is the honest per-pair number —
which pairs are "seamless locally" and which need cloud mode (spec §6).

    voice/.venv/bin/python voice/verify_codeswitch.py                # all pairs
    voice/.venv/bin/python voice/verify_codeswitch.py --pairs es ru --limit 3

Reports (JSON + Markdown) land in tests/reports/.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

# The gate measures the RAW priming effect (PRIMING directly), so the
# per-pair policy in bilingual_priming() can be re-derived from evidence
# whenever the model or the prompt changes.
from multilingual import PRIMING  # noqa: E402
from spans import split_spans  # noqa: E402
from verify_language import (  # noqa: E402
    REPORTS_DIR,
    WHISPER_MODEL,
    normalize,
    synth_mixed,
)

# Eight English carrier phrases per pair, each embedding one span of the
# target language — the natural register of a lesson (asking about a word,
# quoting an answer, checking a phrase).
PHRASES = {
    "es": [
        "How do you say [es]la biblioteca[/es] in English?",
        "Does [es]tengo hambre[/es] mean I am hungry?",
        "Yesterday I learned the word [es]madrugada[/es] in class.",
        "My favorite phrase is [es]buenos días[/es], it sounds friendly.",
        "The waiter said [es]la cuenta, por favor[/es] very fast.",
        "Is it correct to say [es]me gusta mucho[/es] here?",
        "She wrote [es]nos vemos mañana[/es] at the end.",
        "I always forget that [es]el mapa[/es] is masculine.",
    ],
    "fr": [
        "How do you say [fr]la bibliothèque[/fr] in English?",
        "Does [fr]j'ai faim[/fr] mean I am hungry?",
        "Yesterday I learned the word [fr]la gare[/fr] in class.",
        "My favorite phrase is [fr]bonne journée[/fr], it sounds friendly.",
        "The waiter said [fr]l'addition, s'il vous plaît[/fr] very fast.",
        "Is it correct to say [fr]ça me plaît beaucoup[/fr] here?",
        "She wrote [fr]à demain[/fr] at the end.",
        "I always forget that [fr]le musée[/fr] is masculine.",
    ],
    "it": [
        "How do you say [it]la biblioteca[/it] in English?",
        "Does [it]ho fame[/it] mean I am hungry?",
        "Yesterday I learned the word [it]il binario[/it] in class.",
        "My favorite phrase is [it]buongiorno a tutti[/it], it sounds friendly.",
        "The waiter said [it]il conto, per favore[/it] very fast.",
        "Is it correct to say [it]mi piace molto[/it] here?",
        "She wrote [it]a domani[/it] at the end.",
        "I always forget that [it]il problema[/it] is masculine.",
    ],
    "pt": [
        "How do you say [pt]a biblioteca[/pt] in English?",
        "Does [pt]estou com fome[/pt] mean I am hungry?",
        "Yesterday I learned the word [pt]a madrugada[/pt] in class.",
        "My favorite phrase is [pt]bom dia para todos[/pt], it sounds friendly.",
        "The waiter said [pt]a conta, por favor[/pt] very fast.",
        "Is it correct to say [pt]eu gosto muito[/pt] here?",
        "She wrote [pt]até amanhã[/pt] at the end.",
        "I always forget that [pt]o mapa[/pt] is masculine.",
    ],
    "ru": [
        "How do you say [ru]библиотека[/ru] in English?",
        "Does [ru]я хочу есть[/ru] mean I am hungry?",
        "Yesterday I learned the word [ru]вокзал[/ru] in class.",
        "My favorite phrase is [ru]доброе утро[/ru], it sounds friendly.",
        "The waiter said [ru]счёт, пожалуйста[/ru] very fast.",
        "Is it correct to say [ru]мне очень нравится[/ru] here?",
        "She wrote [ru]до завтра[/ru] at the end.",
        "I always forget the stress in [ru]хорошо[/ru].",
    ],
    "zh": [
        "How do you say [zh]图书馆[/zh] in English?",
        "Does [zh]我饿了[/zh] mean I am hungry?",
        "Yesterday I learned the word [zh]火车站[/zh] in class.",
        "My favorite phrase is [zh]早上好[/zh], it sounds friendly.",
        "The waiter said [zh]买单，谢谢[/zh] very fast.",
        "Is it correct to say [zh]我很喜欢[/zh] here?",
        "She wrote [zh]明天见[/zh] at the end.",
        "I always forget the tones in [zh]你好[/zh].",
    ],
}


def transcribe(audio: np.ndarray, prompt: str | None) -> str:
    import mlx_whisper
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL,
                                    language=None, initial_prompt=prompt)
    return result["text"].strip()


def span_survival(span_text: str, transcript: str, code: str) -> float:
    """0-100: how recognizably the embedded span appears in the transcript.

    Longest common substring against the normalized transcript, relative
    to the span's length — 100 means the whole span is present verbatim.
    """
    span_n = normalize(span_text, code)
    trans_n = normalize(transcript, code)
    if not span_n:
        return 0.0
    if span_n in trans_n:
        return 100.0
    match = difflib.SequenceMatcher(None, span_n, trans_n)\
        .find_longest_match(0, len(span_n), 0, len(trans_n))
    return 100.0 * match.size / len(span_n)


def run_pair(code: str, limit: int | None = None) -> dict:
    prompt = PRIMING.get(code)
    rows = []
    for tagged in PHRASES[code][:limit]:
        span = next(s for s in split_spans(tagged, "en") if s.language == code)
        audio = synth_mixed(tagged, "en")
        plain = transcribe(audio, None)
        primed = transcribe(audio, prompt)
        rows.append({
            "phrase": tagged,
            "span": span.text,
            "unprimed": {"heard": plain,
                         "survival": round(span_survival(span.text, plain,
                                                         code), 1)},
            "primed": {"heard": primed,
                       "survival": round(span_survival(span.text, primed,
                                                       code), 1)},
        })

    def avg(kind):
        return round(sum(r[kind]["survival"] for r in rows) / len(rows), 1)

    return {"pair": f"en+{code}", "whisper_model": WHISPER_MODEL,
            "priming": prompt, "unprimed_avg": avg("unprimed"),
            "primed_avg": avg("primed"), "phrases": rows}


def write_report(results: list[dict], tag: str = "") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d") + tag
    path = REPORTS_DIR / f"verify_codeswitch_{stamp}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")

    md = [f"# Code-switch gate — {stamp}", "",
          "Span survival: how much of the embedded foreign phrase came "
          "back recognizably (100 = verbatim).", ""]
    for r in results:
        md.append(f"## {r['pair']} — unprimed {r['unprimed_avg']}%, "
                  f"primed {r['primed_avg']}%")
        md.append("")
        for row in r["phrases"]:
            md.append(f"- “{row['span']}”: {row['unprimed']['survival']}% → "
                      f"{row['primed']['survival']}% primed")
            md.append(f"  - unprimed: {row['unprimed']['heard']}")
            md.append(f"  - primed: {row['primed']['heard']}")
        md.append("")
    (REPORTS_DIR / f"verify_codeswitch_{stamp}.md").write_text(
        "\n".join(md) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="+", default=sorted(PHRASES),
                    choices=sorted(PHRASES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.limit and not args.tag:
        args.tag = "-smoke"

    results = [run_pair(code, args.limit) for code in args.pairs]
    path = write_report(results, args.tag)
    for r in results:
        print(f"{r['pair']}: unprimed {r['unprimed_avg']}%  "
              f"primed {r['primed_avg']}%")
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
