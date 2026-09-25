"""Span tags (T6): how one reply carries two languages.

The tutor's replies mix languages by design — *"'Library' is [es]la
biblioteca[/es]."* Local voices are per-language, so the speech layer
splits a tagged reply into spans and synthesizes each with the right
voice and engine, stitched in order.

The tag convention (also spelled out in the system prompt): wrap a span
that is NOT in the reply's main language in ``[xx]...[/xx]``, where
``xx`` is the two-letter language code. Everything untagged is the main
language.

Parsing rules, in order of importance:

1. **Never crash, and never let a bracket be spoken aloud.** Malformed
   input (unclosed tags, unknown codes, stray closers, tag-only text)
   degrades to plain text in the main language with all tag markers
   stripped.
2. Only a *balanced* ``[xx]...[/xx]`` pair with a known code becomes a
   foreign span. Anything else is treated as if the model had not tagged
   at all.
3. Nesting is not part of the convention; inner tag markers inside a
   balanced span are stripped, and the span keeps the outer language.

This module is dependency-free on purpose — the parser is unit-tested in
the light test venv, hard (T6's definition of done says tag leakage must
be impossible by test).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A balanced pair with a plausible code. Non-greedy, spans newlines.
_PAIR_RX = re.compile(r"\[([a-z]{2})\](.*?)\[/\1\]", re.S)
# Any leftover tag-shaped marker, to be stripped from spoken text. Case-
# insensitive on purpose: [ES] is outside the convention (only lowercase
# pairs make spans), but speaking "bracket E S" aloud would be worse than
# ignoring a model's case slip.
_ORPHAN_RX = re.compile(r"\[/?[a-z]{2}\]", re.I)


@dataclass
class Span:
    language: str
    text: str


def _clean(text: str) -> str:
    """Strip stray tag markers and collapse the whitespace they leave."""
    text = _ORPHAN_RX.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def split_spans(text: str, primary: str,
                known: set[str] | None = None) -> list[Span]:
    """Split a tagged reply into ordered (language, text) spans.

    ``known`` is the set of language codes the speech stack can actually
    speak; a tagged span in an unknown language folds back into the
    primary (the words are still said, just in the main voice — better
    than silence or brackets).
    """
    spans: list[Span] = []

    def push(language: str, chunk: str) -> None:
        chunk = _clean(chunk)
        if not chunk:
            return
        if spans and spans[-1].language == language:
            spans[-1].text += " " + chunk
        else:
            spans.append(Span(language, chunk))

    pos = 0
    for match in _PAIR_RX.finditer(text):
        push(primary, text[pos:match.start()])
        code, inner = match.group(1), match.group(2)
        if known is not None and code not in known:
            push(primary, inner)
        else:
            push(code, inner)
        pos = match.end()
    push(primary, text[pos:])

    if not spans:
        return [Span(primary, "")]
    return spans


def strip_tags(text: str) -> str:
    """The reply as plain text (for logs and transcripts)."""
    return _clean(_PAIR_RX.sub(lambda m: m.group(2), text))
