"""T6: the span parser, tested hard — tag leakage must be impossible."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice"))

from spans import Span, split_spans, strip_tags  # noqa: E402

KNOWN = {"en", "es", "fr", "ru", "zh", "pt", "it", "hi"}


def spans(text, primary="en", known=KNOWN):
    return split_spans(text, primary, known=known)


def test_plain_text_is_one_primary_span():
    assert spans("Hello there.") == [Span("en", "Hello there.")]


def test_single_embedded_span():
    got = spans("The word is [es]la biblioteca[/es], say it!")
    assert got == [Span("en", "The word is"),
                   Span("es", "la biblioteca"),
                   Span("en", ", say it!")]


def test_multiple_spans_and_merging():
    got = spans("[es]Hola[/es] and [es]adiós[/es]")
    assert [s.language for s in got] == ["es", "en", "es"]
    got = spans("[es]Hola[/es][es]amigo[/es]")
    assert got == [Span("es", "Hola amigo")]  # adjacent same-language merge


def test_unclosed_tag_degrades_to_primary():
    got = spans("Say [es]la biblioteca and more")
    assert got == [Span("en", "Say la biblioteca and more")]


def test_stray_closer_is_stripped():
    got = spans("Nice work[/es] there")
    assert got == [Span("en", "Nice work there")]


def test_unknown_code_folds_into_primary():
    got = spans("Hello [xq]whatever[/xq] friend")
    assert got == [Span("en", "Hello whatever friend")]


def test_nested_tags_keep_outer_language():
    got = spans("[es]hola [fr]salut[/fr] amigo[/es]")
    assert got[0].language == "es"
    assert "salut" in got[0].text
    assert "[" not in got[0].text


def test_tag_only_and_empty_inputs():
    assert spans("[es][/es]") == [Span("en", "")]
    assert spans("") == [Span("en", "")]
    assert spans("   ") == [Span("en", "")]


def test_no_bracket_ever_survives():
    nasty = [
        "plain",
        "[es]bien[/es]",
        "[es]bien",
        "bien[/es]",
        "[es][/es]",
        "[es]a[fr]b[/fr]c[/es]",
        "[xx]?[/xx] [es]si[/es] [/fr] [en",
        "one [es]dos[/es] three [ru]четыре[/ru] five",
        "[ES]upper is not a tag[/ES]",
    ]
    for text in nasty:
        for span in spans(text):
            assert "[es]" not in span.text and "[/" not in span.text, \
                f"tag leaked from {text!r}: {span}"
        assert "[/" not in strip_tags(text)


def test_uppercase_is_not_a_tag_but_never_leaks():
    # [ES] is outside the convention; the bracketed text must still not
    # be lost, and lowercase-tag stripping must not eat it silently.
    got = spans("[ES]hola[/ES]")
    assert got[0].language == "en"
    assert "hola" in got[0].text


def test_strip_tags_for_transcripts():
    assert strip_tags("The word is [es]la biblioteca[/es]!") == \
        "The word is la biblioteca!"
