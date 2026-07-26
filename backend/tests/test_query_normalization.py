"""SigLIP query-side text normalization.

SigLIP 2 tokenizes with a case-sensitive Gemma vocabulary and `AutoProcessor`
does not apply the model's canonical lowercase+depunctuate step, so the raw
query text encodes to a measurably worse vector than the normalized one
(benchmark R@1 46.0% -> 53.2% over 1,000 held-out captions).

What these pin is the property that made the change safe to ship: it is a
*no-op* on the short lowercase phrases users actually type, so no query that
works today can be reworded by it.
"""
import pytest

from app.api.search import normalize_query_text


@pytest.mark.parametrize("q", [
    "dog",
    "dog running in snow",
    "two men playing football",
    "a child on a red slide",
])
def test_already_normal_queries_are_returned_unchanged(q):
    """The typical short query is untouched, so it cannot regress."""
    assert normalize_query_text(q) == q


@pytest.mark.parametrize("raw, expected", [
    ("A black dog is running in the snow .", "a black dog is running in the snow"),
    ("Two Men Playing Football!", "two men playing football"),
    ("a child's red slide", "a child s red slide"),
    ("dog,cat", "dog cat"),
    ("  spaced   out  ", "spaced out"),
])
def test_casing_and_punctuation_are_removed(raw, expected):
    assert normalize_query_text(raw) == expected


def test_normalization_is_idempotent():
    """Applying it twice must not drift — search and the benchmark both call it,
    and a non-idempotent normalizer would make them disagree on repeat paths."""
    for q in ["A black dog is running .", "Two Men!", "dog", "a child's toy"]:
        once = normalize_query_text(q)
        assert normalize_query_text(once) == once


def test_a_query_that_is_only_punctuation_survives():
    """Normalizing "???" to "" would hand the encoder an empty string; the
    original is a more useful thing to embed than nothing."""
    assert normalize_query_text("???") == "???"
    assert normalize_query_text("") == ""


def test_unicode_letters_are_kept():
    """`\\w` is unicode-aware, so accented text is lowercased, not stripped."""
    assert normalize_query_text("Café Münster") == "café münster"
