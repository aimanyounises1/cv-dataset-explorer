"""The FTS conjunction: which query terms become hard requirements.

`fts_escape` builds an implicit AND, so every term it keeps is a term a caption
MUST contain. Dropping function words from that conjunction is the change these
tests pin — along with the two ways it must not be "improved" further.
"""
from app import db


def _terms(expr: str) -> list[str]:
    """The quoted terms inside a MATCH expression."""
    return [t.strip('"') for t in expr.split() if t]


def test_stopwords_are_not_required_by_the_match():
    assert _terms(db.fts_escape("a man on a bench")) == ["man", "bench"]


def test_content_words_are_all_still_required():
    """Narrowing the conjunction, not widening it — the terms that remain are
    still ANDed. Widening to OR was measured twice and refuted twice: keyword
    R@10 rises but hybrid MRR falls 0.6313 -> 0.5383."""
    expr = db.fts_escape("dog frisbee beach")
    assert _terms(expr) == ["dog", "frisbee", "beach"]
    assert " OR " not in expr and "OR" not in expr.replace('"', "")


def test_stopword_matching_ignores_case_and_punctuation():
    assert _terms(db.fts_escape("The dog, in The park.")) == ["dog,", "park."]


def test_an_all_stopword_query_keeps_its_terms():
    """"the two" should search for something rather than match everything —
    dropping every term would make the expression empty, which callers treat as
    "no lexical results at all"."""
    assert _terms(db.fts_escape("the two")) == ["the", "two"]


def test_quotes_are_still_escaped():
    """The terms reaching FTS5 are quoted; an embedded quote must be doubled or
    it closes the string and the rest of the query becomes syntax."""
    assert db.fts_escape('say "hi"') == '"say" """hi"""'


def test_empty_query_yields_an_empty_expression():
    assert db.fts_escape("   ") == ""


def test_match_terms_is_what_the_ui_highlights():
    """`matched_terms` drives highlighting, so it has to be the same list the
    conjunction used — otherwise the UI underlines words that were not required."""
    assert db.match_terms("a dog on the beach") == ["dog", "beach"]
    assert db.match_terms("the") == ["the"]
