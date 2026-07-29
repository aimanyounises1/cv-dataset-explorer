"""Refusing a premise costs the premise, not the turn.

The deterministic claim gate used to return its refusal as the whole answer, so
a question carrying one unsupported number threw away the lanes' grounded
findings alongside it — the retrieval had run, was correct, and was discarded.

These pin the splitting rule rather than the wording, because the rule is what
is easy to get wrong: figures must match as whole numbers. A bare substring
check fires inside 4301 and 0.302, which makes the filter either inert or
indiscriminate depending on the sentence, and this repo has been bitten by
exactly that before.
"""
from app.agent.graph import _answer_without

CLAIM = "Summarize the 30% retrieval improvement from last week"
GROUNDED = "Search returned 12 images of dogs jumping into water."


def test_the_grounded_half_of_the_answer_survives():
    answer = f"The 30% improvement is not in any tool result. {GROUNDED}"
    assert _answer_without([CLAIM], answer) == GROUNDED


def test_the_sentence_restating_the_premise_is_dropped():
    assert "30%" not in _answer_without([CLAIM], f"We measured 30% better. {GROUNDED}")


def test_a_figure_is_matched_as_a_number_not_as_a_substring():
    """30 must not match inside 4301 or 0.302 — the trap that made an earlier
    version of this filter inert."""
    kept = _answer_without(["a 30% gain"], "Cluster 4301 has 0.302 mean agreement.")
    assert kept == "Cluster 4301 has 0.302 mean agreement."


def test_a_claim_carrying_no_figures_costs_the_answer_nothing():
    """There is nothing to excise, so nothing is excised: the refusal still
    leads, but the answer is returned whole."""
    assert _answer_without(["it improved a lot"], GROUNDED) == GROUNDED


def test_an_answer_that_is_entirely_the_premise_collapses_to_empty():
    """Then the caller falls back to the refusal alone, which is the old
    behaviour and the right one — there is nothing grounded left to keep."""
    assert _answer_without([CLAIM], "The 30% improvement came from better captions.") == ""


def test_every_unverified_claim_contributes_its_figures():
    answer = "Recall rose 30%. Latency fell 12ms. Search returned 12 images."
    kept = _answer_without(["a 30% gain", "a 12ms saving"], answer)
    assert "30%" not in kept and "12ms" not in kept


def test_decimal_and_thousands_figures_are_matched_whole():
    answer = "Agreement was 0.118 on that sample. The corpus holds 8,000 images."
    kept = _answer_without(["agreement of 0.118"], answer)
    assert "0.118" not in kept and "8,000" in kept
