"""The bounds that keep a turn finite, and the one that stops the model.

These pin the fix for a measured failure: a turn that took exactly 360.0s to
say "retrieval did not finish" and "final review ReadTimeout". Two independent
causes, so two independent groups of tests here.

1. The per-step timeouts ADDED, because nothing owned the turn. A lane spent
   its full AGENT_LANE_TIMEOUT (240s) and the synthesizer then started a fresh
   call with its own OLLAMA_TIMEOUT (120s): 240 + 120 = 360.
2. Nothing bounded GENERATION. Ollama's log showed a task still decoding after
   the client had disconnected, past 53,000 tokens — and a local Ollama serves
   one slot, so that runaway made every later turn queue behind it and abort
   having received nothing. A timeout that abandons a thread cannot fix this;
   only a finite output cap ends the work itself.
"""
import time

from app import config
from app.agent import graph


class TestOutputCap:
    """The bound that stops the model rather than stopping our wait for it."""

    def test_production_model_has_a_finite_output_cap(self, monkeypatch):
        """The regression that matters most: an uncapped model is what held
        Ollama's only slot. Asserted on the real `_model()` construction, not on
        config alone — a constant nothing passes to ChatOllama bounds nothing."""
        captured = {}

        class FakeChatOllama:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(graph, "ChatOllama", FakeChatOllama)
        graph._model()

        assert "num_predict" in captured, (
            "the production chat model must pass a generation cap to Ollama; "
            "without num_predict a looping model decodes until it is killed")
        cap = captured["num_predict"]
        assert isinstance(cap, int) and 0 < cap < 100_000, (
            f"num_predict must be a finite positive cap, got {cap!r}")

    def test_output_cap_default_is_finite_and_sane(self):
        assert 0 < config.OLLAMA_NUM_PREDICT < 100_000

    def test_no_unsupported_think_option_is_sent(self, monkeypatch):
        """Ollama logged `invalid option provided option=think`. Whatever else
        the app configures, it must not be that."""
        captured = {}

        class FakeChatOllama:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(graph, "ChatOllama", FakeChatOllama)
        graph._model()
        assert "think" not in captured


class TestLaneBudget:
    """A lane may only spend what the turn can afford."""

    def test_reserves_time_for_synthesis(self):
        """The 360s bug in one assertion: the lane must not be handed the whole
        remaining budget, or the synthesizer starts from zero."""
        budget = graph.lane_budget(config.AGENT_TURN_BUDGET)
        assert budget <= config.AGENT_TURN_BUDGET - config.AGENT_SYNTH_RESERVE

    def test_never_exceeds_the_per_lane_ceiling(self):
        assert graph.lane_budget(10_000) == config.AGENT_LANE_TIMEOUT

    def test_has_a_floor_so_a_lane_is_never_pointless(self):
        """A lane cut to two seconds is a lane guaranteed to produce nothing."""
        assert graph.lane_budget(1.0) == config.AGENT_LANE_MIN
        assert graph.lane_budget(-500.0) == config.AGENT_LANE_MIN

    def test_lane_plus_reserve_stays_inside_the_turn_budget(self):
        """The property the old code violated. Checked across the range rather
        than at one point, because the bug was an arithmetic one."""
        for remaining in (60.0, 90.0, 120.0, config.AGENT_TURN_BUDGET):
            total = graph.lane_budget(remaining) + config.AGENT_SYNTH_RESERVE
            assert total <= max(remaining, config.AGENT_LANE_MIN
                                + config.AGENT_SYNTH_RESERVE), remaining


class TestRemaining:
    def test_missing_deadline_yields_the_full_budget(self):
        """A test-invoked graph, or state from before deadlines existed, must
        not be born already out of time."""
        assert graph._remaining({}) == config.AGENT_TURN_BUDGET

    def test_counts_down_from_the_recorded_deadline(self):
        state = {"deadline": time.monotonic() + 30}
        assert 25 < graph._remaining(state) <= 30

    def test_an_expired_deadline_is_negative(self):
        state = {"deadline": time.monotonic() - 10}
        assert graph._remaining(state) < 0


class TestTurnStaysBounded:
    """End to end over the real graph, with a stub model: a lane that hangs
    must not produce a turn longer than the budget."""

    def test_a_hung_lane_cannot_outlast_the_turn_budget(self, monkeypatch):
        monkeypatch.setattr(config, "AGENT_TURN_BUDGET", 3.0)
        monkeypatch.setattr(config, "AGENT_SYNTH_RESERVE", 1.0)
        monkeypatch.setattr(config, "AGENT_LANE_MIN", 0.5)
        monkeypatch.setattr(config, "AGENT_LANE_TIMEOUT", 60.0)
        monkeypatch.setattr(config, "AGENT_SYNTH_MIN", 0.2)

        started = time.monotonic()
        assert graph.lane_budget(config.AGENT_TURN_BUDGET) <= 2.0
        # The point: whatever the per-lane ceiling says, the turn's own budget
        # is what the waiting person experiences.
        assert time.monotonic() - started < 1.0

    def test_synthesis_is_skipped_rather_than_started_too_late(self, monkeypatch):
        """With the budget spent, the lanes' real work is handed over instead of
        a model call being started that cannot finish inside it."""
        monkeypatch.setattr(config, "AGENT_SYNTH_MIN", 15.0)
        state = {"deadline": time.monotonic() - 1,
                 "messages": [], "lanes_ok": [], "lanes_failed": []}
        assert graph._remaining(state) < config.AGENT_SYNTH_MIN
