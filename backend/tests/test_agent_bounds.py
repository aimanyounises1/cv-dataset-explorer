"""The library-native bounds that keep an assistant turn finite."""
import asyncio
import json

import pytest
from fastapi import HTTPException

from app import config
from app.agent import graph
from app.api import chat as chat_api
from app.schemas import ChatMessage


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
        assert captured["reasoning"] is False


class HangingGraph:
    async def ainvoke(self, state, config=None):
        await asyncio.sleep(30)
        return state

    async def astream(self, state, config=None, stream_mode=None):
        await asyncio.sleep(30)
        yield "values", state


class TestTurnTimeout:
    """The FastAPI boundary, not timeout arithmetic in graph state, owns the
    total wall clock experienced by the caller."""

    def test_blocking_response_budget_includes_ollama_preflight(self, monkeypatch):
        monkeypatch.setattr(config, "AGENT_TURN_BUDGET", 0.02)

        async def hanging_preflight():
            await asyncio.sleep(30)

        monkeypatch.setattr(chat_api, "_check_ollama", hanging_preflight)
        with pytest.raises(HTTPException) as caught:
            asyncio.run(chat_api.chat(
                [ChatMessage(role="user", content="hello")], None
            ))
        assert caught.value.status_code == 504

    def test_streaming_response_uses_the_same_absolute_deadline(self, monkeypatch):
        monkeypatch.setattr(config, "AGENT_TURN_BUDGET", 0.02)

        async def ready():
            return None

        monkeypatch.setattr(chat_api, "_check_ollama", ready)
        monkeypatch.setattr(chat_api, "_get_graph", HangingGraph)

        async def consume():
            response = await chat_api.chat_stream(
                [ChatMessage(role="user", content="hello")], None
            )
            return [
                json.loads(chunk)
                async for chunk in response.body_iterator
            ]

        events = asyncio.run(consume())
        assert events[-1]["type"] == "error"
        assert "turn timeout" in events[-1]["detail"]

    def test_default_lane_limit_leaves_room_inside_the_turn(self):
        assert 0 < config.AGENT_LANE_TIMEOUT < config.AGENT_TURN_BUDGET


def test_ollama_preflight_requires_the_configured_model_tag(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "qwen3:30b-a3b"}]}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            return Response()

    monkeypatch.setattr(config, "CHAT_MODEL", "qwen3:8b")
    monkeypatch.setattr(chat_api.httpx, "AsyncClient", Client)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(chat_api._check_ollama())
    assert caught.value.status_code == 503
    assert "qwen3:8b" in caught.value.detail
