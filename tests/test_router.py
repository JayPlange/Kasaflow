"""
Tests for services/router.py

This sits a level above the pure unit tests: it checks that the pieces
snap together correctly (LLM decision -> tool execution -> result), by
mocking out understand_customer and execute_tool rather than the raw
OpenAI client. We're testing the WIRING here, not the individual parts
-- those already have their own tests.
"""

from unittest.mock import MagicMock

import pytest

from services import router
from services.llm import ToolSelectionError
from services.tool_executor import ToolExecutionError


def test_route_customer_happy_path(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"product": "ring", "material": "gold", "price": 1200}),
    )

    # Act
    result = router.route_customer("how much is a gold ring?", "session-1")

    # Assert
    assert result == {"product": "ring", "material": "gold", "price": 1200}


def test_route_customer_returns_friendly_error_when_llm_fails(monkeypatch):
    # Arrange: simulate the LLM returning unparseable output
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(side_effect=ToolSelectionError("LLM did not return valid JSON")),
    )

    # Act
    result = router.route_customer("asdkjaslkdj", "session-1")

    # Assert: customer sees a friendly message, not a stack trace
    assert "error" in result
    assert "couldn't understand" in result["error"].lower()


def test_route_customer_returns_friendly_error_when_tool_fails(monkeypatch):
    # Arrange: LLM picks a valid-looking tool, but execution blows up
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(side_effect=ToolExecutionError("Invalid arguments for tool 'get_product_price'")),
    )

    # Act
    result = router.route_customer("how much is that ring?", "session-1")

    # Assert
    assert "error" in result
    assert "something went wrong" in result["error"].lower()


def test_route_customer_rejects_empty_message():
    # Arrange: no mocking needed -- this is rejected before the LLM is ever called

    # Act
    result = router.route_customer("", "session-1")

    # Assert
    assert "error" in result


def test_route_customer_resolves_unknown_material_from_session(monkeypatch):
    # Arrange: turn one establishes "gold" as the material discussed in
    # this session, turn two asks about "that" without naming it again
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "unknown"}}),
    )
    captured_calls = []

    def fake_execute_tool(tool_name, **kwargs):
        captured_calls.append(kwargs)
        return {"product": "ring", "material": kwargs.get("material"), "price": 1200}

    monkeypatch.setattr(router, "execute_tool", fake_execute_tool)

    from services.memory import get_session_store
    get_session_store().set("session-remember", "material", "gold")

    # Act: the model didn't know the material, but this session does
    result = router.route_customer("how much is that ring again?", "session-remember")

    # Assert: the session's remembered material was filled in before the tool ran
    assert captured_calls[0]["material"] == "gold"
    assert result["material"] == "gold"


def test_route_customer_keeps_sessions_isolated(monkeypatch):
    # Arrange: two different sessions asking about "unknown" material
    # must never see each other's remembered context
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "unknown"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        lambda tool_name, **kwargs: {"product": "ring", "material": kwargs.get("material")},
    )

    from services.memory import get_session_store
    get_session_store().set("session-a", "material", "gold")
    get_session_store().set("session-b", "material", "silver")

    # Act
    result_a = router.route_customer("how much is that ring?", "session-a")
    result_b = router.route_customer("how much is that ring?", "session-b")

    # Assert: each session only ever sees its own remembered material
    assert result_a["material"] == "gold"
    assert result_b["material"] == "silver"
