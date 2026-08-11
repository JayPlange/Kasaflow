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


def test_route_customer_handles_multiple_requests_in_one_message(monkeypatch):
    # Arrange: "how much is a gold ring and a silver chain" -- understand_customer
    # returns the additive "requests" (plural) shape instead of one "tool"/"arguments" pair
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={
            "requests": [
                {"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}},
                {"tool": "get_product_price", "arguments": {"product_name": "chain", "material": "silver"}},
            ]
        }),
    )

    def fake_execute_tool(tool_name, **kwargs):
        return {"product": kwargs["product_name"], "material": kwargs["material"], "price": 1200}

    monkeypatch.setattr(router, "execute_tool", fake_execute_tool)

    # Act
    result = router.route_customer("how much is a gold ring and a silver chain", "session-multi")

    # Assert: nothing got silently dropped -- both asks come back, in order
    assert "results" in result
    assert len(result["results"]) == 2
    assert result["results"][0] == {"product": "ring", "material": "gold", "price": 1200}
    assert result["results"][1] == {"product": "chain", "material": "silver", "price": 1200}


def test_route_customer_multi_request_one_failure_does_not_drop_the_others(monkeypatch):
    # Arrange: second sub-request's tool execution blows up
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={
            "requests": [
                {"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}},
                {"tool": "get_product_price", "arguments": {}},
            ]
        }),
    )

    def fake_execute_tool(tool_name, **kwargs):
        if not kwargs:
            raise ToolExecutionError("Invalid arguments for tool 'get_product_price'")
        return {"product": kwargs["product_name"], "material": kwargs["material"], "price": 1200}

    monkeypatch.setattr(router, "execute_tool", fake_execute_tool)

    # Act
    result = router.route_customer("how much is a gold ring and (something malformed)", "session-multi-fail")

    # Assert: the first ask still comes back correctly, the second surfaces
    # its own error instead of taking down the whole reply
    assert result["results"][0]["product"] == "ring"
    assert "error" in result["results"][1]


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


def test_route_customer_does_not_remember_a_category_that_found_nothing(monkeypatch):
    # Arrange: "what bracelets do you have" -- a real category word, but
    # this store doesn't stock any, so recommend_products legitimately
    # returns zero results (not an error, execute_tool doesn't raise).
    # That empty category must not get remembered, or every vague
    # follow-up in the session ("yeah lemme see", "show me something
    # else") would silently inherit the dead category and stay stuck.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "recommend_products", "arguments": {"material": "unknown", "category": "Bracelets"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"recommendations": [], "requested_category": "Bracelets", "available_categories": ["Necklaces", "Rings"]}),
    )

    # Act
    router.route_customer("what bracelets do you have?", "session-empty-category")

    # Assert: nothing was remembered from a call that found nothing
    from services.memory import get_session_store
    assert get_session_store().get("session-empty-category", "category") is None


def test_route_customer_does_not_remember_a_product_price_lookup_that_found_nothing(monkeypatch):
    # Same failure mode as above, via get_product_price's bare None return.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "unicorn pendant", "material": "unknown"}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value=None))

    # Act
    router.route_customer("how much is the unicorn pendant?", "session-empty-product")

    # Assert
    from services.memory import get_session_store
    assert get_session_store().get("session-empty-product", "product_name") is None


def test_route_customer_passes_pending_order_state_to_understand_customer(monkeypatch):
    # Arrange: a proposal already exists for this session (propose_order
    # really ran earlier) -- understand_customer must be told about it so
    # a follow-up "yh" can be correctly resolved to confirm_order instead
    # of guessing blind (see llm.py's _pending_order_state_line()).
    from services import order_tool

    understand = MagicMock(return_value={"tool": "confirm_order", "arguments": {}})
    monkeypatch.setattr(router, "understand_customer", understand)
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"order_confirmation": {}}))
    monkeypatch.setattr(
        router,
        "get_pending_order_summary",
        MagicMock(return_value={"product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0}),
    )

    # Act
    router.route_customer("yh", "session-with-pending-order")

    # Assert
    understand.assert_called_once_with("yh", pending_order={"product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0})


def test_route_customer_passes_none_when_nothing_pending(monkeypatch):
    understand = MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}})
    monkeypatch.setattr(router, "understand_customer", understand)
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"product": "ring", "material": "gold", "price": 1200}))

    # Act: no pending order was ever proposed for this fresh session
    router.route_customer("how much is a gold ring?", "session-fresh")

    # Assert
    understand.assert_called_once_with("how much is a gold ring?", pending_order=None)


def test_route_customer_injects_session_id_for_order_tools(monkeypatch):
    # Arrange: propose_order needs to know which customer's session it's
    # acting on, but that's never something the LLM returns -- see
    # router.py's _SESSION_AWARE_TOOLS
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={
            "tool": "propose_order",
            "arguments": {"product_name": "ring", "material": "gold", "quantity": 1, "delivery_address": "Accra"},
        }),
    )
    captured_calls = []

    def fake_execute_tool(tool_name, **kwargs):
        captured_calls.append(kwargs)
        return {"proposal": {}}

    monkeypatch.setattr(router, "execute_tool", fake_execute_tool)

    # Act
    router.route_customer("I'll take a gold ring, deliver to Accra", "session-order-1")

    # Assert
    assert captured_calls[0]["session_id"] == "session-order-1"


def test_route_customer_does_not_inject_session_id_for_read_only_tools(monkeypatch):
    # Arrange: guard against the injection being accidentally widened to
    # every tool, which would break the five that don't accept session_id
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}}),
    )
    captured_calls = []

    def fake_execute_tool(tool_name, **kwargs):
        captured_calls.append(kwargs)
        return {"product": "ring", "material": "gold", "price": 1200}

    monkeypatch.setattr(router, "execute_tool", fake_execute_tool)

    # Act
    router.route_customer("how much is a gold ring?", "session-1")

    # Assert
    assert "session_id" not in captured_calls[0]
