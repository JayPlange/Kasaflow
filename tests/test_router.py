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

    # Assert: order_draft is None here -- once a full proposal exists,
    # that's the active state, not the draft that led to it (see
    # router.py's route_customer())
    understand.assert_called_once_with(
        "yh",
        pending_order={"product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0},
        order_draft=None,
        pending_intent=None,
        last_action_outcome=None,
        last_priced_product=None,
    )


def test_route_customer_passes_none_when_nothing_pending(monkeypatch):
    understand = MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}})
    monkeypatch.setattr(router, "understand_customer", understand)
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"product": "ring", "material": "gold", "price": 1200}))

    # Act: no pending order was ever proposed for this fresh session
    router.route_customer("how much is a gold ring?", "session-fresh")

    # Assert
    understand.assert_called_once_with(
        "how much is a gold ring?",
        pending_order=None,
        order_draft=None,
        pending_intent=None,
        last_action_outcome=None,
        last_priced_product=None,
    )


def test_route_customer_passes_order_draft_state_when_an_order_is_in_progress(monkeypatch):
    # Arrange: propose_order already asked "how many?" earlier in this
    # session (product_name/material got remembered, quantity didn't
    # exist yet) -- a bare "2" now must be recognised as continuing it.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "propose_order", "arguments": {}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"error": "What address should this be delivered to?"}))
    monkeypatch.setattr(router, "get_pending_order_summary", MagicMock(return_value=None))
    monkeypatch.setattr(
        router,
        "get_order_draft",
        MagicMock(return_value={
            "product_name": "Ring", "material": "18k", "quantity": None,
            "delivery_address": None, "delivery_option": None,
        }),
    )

    # Act
    router.route_customer("2", "session-mid-order")

    # Assert
    from services import router as router_module
    call_kwargs = router_module.understand_customer.call_args.kwargs
    assert call_kwargs["order_draft"]["product_name"] == "Ring"


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


def test_route_customer_injects_session_id_for_cancel_order(monkeypatch):
    # Arrange: cancel_order needs to know which session's last confirmed
    # order to fall back to when the customer doesn't state a number --
    # see router.py's _SESSION_AWARE_TOOLS and order_tool.cancel_order()
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "cancel_order", "arguments": {"order_id": "unknown"}}),
    )
    captured_calls = []

    def fake_execute_tool(tool_name, **kwargs):
        captured_calls.append(kwargs)
        return {"order_cancellation": {"order_id": 6846}}

    monkeypatch.setattr(router, "execute_tool", fake_execute_tool)

    # Act
    router.route_customer("cancel my order", "session-cancel-1")

    # Assert
    assert captured_calls[0]["session_id"] == "session-cancel-1"


def test_route_customer_keeps_cancel_order_session_state_isolated(monkeypatch):
    # Arrange: two different customers cancelling -- session_id must be
    # each caller's own, never leak across sessions
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "cancel_order", "arguments": {"order_id": "6846"}}),
    )
    captured_calls = []

    def fake_execute_tool(tool_name, **kwargs):
        captured_calls.append(kwargs)
        return {"order_cancellation": {"order_id": 6846}}

    monkeypatch.setattr(router, "execute_tool", fake_execute_tool)

    # Act
    router.route_customer("cancel order 6846", "session-A")
    router.route_customer("cancel order 6846", "session-B")

    # Assert: each call carried its own caller's session_id, not a shared/stale one
    assert captured_calls[0]["session_id"] == "session-A"
    assert captured_calls[1]["session_id"] == "session-B"


# ---------------------------------------------------------------------
# converse -- purely conversational messages that need no business tool
# (see llm.py's tool 9 description and router.py's _CONVERSATION_TOOL)
# ---------------------------------------------------------------------

def test_route_customer_returns_converse_reply_directly(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "converse", "arguments": {"reply": "Hey! 👋 How can I help you today?"}}),
    )
    execute_tool = MagicMock()
    monkeypatch.setattr(router, "execute_tool", execute_tool)

    # Act
    result = router.route_customer("hey", "session-1")

    # Assert: no detour through the tool registry at all -- there's no
    # business logic behind converse to execute
    assert result == {"conversation_reply": "Hey! 👋 How can I help you today?"}
    execute_tool.assert_not_called()


def test_route_customer_converse_does_not_touch_session_memory(monkeypatch):
    # Arrange: a converse reply must never be mistaken for a business
    # argument (product, material, delivery address, ...) worth
    # remembering, and must never read/overwrite anything already
    # remembered for an order in progress
    from services.memory import get_session_store
    get_session_store().set("session-mid-chat", "product_name", "Ring")

    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "converse", "arguments": {"reply": "Haha, fair enough!"}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock())

    # Act
    router.route_customer("lol okay", "session-mid-chat")

    # Assert: the remembered product from earlier in the order survives untouched
    assert get_session_store().get("session-mid-chat", "product_name") == "Ring"


def test_route_customer_converse_falls_back_when_llm_omits_a_reply(monkeypatch):
    # Arrange: defensive only -- the model is instructed to always supply
    # a reply for converse (see llm.py), this covers it not doing so
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "converse", "arguments": {}}),
    )

    # Act
    result = router.route_customer("hey", "session-1")

    # Assert: still a real, sendable reply, not a blank message
    assert result["conversation_reply"]


# ---------------------------------------------------------------------
# pending_intent -- a product lookup asked for without a product named
# yet (see memory.set_pending_intent()/get_pending_intent() and llm.py's
# _pending_intent_state_line())
# ---------------------------------------------------------------------

def test_route_customer_sets_pending_intent_when_product_lookup_has_no_product_named(monkeypatch):
    # Arrange: "yeah i wanna see pictures" -- get_product_price runs with
    # product_name genuinely unresolved (fresh session, nothing to fill
    # it from) and finds nothing
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "unknown", "material": "unknown"}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value=None))
    set_pending_intent = MagicMock()
    monkeypatch.setattr(router, "set_pending_intent", set_pending_intent)

    # Act
    router.route_customer("yeah i wanna see pictures", "session-pending-intent-set")

    # Assert
    set_pending_intent.assert_called_once_with("session-pending-intent-set", "get_product_price")


def test_route_customer_does_not_set_pending_intent_for_a_real_product_name_that_just_was_not_found(monkeypatch):
    # Arrange: a genuinely made-up/unstocked product ("unicorn pendant")
    # is a different failure mode from "no product named" -- this must
    # not be remembered as something to resolve on the next message
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "unicorn pendant", "material": "unknown"}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value=None))
    set_pending_intent = MagicMock()
    monkeypatch.setattr(router, "set_pending_intent", set_pending_intent)

    # Act
    router.route_customer("how much is the unicorn pendant?", "session-pending-intent-unicorn")

    # Assert
    set_pending_intent.assert_not_called()


def test_route_customer_clears_pending_intent_once_the_product_lookup_succeeds(monkeypatch):
    # Arrange: the customer follows up naming the product, and it resolves
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "Set Multi Stone Golf Ring, 7g", "material": "unknown"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"product": "Set Multi Stone Golf Ring, 7g", "material": "18k", "price": 12033.0}),
    )
    set_pending_intent = MagicMock()
    monkeypatch.setattr(router, "set_pending_intent", set_pending_intent)

    # Act
    router.route_customer("Set Multi Stone Golf Ring", "session-pending-intent-resolved")

    # Assert: cleared, not left stale for an unrelated later message
    set_pending_intent.assert_called_once_with("session-pending-intent-resolved", None)


def test_route_customer_passes_pending_intent_to_understand_customer(monkeypatch):
    # Arrange
    understand = MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "Ring", "material": "18k"}})
    monkeypatch.setattr(router, "understand_customer", understand)
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"product": "Ring", "material": "18k", "price": 1200}))
    monkeypatch.setattr(router, "get_pending_intent", MagicMock(return_value="get_product_price"))

    # Act
    router.route_customer("this Set Multi Stone Golf Ring, 7g", "session-pending-intent-passthrough")

    # Assert
    call_kwargs = understand.call_args.kwargs
    assert call_kwargs["pending_intent"] == "get_product_price"


def test_route_customer_converse_does_not_read_or_write_pending_intent(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "converse", "arguments": {"reply": "Hey!"}}),
    )
    set_pending_intent = MagicMock()
    monkeypatch.setattr(router, "set_pending_intent", set_pending_intent)
    execute_tool = MagicMock()
    monkeypatch.setattr(router, "execute_tool", execute_tool)

    # Act
    router.route_customer("hey", "session-pending-intent-converse")

    # Assert: converse never touches pending_intent either way
    set_pending_intent.assert_not_called()
    execute_tool.assert_not_called()


# ---------------------------------------------------------------------
# last_action_outcome -- a fully-specified action that still hit a
# genuine, unrecoverable failure (see memory.set_last_action_outcome()
# and llm.py's _last_action_outcome_state_line())
# ---------------------------------------------------------------------

def test_route_customer_passes_last_action_outcome_to_understand_customer(monkeypatch):
    # Arrange
    understand = MagicMock(return_value={"tool": "converse", "arguments": {"reply": "..."}})
    monkeypatch.setattr(router, "understand_customer", understand)
    monkeypatch.setattr(router, "get_last_action_outcome", MagicMock(return_value={"action": "propose_order", "customer_safe_explanation": "x"}))

    # Act
    router.route_customer("why?", "session-1")

    # Assert
    call_kwargs = understand.call_args.kwargs
    assert call_kwargs["last_action_outcome"] == {"action": "propose_order", "customer_safe_explanation": "x"}


def test_route_customer_clears_last_action_outcome_on_a_genuine_success(monkeypatch):
    # Arrange: the customer moves on and successfully looks something up
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "Ring", "material": "18k"}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"product": "Ring", "material": "18k", "price": 1200}))
    set_last_action_outcome = MagicMock()
    monkeypatch.setattr(router, "set_last_action_outcome", set_last_action_outcome)

    # Act
    router.route_customer("how much is the ring?", "session-outcome-clear")

    # Assert
    set_last_action_outcome.assert_called_once_with("session-outcome-clear", None)


def test_route_customer_does_not_clear_last_action_outcome_on_a_business_error(monkeypatch):
    # Arrange: propose_order's own hard-failure return ({"error": ...})
    # must not immediately wipe out the outcome it (order_tool.py) just
    # recorded for this exact call -- see router.py's _tool_succeeded(),
    # which is deliberately stricter than _found_nothing() for this reason
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={
            "tool": "propose_order",
            "arguments": {"product_name": "Ring", "material": "18k", "quantity": 1, "delivery_address": "Accra", "delivery_option": "accra_rider"},
        }),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"error": "Sorry, I can't take orders for that item right now."}))
    set_last_action_outcome = MagicMock()
    monkeypatch.setattr(router, "set_last_action_outcome", set_last_action_outcome)

    # Act
    router.route_customer("place the order", "session-outcome-no-clear")

    # Assert: router itself never called it -- order_tool.py is the only
    # thing that sets a real outcome, on a genuine hard failure
    set_last_action_outcome.assert_not_called()


def test_route_customer_converse_does_not_touch_last_action_outcome(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "converse", "arguments": {"reply": "I'm sorry, here's why..."}}),
    )
    set_last_action_outcome = MagicMock()
    monkeypatch.setattr(router, "set_last_action_outcome", set_last_action_outcome)

    # Act
    router.route_customer("why?", "session-outcome-converse")

    # Assert: converse must never clear an outcome mid-explanation
    set_last_action_outcome.assert_not_called()


# ---------------------------------------------------------------------
# last_priced_product -- the specific product a get_product_price/
# generate_quote call most recently resolved to (see
# memory.set_last_priced_product()/get_last_priced_product() and llm.py's
# _last_priced_product_state_line())
# ---------------------------------------------------------------------

def test_route_customer_passes_last_priced_product_to_understand_customer(monkeypatch):
    # Arrange
    understand = MagicMock(return_value={"tool": "converse", "arguments": {"reply": "..."}})
    monkeypatch.setattr(router, "understand_customer", understand)
    monkeypatch.setattr(router, "get_last_priced_product", MagicMock(return_value="Big White Crown Stone Gold Ring, 14g"))

    # Act
    router.route_customer("what about in 18k", "session-1")

    # Assert
    call_kwargs = understand.call_args.kwargs
    assert call_kwargs["last_priced_product"] == "Big White Crown Stone Gold Ring, 14g"


def test_route_customer_sets_last_priced_product_on_successful_price_lookup(monkeypatch):
    # Arrange: get_product_price resolved a real item
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "Ring", "material": "18k"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"product": "Big White Crown Stone Gold Ring, 14g", "material": "18k", "price": 1200}),
    )
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)

    # Act
    router.route_customer("how much is the ring", "session-price-set")

    # Assert
    set_last_priced_product.assert_called_once_with("session-price-set", "Big White Crown Stone Gold Ring, 14g")


def test_route_customer_sets_last_priced_product_on_successful_quote(monkeypatch):
    # Arrange: generate_quote is the other tool that resolves a specific product
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "generate_quote", "arguments": {"product_name": "Ring", "material": "18k"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"product": "Ring", "material": "18k", "price": 1200, "delivery_options": []}),
    )
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)

    # Act
    router.route_customer("get me a quote for the ring", "session-quote-set")

    # Assert
    set_last_priced_product.assert_called_once_with("session-quote-set", "Ring")


def test_route_customer_does_not_set_last_priced_product_when_lookup_found_nothing(monkeypatch):
    # Arrange: a genuine miss must not overwrite whatever was priced before
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "unicorn pendant", "material": "unknown"}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value=None))
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)

    # Act
    router.route_customer("how much is the unicorn pendant?", "session-price-miss")

    # Assert
    set_last_priced_product.assert_not_called()


def test_route_customer_clears_last_priced_product_on_a_successful_category_browse(monkeypatch):
    # Arrange: recommend_products succeeding means the topic has genuinely
    # moved on from a single priced item to browsing a category
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "recommend_products", "arguments": {"material": "unknown", "category": "Rings"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"recommendations": [{"product": "Ring A"}], "requested_category": "Rings"}),
    )
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)

    # Act
    router.route_customer("what rings do you have?", "session-browse-clear")

    # Assert
    set_last_priced_product.assert_called_once_with("session-browse-clear", None)


def test_route_customer_does_not_clear_last_priced_product_on_an_empty_category_browse(monkeypatch):
    # Arrange: an empty recommend_products result is _found_nothing(), not
    # a real success -- must not clear a genuinely still-active priced item
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "recommend_products", "arguments": {"material": "unknown", "category": "Bracelets"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"recommendations": [], "requested_category": "Bracelets"}),
    )
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)

    # Act
    router.route_customer("what bracelets do you have?", "session-browse-empty")

    # Assert
    set_last_priced_product.assert_not_called()


def test_route_customer_converse_does_not_touch_last_priced_product(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "converse", "arguments": {"reply": "Hey!"}}),
    )
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)
    execute_tool = MagicMock()
    monkeypatch.setattr(router, "execute_tool", execute_tool)

    # Act
    router.route_customer("hey", "session-converse-priced")

    # Assert
    set_last_priced_product.assert_not_called()
    execute_tool.assert_not_called()


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


# ---------------------------------------------------------------------
# _describe_order_corrections / correction_note wiring -- confirmed
# live, 2026-08-19 (Webb): "wait i want to order the 14k rather" after
# material was already "12k" changed the session's remembered material
# correctly, but the very next reply just asked for the next missing
# field (address) with no acknowledgement anything had changed.
# ---------------------------------------------------------------------

def test_describe_order_corrections_returns_none_with_no_prior_draft():
    # A fresh order -- nothing to compare against, so nothing to
    # acknowledge as "changed".
    assert router._describe_order_corrections(None, {"material": "14k"}) is None


def test_describe_order_corrections_returns_none_when_nothing_actually_changed():
    old_draft = {"product_name": "Ring", "material": "18k", "quantity": 2, "delivery_address": None, "delivery_option": None}
    arguments = {"product_name": "Ring", "material": "18k", "quantity": 2}
    assert router._describe_order_corrections(old_draft, arguments) is None


def test_describe_order_corrections_describes_a_single_field_change():
    old_draft = {"product_name": "Ring", "material": "12k", "quantity": 7, "delivery_address": None, "delivery_option": None}
    arguments = {"product_name": "Ring", "material": "14k", "quantity": 7}

    note = router._describe_order_corrections(old_draft, arguments)

    assert note == "Got it, I've updated the karat to 14k."


def test_describe_order_corrections_describes_multiple_field_changes_in_one_sentence():
    # "Actually 14k, make it 6 instead" -- both changed in one message,
    # acknowledged together, not one at a time.
    old_draft = {"product_name": "Ring", "material": "12k", "quantity": 7, "delivery_address": None, "delivery_option": None}
    arguments = {"product_name": "Ring", "material": "14k", "quantity": 6}

    note = router._describe_order_corrections(old_draft, arguments)

    assert note == "Got it, I've updated the karat to 14k and the quantity to 6."


def test_describe_order_corrections_ignores_a_field_still_unknown():
    # material was never known before, and this call still doesn't know
    # it either -- not a correction, nothing to say.
    old_draft = {"product_name": "Ring", "material": None, "quantity": 7, "delivery_address": None, "delivery_option": None}
    arguments = {"product_name": "Ring", "material": "unknown", "quantity": 7}

    assert router._describe_order_corrections(old_draft, arguments) is None


def test_describe_order_corrections_ignores_a_field_answered_for_the_first_time():
    # delivery_address was never known before -- this is propose_order's
    # normal "answering the next missing question" flow, not a
    # correction, even though the value technically differs from None.
    old_draft = {"product_name": "Ring", "material": "18k", "quantity": 7, "delivery_address": None, "delivery_option": None}
    arguments = {"product_name": "Ring", "material": "18k", "quantity": 7, "delivery_address": "East Legon"}

    assert router._describe_order_corrections(old_draft, arguments) is None


def test_route_customer_attaches_correction_note_to_the_result(monkeypatch):
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={
            "tool": "propose_order",
            "arguments": {"product_name": "Ring", "material": "14k", "quantity": 7},
        }),
    )
    monkeypatch.setattr(router, "get_pending_order_summary", MagicMock(return_value=None))
    monkeypatch.setattr(
        router,
        "get_order_draft",
        MagicMock(return_value={
            "product_name": "Ring", "material": "12k", "quantity": 7,
            "delivery_address": None, "delivery_option": None,
        }),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"error": "What address should this be delivered to?"}))

    result = router.route_customer("wait i want to order the 14k rather", "session-correction")

    assert result["correction_note"] == "Got it, I've updated the karat to 14k."
    assert result["error"] == "What address should this be delivered to?"


def test_route_customer_omits_correction_note_when_nothing_changed(monkeypatch):
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={
            "tool": "propose_order",
            "arguments": {"product_name": "Ring", "material": "18k", "quantity": 7, "delivery_address": "East Legon"},
        }),
    )
    monkeypatch.setattr(router, "get_pending_order_summary", MagicMock(return_value=None))
    monkeypatch.setattr(
        router,
        "get_order_draft",
        MagicMock(return_value={
            "product_name": "Ring", "material": "18k", "quantity": 7,
            "delivery_address": None, "delivery_option": None,
        }),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"proposal": {}}))

    result = router.route_customer("east legon", "session-no-correction")

    assert "correction_note" not in result
