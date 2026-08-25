"""
Tests for services/router.py

This sits a level above the pure unit tests: it checks that the pieces
snap together correctly (LLM decision -> tool execution -> result), by
mocking out understand_customer and execute_tool rather than the raw
OpenAI client. We're testing the WIRING here, not the individual parts
-- those already have their own tests.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from services import router
from services.llm import ToolSelectionError
from services.tool_executor import ToolExecutionError


# ---------------------------------------------------------------------
# Turn concurrency: route_customer() must serialize two near-simultaneous
# messages for the SAME session, but must NOT serialize messages for
# different sessions against each other. See memory.SessionStore.
# session_lock()'s docstring for the corruption this prevents -- a
# slower request's write landing after a faster, later request's write,
# silently overwriting the customer's actual last-stated value.
# ---------------------------------------------------------------------

def test_route_customer_serializes_concurrent_calls_for_the_same_session(monkeypatch):
    concurrency = {"current": 0, "max": 0}
    guard = threading.Lock()

    def slow_understand(message, **kwargs):
        with guard:
            concurrency["current"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["current"])
        time.sleep(0.05)
        with guard:
            concurrency["current"] -= 1
        return {"tool": "converse", "arguments": {"reply": "ok"}}

    monkeypatch.setattr(router, "understand_customer", slow_understand)

    threads = [
        threading.Thread(target=router.route_customer, args=(f"message {i}", "session-race"))
        for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # If the lock only covered individual memory.py calls (the pre-fix
    # behaviour), all 5 understand_customer calls could overlap during
    # their sleep -- max would be > 1.
    assert concurrency["max"] == 1


def test_route_customer_does_not_serialize_different_sessions(monkeypatch):
    active = {"count": 0}
    overlapped = {"seen": False}
    guard = threading.Lock()

    def slow_understand(message, **kwargs):
        with guard:
            active["count"] += 1
            if active["count"] > 1:
                overlapped["seen"] = True
        time.sleep(0.05)
        with guard:
            active["count"] -= 1
        return {"tool": "converse", "arguments": {"reply": "ok"}}

    monkeypatch.setattr(router, "understand_customer", slow_understand)

    t1 = threading.Thread(target=router.route_customer, args=("hi", "session-a"))
    t2 = threading.Thread(target=router.route_customer, args=("hi", "session-b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Two different customers must not block on each other -- a global
    # lock would be a correctness fix that also tanks throughput.
    assert overlapped["seen"] is True


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
    # router.py's route_customer()). awaiting_confirmation defaults to
    # False for a session that never had propose_order's own success
    # branch set it True (get_pending_order_summary is mocked here, but
    # is_awaiting_confirmation() isn't -- it reads the real, untouched
    # memory store for this brand-new session_id).
    understand.assert_called_once_with(
        "yh",
        pending_order={"product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0},
        awaiting_confirmation=False,
        order_draft=None,
        pending_intent=None,
        last_action_outcome=None,
        last_priced_product=None,
        just_confirmed_order=None,
        last_presented_products=None,
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
        awaiting_confirmation=False,
        just_confirmed_order=None,
        order_draft=None,
        pending_intent=None,
        last_action_outcome=None,
        last_priced_product=None,
        last_presented_products=None,
    )


def test_route_customer_marks_awaiting_confirmation_after_a_successful_proposal(monkeypatch):
    # P0 fix: propose_order succeeding is the ONLY thing that should let
    # a following bare "yes" be read as confirming this exact order.
    from services.memory import is_awaiting_confirmation

    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={
            "tool": "propose_order",
            "arguments": {
                "product_name": "Ring", "material": "18k", "quantity": 1,
                "delivery_address": "Accra", "delivery_option": "accra_rider",
            },
        }),
    )
    monkeypatch.setattr(
        router, "execute_tool",
        MagicMock(return_value={"proposal": {"product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0}}),
    )

    router.route_customer("I want a ring in 18k, deliver to Accra, rider delivery", "session-just-proposed")

    assert is_awaiting_confirmation("session-just-proposed") is True


def test_route_customer_clears_awaiting_confirmation_after_any_other_tool(monkeypatch):
    # P0 fix: the exact scenario the audit flagged -- an order is
    # proposed and left unconfirmed, then the assistant asks or answers
    # something completely unrelated. A bare "yes" after THAT must not
    # be readable as confirming the stale order, so the flag must clear.
    from services.memory import get_session_store, is_awaiting_confirmation

    get_session_store().set("session-then-browsed", "awaiting_confirmation", True)
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "recommend_products", "arguments": {"category": "Rings"}}),
    )
    monkeypatch.setattr(
        router, "execute_tool",
        MagicMock(return_value={"recommendations": [{"product": "Other Ring", "material": "14k", "price": 900}]}),
    )

    router.route_customer("what other rings do you have", "session-then-browsed")

    assert is_awaiting_confirmation("session-then-browsed") is False


def test_route_customer_clears_awaiting_confirmation_for_a_converse_reply(monkeypatch):
    # Same as above, specifically for converse -- the exact shape of
    # Webb's own example ("want to see a few cheaper options?" -> "yeah").
    from services.memory import get_session_store, is_awaiting_confirmation

    get_session_store().set("session-then-chatted", "awaiting_confirmation", True)
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={
            "tool": "converse",
            "arguments": {"reply": "Want me to show you a few cheaper options?"},
        }),
    )

    router.route_customer("ei that's expensive oo", "session-then-chatted")

    assert is_awaiting_confirmation("session-then-chatted") is False


def test_route_customer_marks_just_confirmed_order_after_a_successful_confirmation(monkeypatch):
    from services.memory import get_just_confirmed_order

    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "confirm_order", "arguments": {}}),
    )
    monkeypatch.setattr(
        router, "execute_tool",
        MagicMock(return_value={"order_confirmation": {"order_id": 777, "total": 2400.0}}),
    )

    router.route_customer("yes", "session-confirming")

    assert get_just_confirmed_order("session-confirming") == {"order_id": 777, "total": 2400.0}


def test_route_customer_clears_just_confirmed_order_on_the_next_unrelated_turn(monkeypatch):
    # The signal must not linger past the one turn it's for -- otherwise
    # a customer's later, unrelated message would keep getting told
    # about an order that was placed several turns ago.
    from services.memory import get_just_confirmed_order, get_session_store

    get_session_store().set("session-moved-on", "just_confirmed_order", {"order_id": 777, "total": 2400.0})
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "chain", "material": "gold"}}),
    )
    monkeypatch.setattr(
        router, "execute_tool",
        MagicMock(return_value={"product": "chain", "material": "gold", "price": 900}),
    )

    router.route_customer("how much is a gold chain", "session-moved-on")

    assert get_just_confirmed_order("session-moved-on") is None


def test_route_customer_reasserts_awaiting_confirmation_after_a_correction_produces_a_new_proposal(monkeypatch):
    # Webb, 2026-08-20: awaiting_confirmation must track "the exact
    # current draft that was most recently proposed", not just "some
    # order exists". A correction that itself produces a full new
    # proposal must flip the flag back to True for THAT proposal, so a
    # following "yes" confirms the corrected order, not nothing at all.
    from services.memory import is_awaiting_confirmation

    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(side_effect=[
            {
                "tool": "propose_order",
                "arguments": {
                    "product_name": "Ring", "material": "14k", "quantity": 2,
                    "delivery_address": "Accra", "delivery_option": "accra_rider",
                },
            },
            {
                "tool": "propose_order",
                "arguments": {
                    "product_name": "unknown", "material": "18k", "quantity": "unknown",
                    "delivery_address": "unknown", "delivery_option": "unknown",
                },
            },
        ]),
    )
    monkeypatch.setattr(
        router, "execute_tool",
        MagicMock(side_effect=[
            {"proposal": {"product": "Ring", "material": "14k", "quantity": 2, "total": 2400.0}},
            {"proposal": {"product": "Ring", "material": "18k", "quantity": 2, "total": 2600.0}},
        ]),
    )

    router.route_customer("2 rings in 14k, deliver to Accra, rider delivery", "session-correction-then-confirm")
    assert is_awaiting_confirmation("session-correction-then-confirm") is True

    router.route_customer("actually make it 18k", "session-correction-then-confirm")
    assert is_awaiting_confirmation("session-correction-then-confirm") is True


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


def test_route_customer_still_passes_order_draft_when_a_stale_pending_order_is_for_a_different_product(monkeypatch):
    # Webb, 2026-08-20, live: a bare "14k" answering propose_order's own
    # "What karat would you like?" misrouted to recommend_products three
    # separate times -- every time with an OLD, unconfirmed proposal for
    # a different product still sitting there. order_draft was previously
    # suppressed unconditionally whenever ANY pending_order existed, so
    # the new order's own in-progress draft (product known, karat
    # missing) never reached the prompt at all, leaving the model with no
    # order_draft-based disambiguation rule for the bare reply. Fixed via
    # _order_draft_matches_pending_order(): only suppress order_draft
    # when it genuinely describes the SAME order as pending_order.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "propose_order", "arguments": {}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"error": "What karat would you like that in?"}))
    monkeypatch.setattr(
        router,
        "get_pending_order_summary",
        MagicMock(return_value={"product": "Gye Nyame White Necklace with Earrings, 30g", "material": "14k", "quantity": 6, "total": 270000.0}),
    )
    monkeypatch.setattr(
        router,
        "get_order_draft",
        MagicMock(return_value={
            "product_name": "Big White Crown Stone Gold Ring, 14g", "material": None,
            "quantity": None, "delivery_address": "Accra", "delivery_option": "accra_rider",
        }),
    )

    # Act
    router.route_customer("14k", "session-stale-pending-different-product")

    # Assert: order_draft for the RING must still reach understand_customer,
    # not be suppressed just because a stale necklace proposal exists.
    call_kwargs = router.understand_customer.call_args.kwargs
    assert call_kwargs["order_draft"] is not None
    assert call_kwargs["order_draft"]["product_name"] == "Big White Crown Stone Gold Ring, 14g"


def test_route_customer_suppresses_order_draft_when_it_matches_the_pending_order(monkeypatch):
    # The guard above must not become overzealous -- when order_draft
    # genuinely IS the same order as pending_order (the common case,
    # e.g. right after a full proposal), it must still be suppressed as
    # before: pending_order's own state line already covers it, and
    # showing both would be redundant.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "confirm_order", "arguments": {}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"order_confirmation": {}}))
    monkeypatch.setattr(
        router,
        "get_pending_order_summary",
        MagicMock(return_value={"product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0}),
    )
    monkeypatch.setattr(
        router,
        "get_order_draft",
        MagicMock(return_value={
            "product_name": "Ring", "material": "18k", "quantity": 1,
            "delivery_address": "Accra", "delivery_option": "accra_rider",
        }),
    )

    # Act
    router.route_customer("yes", "session-matching-draft-and-pending")

    # Assert
    call_kwargs = router.understand_customer.call_args.kwargs
    assert call_kwargs["order_draft"] is None


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


def test_route_customer_injects_session_id_for_get_order_status(monkeypatch):
    # Arrange: get_order_status needs to know which session's last
    # confirmed order to fall back to when the customer doesn't state a
    # number -- see router.py's _SESSION_AWARE_TOOLS and
    # order_tool.get_order_status(), identical reasoning to cancel_order
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_order_status", "arguments": {"order_id": "unknown"}}),
    )
    captured_calls = []

    def fake_execute_tool(tool_name, **kwargs):
        captured_calls.append(kwargs)
        return {"order_status": {"order_id": 6846, "status": "on-hold", "status_label": "received, awaiting payment confirmation", "item_summary": None, "total": None}}

    monkeypatch.setattr(router, "execute_tool", fake_execute_tool)

    # Act
    router.route_customer("where is my order", "session-status-1")

    # Assert
    assert captured_calls[0]["session_id"] == "session-status-1"


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


def test_route_customer_sets_last_priced_product_on_successful_karat_options_lookup(monkeypatch):
    # Arrange: get_product_karat_options is the third tool that resolves
    # a specific product -- a bare karat follow-up right after seeing
    # the options list ("okay what about 12") needs the same
    # continuation handling as one right after a price quote.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_karat_options", "arguments": {"product_name": "Ring"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"product": "Custom Leaf White Gold Necklace, 20g", "karat_options": [{"material": "18k", "price": 34000.0}]}),
    )
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)

    # Act
    router.route_customer("what karat does that come in", "session-karat-options-set")

    # Assert
    set_last_priced_product.assert_called_once_with("session-karat-options-set", "Custom Leaf White Gold Necklace, 20g")


def test_route_customer_does_not_set_last_priced_product_when_karat_options_found_nothing(monkeypatch):
    # Arrange: an empty karat_options list is the "found nothing" case,
    # same as an empty recommendations list -- must not overwrite
    # whatever was priced before.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_karat_options", "arguments": {"product_name": "unicorn pendant"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"product": "unicorn pendant", "karat_options": []}),
    )
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)

    # Act
    router.route_customer("what karat does the unicorn pendant come in", "session-karat-options-miss")

    # Assert
    set_last_priced_product.assert_not_called()


def test_route_customer_sets_last_priced_product_on_successful_weight_lookup(monkeypatch):
    # Arrange: get_product_weight is a fourth tool that resolves a
    # specific product -- a follow-up right after a weight answer needs
    # the same continuation handling as one right after a price quote.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_weight", "arguments": {"product_name": "Ring"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"product": "Set Multi Stone Golf Ring, 7g", "weight": "7g"}),
    )
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)

    # Act
    router.route_customer("how heavy is it", "session-weight-set")

    # Assert
    set_last_priced_product.assert_called_once_with("session-weight-set", "Set Multi Stone Golf Ring, 7g")


def test_route_customer_sets_last_priced_product_even_when_weight_lookup_finds_no_parseable_weight(monkeypatch):
    # A product with no weight in its catalogue name was still genuinely
    # identified -- unlike an empty karat_options list, this is not the
    # "found nothing" case, and last_priced_product should still be set
    # so the active-product context carries forward.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_weight", "arguments": {"product_name": "Custom Butterfly Gold Ring"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"product": "Custom Butterfly Gold Ring", "weight": None}),
    )
    set_last_priced_product = MagicMock()
    monkeypatch.setattr(router, "set_last_priced_product", set_last_priced_product)

    # Act
    router.route_customer("how heavy is the custom butterfly ring", "session-weight-no-data")

    # Assert
    set_last_priced_product.assert_called_once_with("session-weight-no-data", "Custom Butterfly Gold Ring")


# ---------------------------------------------------------------------
# last_presented_products -- a successful recommend_products call must
# remember, via the SAME selection response_formatter.py renders with,
# exactly what this customer was just shown (task #181).
# ---------------------------------------------------------------------

def test_route_customer_sets_last_presented_products_on_successful_recommend(monkeypatch):
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "recommend_products", "arguments": {"material": "unknown", "category": "Rings"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"recommendations": [{"product": "Ring A", "category": "Rings"}]}),
    )
    set_last_presented_products = MagicMock()
    monkeypatch.setattr(router, "set_last_presented_products", set_last_presented_products)

    router.route_customer("what rings do you have?", "session-presented-set")

    set_last_presented_products.assert_called_once()
    call_session_id, call_groups = set_last_presented_products.call_args.args
    assert call_session_id == "session-presented-set"
    assert call_groups == [("Ring A", [{"product": "Ring A", "category": "Rings"}])]


def test_route_customer_does_not_set_last_presented_products_on_an_empty_recommend(monkeypatch):
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
    set_last_presented_products = MagicMock()
    monkeypatch.setattr(router, "set_last_presented_products", set_last_presented_products)

    router.route_customer("what bracelets do you have?", "session-presented-miss")

    set_last_presented_products.assert_not_called()


def test_route_customer_does_not_set_last_presented_products_for_other_tools(monkeypatch):
    # A single-item lookup (price, karat options, weight, ...) must not
    # overwrite the remembered list -- only recommend_products does.
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "Ring", "material": "18k"}}),
    )
    monkeypatch.setattr(
        router,
        "execute_tool",
        MagicMock(return_value={"product": "Ring", "material": "18k", "price": 1200}),
    )
    set_last_presented_products = MagicMock()
    monkeypatch.setattr(router, "set_last_presented_products", set_last_presented_products)

    router.route_customer("how much is the ring", "session-presented-untouched")

    set_last_presented_products.assert_not_called()


def test_route_customer_reads_last_presented_products_and_threads_it_to_understand_customer(monkeypatch):
    understand_customer = MagicMock(
        return_value={"tool": "get_product_price", "arguments": {"product_name": "Ring A", "material": "unknown"}}
    )
    monkeypatch.setattr(router, "understand_customer", understand_customer)
    monkeypatch.setattr(
        router, "execute_tool",
        MagicMock(return_value={"product": "Ring A", "karat_options": [{"material": "18k", "price": 1200}]}),
    )
    stored = {"generation": 1, "items": [{"position": 1, "product_name": "Ring A", "category": "Rings"}]}
    monkeypatch.setattr(router, "get_last_presented_products", MagicMock(return_value=stored))

    router.route_customer("how much is the first one", "session-presented-read")

    assert understand_customer.call_args.kwargs["last_presented_products"] == stored


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
# _order_draft_matches_pending_order -- see route_customer()'s
# order_draft computation. Webb, 2026-08-20, live: a stale pending_order
# for one product unconditionally suppressed order_draft for a
# different, in-progress order, leaving a bare karat reply with no
# disambiguation context at all.
# ---------------------------------------------------------------------

def test_order_draft_matches_pending_order_returns_false_when_either_side_is_none():
    assert router._order_draft_matches_pending_order(None, {"product": "Ring"}) is False
    assert router._order_draft_matches_pending_order({"product_name": "Ring"}, None) is False
    assert router._order_draft_matches_pending_order(None, None) is False


def test_order_draft_matches_pending_order_true_for_the_same_product(monkeypatch):
    order_draft = {"product_name": "Ring", "material": "18k"}
    pending_order = {"product": "Ring", "material": "18k"}
    assert router._order_draft_matches_pending_order(order_draft, pending_order) is True


def test_order_draft_matches_pending_order_is_case_and_whitespace_insensitive():
    order_draft = {"product_name": "  Ring  "}
    pending_order = {"product": "RING"}
    assert router._order_draft_matches_pending_order(order_draft, pending_order) is True


def test_order_draft_matches_pending_order_false_for_a_different_product():
    # The exact case that reached the model with zero disambiguation
    # context: a stale necklace proposal still pending while the ring's
    # own draft is in progress -- these must NOT be treated as the same
    # order.
    order_draft = {"product_name": "Big White Crown Stone Gold Ring, 14g"}
    pending_order = {"product": "Gye Nyame White Necklace with Earrings, 30g"}
    assert router._order_draft_matches_pending_order(order_draft, pending_order) is False


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


def test_describe_order_corrections_returns_none_for_a_genuinely_different_product():
    # A different, explicitly-named product means this message is
    # describing something else entirely, not correcting the order on
    # file -- diffing the other fields against it would be misleading
    # (confirmed live, 2026-08-20: exactly this produced a fabricated
    # correction_note for a product the customer never mentioned).
    old_draft = {
        "product_name": "Custom Gye Nyame Gold Necklace with Earrings, 20g", "material": "14k",
        "quantity": 1, "delivery_address": "Tamale", "delivery_option": "kumasi_rider",
    }
    arguments = {
        "product_name": "Solid Cross Chains White Gold Necklace, 20g", "material": "14k",
        "quantity": 1, "delivery_address": "Accra", "delivery_option": "accra_rider",
    }

    assert router._describe_order_corrections(old_draft, arguments) is None


def test_describe_order_corrections_still_fires_for_the_same_product_while_pending():
    # The case the router-level "skip whenever pending_order exists"
    # guard used to wrongly block too: a genuine correction to the SAME
    # order, made while it's already fully proposed and awaiting
    # confirmation, must still be acknowledged.
    old_draft = {
        "product_name": "Ring", "material": "14k", "quantity": 2,
        "delivery_address": "Accra", "delivery_option": "accra_rider",
    }
    arguments = {
        "product_name": "unknown", "material": "18k", "quantity": "unknown",
        "delivery_address": "unknown", "delivery_option": "unknown",
    }

    note = router._describe_order_corrections(old_draft, arguments)

    assert note == "Got it, I've updated the karat to 18k."


def test_describe_order_corrections_treats_an_unstated_product_as_the_same_one():
    # product_name "unknown" (not restated) must not itself be read as
    # "a different product" -- only an EXPLICITLY different name should
    # suppress the correction note.
    old_draft = {"product_name": "Ring", "material": "12k", "quantity": 7, "delivery_address": None, "delivery_option": None}
    arguments = {"product_name": "unknown", "material": "14k", "quantity": 7}

    note = router._describe_order_corrections(old_draft, arguments)

    assert note == "Got it, I've updated the karat to 14k."


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


def test_route_customer_omits_correction_note_for_a_genuinely_different_product(monkeypatch):
    # Confirmed live, 2026-08-20: with an unconfirmed Tamale order still
    # pending, a customer's complete new order (different product,
    # deliver to Accra) got a correction_note referencing the OLD
    # pending order's product -- something this message never even
    # mentioned correcting. The fix lives in _describe_order_corrections()
    # itself: a different, explicitly-named product means this is a
    # fresh order, not a correction, regardless of whether a pending
    # order happens to exist (an earlier version of this fix gated the
    # whole computation on "no pending_order", which also silently broke
    # the much more common case -- correcting a field of an order that's
    # already fully proposed and pending confirmation).
    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={
            "tool": "propose_order",
            "arguments": {
                "product_name": "Solid Cross Chains White Gold Necklace, 20g", "material": "14k",
                "quantity": 1, "delivery_address": "Accra", "delivery_option": "accra_rider",
            },
        }),
    )
    monkeypatch.setattr(
        router,
        "get_pending_order_summary",
        MagicMock(return_value={
            "product": "Custom Gye Nyame Gold Necklace with Earrings, 20g",
            "material": "14k", "quantity": 1, "total": 26000.0,
        }),
    )
    monkeypatch.setattr(
        router,
        "get_order_draft",
        MagicMock(return_value={
            "product_name": "Custom Gye Nyame Gold Necklace with Earrings, 20g", "material": "14k",
            "quantity": 1, "delivery_address": "Tamale", "delivery_option": "kumasi_rider",
        }),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"proposal": {}}))

    result = router.route_customer(
        "I'd like to order 1 Solid Cross Chains White Gold Necklace, 20g in 14k, deliver to Accra, "
        "rider delivery within Accra", "session-with-pending-order",
    )

    assert "correction_note" not in result


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


# ---------------------------------------------------------------------
# Per-turn debug trace: instrumentation Webb asked for (2026-08-21) so a
# failing live turn can be diagnosed from what the LLM and the app
# actually did, rather than guessed at ("that's just the model") after
# the fact. See router._log_turn_trace()'s docstring. These tests only
# check the trace itself is complete and correct -- not that any
# particular turn behaves a certain way (already covered elsewhere).
# ---------------------------------------------------------------------

def test_route_customer_logs_a_complete_turn_trace_on_success(monkeypatch, caplog):
    import json
    import logging

    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "recommend_products", "arguments": {"category": "rings"}}),
    )
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"products": []}))

    with caplog.at_level(logging.INFO, logger="services.router"):
        result = router.route_customer("show me rings", "trace-session-success")

    traces = [r.message for r in caplog.records if r.message.startswith("KASAFLOW_TURN_TRACE")]
    assert len(traces) == 1
    payload = json.loads(traces[0][len("KASAFLOW_TURN_TRACE "):])

    assert payload["session_id"] == "trace-session-success"
    assert payload["customer_message"] == "show me rings"
    assert payload["llm_structured_output"] == {"tool": "recommend_products", "arguments": {"category": "rings"}}
    assert payload["resolved_arguments"] == {"category": "rings"}
    assert payload["tool_result"] == {"products": []}
    assert payload["final_result"] == result
    for key in ("pending_order", "order_draft", "pending_intent", "last_action_outcome",
                "last_priced_product", "awaiting_confirmation", "just_confirmed_order"):
        assert key in payload["pre_tool_state"]
        assert key in payload["post_tool_state"]


def test_route_customer_logs_a_turn_trace_when_the_tool_raises(monkeypatch, caplog):
    import json
    import logging

    monkeypatch.setattr(
        router,
        "understand_customer",
        MagicMock(return_value={"tool": "recommend_products", "arguments": {"category": "rings"}}),
    )

    def _raise(tool, **kwargs):
        raise ToolExecutionError("boom")

    monkeypatch.setattr(router, "execute_tool", _raise)

    with caplog.at_level(logging.INFO, logger="services.router"):
        result = router.route_customer("show me rings", "trace-session-error")

    assert result == {"error": "Something went wrong while processing your request."}
    traces = [r.message for r in caplog.records if r.message.startswith("KASAFLOW_TURN_TRACE")]
    assert len(traces) == 1
    payload = json.loads(traces[0][len("KASAFLOW_TURN_TRACE "):])
    assert payload["tool_result"] == {"exception": "boom"}
    assert payload["final_result"] == {"error": "Something went wrong while processing your request."}


def test_route_customer_logs_a_turn_trace_when_tool_selection_fails(monkeypatch, caplog):
    import json
    import logging

    def _raise(*args, **kwargs):
        raise ToolSelectionError("could not decide")

    monkeypatch.setattr(router, "understand_customer", _raise)

    with caplog.at_level(logging.INFO, logger="services.router"):
        result = router.route_customer("garbled input", "trace-session-tool-selection-error")

    assert result == {"error": "I couldn't understand that request. Could you rephrase it?"}
    traces = [r.message for r in caplog.records if r.message.startswith("KASAFLOW_TURN_TRACE")]
    assert len(traces) == 1
    payload = json.loads(traces[0][len("KASAFLOW_TURN_TRACE "):])
    assert payload["llm_structured_output"] == {"error": "could not decide"}
    assert payload["final_result"] == {"error": "I couldn't understand that request. Could you rephrase it?"}


# ---------------------------------------------------------------------
# awaiting_field deterministic short-circuit (Webb, 2026-08-21, P0.4):
# a direct answer to propose_order's own immediately preceding question
# resolves without ever calling understand_customer(). See
# router._try_resolve_awaiting_field()'s docstring for why this exists
# (a bare "14k" misrouting to recommend_products, live, three times,
# despite extensive existing prompt guidance covering exactly this
# case) and memory.set_awaiting_field()'s docstring for its lifetime.
#
# Pure-function tests first (no session state, no LLM, no tool
# execution -- just the pattern match itself), then integration tests
# proving understand_customer is never actually called when it fires.
# ---------------------------------------------------------------------

def test_try_resolve_awaiting_field_returns_none_when_nothing_is_awaited():
    assert router._try_resolve_awaiting_field(None, "14k") is None


@pytest.mark.parametrize("reply, expected_karat", [("12", "12"), ("12k", "12"), ("18K", "18"), (" 14k ", "14")])
def test_try_resolve_awaiting_field_resolves_a_bare_karat(reply, expected_karat):
    result = router._try_resolve_awaiting_field("material", reply)
    assert result["tool"] == "propose_order"
    assert result["arguments"]["material"] == f"{expected_karat}k"
    assert result["arguments"]["product_name"] == "unknown"
    assert result["arguments"]["quantity"] == "unknown"


def test_try_resolve_awaiting_field_does_not_resolve_a_compound_material_message():
    # "12k, 2 of them" needs the LLM's own compound-correction handling
    # (see llm.py's _order_draft_state_line()) -- the whole message must
    # match, not just a leading substring.
    assert router._try_resolve_awaiting_field("material", "12k, 2 of them") is None


@pytest.mark.parametrize("reply, expected_qty", [("2", 2), ("2 pieces", 2), ("3 of them", 3), ("5 please", 5)])
def test_try_resolve_awaiting_field_resolves_a_bare_quantity(reply, expected_qty):
    result = router._try_resolve_awaiting_field("quantity", reply)
    assert result["tool"] == "propose_order"
    assert result["arguments"]["quantity"] == expected_qty
    assert result["arguments"]["material"] == "unknown"


@pytest.mark.parametrize("reply", ["14k", "18k", "12k please"])
def test_try_resolve_awaiting_field_does_not_resolve_a_karat_reply_as_quantity(reply):
    # Webb, 2026-08-21: "assistant asked for quantity -> '14k' must not
    # become quantity" -- a trailing "k" must not be silently parsed as
    # a digit count. _BARE_QUANTITY_RE's optional trailing-word group
    # only ever matches "pieces"/"pcs"/"of them"/"please", never a bare
    # "k", so this simply doesn't match and falls through to the LLM
    # (which has the order_draft context to tell these apart -- see
    # llm.py's own bare-number disambiguation rule) rather than being
    # guessed at here. Note a bare digit WITHOUT "k" ("18" alone, no
    # material context at all) correctly DOES resolve as quantity when
    # quantity is what's actually awaited -- awaiting_field itself is
    # what disambiguates a bare number here, unlike the LLM's harder
    # context-free version of the same problem.
    assert router._try_resolve_awaiting_field("quantity", reply) is None


@pytest.mark.parametrize("reply", ["four", "six pieces", "a couple"])
def test_try_resolve_awaiting_field_does_not_resolve_a_word_number_as_quantity(reply):
    # Deliberate scope boundary, not a gap: this resolver only ever
    # matches digit patterns. Word-form numbers fall through to the LLM
    # path exactly as before this existed -- safe (never wrong), just
    # not sped up for this case. Keeping the deterministic path narrow
    # is the point (Webb, 2026-08-21: "it must remain narrow").
    assert router._try_resolve_awaiting_field("quantity", reply) is None


@pytest.mark.parametrize("reply", [
    "yes", "Yeah", "yh", "ok", "confirm", "go ahead", "place it", "sure.", "yes, confirm", "Yes, confirm!",
])
def test_try_resolve_awaiting_field_resolves_a_bare_confirmation(reply):
    result = router._try_resolve_awaiting_field("confirmation", reply)
    assert result == {"tool": "confirm_order", "arguments": {}, "_source": "awaiting_field:confirmation"}


def test_try_resolve_awaiting_field_does_not_resolve_an_ambiguous_confirmation_reply():
    # Not in the canonical set -- a hedge or a longer sentence must go
    # through the LLM's own, already-tested confirmation guidance, not
    # be guessed at here.
    assert router._try_resolve_awaiting_field("confirmation", "maybe, how much would it be") is None


@pytest.mark.parametrize("reply", ["East Legon", "12 Cantonments Road, Accra", "Kasoa"])
def test_try_resolve_awaiting_field_resolves_a_bare_address(reply):
    result = router._try_resolve_awaiting_field("delivery_address", reply)
    assert result["tool"] == "propose_order"
    assert result["arguments"]["delivery_address"] == reply
    assert result["arguments"]["material"] == "unknown"


@pytest.mark.parametrize("reply, expected_address, expected_option", [
    ("East Legon, accra_rider", "East Legon", "accra_rider"),
    ("accra_rider, East Legon", "East Legon", "accra_rider"),
    ("deliver to Kumasi, kumasi_rider", "Kumasi", "kumasi_rider"),
    ("international, 221B Baker Street", "221B Baker Street", "international"),
])
def test_try_resolve_awaiting_field_splits_a_trailing_delivery_option_from_the_address(
    reply, expected_address, expected_option,
):
    # Caught live, 2026-08-24 (Webb): answering "What address should this
    # be delivered to?" with "East Legon, accra_rider" stored the WHOLE
    # string verbatim as delivery_address (nothing here does semantic
    # splitting) -- "Delivery to East Legon, accra_rider" in the
    # customer-facing proposal, with delivery_option left "unknown" for
    # propose_order to separately re-infer from the address text. Must
    # recognise the literal option key wherever it appears in the reply,
    # strip it out of the address, and return it as delivery_option
    # directly instead.
    result = router._try_resolve_awaiting_field("delivery_address", reply)
    assert result["arguments"]["delivery_address"] == expected_address
    assert result["arguments"]["delivery_option"] == expected_option


def test_try_resolve_awaiting_field_falls_through_when_the_reply_is_just_an_option_token():
    # "accra_rider" on its own has no place name left after the option
    # token is stripped out -- fall through to the LLM rather than store
    # an empty address.
    assert router._try_resolve_awaiting_field("delivery_address", "accra_rider") is None


@pytest.mark.parametrize("reply, expected_address", [
    ("deliver to East Legon", "East Legon"),
    ("Delivery to Kumasi", "Kumasi"),
    ("send it to Kasoa", "Kasoa"),
    ("ship to 12 Cantonments Road, Accra", "12 Cantonments Road, Accra"),
    ("please deliver to Tema", "Tema"),
])
def test_try_resolve_awaiting_field_strips_a_delivery_preamble_from_a_bare_address(reply, expected_address):
    # Caught live, 2026-08-24 (Webb): "deliver to East Legon" answering
    # "What address should this be delivered to?" was stored verbatim,
    # including the "deliver to" the customer echoed from the question
    # itself -- producing "Delivery to deliver to East Legon" in the
    # customer-facing proposal once propose_order's own "Delivery to "
    # prefix was added on top. The LLM path already strips this correctly
    # (semantic extraction is its job); this deterministic bypass does no
    # extraction at all, so it needs its own, narrow preamble strip.
    result = router._try_resolve_awaiting_field("delivery_address", reply)
    assert result["arguments"]["delivery_address"] == expected_address


def test_try_resolve_awaiting_field_falls_through_when_the_whole_reply_is_just_the_preamble():
    # "deliver to" on its own has no place name left after stripping the
    # preamble -- storing "" as a "valid" address would be worse than
    # falling through to the LLM, which can at least ask a clarifying
    # question with full order_draft context.
    assert router._try_resolve_awaiting_field("delivery_address", "deliver to") is None


@pytest.mark.parametrize("reply", ["confirm", "Confirm!", "yes", "ok", "go ahead", "place it"])
def test_try_resolve_awaiting_field_does_not_treat_a_bare_agreement_word_as_an_address(reply):
    # Caught live, 2026-08-21 (Webb's own first trace run): a stray
    # "confirm" sent while delivery_address was still awaited (the
    # previous turn's address extraction had already failed, so the
    # question was still open) was stored as the literal delivery
    # address "confirm" -- satisfying propose_order's own "is an address
    # present" check, so it moved straight on to asking about
    # delivery_option instead of recognising the customer was trying to
    # confirm. None of the other exclusions in this branch (question
    # marks, correction words, karat/quantity shapes) catch a bare
    # agreement word, because there's nothing address-SPECIFIC about it
    # -- it just isn't descriptive content for ANY field. A regression
    # in this feature's own first version, found from Webb's live
    # transcript, not a pre-existing bug.
    assert router._try_resolve_awaiting_field("delivery_address", reply) is None


def test_try_resolve_awaiting_field_does_not_resolve_a_karat_correction_as_an_address():
    # Webb, 2026-08-21: "assistant asked for address -> '14k, make it 6'
    # must go through the normal correction path, not become an
    # address." Already covered by the disqualifying-words/embedded-
    # karat parametrized test above (test_try_resolve_awaiting_field_
    # does_not_guess_at_a_non_address_reply) -- named separately here so
    # this specific scenario, called out explicitly, has its own visible
    # regression test rather than only living inside a parametrize list.
    assert router._try_resolve_awaiting_field("delivery_address", "14k, make it 6") is None


@pytest.mark.parametrize("reply", [
    "why do you need that?",
    "actually 14k, make it 6 instead",
    "wait, can I change the karat first",
    "12k",
    "2 of them",
    "no, I gave you the address already",
])
def test_try_resolve_awaiting_field_does_not_guess_at_a_non_address_reply(reply):
    # Every one of these is a real, live-plausible reply that is NOT an
    # address -- a question, a correction naming a different field, a
    # bare karat/quantity that belongs to a different awaited field, or
    # pushback. All must fall through to the LLM's own order_draft-aware
    # handling rather than being stored as a literal delivery address.
    assert router._try_resolve_awaiting_field("delivery_address", reply) is None


@pytest.mark.parametrize("reply", ["Accra", "Kumasi", "Nima, Accra"])
def test_try_resolve_awaiting_field_resolves_a_bare_reply_when_delivery_option_is_awaited(reply):
    # Task #168, Webb live 2026-08-24: propose_order's ambiguous-delivery-
    # option fallback ("Would you like rider delivery within Accra...")
    # previously never tagged awaiting_field at all, unlike material/
    # quantity/delivery_address -- so a simple reply restating the place
    # had no deterministic bypass to catch it. Shares the delivery_address
    # branch's extraction on purpose (see that branch's own comment):
    # propose_order() re-derives whichever of address/option is missing
    # from the other regardless of which one this bypass fills in.
    result = router._try_resolve_awaiting_field("delivery_option", reply)
    assert result["tool"] == "propose_order"
    assert result["arguments"]["delivery_address"] == reply
    assert result["_source"] == "awaiting_field:delivery_option"


def test_try_resolve_awaiting_field_falls_through_to_llm_for_a_pushback_shaped_delivery_option_reply():
    # The real live message ("nima is obviously in Accra so why would you
    # ask this \"Would you like...?\"") contains a "?" (quoting the
    # system's own question back), which correctly excludes it from this
    # deterministic bypass -- pushback this compound is exactly what the
    # LLM path, not a regex, should be resolving. See llm.py's
    # _order_draft_state_line() for the prompt-side half of this fix.
    reply = 'nima is obviously in Accra so why would you ask this "Would you like rider delivery within Accra, rider delivery within Kumasi, or shipping outside Ghana?"'
    assert router._try_resolve_awaiting_field("delivery_option", reply) is None


@pytest.mark.parametrize("reply", [
    "yes", "Yeah", "yh", "ok", "sure.", "please", "yes, confirm",
])
def test_try_resolve_awaiting_field_resolves_a_delivery_interest_affirmative(reply):
    # Confirmed live, 2026-08-22: "Want to know about delivery too?"
    # followed by "yes" fell through to converse instead of showing
    # delivery options, because nothing tracked that this specific
    # question was open. Same canonical affirmative set as the
    # "confirmation" branch -- this is the identical shape of question.
    result = router._try_resolve_awaiting_field("delivery_interest", reply)
    assert result == {
        "tool": "get_delivery_information",
        "arguments": {},
        "_source": "awaiting_field:delivery_interest",
    }


@pytest.mark.parametrize("reply", ["no", "no thanks", "not now", "how much is a chain"])
def test_try_resolve_awaiting_field_does_not_guess_at_a_non_affirmative_delivery_reply(reply):
    # A "no", a new question, or anything else not in the canonical
    # affirmative set must fall through to the LLM path rather than
    # being forced into showing delivery info.
    assert router._try_resolve_awaiting_field("delivery_interest", reply) is None


def test_route_customer_sets_awaiting_field_after_a_bare_price_reply(monkeypatch):
    # get_product_price's bare-price shape (no delivery_options key)
    # always ends with "Want to know about delivery too?" -- this must
    # track that the question is now open. See router.py's _PRICE_TOOL
    # branch alongside the existing propose_order awaiting_field logic.
    monkeypatch.setattr(
        router, "understand_customer",
        MagicMock(return_value={"tool": "get_product_price", "arguments": {"product_name": "Ring", "material": "18k"}}),
    )
    monkeypatch.setattr(
        router, "execute_tool",
        MagicMock(return_value={"product": "Big White Crown Stone Gold Ring, 14g", "material": "18k", "price": 24066.0}),
    )
    session_id = "delivery-interest-gets-set"

    router.route_customer("how much is the ring in 18k", session_id)

    assert router.get_awaiting_field(session_id) == "delivery_interest"


def test_route_customer_does_not_set_delivery_interest_for_a_combined_quote(monkeypatch):
    # generate_quote's shape already includes delivery_options up front
    # -- it has already answered the delivery question, so there is
    # nothing left open to track.
    monkeypatch.setattr(
        router, "understand_customer",
        MagicMock(return_value={"tool": "generate_quote", "arguments": {"product_name": "Ring", "material": "18k"}}),
    )
    monkeypatch.setattr(
        router, "execute_tool",
        MagicMock(return_value={"product": "Ring", "material": "18k", "price": 24066.0, "delivery_options": []}),
    )
    session_id = "delivery-interest-not-set-for-quote"

    router.route_customer("give me a full quote for the ring in 18k", session_id)

    assert router.get_awaiting_field(session_id) is None


def test_route_customer_resolves_a_bare_yes_to_delivery_info_after_a_price_reply(monkeypatch):
    # End-to-end: the exact live scenario -- a plain price reply, then a
    # bare "yes", must resolve to delivery info without ever calling the
    # LLM for the second turn.
    understand_customer_mock = MagicMock(
        return_value={"tool": "get_product_price", "arguments": {"product_name": "Ring", "material": "18k"}}
    )
    monkeypatch.setattr(router, "understand_customer", understand_customer_mock)
    monkeypatch.setattr(
        router, "execute_tool",
        MagicMock(side_effect=[
            {"product": "Big White Crown Stone Gold Ring, 14g", "material": "18k", "price": 24066.0},
            {"delivery_options": [{"key": "accra_rider", "label": "rider delivery within Accra"}]},
        ]),
    )
    session_id = "delivery-interest-end-to-end"

    router.route_customer("how much is the ring in 18k", session_id)
    assert understand_customer_mock.call_count == 1

    result = router.route_customer("yes", session_id)
    assert understand_customer_mock.call_count == 1  # bypassed, not called again
    assert "delivery_options" in result


def test_route_customer_never_calls_understand_customer_when_the_bypass_fires(monkeypatch):
    understand_customer_mock = MagicMock()
    monkeypatch.setattr(router, "understand_customer", understand_customer_mock)
    monkeypatch.setattr(router, "get_awaiting_field", MagicMock(return_value="material"))
    monkeypatch.setattr(router, "execute_tool", MagicMock(return_value={"error": "How many would you like?", "awaiting_field": "quantity"}))

    result = router.route_customer("14k", "bypass-session")

    understand_customer_mock.assert_not_called()
    assert result["error"] == "How many would you like?"
    # The internal routing key must never leak into the customer-facing
    # result -- see _execute_single()'s awaiting_field stripping.
    assert "awaiting_field" not in result


def test_route_customer_falls_through_to_the_llm_when_the_bypass_does_not_match(monkeypatch):
    understand_customer_mock = MagicMock(return_value={"tool": "converse", "arguments": {"reply": "Sure!"}})
    monkeypatch.setattr(router, "understand_customer", understand_customer_mock)
    monkeypatch.setattr(router, "get_awaiting_field", MagicMock(return_value="material"))

    router.route_customer("what karats do you have available?", "no-bypass-session")

    understand_customer_mock.assert_called_once()


def test_awaiting_field_does_not_survive_an_unrelated_turn_in_between(monkeypatch):
    # "Immediately preceding question" (Webb's own phrasing) -- a
    # converse reply in between means the next bare "12k" is no longer
    # answering propose_order's karat question, so it must go back
    # through the LLM rather than being deterministically resolved
    # against a field that's no longer the live topic.
    understand_customer_mock = MagicMock(side_effect=[
        {"tool": "propose_order", "arguments": {
            "product_name": "Ring", "material": "unknown", "quantity": 1,
            "delivery_address": "Accra", "delivery_option": "accra_rider"}},
        {"tool": "converse", "arguments": {"reply": "Sure, happy to help!"}},
        {"tool": "recommend_products", "arguments": {"category": "Rings", "material": "unknown"}},
    ])
    monkeypatch.setattr(router, "understand_customer", understand_customer_mock)
    monkeypatch.setattr(router, "execute_tool", MagicMock(side_effect=[
        {"error": "What karat would you like that in?", "awaiting_field": "material"},
        {"conversation_reply": "Sure, happy to help!"},
        {"recommendations": []},
    ]))
    session_id = "awaiting-field-does-not-linger"

    router.route_customer("I'd like to order a Ring, 1, deliver to Accra, rider delivery", session_id)
    assert router.get_awaiting_field(session_id) == "material"
    assert understand_customer_mock.call_count == 1

    router.route_customer("thanks for your help", session_id)
    assert router.get_awaiting_field(session_id) is None
    assert understand_customer_mock.call_count == 2

    # A bare "12k" now must go through the LLM (recommend_products, per
    # the canned mock) rather than being silently resolved as the
    # karat -- proving the deterministic path did NOT fire now that
    # awaiting_field has reset.
    router.route_customer("12k", session_id)
    assert understand_customer_mock.call_count == 3
