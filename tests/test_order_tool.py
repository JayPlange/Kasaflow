"""
Unit tests for services/order_tool.py -- the action layer.

Mocks requests.post directly (same rule as every other test file here:
never call a real external API in a unit test) and swaps the module's
session store for a fresh one per test, the same isolation pattern
test_memory.py already uses for SessionStore itself.
"""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from services import order_tool
from services.memory import SessionStore


@pytest.fixture(autouse=True)
def fresh_session_store(monkeypatch):
    """Every test gets an empty store -- propose_order/confirm_order
    read and write session state, so tests must not see each other's."""
    store = SessionStore()
    monkeypatch.setattr(order_tool, "_store", store)
    return store


def _mock_product_lookup(monkeypatch, product=None):
    fake = MagicMock(return_value=product)
    monkeypatch.setattr(order_tool, "get_product_price", fake)
    return fake


def _woocommerce_settings(monkeypatch, **overrides):
    defaults = dict(
        woocommerce_url="https://adomdejeweller.com",
        woocommerce_orders_consumer_key="ck_test",
        woocommerce_orders_consumer_secret="cs_test",
    )
    defaults.update(overrides)
    monkeypatch.setattr(order_tool, "settings", replace(order_tool.settings, **defaults))


def _mock_post(monkeypatch, order_id=987, status_code=201):
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"id": order_id, "status": "on-hold"}
    fake_post = MagicMock(return_value=fake_response)
    monkeypatch.setattr(order_tool.requests, "post", fake_post)
    return fake_post


# ---------------------------------------------------------------------
# propose_order
# ---------------------------------------------------------------------

def test_propose_order_returns_priced_proposal(monkeypatch):
    # Arrange
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "product": "Ring", "material": "18k", "price": 1200.0},
    )

    # Act
    result = order_tool.propose_order("ring", "18k", 2, "12 Cantonments Road, Accra", "accra_rider", "session-1")

    # Assert: total is product cost only -- delivery isn't priced
    # automatically (see order_tool.propose_order()'s docstring)
    proposal = result["proposal"]
    assert proposal["product"] == "Ring"
    assert proposal["quantity"] == 2
    assert proposal["subtotal"] == 2400.0
    assert proposal["total"] == 2400.0
    assert proposal["product_id"] == 42
    assert proposal["variation_id"] is None
    assert proposal["status"] == "pending"
    assert proposal["delivery_option"] == "accra_rider"
    assert proposal["delivery_option_label"] == "rider delivery within Accra"
    assert "token" in proposal


def test_propose_order_carries_variation_id_when_present(monkeypatch):
    # Arrange
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "variation_id": 99, "product": "Ring", "material": "18k", "price": 1200.0},
    )

    # Act
    result = order_tool.propose_order("ring", "18k", 1, "Accra", "accra_rider", "session-1")

    # Assert
    assert result["proposal"]["variation_id"] == 99


def test_propose_order_stores_pending_order_in_session(monkeypatch, fresh_session_store):
    # Arrange
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "product": "Ring", "material": "18k", "price": 1200.0},
    )

    # Act
    order_tool.propose_order("ring", "18k", 1, "Accra", "accra_rider", "session-1")

    # Assert
    stored = fresh_session_store.get("session-1", order_tool._PENDING_ORDER_KEY)
    assert stored is not None
    assert stored["product"] == "Ring"


@pytest.mark.parametrize("product_name", ["", "   ", "unknown", "UNKNOWN", None])
def test_propose_order_rejects_missing_product_name(monkeypatch, product_name):
    # Arrange: "place an order" with no item named yet -- this must be
    # asked first, before quantity/address/delivery_option, or the
    # customer gets walked through irrelevant questions for a product
    # that was never identified (confirmed live, 2026-08-12).
    lookup = _mock_product_lookup(monkeypatch, {"id": 1, "product": "Ring", "material": "18k", "price": 100})

    # Act
    result = order_tool.propose_order(product_name, "18k", 1, "Accra", "accra_rider", "session-1")

    # Assert
    assert "error" in result
    assert "which item" in result["error"].lower()
    lookup.assert_not_called()


@pytest.mark.parametrize("material", ["", "   ", "unknown", "UNKNOWN", None])
def test_propose_order_rejects_missing_material(monkeypatch, material):
    # Arrange: a named product but no stated karat -- must ask rather
    # than let the semantic-search fallback silently guess one (and
    # possibly quote/order the wrong price).
    lookup = _mock_product_lookup(monkeypatch, {"id": 1, "product": "Ring", "material": "18k", "price": 100})

    # Act
    result = order_tool.propose_order("ring", material, 1, "Accra", "accra_rider", "session-1")

    # Assert
    assert "error" in result
    assert "karat" in result["error"].lower()
    lookup.assert_not_called()


@pytest.mark.parametrize("quantity", ["unknown", "zero", 0, -1, None])
def test_propose_order_rejects_invalid_quantity(monkeypatch, quantity):
    # Arrange: product lookup should never even be reached
    lookup = _mock_product_lookup(monkeypatch, {"id": 1, "product": "Ring", "material": "18k", "price": 100})

    # Act
    result = order_tool.propose_order("ring", "18k", quantity, "Accra", "accra_rider", "session-1")

    # Assert
    assert "error" in result
    lookup.assert_not_called()


@pytest.mark.parametrize("address", ["", "   ", "unknown", "UNKNOWN"])
def test_propose_order_rejects_missing_delivery_address(monkeypatch, address):
    # Arrange
    lookup = _mock_product_lookup(monkeypatch, {"id": 1, "product": "Ring", "material": "18k", "price": 100})

    # Act
    result = order_tool.propose_order("ring", "18k", 1, address, "accra_rider", "session-1")

    # Assert
    assert "error" in result
    lookup.assert_not_called()


@pytest.mark.parametrize("delivery_option", ["", "   ", "unknown", "UNKNOWN", None, "accra", "lagos"])
def test_propose_order_rejects_missing_or_invalid_delivery_option(monkeypatch, delivery_option):
    # Arrange: product lookup should never even be reached -- this is
    # asked before we bother looking anything up, same as quantity/address
    lookup = _mock_product_lookup(monkeypatch, {"id": 1, "product": "Ring", "material": "18k", "price": 100})

    # Act
    result = order_tool.propose_order("ring", "18k", 1, "Accra", delivery_option, "session-1")

    # Assert: asks using the real option labels, doesn't invent its own wording
    assert "error" in result
    assert "rider delivery within Accra" in result["error"]
    assert "rider delivery within Kumasi" in result["error"]
    assert "shipping outside Ghana" in result["error"]
    assert "or shipping outside Ghana" in result["error"]  # not a bare comma list
    lookup.assert_not_called()


def test_propose_order_returns_error_when_product_not_found(monkeypatch):
    # Arrange
    _mock_product_lookup(monkeypatch, None)

    # Act
    result = order_tool.propose_order("bracelet", "platinum", 1, "Accra", "accra_rider", "session-1")

    # Assert
    assert "couldn't find" in result["error"].lower()


def test_propose_order_returns_error_when_product_missing_woocommerce_id(monkeypatch):
    # Arrange: simulates a catalogue synced before the id/variation_id
    # fields existed (see woocommerce_sync.py's build_catalogue())
    _mock_product_lookup(
        monkeypatch,
        {"product": "Ring", "material": "18k", "price": 1200.0},  # no "id"
    )

    # Act
    result = order_tool.propose_order("ring", "18k", 1, "Accra", "accra_rider", "session-1")

    # Assert: fails now, with a clear reason, rather than letting the
    # customer confirm an order that can never actually be created
    assert "error" in result


# ---------------------------------------------------------------------
# get_pending_order_summary
# ---------------------------------------------------------------------

def test_get_pending_order_summary_returns_none_when_nothing_pending():
    assert order_tool.get_pending_order_summary("session-never-seen") is None


def test_get_pending_order_summary_reflects_a_proposed_order(monkeypatch):
    # Arrange
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "product": "Ring", "material": "18k", "price": 1200.0},
    )
    order_tool.propose_order("ring", "18k", 2, "Accra", "accra_rider", "session-1")

    # Act
    summary = order_tool.get_pending_order_summary("session-1")

    # Assert: this is what router.py hands the LLM so it actually knows
    # there's something to confirm, rather than guessing blind -- see
    # llm.py's _pending_order_state_line(). total is product cost only.
    assert summary == {"product": "Ring", "material": "18k", "quantity": 2, "total": 2400.0}


def test_get_pending_order_summary_clears_after_confirmation(monkeypatch):
    # Arrange
    _woocommerce_settings(monkeypatch)
    _mock_post(monkeypatch, order_id=555)
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "product": "Ring", "material": "18k", "price": 1200.0},
    )
    order_tool.propose_order("ring", "18k", 1, "Accra", "accra_rider", "session-1")
    order_tool.confirm_order("session-1")

    # Act / Assert: nothing left pending once it's gone through
    assert order_tool.get_pending_order_summary("session-1") is None


# ---------------------------------------------------------------------
# confirm_order
# ---------------------------------------------------------------------

def test_confirm_order_returns_error_when_nothing_pending():
    # Act
    result = order_tool.confirm_order("session-never-seen")

    # Assert: open-ended, not "want a quote?" -- this also fires for a
    # bare "yh" that was never about an order at all (see the message's
    # own comment in order_tool.py for why)
    assert "anything pending to confirm" in result["error"].lower()


def test_confirm_order_creates_order_and_clears_pending_state(monkeypatch, fresh_session_store):
    # Arrange
    _woocommerce_settings(monkeypatch)
    fake_post = _mock_post(monkeypatch, order_id=555)
    fresh_session_store.set(
        "session-1",
        order_tool._PENDING_ORDER_KEY,
        {
            "token": "abc-123",
            "status": "pending",
            "product_id": 42,
            "variation_id": None,
            "product": "Ring",
            "material": "18k",
            "quantity": 2,
            "total": 2400.0,
            "delivery_address": "Accra",
            "delivery_option": "accra_rider",
            "delivery_option_label": "rider delivery within Accra",
        },
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: WooCommerce actually got called with the right line item
    fake_post.assert_called_once()
    _, kwargs = fake_post.call_args
    assert kwargs["json"]["status"] == "on-hold"
    assert kwargs["json"]["line_items"] == [{"product_id": 42, "quantity": 2}]
    # The chosen delivery option is written onto the order itself (see
    # order_tool.py's confirm_order() docstring), not just sent as a
    # one-off notification -- so it's still visible if that fails
    assert "rider delivery within Accra" in kwargs["json"]["customer_note"]
    assert {"key": "kasaflow_delivery_option", "value": "accra_rider"} in kwargs["json"]["meta_data"]

    # Assert: customer-facing result and session state
    assert result["order_confirmation"]["order_id"] == 555
    assert result["order_confirmation"]["delivery_option_label"] == "rider delivery within Accra"
    assert fresh_session_store.get("session-1", order_tool._PENDING_ORDER_KEY) is None
    assert fresh_session_store.get("session-1", order_tool._LAST_CONFIRMED_KEY)["order_id"] == 555


def test_confirm_order_includes_variation_id_when_present(monkeypatch, fresh_session_store):
    # Arrange
    _woocommerce_settings(monkeypatch)
    fake_post = _mock_post(monkeypatch)
    fresh_session_store.set(
        "session-1",
        order_tool._PENDING_ORDER_KEY,
        {
            "token": "abc-123", "status": "pending", "product_id": 42, "variation_id": 99,
            "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0,
            "delivery_address": "Accra",
        },
    )

    # Act
    order_tool.confirm_order("session-1")

    # Assert
    _, kwargs = fake_post.call_args
    assert kwargs["json"]["line_items"] == [{"product_id": 42, "variation_id": 99, "quantity": 1}]


def test_confirm_order_resends_confirmation_for_a_duplicate_confirm(monkeypatch, fresh_session_store):
    # Arrange: the first confirm already went through and cleared the
    # pending order -- simulates a duplicated WhatsApp webhook delivery
    fresh_session_store.set(
        "session-1", order_tool._LAST_CONFIRMED_KEY,
        {"order_id": 555, "total": 2425.0, "delivery_address": "Accra"},
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: same confirmation again, not "nothing to confirm" (which
    # would wrongly read as if the first order never happened)
    assert result["order_confirmation"]["order_id"] == 555


def test_confirm_order_blocks_a_second_call_while_first_is_in_flight(monkeypatch, fresh_session_store):
    # Arrange: status "submitting" simulates a first confirm_order call
    # still waiting on WooCommerce's response
    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "submitting", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0, "delivery_address": "Accra"},
    )
    fake_post = _mock_post(monkeypatch)

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: no second request fired
    assert "error" in result
    fake_post.assert_not_called()


def test_confirm_order_resets_to_pending_on_woocommerce_failure(monkeypatch, fresh_session_store):
    # Arrange
    _woocommerce_settings(monkeypatch)
    fake_post = MagicMock(side_effect=RuntimeError("connection reset"))
    monkeypatch.setattr(order_tool.requests, "post", fake_post)
    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0, "delivery_address": "Accra"},
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: customer sees a clean error, and the order is retryable
    # (status back to "pending", not stuck "submitting" forever)
    assert "error" in result
    pending = fresh_session_store.get("session-1", order_tool._PENDING_ORDER_KEY)
    assert pending["status"] == "pending"


def test_confirm_order_recovers_from_timeout_when_order_was_actually_created(monkeypatch, fresh_session_store):
    # Arrange: the POST times out (response lost), but WooCommerce
    # actually created the order -- the lookup should find it and
    # confirm using the real order, not fail or double-create
    _woocommerce_settings(monkeypatch)
    fake_post = MagicMock(side_effect=order_tool.requests.exceptions.Timeout("timed out"))
    monkeypatch.setattr(order_tool.requests, "post", fake_post)

    fake_get_response = MagicMock()
    fake_get_response.raise_for_status.return_value = None
    fake_get_response.json.return_value = [{"id": 777, "status": "on-hold"}]
    fake_get = MagicMock(return_value=fake_get_response)
    monkeypatch.setattr(order_tool.requests, "get", fake_get)

    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0, "delivery_address": "Accra"},
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: recovered the real order, no duplicate created, pending cleared
    assert result["order_confirmation"]["order_id"] == 777
    fake_post.assert_called_once()  # never retried the write itself
    _, kwargs = fake_get.call_args
    assert kwargs["params"]["search"] == "abc-123"
    assert fresh_session_store.get("session-1", order_tool._PENDING_ORDER_KEY) is None


def test_confirm_order_falls_back_to_pending_when_timeout_lookup_finds_nothing(monkeypatch, fresh_session_store):
    # Arrange: POST times out, and the order genuinely wasn't created
    _woocommerce_settings(monkeypatch)
    monkeypatch.setattr(
        order_tool.requests, "post",
        MagicMock(side_effect=order_tool.requests.exceptions.Timeout("timed out")),
    )
    fake_get_response = MagicMock()
    fake_get_response.raise_for_status.return_value = None
    fake_get_response.json.return_value = []  # nothing found
    monkeypatch.setattr(order_tool.requests, "get", MagicMock(return_value=fake_get_response))

    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0, "delivery_address": "Accra"},
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: genuinely unresolved -- back to "pending" so a customer
    # retry can actually place the order, not stuck as "submitting" forever
    assert "error" in result
    pending = fresh_session_store.get("session-1", order_tool._PENDING_ORDER_KEY)
    assert pending["status"] == "pending"


def test_confirm_order_does_not_look_up_on_non_timeout_failures(monkeypatch, fresh_session_store):
    # Arrange: a clearly-failed request (not ambiguous) -- e.g. WooCommerce
    # rejected it outright. No lookup should be attempted for this.
    _woocommerce_settings(monkeypatch)
    monkeypatch.setattr(
        order_tool.requests, "post",
        MagicMock(side_effect=order_tool.requests.exceptions.ConnectionError("refused")),
    )
    fake_get = MagicMock()
    monkeypatch.setattr(order_tool.requests, "get", fake_get)

    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0, "delivery_address": "Accra"},
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: no lookup attempted -- only Timeout is treated as ambiguous
    assert "error" in result
    fake_get.assert_not_called()


def test_create_order_payload_includes_token_in_meta_data(monkeypatch, fresh_session_store):
    # Arrange
    _woocommerce_settings(monkeypatch)
    fake_post = _mock_post(monkeypatch)
    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1200.0, "delivery_address": "Accra",
         "delivery_option": "accra_rider", "delivery_option_label": "rider delivery within Accra"},
    )

    # Act
    order_tool.confirm_order("session-1")

    # Assert: a structured, purpose-built lookup key, not just free text
    _, kwargs = fake_post.call_args
    assert {"key": "kasaflow_order_token", "value": "abc-123"} in kwargs["json"]["meta_data"]


def test_create_order_payload_gracefully_handles_a_missing_delivery_option(monkeypatch, fresh_session_store):
    # Arrange: defensive -- shouldn't happen given propose_order() always
    # validates it first, but a payload built from a hand-constructed
    # pending order (as these tests do) must not crash if it's absent
    _woocommerce_settings(monkeypatch)
    fake_post = _mock_post(monkeypatch)
    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1200.0, "delivery_address": "Accra"},
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: no crash, order still goes through
    assert "order_confirmation" in result


def test_confirm_order_raises_clear_error_when_orders_config_missing(monkeypatch, fresh_session_store):
    # Arrange: WooCommerce order-write credentials not configured
    monkeypatch.setattr(
        order_tool,
        "settings",
        replace(
            order_tool.settings,
            woocommerce_orders_consumer_key=None,
            woocommerce_orders_consumer_secret=None,
        ),
    )
    fake_post = _mock_post(monkeypatch)
    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0, "delivery_address": "Accra"},
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: fails cleanly before ever attempting the request
    assert "error" in result
    fake_post.assert_not_called()


# ---------------------------------------------------------------------
# staff notification -- delivery isn't automated (see delivery_tool.py),
# so confirm_order() has to actually tell a human once an order goes
# through, or nothing ever gets delivered.
# ---------------------------------------------------------------------

def _pending_order(**overrides):
    base = {
        "token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
        "product": "Ring", "material": "18k", "quantity": 1, "total": 1200.0,
        "delivery_address": "12 Cantonments Road, Accra",
        "delivery_option": "accra_rider", "delivery_option_label": "rider delivery within Accra",
    }
    base.update(overrides)
    return base


def test_confirm_order_notifies_staff_when_phone_is_configured(monkeypatch, fresh_session_store):
    # Arrange
    _woocommerce_settings(monkeypatch, staff_notification_phone="233509764406")
    _mock_post(monkeypatch, order_id=555)
    fresh_session_store.set("session-233500000000", order_tool._PENDING_ORDER_KEY, _pending_order())
    send_mock = MagicMock()
    monkeypatch.setattr(order_tool, "send_text_message", send_mock)

    # Act
    order_tool.confirm_order("session-233500000000")

    # Assert: staff got pinged with the order id, delivery choice, and
    # a way to reach the customer back
    send_mock.assert_called_once()
    to, body = send_mock.call_args[0]
    assert to == "233509764406"
    assert "555" in body
    assert "rider delivery within Accra" in body
    assert "session-233500000000" in body


def test_confirm_order_warns_but_still_succeeds_when_staff_phone_not_configured(monkeypatch, fresh_session_store, caplog):
    # Arrange: no staff_notification_phone set at all
    _woocommerce_settings(monkeypatch)
    _mock_post(monkeypatch, order_id=555)
    fresh_session_store.set("session-1", order_tool._PENDING_ORDER_KEY, _pending_order())
    send_mock = MagicMock()
    monkeypatch.setattr(order_tool, "send_text_message", send_mock)

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: the customer's order still succeeds -- a missing
    # notification channel is a config gap, not the customer's problem
    assert result["order_confirmation"]["order_id"] == 555
    send_mock.assert_not_called()


def test_confirm_order_succeeds_even_when_staff_notification_fails(monkeypatch, fresh_session_store):
    # Arrange: staff is configured, but the WhatsApp send itself fails
    # (rate limit, bad token, whatever) -- this must be best-effort,
    # never allowed to take down an order that already went through
    _woocommerce_settings(monkeypatch, staff_notification_phone="233509764406")
    _mock_post(monkeypatch, order_id=555)
    fresh_session_store.set("session-1", order_tool._PENDING_ORDER_KEY, _pending_order())
    monkeypatch.setattr(
        order_tool, "send_text_message",
        MagicMock(side_effect=order_tool.WhatsAppError("rate limited")),
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: customer still gets a clean confirmation
    assert result["order_confirmation"]["order_id"] == 555


def test_confirm_order_does_not_renotify_staff_on_a_duplicate_confirm(monkeypatch, fresh_session_store):
    # Arrange: the order already went through -- this is the "resend the
    # same confirmation" path, not a fresh one, so staff shouldn't get
    # pinged again for the same order
    fresh_session_store.set(
        "session-1", order_tool._LAST_CONFIRMED_KEY,
        {"order_id": 555, "total": 1200.0, "delivery_address": "Accra", "delivery_option_label": "rider delivery within Accra"},
    )
    send_mock = MagicMock()
    monkeypatch.setattr(order_tool, "send_text_message", send_mock)

    # Act
    order_tool.confirm_order("session-1")

    # Assert
    send_mock.assert_not_called()
