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


def _mock_delivery(monkeypatch, delivery_time="2-5 business days", shipping_cost=25):
    monkeypatch.setattr(
        order_tool,
        "get_delivery_information",
        MagicMock(return_value={"delivery_time": delivery_time, "shipping_cost": shipping_cost}),
    )


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
    _mock_delivery(monkeypatch)

    # Act
    result = order_tool.propose_order("ring", "18k", 2, "12 Cantonments Road, Accra", "session-1")

    # Assert
    proposal = result["proposal"]
    assert proposal["product"] == "Ring"
    assert proposal["quantity"] == 2
    assert proposal["subtotal"] == 2400.0
    assert proposal["total"] == 2425.0  # subtotal + GH₵25 shipping
    assert proposal["product_id"] == 42
    assert proposal["variation_id"] is None
    assert proposal["status"] == "pending"
    assert "token" in proposal


def test_propose_order_carries_variation_id_when_present(monkeypatch):
    # Arrange
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "variation_id": 99, "product": "Ring", "material": "18k", "price": 1200.0},
    )
    _mock_delivery(monkeypatch)

    # Act
    result = order_tool.propose_order("ring", "18k", 1, "Accra", "session-1")

    # Assert
    assert result["proposal"]["variation_id"] == 99


def test_propose_order_stores_pending_order_in_session(monkeypatch, fresh_session_store):
    # Arrange
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "product": "Ring", "material": "18k", "price": 1200.0},
    )
    _mock_delivery(monkeypatch)

    # Act
    order_tool.propose_order("ring", "18k", 1, "Accra", "session-1")

    # Assert
    stored = fresh_session_store.get("session-1", order_tool._PENDING_ORDER_KEY)
    assert stored is not None
    assert stored["product"] == "Ring"


@pytest.mark.parametrize("quantity", ["unknown", "zero", 0, -1, None])
def test_propose_order_rejects_invalid_quantity(monkeypatch, quantity):
    # Arrange: product lookup should never even be reached
    lookup = _mock_product_lookup(monkeypatch, {"id": 1, "product": "Ring", "material": "18k", "price": 100})

    # Act
    result = order_tool.propose_order("ring", "18k", quantity, "Accra", "session-1")

    # Assert
    assert "error" in result
    lookup.assert_not_called()


@pytest.mark.parametrize("address", ["", "   ", "unknown", "UNKNOWN"])
def test_propose_order_rejects_missing_delivery_address(monkeypatch, address):
    # Arrange
    lookup = _mock_product_lookup(monkeypatch, {"id": 1, "product": "Ring", "material": "18k", "price": 100})

    # Act
    result = order_tool.propose_order("ring", "18k", 1, address, "session-1")

    # Assert
    assert "error" in result
    lookup.assert_not_called()


def test_propose_order_returns_error_when_product_not_found(monkeypatch):
    # Arrange
    _mock_product_lookup(monkeypatch, None)

    # Act
    result = order_tool.propose_order("bracelet", "platinum", 1, "Accra", "session-1")

    # Assert
    assert "couldn't find" in result["error"].lower()


def test_propose_order_returns_error_when_product_missing_woocommerce_id(monkeypatch):
    # Arrange: simulates a catalogue synced before the id/variation_id
    # fields existed (see woocommerce_sync.py's build_catalogue())
    _mock_product_lookup(
        monkeypatch,
        {"product": "Ring", "material": "18k", "price": 1200.0},  # no "id"
    )
    _mock_delivery(monkeypatch)

    # Act
    result = order_tool.propose_order("ring", "18k", 1, "Accra", "session-1")

    # Assert: fails now, with a clear reason, rather than letting the
    # customer confirm an order that can never actually be created
    assert "error" in result


# ---------------------------------------------------------------------
# confirm_order
# ---------------------------------------------------------------------

def test_confirm_order_returns_error_when_nothing_pending():
    # Act
    result = order_tool.confirm_order("session-never-seen")

    # Assert
    assert "nothing to confirm" in result["error"].lower()


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
            "total": 2425.0,
            "delivery_address": "Accra",
        },
    )

    # Act
    result = order_tool.confirm_order("session-1")

    # Assert: WooCommerce actually got called with the right line item
    fake_post.assert_called_once()
    _, kwargs = fake_post.call_args
    assert kwargs["json"]["status"] == "on-hold"
    assert kwargs["json"]["line_items"] == [{"product_id": 42, "quantity": 2}]

    # Assert: customer-facing result and session state
    assert result["order_confirmation"]["order_id"] == 555
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
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0, "delivery_address": "Accra"},
    )

    # Act
    order_tool.confirm_order("session-1")

    # Assert: a structured, purpose-built lookup key, not just free text
    _, kwargs = fake_post.call_args
    assert kwargs["json"]["meta_data"] == [{"key": "kasaflow_order_token", "value": "abc-123"}]


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
