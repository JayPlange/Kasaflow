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
        # Explicitly cleared, not left to whatever's in the real .env --
        # this repo now has a real STAFF_NOTIFICATION_PHONE configured
        # (needed for the live store), and without this override every
        # test using this helper would silently start firing real
        # send_text_message() calls through the module-global `requests`
        # mock, changing call counts in tests that never meant to
        # exercise notification at all. Tests that actually want to
        # assert notification behaviour pass staff_notification_phone=
        # explicitly via **overrides, same as before.
        staff_notification_phone=None,
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


# ---------------------------------------------------------------------
# delivery_option/address mismatch -- see delivery_tool.
# delivery_option_matches_address() for the underlying check. Confirmed
# live, 2026-08-14 (Tamale/kumasi_rider) and 2026-08-18 (Cape Coast/
# international): a mismatch must soften the customer-facing label
# rather than assert a delivery arrangement that isn't real.
# ---------------------------------------------------------------------

def test_propose_order_softens_label_when_international_is_a_real_ghanaian_address(monkeypatch):
    # Arrange: "international" wrongly picked for Cape Coast, a real
    # Ghanaian city -- must not tell the customer it's shipping "outside
    # Ghana" when it plainly isn't.
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "product": "Ring", "material": "18k", "price": 1200.0},
    )

    # Act
    result = order_tool.propose_order("ring", "18k", 1, "Cape Coast", "international", "session-1")

    # Assert: the raw key is preserved (it's still what the customer/LLM
    # chose), but the label shown to the customer is softened
    proposal = result["proposal"]
    assert proposal["delivery_option"] == "international"
    assert proposal["delivery_option_label"] == (
        "a delivery arrangement to be confirmed by our team (this address doesn't "
        "match our usual shipping outside Ghana zone)"
    )


def test_propose_order_calls_geocoding_tools_resolve_delivery_match(monkeypatch):
    # Wiring check: propose_order must go through
    # geocoding_tool.resolve_delivery_match() (which itself falls back
    # to the offline heuristic when geocoding isn't configured), not
    # delivery_tool.delivery_option_matches_address() directly -- see
    # order_tool.py's import and the comment above this call site.
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "product": "Ring", "material": "18k", "price": 1200.0},
    )
    resolve_delivery_match = MagicMock(return_value=True)
    monkeypatch.setattr(order_tool, "resolve_delivery_match", resolve_delivery_match)

    order_tool.propose_order("ring", "18k", 1, "East Legon", "accra_rider", "session-1")

    resolve_delivery_match.assert_called_once_with("accra_rider", "East Legon")


def test_propose_order_does_not_soften_a_genuine_international_address(monkeypatch):
    # Arrange: a real address outside Ghana -- "international" is correct here
    _mock_product_lookup(
        monkeypatch,
        {"id": 42, "product": "Ring", "material": "18k", "price": 1200.0},
    )

    # Act
    result = order_tool.propose_order("ring", "18k", 1, "221B Baker Street, London", "international", "session-1")

    # Assert
    assert result["proposal"]["delivery_option_label"] == "shipping outside Ghana"


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


def test_propose_order_records_last_action_outcome_when_product_missing_woocommerce_id(monkeypatch):
    # Arrange: this is the one propose_order failure a customer did
    # nothing wrong to cause -- see memory.set_last_action_outcome()'s
    # docstring for why a follow-up "why?" needs this recorded
    _mock_product_lookup(
        monkeypatch,
        {"product": "Ring", "material": "18k", "price": 1200.0},  # no "id"
    )
    set_last_action_outcome = MagicMock()
    monkeypatch.setattr(order_tool, "set_last_action_outcome", set_last_action_outcome)

    # Act
    order_tool.propose_order("ring", "18k", 1, "Accra", "accra_rider", "session-1")

    # Assert
    set_last_action_outcome.assert_called_once()
    session_id, outcome = set_last_action_outcome.call_args[0]
    assert session_id == "session-1"
    assert outcome["action"] == "propose_order"
    assert "customer_safe_explanation" in outcome


def test_propose_order_does_not_record_last_action_outcome_for_a_recoverable_prompt(monkeypatch):
    # Arrange: a plain "how many would you like?" is self-explanatory --
    # only the genuinely unrecoverable failure above needs this
    lookup = _mock_product_lookup(monkeypatch, {"id": 1, "product": "Ring", "material": "18k", "price": 100})
    set_last_action_outcome = MagicMock()
    monkeypatch.setattr(order_tool, "set_last_action_outcome", set_last_action_outcome)

    # Act
    order_tool.propose_order("ring", "18k", "unknown", "Accra", "accra_rider", "session-1")

    # Assert
    set_last_action_outcome.assert_not_called()
    lookup.assert_not_called()


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


def test_confirm_order_records_last_action_outcome_on_a_non_timeout_failure(monkeypatch, fresh_session_store):
    # Arrange: same failure as above, checking the other branch that sets
    # the same shared outcome constant
    _woocommerce_settings(monkeypatch)
    monkeypatch.setattr(order_tool.requests, "post", MagicMock(side_effect=RuntimeError("connection reset")))
    set_last_action_outcome = MagicMock()
    monkeypatch.setattr(order_tool, "set_last_action_outcome", set_last_action_outcome)
    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0, "delivery_address": "Accra"},
    )

    # Act
    order_tool.confirm_order("session-1")

    # Assert
    set_last_action_outcome.assert_called_once()
    session_id, outcome = set_last_action_outcome.call_args[0]
    assert session_id == "session-1"
    assert outcome["action"] == "confirm_order"


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


def test_confirm_order_records_last_action_outcome_on_unresolved_timeout(monkeypatch, fresh_session_store):
    # Arrange: same scenario as above -- a genuine, unrecoverable-by-the-
    # customer failure a follow-up "why?" needs explained honestly
    _woocommerce_settings(monkeypatch)
    monkeypatch.setattr(
        order_tool.requests, "post",
        MagicMock(side_effect=order_tool.requests.exceptions.Timeout("timed out")),
    )
    fake_get_response = MagicMock()
    fake_get_response.raise_for_status.return_value = None
    fake_get_response.json.return_value = []
    monkeypatch.setattr(order_tool.requests, "get", MagicMock(return_value=fake_get_response))
    set_last_action_outcome = MagicMock()
    monkeypatch.setattr(order_tool, "set_last_action_outcome", set_last_action_outcome)

    fresh_session_store.set(
        "session-1", order_tool._PENDING_ORDER_KEY,
        {"token": "abc-123", "status": "pending", "product_id": 42, "variation_id": None,
         "product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0, "delivery_address": "Accra"},
    )

    # Act
    order_tool.confirm_order("session-1")

    # Assert
    set_last_action_outcome.assert_called_once()
    session_id, outcome = set_last_action_outcome.call_args[0]
    assert session_id == "session-1"
    assert outcome["action"] == "confirm_order"
    assert "customer_safe_explanation" in outcome


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


# ---------------------------------------------------------------------
# cancel_order -- see order_tool.py's cancel_order() docstring: prefers
# an explicit order number, falls back to this session's last confirmed
# order, and always re-checks the order's live WooCommerce status rather
# than trusting what this session last knew about it.
# ---------------------------------------------------------------------

def _mock_get_order(monkeypatch, status="on-hold", order_id=6846, status_code=200):
    fake_response = MagicMock()
    fake_response.status_code = status_code
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"id": order_id, "status": status}
    fake_get = MagicMock(return_value=fake_response)
    monkeypatch.setattr(order_tool.requests, "get", fake_get)
    return fake_get


def _mock_put(monkeypatch, status_code=200):
    fake_response = MagicMock()
    fake_response.status_code = status_code
    fake_response.raise_for_status.return_value = None
    fake_put = MagicMock(return_value=fake_response)
    monkeypatch.setattr(order_tool.requests, "put", fake_put)
    return fake_put


def test_cancel_order_succeeds_with_an_explicit_order_id(monkeypatch, fresh_session_store):
    # Arrange
    _woocommerce_settings(monkeypatch)
    _mock_get_order(monkeypatch, status="on-hold", order_id=6846)
    fake_put = _mock_put(monkeypatch)

    # Act
    result = order_tool.cancel_order("session-1", order_id="6846")

    # Assert: cancelled the exact order the customer named
    assert result["order_cancellation"]["order_id"] == 6846
    _, kwargs = fake_put.call_args
    assert kwargs["json"] == {"status": "cancelled"}
    assert "/orders/6846" in fake_put.call_args[0][0]


def test_cancel_order_falls_back_to_last_confirmed_order_when_no_id_given(monkeypatch, fresh_session_store):
    # Arrange: the LLM passes "unknown" when the customer didn't state a
    # number -- see llm.py's cancel_order guidance
    _woocommerce_settings(monkeypatch)
    fresh_session_store.set(
        "session-1", order_tool._LAST_CONFIRMED_KEY,
        {"order_id": 6846, "total": 1200.0, "delivery_address": "Accra", "delivery_option_label": "rider delivery within Accra"},
    )
    fake_get = _mock_get_order(monkeypatch, status="on-hold", order_id=6846)
    _mock_put(monkeypatch)

    # Act
    result = order_tool.cancel_order("session-1", order_id="unknown")

    # Assert
    assert result["order_cancellation"]["order_id"] == 6846
    assert "/orders/6846" in fake_get.call_args[0][0]


def test_cancel_order_asks_for_a_number_when_nothing_is_on_file(monkeypatch, fresh_session_store):
    # Arrange: no order_id given, and nothing to fall back to either
    _woocommerce_settings(monkeypatch)

    # Act
    result = order_tool.cancel_order("session-1", order_id="unknown")

    # Assert
    assert "error" in result
    assert "order number" in result["error"].lower()


def test_cancel_order_returns_none_resolved_id_for_an_unparseable_order_id(monkeypatch, fresh_session_store):
    # Arrange: a garbled order number -- deliberately does NOT fall back
    # to last_confirmed_order, since the customer did name something,
    # just not something usable (see _resolve_order_id())
    _woocommerce_settings(monkeypatch)
    fresh_session_store.set(
        "session-1", order_tool._LAST_CONFIRMED_KEY,
        {"order_id": 6846, "total": 1200.0, "delivery_address": "Accra", "delivery_option_label": None},
    )
    fake_get = _mock_get_order(monkeypatch)

    # Act
    result = order_tool.cancel_order("session-1", order_id="order number six")

    # Assert
    assert "error" in result
    fake_get.assert_not_called()


def test_cancel_order_reports_not_found_for_a_404(monkeypatch, fresh_session_store):
    # Arrange
    _woocommerce_settings(monkeypatch)
    _mock_get_order(monkeypatch, status_code=404)

    # Act
    result = order_tool.cancel_order("session-1", order_id="9999")

    # Assert
    assert "error" in result
    assert "9999" in result["error"]


def test_cancel_order_is_idempotent_when_already_cancelled(monkeypatch, fresh_session_store):
    # Arrange: a duplicated "cancel" message, most likely -- same
    # WhatsApp-delivery-duplication rationale as confirm_order()'s own
    # idempotency handling above
    _woocommerce_settings(monkeypatch)
    _mock_get_order(monkeypatch, status="cancelled", order_id=6846)
    fake_put = _mock_put(monkeypatch)

    # Act
    result = order_tool.cancel_order("session-1", order_id="6846")

    # Assert: reported as already-cancelled, never re-issues the cancel write
    assert result == {"order_already_cancelled": {"order_id": 6846}}
    fake_put.assert_not_called()


def test_cancel_order_escalates_a_non_cancellable_status_and_notifies_staff(monkeypatch, fresh_session_store):
    # Arrange: e.g. already shipped/completed/refunded -- not a status
    # this tool will touch automatically (see _CANCELLABLE_STATUSES)
    _woocommerce_settings(monkeypatch, staff_notification_phone="233509764406")
    _mock_get_order(monkeypatch, status="completed", order_id=6846)
    fake_put = _mock_put(monkeypatch)
    send_mock = MagicMock()
    monkeypatch.setattr(order_tool, "send_text_message", send_mock)

    # Act
    result = order_tool.cancel_order("session-233500000000", order_id="6846")

    # Assert: handed to staff, not silently refused or force-cancelled
    assert result == {"order_escalation": {"order_id": 6846, "status": "completed"}}
    fake_put.assert_not_called()
    send_mock.assert_called_once()
    to, body = send_mock.call_args[0]
    assert to == "233509764406"
    assert "6846" in body
    assert "completed" in body


def test_cancel_order_escalation_warns_but_does_not_fail_when_staff_phone_not_configured(monkeypatch, fresh_session_store, caplog):
    # Arrange: no staff_notification_phone -- config gap, not the
    # customer's problem, same principle as confirm_order()'s own
    # staff-notification tests above
    _woocommerce_settings(monkeypatch)
    _mock_get_order(monkeypatch, status="completed", order_id=6846)
    send_mock = MagicMock()
    monkeypatch.setattr(order_tool, "send_text_message", send_mock)

    # Act
    result = order_tool.cancel_order("session-1", order_id="6846")

    # Assert
    assert result == {"order_escalation": {"order_id": 6846, "status": "completed"}}
    send_mock.assert_not_called()


def test_cancel_order_reports_a_clean_error_when_the_lookup_fails(monkeypatch, fresh_session_store):
    # Arrange: a genuine WooCommerce/network failure, not a 404
    _woocommerce_settings(monkeypatch)
    monkeypatch.setattr(order_tool.requests, "get", MagicMock(side_effect=RuntimeError("connection reset")))
    fake_put = _mock_put(monkeypatch)

    # Act
    result = order_tool.cancel_order("session-1", order_id="6846")

    # Assert
    assert "error" in result
    fake_put.assert_not_called()


def test_cancel_order_reports_a_clean_error_when_the_cancel_write_fails(monkeypatch, fresh_session_store):
    # Arrange: order is found and cancellable, but the PUT itself fails
    _woocommerce_settings(monkeypatch)
    _mock_get_order(monkeypatch, status="on-hold", order_id=6846)
    monkeypatch.setattr(order_tool.requests, "put", MagicMock(side_effect=RuntimeError("connection reset")))
    send_mock = MagicMock()
    monkeypatch.setattr(order_tool, "send_text_message", send_mock)

    # Act
    result = order_tool.cancel_order("session-1", order_id="6846")

    # Assert: never claims success, never notifies staff of a cancellation
    # that didn't actually happen
    assert "error" in result
    send_mock.assert_not_called()


def test_cancel_order_notifies_staff_and_clears_last_confirmed_on_success(monkeypatch, fresh_session_store):
    # Arrange: cancelling the session's own last-confirmed order should
    # clear that fallback -- see cancel_order()'s docstring on why it's
    # no longer the right order to fall back to afterwards
    _woocommerce_settings(monkeypatch, staff_notification_phone="233509764406")
    fresh_session_store.set(
        "session-1", order_tool._LAST_CONFIRMED_KEY,
        {"order_id": 6846, "total": 1200.0, "delivery_address": "Accra", "delivery_option_label": "rider delivery within Accra"},
    )
    _mock_get_order(monkeypatch, status="on-hold", order_id=6846)
    _mock_put(monkeypatch)
    send_mock = MagicMock()
    monkeypatch.setattr(order_tool, "send_text_message", send_mock)

    # Act
    result = order_tool.cancel_order("session-1", order_id="6846")

    # Assert
    assert result == {"order_cancellation": {"order_id": 6846}}
    send_mock.assert_called_once()
    to, body = send_mock.call_args[0]
    assert to == "233509764406"
    assert "6846" in body
    assert "session-1" in body
    assert fresh_session_store.get("session-1", order_tool._LAST_CONFIRMED_KEY) is None


def test_cancel_order_does_not_clear_last_confirmed_when_cancelling_a_different_order(monkeypatch, fresh_session_store):
    # Arrange: session's last confirmed order is #111, but the customer
    # explicitly names a different order (#6846) to cancel -- #111 is
    # still the right fallback for a future bare "cancel my order"
    _woocommerce_settings(monkeypatch)
    fresh_session_store.set(
        "session-1", order_tool._LAST_CONFIRMED_KEY,
        {"order_id": 111, "total": 500.0, "delivery_address": "Accra", "delivery_option_label": None},
    )
    _mock_get_order(monkeypatch, status="on-hold", order_id=6846)
    _mock_put(monkeypatch)

    # Act
    order_tool.cancel_order("session-1", order_id="6846")

    # Assert
    assert fresh_session_store.get("session-1", order_tool._LAST_CONFIRMED_KEY)["order_id"] == 111
