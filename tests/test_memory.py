"""
Unit tests for services/memory.py -- the per-session store, and the
fill_missing_context / remember_context helpers router.py calls.

These are pure, no mocking of the OpenAI client involved: they test that
sessions stay isolated from each other and that entries expire, which is
the exact bug the old shared-dict version had.
"""

from services.memory import (
    SessionStore,
    fill_missing_context,
    get_last_action_outcome,
    get_last_priced_product,
    get_order_draft,
    get_pending_intent,
    remember_context,
    set_last_action_outcome,
    set_last_priced_product,
    set_pending_intent,
)


# ---------------------------------------------------------------------
# SessionStore: isolation and expiry
# ---------------------------------------------------------------------

def test_set_and_get_round_trips_a_value():
    store = SessionStore()
    store.set("session-1", "material", "gold")
    assert store.get("session-1", "material") == "gold"


def test_get_returns_default_for_unknown_session():
    store = SessionStore()
    assert store.get("never-seen", "material") is None
    assert store.get("never-seen", "material", default="fallback") == "fallback"


def test_sessions_are_isolated_from_each_other():
    # Arrange: this is the exact bug the old shared conversation_memory
    # dict had -- one session's write must never be visible to another
    store = SessionStore()
    store.set("session-a", "material", "gold")
    store.set("session-b", "material", "silver")

    # Assert
    assert store.get("session-a", "material") == "gold"
    assert store.get("session-b", "material") == "silver"


def test_entries_expire_after_ttl():
    # Arrange: a store with a TTL of 0 means anything set is already
    # expired by the time it's read back
    store = SessionStore(ttl_seconds=0)
    store.set("session-1", "material", "gold")

    # Act / Assert
    assert store.get("session-1", "material") is None


def test_clear_removes_a_session():
    store = SessionStore()
    store.set("session-1", "material", "gold")
    store.clear("session-1")
    assert store.get("session-1", "material") is None


def test_active_session_count_ignores_expired_sessions():
    store = SessionStore()
    live = SessionStore(ttl_seconds=0)
    store.set("session-1", "material", "gold")
    store.set("session-2", "material", "silver")
    live.set("session-1", "material", "gold")

    assert store.active_session_count() == 2
    assert live.active_session_count() == 0


# ---------------------------------------------------------------------
# fill_missing_context / remember_context
# ---------------------------------------------------------------------

def test_fill_missing_context_leaves_known_arguments_untouched():
    arguments = {"product_name": "ring", "material": "gold"}
    result = fill_missing_context("session-1", arguments)
    assert result == {"product_name": "ring", "material": "gold"}


def test_fill_missing_context_resolves_unknown_from_the_session(monkeypatch):
    from services import memory
    store = SessionStore()
    store.set("session-1", "material", "gold")
    monkeypatch.setattr(memory, "_store", store)

    arguments = {"product_name": "ring", "material": "unknown"}
    result = fill_missing_context("session-1", arguments)

    assert result["material"] == "gold"


def test_fill_missing_context_leaves_unknown_when_session_has_nothing(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    arguments = {"product_name": "unknown", "material": "unknown"}
    result = fill_missing_context("new-session", arguments)

    # Nothing to fill from, so it stays "unknown" rather than inventing a guess
    assert result["product_name"] == "unknown"
    assert result["material"] == "unknown"


def test_remember_context_only_stores_resolved_values(monkeypatch):
    from services import memory
    store = SessionStore()
    monkeypatch.setattr(memory, "_store", store)

    remember_context("session-1", {"product_name": "ring", "material": "unknown"})

    assert store.get("session-1", "product_name") == "ring"
    assert store.get("session-1", "material") is None


def test_delivery_option_round_trips_across_turns(monkeypatch):
    # A customer who already said "deliver to Accra" earlier shouldn't
    # have to repeat it when they get to actually placing the order.
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    remember_context("session-1", {"product_name": "ring", "delivery_option": "accra_rider"})
    result = fill_missing_context("session-1", {"product_name": "ring", "delivery_option": "unknown"})

    assert result["delivery_option"] == "accra_rider"


def test_remember_context_stores_a_non_string_quantity(monkeypatch):
    # The LLM returns quantity as a real JSON number, not a string --
    # the old `isinstance(value, str)` gate in remember_context silently
    # dropped it every time, so a customer's "2" never survived to the
    # next turn (confirmed live, 2026-08-12; this is the actual root
    # cause of that bug).
    from services import memory
    store = SessionStore()
    monkeypatch.setattr(memory, "_store", store)

    remember_context("session-1", {"product_name": "ring", "quantity": 2})

    assert store.get("session-1", "quantity") == 2


def test_remember_context_does_not_clear_a_remembered_value_when_key_is_absent(monkeypatch):
    # A tool call that doesn't take `delivery_address` at all (e.g. a
    # plain price lookup) must not wipe out an address remembered from
    # an earlier turn -- only an explicit "unknown" should ever clear it.
    from services import memory
    store = SessionStore()
    store.set("session-1", "delivery_address", "12 Cantonments Road, Accra")
    monkeypatch.setattr(memory, "_store", store)

    remember_context("session-1", {"product_name": "ring", "material": "gold"})

    assert store.get("session-1", "delivery_address") == "12 Cantonments Road, Accra"


def test_delivery_address_round_trips_across_turns(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    remember_context("session-1", {"delivery_address": "12 Cantonments Road, Accra"})
    result = fill_missing_context("session-1", {"delivery_address": "unknown"})

    assert result["delivery_address"] == "12 Cantonments Road, Accra"


# ---------------------------------------------------------------------
# get_order_draft
# ---------------------------------------------------------------------

def test_get_order_draft_returns_none_when_nothing_given_yet():
    assert get_order_draft("session-never-seen") is None


def test_get_order_draft_reflects_partial_progress(monkeypatch):
    # "I'd like to place an order" -> propose_order asked "how many?" --
    # this is what the prompt needs to recognise the customer's next
    # bare "2" as continuing that, not starting something new.
    from services import memory
    store = SessionStore()
    monkeypatch.setattr(memory, "_store", store)

    remember_context("session-1", {"product_name": "Custom Leaf White Gold Necklace, 20g", "material": "14k"})

    draft = get_order_draft("session-1")

    assert draft["product_name"] == "Custom Leaf White Gold Necklace, 20g"
    assert draft["material"] == "14k"
    assert draft["quantity"] is None
    assert draft["delivery_address"] is None
    assert draft["delivery_option"] is None


# ---------------------------------------------------------------------
# pending_intent
# ---------------------------------------------------------------------

def test_get_pending_intent_returns_none_when_nothing_set():
    assert get_pending_intent("session-never-seen") is None


def test_pending_intent_round_trips(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_pending_intent("session-1", "get_product_price")

    assert get_pending_intent("session-1") == "get_product_price"


def test_pending_intent_can_be_cleared(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_pending_intent("session-1", "get_product_price")
    set_pending_intent("session-1", None)

    assert get_pending_intent("session-1") is None


# ---------------------------------------------------------------------
# last_action_outcome
# ---------------------------------------------------------------------

def test_get_last_action_outcome_returns_none_when_nothing_set():
    assert get_last_action_outcome("session-never-seen") is None


def test_last_action_outcome_round_trips(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())
    outcome = {"action": "propose_order", "customer_safe_explanation": "reason"}

    set_last_action_outcome("session-1", outcome)

    assert get_last_action_outcome("session-1") == outcome


def test_last_action_outcome_can_be_cleared(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_last_action_outcome("session-1", {"action": "propose_order", "customer_safe_explanation": "reason"})
    set_last_action_outcome("session-1", None)

    assert get_last_action_outcome("session-1") is None


# ---------------------------------------------------------------------
# last_priced_product
# ---------------------------------------------------------------------

def test_get_last_priced_product_returns_none_when_nothing_set():
    assert get_last_priced_product("session-never-seen") is None


def test_last_priced_product_round_trips(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_last_priced_product("session-1", "Big White Crown Stone Gold Ring, 14g")

    assert get_last_priced_product("session-1") == "Big White Crown Stone Gold Ring, 14g"


def test_last_priced_product_can_be_cleared(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_last_priced_product("session-1", "Big White Crown Stone Gold Ring, 14g")
    set_last_priced_product("session-1", None)

    assert get_last_priced_product("session-1") is None


def test_last_priced_product_sessions_stay_isolated(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_last_priced_product("session-a", "Ring A")
    set_last_priced_product("session-b", "Ring B")

    assert get_last_priced_product("session-a") == "Ring A"
    assert get_last_priced_product("session-b") == "Ring B"


def test_context_round_trips_across_two_simulated_turns(monkeypatch):
    # Arrange: turn one resolves and remembers gold; turn two arrives
    # with "unknown" and should recover it from the session
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    remember_context("session-1", {"product_name": "ring", "material": "gold"})
    result = fill_missing_context("session-1", {"product_name": "ring", "material": "unknown"})

    assert result["material"] == "gold"
