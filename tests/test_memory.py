"""
Unit tests for services/memory.py -- the per-session store, and the
fill_missing_context / remember_context helpers router.py calls.

These are pure, no mocking of the OpenAI client involved: they test that
sessions stay isolated from each other and that entries expire, which is
the exact bug the old shared-dict version had.
"""

from services.memory import (
    AWAITING_FIELDS,
    SessionStore,
    fill_missing_context,
    get_awaiting_field,
    get_last_action_outcome,
    get_last_presented_products,
    get_last_priced_product,
    get_order_draft,
    get_pending_intent,
    increment_weight_ask_count,
    is_awaiting_confirmation,
    remember_context,
    set_awaiting_confirmation,
    set_awaiting_field,
    set_last_action_outcome,
    set_last_presented_products,
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


def test_fill_missing_context_does_not_carry_product_specific_fields_to_a_different_product(monkeypatch):
    # Webb, 2026-08-20, check #6: "order Product A in 14k, 6 of them...
    # actually I'll take Product B". Product B's own missing fields must
    # NOT get silently completed from Product A's memory -- material and
    # quantity describe Product A's order specifically, not a fact still
    # true for a different item.
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())
    remember_context("session-1", {
        "product_name": "Product A", "material": "14k", "quantity": 6,
        "delivery_address": "Accra", "delivery_option": "accra_rider",
    })

    result = fill_missing_context("session-1", {
        "product_name": "Product B", "material": "unknown", "quantity": "unknown",
        "delivery_address": "unknown", "delivery_option": "unknown",
    })

    assert result["product_name"] == "Product B"
    assert result["material"] == "unknown"
    assert result["quantity"] == "unknown"


def test_fill_missing_context_still_carries_order_level_delivery_fields_across_a_product_switch(monkeypatch):
    # Webb, 2026-08-20, check #6 follow-up: unlike karat/quantity,
    # delivery_address and delivery_option are facts about the customer's
    # order as a whole, not the specific item -- "deliver to Accra, rider
    # delivery" doesn't stop being true just because the customer swapped
    # which ring they want. These must keep resolving from memory even
    # when the product itself has changed, so the customer isn't asked to
    # restate an address they already gave earlier in the same
    # conversation.
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())
    remember_context("session-1", {
        "product_name": "Product A", "material": "14k", "quantity": 6,
        "delivery_address": "Accra", "delivery_option": "accra_rider",
    })

    result = fill_missing_context("session-1", {
        "product_name": "Product B", "material": "unknown", "quantity": "unknown",
        "delivery_address": "unknown", "delivery_option": "unknown",
    })

    assert result["delivery_address"] == "Accra"
    assert result["delivery_option"] == "accra_rider"


def test_fill_missing_context_does_not_backfill_delivery_option_when_the_address_changes(monkeypatch):
    # Webb, 2026-08-20, live: a customer moved from an Accra order to a
    # Kumasi one and later gave a brand-new Accra-area address (Kasoa) --
    # every reply kept saying "doesn't match our usual rider delivery
    # within Kumasi zone", because delivery_option ("kumasi_rider") was
    # being silently carried over from the OLD address instead of
    # re-derived for the new one. delivery_option is a property of the
    # address, not an independent fact -- once delivery_address genuinely
    # changes, delivery_option must come back "unknown" so
    # order_tool.propose_order()'s own infer_delivery_option() call runs
    # fresh, rather than being skipped because a stale-but-valid key was
    # already sitting there.
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())
    remember_context("session-1", {
        "product_name": "Ring", "delivery_address": "Kumasi", "delivery_option": "kumasi_rider",
    })

    result = fill_missing_context("session-1", {
        "product_name": "Ring", "delivery_address": "Kasoa", "delivery_option": "unknown",
    })

    assert result["delivery_address"] == "Kasoa"
    assert result["delivery_option"] == "unknown"


def test_fill_missing_context_keeps_an_explicitly_restated_delivery_option_on_an_address_change(monkeypatch):
    # The guard above must not overreach -- if the customer explicitly
    # states BOTH a new address and a delivery option in the same
    # message, that stated option is this call's own answer, not a
    # leftover, and must be used as-is.
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())
    remember_context("session-1", {
        "product_name": "Ring", "delivery_address": "Kumasi", "delivery_option": "kumasi_rider",
    })

    result = fill_missing_context("session-1", {
        "product_name": "Ring", "delivery_address": "East Legon", "delivery_option": "accra_rider",
    })

    assert result["delivery_address"] == "East Legon"
    assert result["delivery_option"] == "accra_rider"


def test_remember_context_clears_delivery_option_when_the_address_changes(monkeypatch):
    # Write-side counterpart: once the OLD delivery_option is cleared
    # from the store on a genuine address change, it must not just get
    # backfilled right back in on the very next turn either.
    from services import memory
    store = SessionStore()
    monkeypatch.setattr(memory, "_store", store)
    remember_context("session-1", {
        "product_name": "Ring", "delivery_address": "Kumasi", "delivery_option": "kumasi_rider",
    })

    remember_context("session-1", {
        "product_name": "Ring", "delivery_address": "Kasoa", "delivery_option": "unknown",
    })

    assert store.get("session-1", "delivery_address") == "Kasoa"
    assert store.get("session-1", "delivery_option") is None


def test_fill_missing_context_still_fills_for_the_same_product(monkeypatch):
    # The guard must not become overzealous -- continuing the SAME
    # product's order (the overwhelmingly common case) must still work
    # exactly as before.
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())
    remember_context("session-1", {
        "product_name": "Ring", "material": "14k", "quantity": 2, "delivery_address": "Accra",
    })
    result = fill_missing_context("session-1", {
        "product_name": "Ring", "material": "unknown", "quantity": "unknown", "delivery_address": "unknown",
    })

    assert result["material"] == "14k"
    assert result["quantity"] == 2
    assert result["delivery_address"] == "Accra"


def test_fill_missing_context_still_fills_when_no_product_is_named_at_all(monkeypatch):
    # A bare reply ("14k") never restates product_name at all -- the
    # arguments dict for a call like this has no "product_name" key at
    # all, so the guard must not mistake that absence for "a different
    # product" and must still resolve normally, exactly like continuing
    # the same product.
    from services import memory
    store = SessionStore()
    monkeypatch.setattr(memory, "_store", store)
    remember_context("session-1", {"product_name": "Ring", "material": "12k", "quantity": 2})

    result = fill_missing_context("session-1", {"material": "unknown"})

    assert result["material"] == "12k"


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


def test_remember_context_clears_old_product_specific_fields_on_a_product_switch(monkeypatch):
    # Write-side counterpart to the fill_missing_context tests above: once
    # a call names a genuinely different product, the OLD product's
    # material/quantity must not keep sitting in the store, or
    # get_order_draft() (via _describe_order_corrections()) would still
    # attribute Product A's karat/quantity to Product B once Product B's
    # own product_name lands. delivery_address/delivery_option are
    # order-level, not item-level, so they must survive untouched.
    from services import memory
    store = SessionStore()
    monkeypatch.setattr(memory, "_store", store)
    remember_context("session-1", {
        "product_name": "Product A", "material": "14k", "quantity": 6,
        "delivery_address": "Accra", "delivery_option": "accra_rider",
    })

    remember_context("session-1", {
        "product_name": "Product B", "material": "unknown", "quantity": "unknown",
        "delivery_address": "unknown", "delivery_option": "unknown",
    })

    assert store.get("session-1", "product_name") == "Product B"
    assert store.get("session-1", "material") is None
    assert store.get("session-1", "quantity") is None
    assert store.get("session-1", "delivery_address") == "Accra"
    assert store.get("session-1", "delivery_option") == "accra_rider"


def test_remember_context_keeps_explicitly_restated_fields_on_a_product_switch(monkeypatch):
    # The clearing above must not become overzealous: "actually I'll take
    # Product B, still 14k and 6 pieces" explicitly restates material and
    # quantity in the SAME call that names the new product -- those
    # values are this call's own arguments, not leftovers from Product A,
    # and must be stored normally rather than cleared.
    from services import memory
    store = SessionStore()
    monkeypatch.setattr(memory, "_store", store)
    remember_context("session-1", {
        "product_name": "Product A", "material": "14k", "quantity": 6,
    })

    remember_context("session-1", {
        "product_name": "Product B", "material": "14k", "quantity": 6,
    })

    assert store.get("session-1", "product_name") == "Product B"
    assert store.get("session-1", "material") == "14k"
    assert store.get("session-1", "quantity") == 6


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
# session_lock -- per-session turn lock (see router.route_customer())
# ---------------------------------------------------------------------

def test_session_lock_returns_the_same_lock_object_for_the_same_session():
    store = SessionStore()
    assert store.session_lock("session-1") is store.session_lock("session-1")


def test_session_lock_returns_different_lock_objects_for_different_sessions():
    store = SessionStore()
    assert store.session_lock("session-1") is not store.session_lock("session-2")


# ---------------------------------------------------------------------
# awaiting_confirmation -- P0 fix: a bare "yes" must only confirm the
# session's pending order when proposing it was the last thing that
# happened, not whenever a pending order merely exists somewhere.
# ---------------------------------------------------------------------

def test_is_awaiting_confirmation_defaults_to_false_when_nothing_set():
    assert is_awaiting_confirmation("session-never-seen") is False


def test_awaiting_confirmation_round_trips(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_awaiting_confirmation("session-1", True)

    assert is_awaiting_confirmation("session-1") is True


def test_awaiting_confirmation_can_be_cleared(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_awaiting_confirmation("session-1", True)
    set_awaiting_confirmation("session-1", False)

    assert is_awaiting_confirmation("session-1") is False


# ---------------------------------------------------------------------
# awaiting_field -- P0.4. "product_name" joined the AWAITING_FIELDS set
# 2026-08-30 (task #60, Webb): before this, propose_order's missing-item
# error was the one of its five field checks that never tagged
# awaiting_field at all, so set_awaiting_field() had never been asked to
# store it -- confirming it doesn't raise (AWAITING_FIELDS was previously
# a closed set that explicitly excluded it) is the actual regression this
# guards, not just a round-trip.
# ---------------------------------------------------------------------

def test_awaiting_fields_includes_product_name():
    assert "product_name" in AWAITING_FIELDS


def test_awaiting_field_product_name_round_trips(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_awaiting_field("session-1", "product_name")

    assert get_awaiting_field("session-1") == "product_name"


def test_set_awaiting_field_still_rejects_an_unknown_value(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    try:
        set_awaiting_field("session-1", "not_a_real_field")
        assert False, "expected ValueError"
    except ValueError:
        pass


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


# ---------------------------------------------------------------------
# weight_ask_count -- Webb, 2026-08-30 live transcript: three follow-ups
# about the same product's weight ("that's the weight?", "is that really
# 1g?", "how many grams is that?") all came back character-for-character
# identical. response_formatter.py keys its phrasing variant off this.
# ---------------------------------------------------------------------

def test_increment_weight_ask_count_starts_at_one(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    assert increment_weight_ask_count("session-1") == 1


def test_increment_weight_ask_count_climbs_on_each_call(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    assert increment_weight_ask_count("session-1") == 1
    assert increment_weight_ask_count("session-1") == 2
    assert increment_weight_ask_count("session-1") == 3


def test_increment_weight_ask_count_sessions_stay_isolated(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    increment_weight_ask_count("session-a")
    increment_weight_ask_count("session-a")
    increment_weight_ask_count("session-b")

    assert increment_weight_ask_count("session-a") == 3
    assert increment_weight_ask_count("session-b") == 2


# ---------------------------------------------------------------------
# last_presented_products
# ---------------------------------------------------------------------

def test_get_last_presented_products_returns_none_when_nothing_set():
    assert get_last_presented_products("session-never-seen") is None


def test_last_presented_products_round_trips_shape(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    groups = [
        ("Gold Hoop Earrings, 5g", [{"category": "Earrings", "material": "14k"}]),
        ("Silver Chain Necklace, 20g", [{"category": "Necklaces", "material": "18k"}]),
    ]
    set_last_presented_products("session-1", groups)

    stored = get_last_presented_products("session-1")
    assert stored == {
        "generation": 1,
        "items": [
            {"position": 1, "product_name": "Gold Hoop Earrings, 5g", "category": "Earrings"},
            {"position": 2, "product_name": "Silver Chain Necklace, 20g", "category": "Necklaces"},
        ],
    }


def test_last_presented_products_omits_price_and_material(monkeypatch):
    # Webb, 2026-08-25: the stored entry must never carry price/karat as
    # authoritative state -- only position/product_name/category.
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    groups = [("Ring", [{"category": "Rings", "material": "14k", "price": "1200"}])]
    set_last_presented_products("session-1", groups)

    item = get_last_presented_products("session-1")["items"][0]
    assert set(item.keys()) == {"position", "product_name", "category"}


def test_last_presented_products_generation_increments_on_each_write(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_last_presented_products("session-1", [("Ring", [{"category": "Rings"}])])
    set_last_presented_products("session-1", [("Necklace", [{"category": "Necklaces"}])])

    assert get_last_presented_products("session-1")["generation"] == 2


def test_last_presented_products_overwritten_wholesale_by_next_call(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_last_presented_products("session-1", [("Ring", [{"category": "Rings"}])])
    set_last_presented_products("session-1", [("Necklace", [{"category": "Necklaces"}])])

    items = get_last_presented_products("session-1")["items"]
    assert len(items) == 1
    assert items[0]["product_name"] == "Necklace"


def test_last_presented_products_not_cleared_by_clear_order_state(monkeypatch):
    from services.memory import clear_order_state
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_last_presented_products("session-1", [("Ring", [{"category": "Rings"}])])
    clear_order_state("session-1")

    assert get_last_presented_products("session-1") is not None


def test_last_presented_products_sessions_stay_isolated(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_last_presented_products("session-a", [("Ring A", [{"category": "Rings"}])])
    set_last_presented_products("session-b", [("Ring B", [{"category": "Rings"}])])

    assert get_last_presented_products("session-a")["items"][0]["product_name"] == "Ring A"
    assert get_last_presented_products("session-b")["items"][0]["product_name"] == "Ring B"


def test_last_presented_products_handles_missing_category(monkeypatch):
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    set_last_presented_products("session-1", [("Ring", [{}])])

    assert get_last_presented_products("session-1")["items"][0]["category"] is None


def test_context_round_trips_across_two_simulated_turns(monkeypatch):
    # Arrange: turn one resolves and remembers gold; turn two arrives
    # with "unknown" and should recover it from the session
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    remember_context("session-1", {"product_name": "ring", "material": "gold"})
    result = fill_missing_context("session-1", {"product_name": "ring", "material": "unknown"})

    assert result["material"] == "gold"
