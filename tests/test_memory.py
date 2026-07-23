"""
Unit tests for services/memory.py -- the per-session store, and the
fill_missing_context / remember_context helpers router.py calls.

These are pure, no mocking of the OpenAI client involved: they test that
sessions stay isolated from each other and that entries expire, which is
the exact bug the old shared-dict version had.
"""

from services.memory import SessionStore, fill_missing_context, remember_context


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


def test_context_round_trips_across_two_simulated_turns(monkeypatch):
    # Arrange: turn one resolves and remembers gold; turn two arrives
    # with "unknown" and should recover it from the session
    from services import memory
    monkeypatch.setattr(memory, "_store", SessionStore())

    remember_context("session-1", {"product_name": "ring", "material": "gold"})
    result = fill_missing_context("session-1", {"product_name": "ring", "material": "unknown"})

    assert result["material"] == "gold"
