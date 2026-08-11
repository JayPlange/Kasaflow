"""
Per-session conversation memory.

Wired in now. The previous version of this file kept a single dict
shared by every request the process handled, which meant two different
customers' in-flight requests could silently overwrite each other's
context -- unsafe under any real concurrency, so it was deliberately
left unregistered as a tool.

This version keys memory by a session_id the caller provides (or that
main.py generates for them on their first request and hands back), so
one customer's context can never leak into another's. Sessions expire
after a period of inactivity so a long-running process doesn't
accumulate state forever.
"""

import threading
import time

_SESSION_TTL_SECONDS = 30 * 60  # 30 minutes of inactivity


class SessionStore:
    """Thread-safe, per-session key/value store with TTL-based expiry."""

    def __init__(self, ttl_seconds: float = _SESSION_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}  # session_id -> {"data": {...}, "expires_at": float}

    def _is_expired(self, entry: dict, now: float) -> bool:
        return entry["expires_at"] < now

    def get(self, session_id: str, key: str, default=None):
        now = time.monotonic()
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None or self._is_expired(entry, now):
                return default
            return entry["data"].get(key, default)

    def set(self, session_id: str, key: str, value) -> None:
        now = time.monotonic()
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None or self._is_expired(entry, now):
                entry = {"data": {}, "expires_at": now + self._ttl}
                self._sessions[session_id] = entry
            entry["data"][key] = value
            entry["expires_at"] = now + self._ttl

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def active_session_count(self) -> int:
        now = time.monotonic()
        with self._lock:
            return sum(1 for entry in self._sessions.values() if not self._is_expired(entry, now))


# Module-level store shared by the app process. Safe now because every
# access is keyed by session_id and guarded by a lock, unlike the old
# shared dict this replaced.
_store = SessionStore()

# Arguments worth remembering across turns within the same session, so
# "how much is shipping for that one" can resolve without the customer
# repeating themselves. Not every tool takes all of these, callers only
# fill and save whichever keys are actually present in that tool's
# arguments. "category" lets a follow-up like "what about in 14k?"
# after "what rings do you have" remember "Rings" without repeating it.
# "delivery_option" means a customer who already said "deliver to Accra"
# earlier in the conversation doesn't have to repeat it when they get to
# actually placing the order. "quantity" and "delivery_address" close a
# real gap propose_order's own clarifying questions ("How many would
# you like?", "What address?") otherwise fell into: without these two
# remembered too, a customer's bare "2" or bare address in reply never
# stuck, and the next turn re-asked the same question forever (confirmed
# live, 2026-08-12) -- see get_order_draft() below for the other half of
# this fix.
_REMEMBERED_KEYS = ("product_name", "material", "category", "delivery_option", "quantity", "delivery_address")

# Which of the remembered keys matter for an in-progress order, and the
# words used to describe each to the LLM (see get_order_draft() and
# llm.py's _order_draft_state_line()).
_ORDER_DRAFT_KEYS = ("product_name", "material", "quantity", "delivery_address", "delivery_option")


def _is_unknown(value) -> bool:
    return isinstance(value, str) and value.strip().lower() == "unknown"


def fill_missing_context(session_id: str, arguments: dict) -> dict:
    """Resolve any argument the model marked "unknown" using this session's last-known values.

    The prompt in services/llm.py tells the model to return "unknown"
    rather than guess when it cannot determine a product or material
    from the message alone -- this is what turns that sentinel into an
    actual resolved value from earlier in the same conversation.

    Returns a new dict rather than mutating the one passed in. The
    caller's arguments dict may be referenced elsewhere (tests and
    callers that snapshot the LLM's raw output are the obvious cases),
    so mutating it in place risks a session's resolved value silently
    leaking into code that still expects the original "unknown".
    """
    resolved = dict(arguments)
    for key in _REMEMBERED_KEYS:
        if key in resolved and _is_unknown(resolved[key]):
            remembered = _store.get(session_id, key)
            if remembered is not None:
                resolved[key] = remembered
    return resolved


def remember_context(session_id: str, arguments: dict) -> None:
    """Persist any resolved (non-"unknown") arguments for later turns in this session.

    Deliberately not gated on `isinstance(value, str)` -- quantity comes
    back from the LLM as a real JSON number (an int), not a string, and
    the old `isinstance(value, str)` check silently dropped it every
    time, which is exactly why quantity never used to survive to the
    next turn. Only skips a key that's genuinely absent from this call's
    arguments (a different tool that doesn't take it) or explicitly
    "unknown" -- never overwrites a real remembered value with a
    stale/missing one.
    """
    for key in _REMEMBERED_KEYS:
        if key not in arguments:
            continue
        value = arguments[key]
        if not _is_unknown(value):
            _store.set(session_id, key, value)


def get_order_draft(session_id: str) -> dict | None:
    """Snapshot of this session's remembered order-relevant slots.

    Exists for llm.py's prompt (_order_draft_state_line()): the model
    otherwise sees only the customer's current message, no conversation
    history, so once propose_order has asked "How many would you like?"
    a bare "2" in reply is unresolvable on its own -- the model has no
    way to know that's an answer to a question it can't see. Handing it
    this snapshot lets it recognise a short reply as continuing an order
    already in progress, rather than guessing at a different tool
    entirely (confirmed live, 2026-08-12: it did guess wrong).

    Returns None when nothing order-relevant has been given yet, so the
    prompt can skip this section entirely for a fresh conversation.
    """
    draft = {key: _store.get(session_id, key) for key in _ORDER_DRAFT_KEYS}
    if not any(value is not None for value in draft.values()):
        return None
    return draft


def get_session_store() -> SessionStore:
    """Exposed so tests (and, later, a healthcheck or admin endpoint) can
    inspect the store without reaching into the module-level `_store`
    directly."""
    return _store
