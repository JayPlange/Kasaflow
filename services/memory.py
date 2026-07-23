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
# repeating themselves. Not every tool takes both, callers only fill and
# save whichever keys are actually present in that tool's arguments.
_REMEMBERED_KEYS = ("product_name", "material")


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
    """Persist any resolved (non-"unknown") arguments for later turns in this session."""
    for key in _REMEMBERED_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and not _is_unknown(value):
            _store.set(session_id, key, value)


def get_session_store() -> SessionStore:
    """Exposed so tests (and, later, a healthcheck or admin endpoint) can
    inspect the store without reaching into the module-level `_store`
    directly."""
    return _store
