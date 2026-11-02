"""
Per-turn guardrail events, kept on the current thread.

router.py opens a turn with begin(), anything during the turn can call
add_event(), and router.py reads events() when it logs the turn
(KASAFLOW_TURN_METRICS). The eval runner reads the same events, so a
guardrail that fired in production and one that fired in an eval are
recorded the same way.

A turn runs start to finish on one thread while holding the session
lock (router.route_customer()), so thread-local storage is enough.
"""

from __future__ import annotations

import threading

_local = threading.local()


def begin() -> None:
    _local.events = []


def add_event(name: str, **data) -> None:
    events = getattr(_local, "events", None)
    if events is None:
        return
    events.append({"event": name, **data})


def events() -> list[dict]:
    return list(getattr(_local, "events", None) or [])


def end() -> None:
    _local.events = None
