"""
Handing a conversation to a person.

Once a session is escalated, the assistant stops acting on it: no model
call, no tools, no order writes. Every later message from that customer
gets a short holding reply, and staff are notified on WhatsApp
(STAFF_NOTIFICATION_PHONE) with the reason and enough context to pick
the conversation up. The handoff lasts as long as the session does
(memory.py's inactivity TTL), so a customer who goes quiet and comes
back much later starts fresh.

What triggers an escalation (each one is a reason code below):

- customer_request: the customer asks for a person. Detected in code
  (wants_human()) before the model runs, so "can I talk to someone"
  never depends on the model choosing the right tool. The model can
  also choose escalate_to_human itself for requests the regex misses.
- model_requested: the model chose escalate_to_human (complaints,
  disputes, anything it judges it can't handle).
- repeated_failures: this many turns in a row failed outright
  (settings.escalation_failure_threshold).
- repeated_manipulation: this many flagged messages in one session
  (settings.escalation_injection_threshold); see guardrails.py.
- order_review: an order is over the auto-confirm limits
  (settings.order_auto_confirm_max_total / order_max_quantity).
- total_mismatch: the store's order total didn't match the total the
  customer agreed to; see order_tool.confirm_order().

Notification is best effort, the same posture as order_tool.py's staff
notifications: a failed WhatsApp send is logged loudly but never turns
into an error the customer sees. The holding reply is honest either way
("our team has been notified" is only said when a send was attempted).
"""

from __future__ import annotations

import logging
import re
import time

from app.config import settings
from services.memory import _SESSION_TTL_SECONDS, get_session_store
from services.whatsapp_client import WhatsAppError, send_text_message

logger = logging.getLogger(__name__)

_ESCALATION_KEY = "escalation"
_FAILURE_COUNT_KEY = "consecutive_failures"
_FLAGGED_COUNT_KEY = "flagged_message_count"

REASON_LABELS = {
    "customer_request": "customer asked for a person",
    "model_requested": "assistant judged it needs a person",
    "repeated_failures": "assistant failed several turns in a row",
    "repeated_manipulation": "repeated messages trying to manipulate the assistant",
    "order_review": "order is over the auto-confirm limit",
    "total_mismatch": "store total differs from the quoted total",
}

_WANTS_HUMAN_RE = re.compile(
    r"\b(speak|talk|chat)\s+(to|with)\s+(a|an|the|your|some|someone|somebody)?\s*"
    r"(human|person|real person|agent|manager|owner|staff|someone|somebody|"
    r"customer (service|care|support)|representative|rep|salesperson)\b"
    r"|\b(real|actual|live)\s+(person|human|agent)\b"
    r"|\b(human|agent|person)\s+please\b"
    r"|\b(connect|transfer|put)\s+me\s+(to|through to|with)\s+(a|an|the|your|someone)\b"
    r"|\bcall\s+me\b",
    re.IGNORECASE,
)


def wants_human(message: str | None) -> bool:
    return bool(message) and bool(_WANTS_HUMAN_RE.search(message))


def get_escalation(session_id: str) -> dict | None:
    return get_session_store().get(session_id, _ESCALATION_KEY)


def is_escalated(session_id: str) -> bool:
    return get_escalation(session_id) is not None


def escalate(session_id: str, reason: str, detail: str | None = None, context: dict | None = None) -> dict:
    """Mark the session as handed off and notify staff once.

    Returns the result shape response_formatter.py turns into the
    customer's reply: {"human_handoff": {...}}. Calling it again for an
    already-escalated session does not notify again unless
    settings.escalation_renotify_seconds have passed.
    """
    store = get_session_store()
    now = time.time()
    existing = store.get(session_id, _ESCALATION_KEY)
    if existing is not None:
        last = existing.get("notified_at") or 0
        if now - last < settings.escalation_renotify_seconds:
            return {"human_handoff": {"reason": existing["reason"], "already_escalated": True,
                                      "staff_notified": existing.get("staff_notified", False)}}

    notified = _notify_staff(session_id, reason, detail, context)
    record = {
        "reason": reason,
        "detail": detail,
        "at": existing["at"] if existing else now,
        "notified_at": now,
        "staff_notified": notified,
    }
    store.set(session_id, _ESCALATION_KEY, record)
    logger.info("KASAFLOW_ESCALATION session=%s reason=%s notified=%s", session_id, reason, notified)
    return {"human_handoff": {"reason": reason, "already_escalated": existing is not None,
                              "staff_notified": notified}}


def holding_result(session_id: str, message: str) -> dict:
    """What an escalated session gets instead of a normal turn.

    Re-notifies staff (rate limited inside escalate()) so a customer who
    keeps writing isn't silently ignored if the first notification was
    missed.
    """
    current = get_escalation(session_id) or {}
    return escalate(
        session_id,
        current.get("reason", "customer_request"),
        detail=f"Customer wrote again: {message[:200]}",
    )


def _notify_staff(session_id: str, reason: str, detail: str | None, context: dict | None) -> bool:
    if not settings.staff_notification_phone:
        logger.warning(
            "STAFF_NOTIFICATION_PHONE not configured -- session %s needs a person (%s) but nobody "
            "was notified.",
            session_id, reason,
        )
        return False

    lines = [
        "KasaFlow handoff: a customer needs a person.",
        f"Reason: {REASON_LABELS.get(reason, reason)}",
        f"Customer WhatsApp: {session_id}",
    ]
    if detail:
        lines.append(f"Detail: {detail[:300]}")
    pending = (context or {}).get("pending_order")
    if pending:
        lines.append(
            f"Pending order: {pending.get('quantity')} x {pending.get('material')} "
            f"{pending.get('product')}, GH₵{float(pending.get('total') or 0):,.2f}"
        )
    lines.append(
        "The assistant will not act on this conversation until it has been quiet for "
        f"about {_SESSION_TTL_SECONDS // 60} minutes."
    )
    try:
        send_text_message(settings.staff_notification_phone, "\n".join(lines))
        return True
    except WhatsAppError as e:
        logger.error("Failed to notify staff about handoff for session %s: %s", session_id, e)
        return False


# --- Counters feeding the automatic triggers --------------------------------

def record_failure(session_id: str) -> int:
    store = get_session_store()
    count = (store.get(session_id, _FAILURE_COUNT_KEY) or 0) + 1
    store.set(session_id, _FAILURE_COUNT_KEY, count)
    return count


def reset_failures(session_id: str) -> None:
    get_session_store().set(session_id, _FAILURE_COUNT_KEY, 0)


def record_flagged_message(session_id: str) -> int:
    store = get_session_store()
    count = (store.get(session_id, _FLAGGED_COUNT_KEY) or 0) + 1
    store.set(session_id, _FLAGGED_COUNT_KEY, count)
    return count


def over_failure_threshold(count: int) -> bool:
    threshold = settings.escalation_failure_threshold
    return threshold > 0 and count >= threshold


def over_flag_threshold(count: int) -> bool:
    threshold = settings.escalation_injection_threshold
    return threshold > 0 and count >= threshold


def escalate_to_human(session_id: str, reason: str | None = None) -> dict:
    """Tool entry point for the model's escalate_to_human choice."""
    from services.order_tool import get_pending_order_summary

    return escalate(
        session_id, "model_requested", detail=reason or None,
        context={"pending_order": get_pending_order_summary(session_id)},
    )
