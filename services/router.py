"""
Top-level orchestration: customer message -> tool selection -> tool
execution -> result. This is the only place that decides what the
customer sees when something goes wrong upstream.
"""

import logging

from services.llm import ToolSelectionError, understand_customer
from services.memory import (
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
from services.order_tool import get_pending_order_summary
from services.tool_executor import ToolExecutionError, execute_tool

logger = logging.getLogger(__name__)

# propose_order/confirm_order/cancel_order (services/order_tool.py) all
# need to know which customer's session they're acting on, but that's
# never something the LLM decides or the customer states -- it comes
# from the channel (WhatsApp) this message arrived on. Every other
# registered tool's **kwargs contract is exactly whatever the LLM
# returned, nothing more, so session_id is injected only for these three
# by name rather than added to every tool call: passing an unexpected
# session_id kwarg to any of the other five would raise inside
# tool_executor.py's TypeError handler.
_SESSION_AWARE_TOOLS = {"propose_order", "confirm_order", "cancel_order"}

# converse (services/llm.py) isn't a real registered tool -- there's no
# deterministic business logic behind it, no lookup, nothing to execute.
# The LLM writes the actual customer-facing reply itself as part of the
# same tool-selection call, precisely because there's no business fact
# involved that a tool would need to ground it in. Handled entirely here,
# before execute_tool()/tool_registry.py ever see it: registering it as a
# "tool" that just echoes back its own argument would be a pointless
# round-trip through machinery built for looking things up, and would
# wrongly make it eligible for fill_missing_context()/remember_context()
# below, which exist to resolve/save business arguments (product,
# material, delivery address, ...) -- a converse reply has none of those,
# and a stray "reply" key must never leak into or overwrite that state.
_CONVERSATION_TOOL = "converse"
_CONVERSATION_FALLBACK_REPLY = "Hey! How can I help you today?"

# The two tools whose only failure mode relevant here is "customer wants
# this, but hasn't named a product yet" -- see memory.set_pending_intent()
# and llm.py's _pending_intent_state_line() for why that specific gap
# needs to be tracked across turns.
_PENDING_INTENT_TOOLS = {"get_product_price", "generate_quote"}

# Both return {"product": <resolved catalogue name>, ...} on a genuine
# match (see product_tool.get_product_price() and
# quote_service.generate_quote()) -- the same two tools tracked by
# _PENDING_INTENT_TOOLS above, but for the opposite case: a lookup that
# DID resolve. See memory.set_last_priced_product() and llm.py's
# _last_priced_product_state_line() for why a bare karat-only follow-up
# needs this remembered.
_PRICING_TOOLS = {"get_product_price", "generate_quote"}

# A genuine category browse means the topic has moved on from one
# specific priced item -- see memory.set_last_priced_product()'s
# docstring for why this specifically clears rather than leaves the
# old value in place.
_RECOMMEND_TOOL = "recommend_products"

# propose_order is the only tool whose arguments can genuinely "correct"
# something the customer already gave earlier in the same order -- see
# _describe_order_corrections() below.
_ORDER_TOOL = "propose_order"

_ORDER_CORRECTION_FIELDS = {
    "product_name": "the item",
    "material": "the karat",
    "quantity": "the quantity",
    "delivery_address": "the delivery address",
}


def _found_nothing(result: dict | None) -> bool:
    """True when a tool ran successfully but didn't actually find what
    the customer asked about: get_product_price's bare None, an empty
    recommend_products category, or generate_quote's "couldn't find that
    product" message.

    Exists because execute_tool() only raises on a genuine failure --
    an empty/no-match result is a normal, successful return, so without
    this check remember_context() below would happily save a category
    or product that produced nothing. A customer asking for something
    genuinely unstocked (say "bracelets") would then have that dead
    category silently remembered, trapping every vague follow-up
    ("show me something else", "yeah lemme see") in the same no-match
    state until they explicitly named a real category again -- exactly
    the "never learn an invalid value" promise below is meant to rule
    out, but previously didn't for this specific case."""
    if result is None:
        return True
    if "recommendations" in result and not result["recommendations"]:
        return True
    if "message" in result and "product" not in result:
        return True
    return False


def route_customer(message: str, session_id: str) -> dict:
    try:
        # Tells the LLM whether this session actually has anything
        # pending to confirm -- see llm.py's _pending_order_state_line()
        # for why a bare "yh"/"yeah" is unresolvable without it.
        pending_order = get_pending_order_summary(session_id)
        # One step earlier in the same problem: an order that's been
        # started (propose_order asked "how many?") but isn't priced
        # yet. Only relevant while there's no full pending_order --
        # once a proposal exists, that's the active state, not the
        # draft that led to it. See llm.py's _order_draft_state_line().
        order_draft = None if pending_order else get_order_draft(session_id)
        # A product lookup the customer asked for but hadn't named a
        # product for yet ("yeah i wanna see pictures") -- see
        # memory.get_pending_intent() and llm.py's
        # _pending_intent_state_line() for why a follow-up naming the
        # product needs this to avoid asking the customer to repeat
        # themselves a second time.
        pending_intent = get_pending_intent(session_id)
        # A real business action (usually propose_order/confirm_order)
        # that was fully specified and still hit a genuine, unrecoverable
        # failure -- different axis from all three above, which are about
        # missing information. See memory.get_last_action_outcome() and
        # llm.py's _last_action_outcome_state_line().
        last_action_outcome = get_last_action_outcome(session_id)
        # The specific product a get_product_price/generate_quote call
        # most recently resolved to, so a bare karat-only follow-up
        # ("what about in 18k") can re-quote the same item instead of
        # falling through to recommend_products. See
        # memory.get_last_priced_product() and llm.py's
        # _last_priced_product_state_line().
        last_priced_product = get_last_priced_product(session_id)
        tool_request = understand_customer(
            message,
            pending_order=pending_order,
            order_draft=order_draft,
            pending_intent=pending_intent,
            last_action_outcome=last_action_outcome,
            last_priced_product=last_priced_product,
        )
    except ValueError as e:
        return {"error": str(e)}
    except ToolSelectionError as e:
        logger.error("Tool selection failed: %s", e)
        return {"error": "I couldn't understand that request. Could you rephrase it?"}

    # understand_customer() only returns "requests" (plural) when the
    # message genuinely contained more than one distinct ask -- see
    # llm.py. The single-request path below is completely unchanged
    # from before that existed, so route_customer()'s original,
    # documented contract-stable shape is untouched for every message
    # that doesn't need splitting (still the overwhelming majority).
    if "requests" in tool_request:
        results = [_execute_single(req, session_id) for req in tool_request["requests"]]
        return {"results": results}

    return _execute_single(tool_request, session_id)


def _execute_single(tool_request: dict, session_id: str) -> dict:
    if tool_request["tool"] == _CONVERSATION_TOOL:
        return _handle_conversation(tool_request["arguments"])

    # Resolve any "this" / "that one" reference the model couldn't
    # answer from the message alone against what this session last
    # talked about, before the arguments ever reach a tool.
    arguments = fill_missing_context(session_id, tool_request["arguments"])

    if tool_request["tool"] in _SESSION_AWARE_TOOLS:
        arguments = {**arguments, "session_id": session_id}

    # Read BEFORE execute_tool()/remember_context() below overwrite it --
    # this is deliberately the state as it was going into this call, so
    # it can be compared against `arguments` (this call's resolved
    # values) to detect a genuine correction. See
    # _describe_order_corrections()'s docstring.
    correction_note = None
    if tool_request["tool"] == _ORDER_TOOL:
        correction_note = _describe_order_corrections(get_order_draft(session_id), arguments)

    try:
        result = execute_tool(tool_request["tool"], **arguments)
    except ToolExecutionError as e:
        logger.error("Tool execution failed: %s", e)
        return {"error": "Something went wrong while processing your request."}

    # Only remember context once the tool has actually run against it
    # AND found something -- see _found_nothing()'s docstring for the
    # concrete failure mode this avoids.
    if not _found_nothing(result):
        remember_context(session_id, arguments)

    _update_pending_intent(session_id, tool_request["tool"], arguments, result)
    _update_last_priced_product(session_id, tool_request["tool"], result)

    if _tool_succeeded(result):
        # A genuine success means whatever failed before is no longer
        # the active topic -- see memory.set_last_action_outcome()'s
        # docstring. Deliberately stricter than "not _found_nothing()":
        # that treats any {"error": ...} shape (including the very
        # failure this session just recorded, e.g. propose_order's own
        # no-id return) as "found something", which would wipe out the
        # outcome on the exact same call that just set it.
        set_last_action_outcome(session_id, None)

    if correction_note and isinstance(result, dict):
        result = {**result, "correction_note": correction_note}

    return result


def _describe_order_corrections(old_draft: dict | None, arguments: dict) -> str | None:
    """Builds a short acknowledgement sentence when this propose_order
    call changes a field the customer had already given earlier in the
    same order (e.g. "wait, 14k rather" after material was already
    "12k") -- confirmed live, 2026-08-19 (Webb): the correction was
    applied to session memory correctly, but the very next reply just
    asked for the next missing field with no acknowledgement anything
    had changed, which read as the assistant not having registered the
    change at all.

    Deliberately doesn't force a "please confirm this change" round
    trip -- an extra yes/no turn for an unambiguous correction is
    friction a real assistant wouldn't add (Webb and a second AI's
    review of the same transcript both flagged this independently,
    2026-08-19). This just states what changed; response_formatter.py
    prepends it to whatever reply would already be sent (the next
    missing-field question, or the full proposal if everything's now
    known), so the conversation continues normally afterwards.

    Returns None when there's nothing to acknowledge: no prior draft at
    all (a fresh order, not a correction -- see get_order_draft()'s
    None case), or none of the fields this call resolved actually
    differ from what was already known."""
    if not old_draft:
        return None

    changed = []
    for key, label in _ORDER_CORRECTION_FIELDS.items():
        old_value = old_draft.get(key)
        new_value = arguments.get(key)
        if old_value is None or new_value is None:
            continue
        if isinstance(new_value, str) and new_value.strip().lower() == "unknown":
            continue
        if str(old_value).strip().lower() == str(new_value).strip().lower():
            continue
        changed.append(f"{label} to {new_value}")

    if not changed:
        return None
    if len(changed) == 1:
        return f"Got it, I've updated {changed[0]}."
    return f"Got it, I've updated {' and '.join(changed)}."


def _tool_succeeded(result: dict | None) -> bool:
    if result is None or "error" in result:
        return False
    return not _found_nothing(result)


def _update_pending_intent(session_id: str, tool_name: str, arguments: dict, result: dict | None) -> None:
    if tool_name in _PENDING_INTENT_TOOLS and _found_nothing(result):
        product_name = arguments.get("product_name")
        if product_name is None or str(product_name).strip().lower() == "unknown":
            # Missing the product itself, specifically -- not a real name
            # that just didn't match anything (a made-up item, a typo).
            # Only this case means "ask again once they tell you which
            # one", so only this case is worth remembering.
            set_pending_intent(session_id, tool_name)
            return

    if not _found_nothing(result):
        # Whatever was pending (if anything) is now resolved or has been
        # superseded by a successful, different request -- either way,
        # stale intent left behind would risk misreading an unrelated
        # later message as still answering it.
        set_pending_intent(session_id, None)


def _update_last_priced_product(session_id: str, tool_name: str, result: dict | None) -> None:
    if tool_name in _PRICING_TOOLS and _tool_succeeded(result):
        # get_product_price/generate_quote's success shape always
        # includes "product" -- see _found_nothing()'s "message" without
        # "product" check above, which is exactly what rules the failure
        # case out here.
        set_last_priced_product(session_id, result.get("product"))
    elif tool_name == _RECOMMEND_TOOL and _tool_succeeded(result):
        set_last_priced_product(session_id, None)


def _handle_conversation(arguments: dict) -> dict:
    # Defensive only: the LLM is instructed to always write a real reply
    # for converse (see llm.py's tool 8 description) since there's no
    # deterministic fallback that could construct one -- an empty/missing
    # reply here means the model didn't follow that, not a real business
    # state to recover from, so a generic greeting is the only sane default.
    reply = arguments.get("reply") if isinstance(arguments, dict) else None
    reply = str(reply).strip() if reply else ""
    return {"conversation_reply": reply or _CONVERSATION_FALLBACK_REPLY}
