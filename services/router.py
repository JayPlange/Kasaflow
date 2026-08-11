"""
Top-level orchestration: customer message -> tool selection -> tool
execution -> result. This is the only place that decides what the
customer sees when something goes wrong upstream.
"""

import logging

from services.llm import ToolSelectionError, understand_customer
from services.memory import fill_missing_context, get_order_draft, remember_context
from services.order_tool import get_pending_order_summary
from services.tool_executor import ToolExecutionError, execute_tool

logger = logging.getLogger(__name__)

# propose_order/confirm_order (services/order_tool.py) both need to know
# which customer's session they're acting on, but that's never something
# the LLM decides or the customer states -- it comes from the channel
# (WhatsApp) this message arrived on. Every other registered tool's
# **kwargs contract is exactly whatever the LLM returned, nothing more,
# so session_id is injected only for these two by name rather than added
# to every tool call: passing an unexpected session_id kwarg to any of
# the other five would raise inside tool_executor.py's TypeError handler.
_SESSION_AWARE_TOOLS = {"propose_order", "confirm_order"}


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
        tool_request = understand_customer(message, pending_order=pending_order, order_draft=order_draft)
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
    # Resolve any "this" / "that one" reference the model couldn't
    # answer from the message alone against what this session last
    # talked about, before the arguments ever reach a tool.
    arguments = fill_missing_context(session_id, tool_request["arguments"])

    if tool_request["tool"] in _SESSION_AWARE_TOOLS:
        arguments = {**arguments, "session_id": session_id}

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

    return result
