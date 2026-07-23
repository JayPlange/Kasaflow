"""
Top-level orchestration: customer message -> tool selection -> tool
execution -> result. This is the only place that decides what the
customer sees when something goes wrong upstream.
"""

import logging

from services.llm import ToolSelectionError, understand_customer
from services.memory import fill_missing_context, remember_context
from services.tool_executor import ToolExecutionError, execute_tool

logger = logging.getLogger(__name__)


def route_customer(message: str, session_id: str) -> dict:
    try:
        tool_request = understand_customer(message)
    except ValueError as e:
        return {"error": str(e)}
    except ToolSelectionError as e:
        logger.error("Tool selection failed: %s", e)
        return {"error": "I couldn't understand that request. Could you rephrase it?"}

    # Resolve any "this" / "that one" reference the model couldn't
    # answer from the message alone against what this session last
    # talked about, before the arguments ever reach a tool.
    arguments = fill_missing_context(session_id, tool_request["arguments"])

    try:
        result = execute_tool(tool_request["tool"], **arguments)
    except ToolExecutionError as e:
        logger.error("Tool execution failed: %s", e)
        return {"error": "Something went wrong while processing your request."}

    # Only remember context once the tool has actually run against it,
    # so a session never "learns" a product/material that turned out to
    # be invalid.
    remember_context(session_id, arguments)

    return result
