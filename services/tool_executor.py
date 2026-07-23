"""
Executes a tool by name with the arguments the LLM decided on.

This is the boundary between "LLM decided what to do" and "deterministic
Python code actually does it". Errors from tool functions are caught
here and turned into a single, predictable exception type so callers
(router.py) don't need to know about every possible tool's failure modes.
"""

import logging

from services.tool_registry import TOOLS

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when a registered tool fails or is called incorrectly."""


def execute_tool(tool_name: str, **kwargs) -> dict:
    tool = TOOLS.get(tool_name)

    if not tool:
        logger.warning("Unknown tool requested: %s", tool_name)
        return {"error": f"Tool '{tool_name}' not found"}

    try:
        return tool(**kwargs)
    except TypeError as e:
        # Most commonly: LLM returned the wrong/missing argument names.
        logger.error("Tool '%s' called with bad arguments %s: %s", tool_name, kwargs, e)
        raise ToolExecutionError(f"Invalid arguments for tool '{tool_name}'") from e
    except Exception as e:
        logger.exception("Tool '%s' raised an unexpected error", tool_name)
        raise ToolExecutionError(f"Tool '{tool_name}' failed: {e}") from e
