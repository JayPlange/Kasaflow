"""
Unit tests for services/tool_executor.py

This module is the safety net between "LLM decided what to call" and
"real Python function actually ran". These tests exist specifically to
prove that net catches things -- an unknown tool name, wrong arguments,
and a tool that blows up internally should never crash the whole app.
"""

import pytest

from services.tool_executor import ToolExecutionError, execute_tool


def test_execute_tool_runs_known_tool_successfully():
    # Arrange: get_delivery_information is a real registered tool that
    # takes no arguments

    # Act
    result = execute_tool("get_delivery_information")

    # Assert
    assert result["delivery_time"] == "2-5 business days"


def test_execute_tool_returns_error_dict_for_unknown_tool():
    # Arrange: a tool name that was never registered -- this simulates
    # the LLM hallucinating a tool that doesn't exist

    # Act
    result = execute_tool("send_email_to_customer")

    # Assert: should degrade gracefully, not raise
    assert result == {"error": "Tool 'send_email_to_customer' not found"}


def test_execute_tool_raises_tool_execution_error_for_bad_arguments():
    # Arrange: get_delivery_information takes zero arguments, so passing
    # one simulates the LLM guessing the wrong argument names

    # Act / Assert
    with pytest.raises(ToolExecutionError):
        execute_tool("get_delivery_information", unexpected_arg="oops")


def test_execute_tool_wraps_unexpected_tool_failures(monkeypatch):
    # Arrange: register a fake tool that always blows up, to simulate a
    # bug inside a tool's own logic (e.g. a future tool calling a flaky
    # external API)
    from services import tool_registry

    def broken_tool():
        raise RuntimeError("simulated failure inside a tool")

    monkeypatch.setitem(tool_registry.TOOLS, "broken_tool", broken_tool)

    # Act / Assert
    with pytest.raises(ToolExecutionError):
        execute_tool("broken_tool")
