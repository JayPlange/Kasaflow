"""
Unit tests for services/llm.py

The golden rule here: never call the real OpenAI API in a unit test.
It costs money, it's slow, and it can fail for reasons that have
nothing to do with a bug in your code. Instead we use a "mock" -- a
stand-in object that pretends to be the OpenAI client and returns
exactly what we tell it to, so we're testing OUR code's reaction to
the response, not OpenAI's actual behavior.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services import llm
from services.llm import ToolSelectionError


# ---------------------------------------------------------------------
# _parse_tool_request: pure function, no API involved, no mocking needed
# ---------------------------------------------------------------------

def test_parse_tool_request_handles_clean_json():
    # Arrange
    raw = '{"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}}'

    # Act
    result = llm._parse_tool_request(raw)

    # Assert
    assert result["tool"] == "get_product_price"
    assert result["arguments"]["material"] == "gold"


def test_parse_tool_request_strips_markdown_fences():
    # Arrange: models sometimes wrap JSON in ```json fences even when told not to
    raw = '```json\n{"tool": "get_delivery_information", "arguments": {}}\n```'

    # Act
    result = llm._parse_tool_request(raw)

    # Assert
    assert result["tool"] == "get_delivery_information"


def test_parse_tool_request_raises_on_invalid_json():
    # Arrange
    raw = "Sure! Here is the tool you need: get_product_price"

    # Act / Assert
    with pytest.raises(ToolSelectionError):
        llm._parse_tool_request(raw)


def test_parse_tool_request_raises_when_keys_missing():
    # Arrange: valid JSON, but missing the "arguments" key
    raw = '{"tool": "get_product_price"}'

    # Act / Assert
    with pytest.raises(ToolSelectionError):
        llm._parse_tool_request(raw)


# ---------------------------------------------------------------------
# understand_customer: mocks the OpenAI client entirely
# ---------------------------------------------------------------------

def _mock_openai_response(output_text: str):
    """Builds a fake response object shaped like the real OpenAI SDK's."""
    return SimpleNamespace(output_text=output_text)


def test_understand_customer_returns_parsed_tool_request(monkeypatch):
    # Arrange: replace the real OpenAI client with a mock that returns
    # a canned response instead of calling the network
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_openai_response(
        '{"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}}'
    )
    monkeypatch.setattr(llm, "client", fake_client)

    # Act
    result = llm.understand_customer("how much is a gold ring?")

    # Assert
    assert result["tool"] == "get_product_price"
    fake_client.responses.create.assert_called_once()


def test_understand_customer_rejects_empty_message():
    # Arrange: no mocking needed, this should fail before ever touching the client

    # Act / Assert
    with pytest.raises(ValueError):
        llm.understand_customer("   ")


def test_understand_customer_raises_tool_selection_error_on_bad_json(monkeypatch):
    # Arrange: the mock "AI" returns garbage
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_openai_response("not json at all")
    monkeypatch.setattr(llm, "client", fake_client)

    # Act / Assert
    with pytest.raises(ToolSelectionError):
        llm.understand_customer("how much is a gold ring?")
