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
# _parse_tool_request: the additive "requests" (plural) shape for
# messages that contain more than one distinct ask
# ---------------------------------------------------------------------

def test_parse_tool_request_handles_multi_request_shape():
    # Arrange: "how much is a gold ring and a silver chain"
    raw = (
        '{"requests": ['
        '{"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}},'
        '{"tool": "get_product_price", "arguments": {"product_name": "chain", "material": "silver"}}'
        ']}'
    )

    # Act
    result = llm._parse_tool_request(raw)

    # Assert
    assert "requests" in result
    assert len(result["requests"]) == 2
    assert result["requests"][0]["arguments"]["product_name"] == "ring"
    assert result["requests"][1]["arguments"]["product_name"] == "chain"


def test_parse_tool_request_raises_when_requests_is_empty():
    # Arrange: model returned the multi-request shape with nothing in it
    raw = '{"requests": []}'

    # Act / Assert
    with pytest.raises(ToolSelectionError):
        llm._parse_tool_request(raw)


def test_parse_tool_request_raises_when_a_request_entry_is_malformed():
    # Arrange: second entry is missing "arguments"
    raw = (
        '{"requests": ['
        '{"tool": "get_product_price", "arguments": {"product_name": "ring", "material": "gold"}},'
        '{"tool": "get_delivery_information"}'
        ']}'
    )

    # Act / Assert
    with pytest.raises(ToolSelectionError):
        llm._parse_tool_request(raw)


def test_parse_tool_request_truncates_when_over_the_cap(monkeypatch):
    # Arrange: more distinct asks than we're willing to fan out to tools for
    monkeypatch.setattr(llm, "MAX_REQUESTS_PER_MESSAGE", 2)
    raw = (
        '{"requests": ['
        '{"tool": "get_delivery_information", "arguments": {}},'
        '{"tool": "get_delivery_information", "arguments": {}},'
        '{"tool": "get_delivery_information", "arguments": {}}'
        ']}'
    )

    # Act
    result = llm._parse_tool_request(raw)

    # Assert: capped, not rejected outright -- answer what we safely can
    assert len(result["requests"]) == 2


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


# ---------------------------------------------------------------------
# pending-order context: a bare "yh"/"yeah" is unresolvable without
# knowing whether this session actually has anything to confirm -- see
# _pending_order_state_line()'s docstring
# ---------------------------------------------------------------------

def test_prompt_tells_the_model_nothing_is_pending_by_default():
    prompt = llm._build_prompt("yh", pending_order=None, order_draft=None)
    assert "does NOT currently have any pending order" in prompt
    assert "Do not use confirm_order" in prompt


def test_prompt_describes_a_real_pending_order():
    pending = {"product": "Ring", "material": "18k", "quantity": 2, "total": 2425.0}
    prompt = llm._build_prompt("yh", pending_order=pending, order_draft=None)
    assert "pending order awaiting confirmation" in prompt
    assert "2 x 18k Ring" in prompt
    assert "2,425.00" in prompt


def test_understand_customer_passes_pending_order_through_to_the_prompt(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_openai_response(
        '{"tool": "confirm_order", "arguments": {}}'
    )
    monkeypatch.setattr(llm, "client", fake_client)
    pending = {"product": "Ring", "material": "18k", "quantity": 1, "total": 1225.0}

    # Act
    llm.understand_customer("yh", pending_order=pending)

    # Assert: the actual prompt sent to the model reflects the pending order
    sent_prompt = fake_client.responses.create.call_args.kwargs["input"]
    assert "1 x 18k Ring" in sent_prompt


# ---------------------------------------------------------------------
# order-draft context: a bare "2" or a bare address is unresolvable
# without knowing an order is already in progress -- see
# _order_draft_state_line()'s docstring
# ---------------------------------------------------------------------

def test_prompt_omits_the_order_draft_section_when_nothing_in_progress():
    prompt = llm._build_prompt("2", pending_order=None, order_draft=None)
    assert "order in progress" not in prompt


def test_prompt_describes_a_partial_order_draft():
    draft = {
        "product_name": "Custom Leaf White Gold Necklace, 20g",
        "material": "14k",
        "quantity": None,
        "delivery_address": None,
        "delivery_option": None,
    }
    prompt = llm._build_prompt("2", pending_order=None, order_draft=draft)
    assert "order in progress" in prompt
    assert "product=Custom Leaf White Gold Necklace, 20g" in prompt
    assert "material/karat=14k" in prompt
    assert "Still missing: quantity, delivery address, delivery option" in prompt


def test_prompt_omits_order_draft_section_once_everything_is_known():
    # Nothing left for a short reply to be answering -- propose_order
    # itself is the next step, not another round of "what's missing".
    draft = {
        "product_name": "Ring", "material": "18k", "quantity": 2,
        "delivery_address": "Accra", "delivery_option": "accra_rider",
    }
    prompt = llm._build_prompt("2", pending_order=None, order_draft=draft)
    assert "order in progress" not in prompt


def test_understand_customer_passes_order_draft_through_to_the_prompt(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_openai_response(
        '{"tool": "propose_order", "arguments": {}}'
    )
    monkeypatch.setattr(llm, "client", fake_client)
    draft = {
        "product_name": "Ring", "material": "18k", "quantity": None,
        "delivery_address": None, "delivery_option": None,
    }

    # Act
    llm.understand_customer("2", order_draft=draft)

    # Assert
    sent_prompt = fake_client.responses.create.call_args.kwargs["input"]
    assert "product=Ring" in sent_prompt


# ---------------------------------------------------------------------
# converse -- the eighth outcome, for purely conversational messages
# that need no business tool (see llm.py's tool 8 description)
# ---------------------------------------------------------------------

def test_prompt_includes_converse_tool_guidance():
    prompt = llm._build_prompt("hey", pending_order=None, order_draft=None)
    assert "8. converse" in prompt
    assert "reply" in prompt
    assert "NOT_FOUND" not in prompt  # guardrail language stays out of the prompt itself


def test_understand_customer_parses_a_converse_response(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_openai_response(
        '{"tool": "converse", "arguments": {"reply": "Hey! How can I help you today?"}}'
    )
    monkeypatch.setattr(llm, "client", fake_client)

    # Act
    result = llm.understand_customer("hey")

    # Assert
    assert result["tool"] == "converse"
    assert result["arguments"]["reply"] == "Hey! How can I help you today?"


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
