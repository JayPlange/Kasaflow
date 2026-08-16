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


def test_prompt_lets_the_model_correct_an_already_known_order_draft_field():
    # A customer correcting themselves ("sorry, delivery within kumasi"
    # after already saying Accra) must not get told to keep the old
    # value -- confirmed live, 2026-08-13: the correction got routed to
    # a fresh get_delivery_information question instead of updating the
    # order, because this instruction only ever said "keep every
    # already-known value", with no exception for a correction.
    draft = {
        "product_name": "Ring", "material": None, "quantity": 2,
        "delivery_address": "Suame, Kumasi", "delivery_option": "accra_rider",
    }
    prompt = llm._build_prompt("sorry, delivery within kumasi", pending_order=None, order_draft=draft)
    assert "clearly states a different value" in prompt
    assert "do not silently keep a value they just corrected" in prompt


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
# converse -- the ninth outcome, for purely conversational messages
# that need no business tool (see llm.py's tool 9 description). It moved
# from 8th to 9th when cancel_order was added as tool 8.
# ---------------------------------------------------------------------

def test_prompt_includes_converse_tool_guidance():
    prompt = llm._build_prompt("hey", pending_order=None, order_draft=None)
    assert "9. converse" in prompt
    assert "reply" in prompt
    assert "NOT_FOUND" not in prompt  # guardrail language stays out of the prompt itself


def test_prompt_tells_the_model_to_use_recommend_products_for_category_photo_requests():
    # "necklace images" etc. -- a category is enough to act on, must not
    # fall through to get_product_price("unknown") or converse
    prompt = llm._build_prompt("necklace images", pending_order=None, order_draft=None)
    assert "recommend_products with category" in prompt.lower() or "use recommend_products" in prompt.lower()


# ---------------------------------------------------------------------
# cancel_order -- tool 8, added 2026-08-16 alongside the delivery-address
# mismatch check and the "placed" wording softening (see order_tool.py's
# cancel_order() and llm.py's tool 8 description)
# ---------------------------------------------------------------------

def test_prompt_includes_cancel_order_tool_guidance():
    prompt = llm._build_prompt("hey", pending_order=None, order_draft=None)
    assert "8. cancel_order" in prompt
    assert "order_id" in prompt


def test_understand_customer_parses_a_cancel_order_response_with_no_number(monkeypatch):
    # Arrange: customer didn't state an order number -- the LLM must
    # pass "unknown", never invent one (see llm.py's cancel_order
    # guidance and order_tool._resolve_order_id()'s fallback)
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_openai_response(
        '{"tool": "cancel_order", "arguments": {"order_id": "unknown"}}'
    )
    monkeypatch.setattr(llm, "client", fake_client)

    # Act
    result = llm.understand_customer("cancel my order")

    # Assert
    assert result["tool"] == "cancel_order"
    assert result["arguments"]["order_id"] == "unknown"


def test_understand_customer_parses_a_cancel_order_response_with_a_number(monkeypatch):
    # Arrange: customer gave an explicit order number
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_openai_response(
        '{"tool": "cancel_order", "arguments": {"order_id": "6846"}}'
    )
    monkeypatch.setattr(llm, "client", fake_client)

    # Act
    result = llm.understand_customer("please cancel order 6846")

    # Assert
    assert result["tool"] == "cancel_order"
    assert result["arguments"]["order_id"] == "6846"


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


# ---------------------------------------------------------------------
# pending_intent -- a product lookup the customer asked for but hadn't
# named a product for yet (see llm.py's _pending_intent_state_line())
# ---------------------------------------------------------------------

def test_prompt_omits_pending_intent_section_when_nothing_pending():
    prompt = llm._build_prompt("this one", pending_order=None, order_draft=None, pending_intent=None)
    assert "hadn't named a specific product" not in prompt


def test_prompt_describes_a_pending_intent():
    prompt = llm._build_prompt(
        "this Set Multi Stone Golf Ring, 7g",
        pending_order=None, order_draft=None, pending_intent="get_product_price",
    )
    assert "hadn't named a specific product" in prompt
    assert "call get_product_price with that product name" in prompt
    assert "do not use converse" in prompt.lower()


def test_understand_customer_passes_pending_intent_through_to_the_prompt(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_openai_response(
        '{"tool": "get_product_price", "arguments": {"product_name": "Set Multi Stone Golf Ring, 7g", "material": "unknown"}}'
    )
    monkeypatch.setattr(llm, "client", fake_client)

    # Act
    llm.understand_customer("this Set Multi Stone Golf Ring, 7g", pending_intent="get_product_price")

    # Assert
    sent_prompt = fake_client.responses.create.call_args.kwargs["input"]
    assert "hadn't named a specific product" in sent_prompt


# ---------------------------------------------------------------------
# last_action_outcome -- a fully-specified business action that still
# failed for a real reason (see llm.py's _last_action_outcome_state_line())
# ---------------------------------------------------------------------

def test_prompt_omits_last_action_outcome_section_when_none():
    prompt = llm._build_prompt("why?", pending_order=None, order_draft=None, last_action_outcome=None)
    assert "just failed" not in prompt


def test_prompt_describes_a_last_action_outcome():
    outcome = {
        "action": "propose_order",
        "customer_safe_explanation": "I can't take an order for that item right now.",
    }
    prompt = llm._build_prompt("why?", pending_order=None, order_draft=None, last_action_outcome=outcome)
    assert "just failed" in prompt
    assert "I can't take an order for that item right now." in prompt
    assert "do not say you haven't seen anything" in prompt.lower()


def test_understand_customer_passes_last_action_outcome_through_to_the_prompt(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_openai_response(
        '{"tool": "converse", "arguments": {"reply": "..."}}'
    )
    monkeypatch.setattr(llm, "client", fake_client)
    outcome = {"action": "propose_order", "customer_safe_explanation": "reason given to the customer"}

    # Act
    llm.understand_customer("why?", last_action_outcome=outcome)

    # Assert
    sent_prompt = fake_client.responses.create.call_args.kwargs["input"]
    assert "reason given to the customer" in sent_prompt


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
