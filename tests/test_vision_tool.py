"""
Unit tests for services/vision_tool.py

Same golden rule as test_llm.py and test_embeddings_client.py: never
call the real OpenAI API here. We mock client.responses.create and
check our own request-shape, retry, and sentinel-handling logic, not
OpenAI's actual behaviour -- that's what a live regression check is
for, and this module hasn't had one yet (see its module docstring).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError

from services import vision_tool
from services.vision_tool import VisionServiceError, describe_product_image


def _mock_vision_response(output_text: str):
    return SimpleNamespace(output_text=output_text)


def test_describe_product_image_returns_description_on_success(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_vision_response("gold twist ring")
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    result = describe_product_image(b"fake-image-bytes")

    # Assert
    assert result == "gold twist ring"


def test_describe_product_image_sends_base64_data_url_and_prompt(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_vision_response("silver chain")
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    describe_product_image(b"\x89PNG-fake-bytes", mime_type="image/png")

    # Assert: correct multimodal Responses API shape -- one user message
    # with a text block (the instructions) and an input_image block
    # carrying the image as a base64 data URL, not a hosted URL (nothing
    # to host a customer's photo at)
    _, kwargs = fake_client.responses.create.call_args
    content = kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_describe_product_image_returns_empty_string_for_not_jewellery(monkeypatch):
    # Arrange: the model was told to return this sentinel for photos
    # that clearly aren't jewellery at all
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_vision_response("NOT_JEWELLERY")
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    result = describe_product_image(b"fake-image-bytes")

    # Assert: caller (whatsapp_routes.py) treats "" as "couldn't tell what that was"
    assert result == ""


def test_describe_product_image_retries_on_transient_error_then_succeeds(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.side_effect = [
        APIConnectionError(request=MagicMock()),
        _mock_vision_response("gold ring"),
    ]
    monkeypatch.setattr(vision_tool, "client", fake_client)
    monkeypatch.setattr(vision_tool.time, "sleep", lambda seconds: None)

    # Act
    result = describe_product_image(b"fake-image-bytes")

    # Assert
    assert result == "gold ring"
    assert fake_client.responses.create.call_count == 2


def test_describe_product_image_raises_after_exhausting_retries(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.side_effect = APIConnectionError(request=MagicMock())
    monkeypatch.setattr(vision_tool, "client", fake_client)
    monkeypatch.setattr(vision_tool.time, "sleep", lambda seconds: None)

    # Act / Assert
    with pytest.raises(VisionServiceError):
        describe_product_image(b"fake-image-bytes")
