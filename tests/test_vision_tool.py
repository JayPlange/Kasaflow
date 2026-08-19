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


def test_describe_product_image_prompt_allows_describing_shape_instead_of_guessing_category(monkeypatch):
    # Confirmed live, 2026-08-18: a photo of loose teardrop-shaped
    # charms on a workbench (not an assembled piece) was confidently
    # described as a "cuff bracelet" -- wrong item type, which then
    # steered the downstream candidate search away from the real
    # matching product entirely. The prompt must tell the model it's
    # allowed to describe shape/material instead of forcing a category
    # guess when the photo doesn't clearly show one.
    assert "do not guess a category" in vision_tool._PROMPT.lower()


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


# ---------------------------------------------------------------------
# match_photo_to_candidates -- a second, distinct vision-API use, one
# customer photo compared against a short list of catalogue candidate
# photos to pick the same physical item, not just describe a photo in
# isolation. Same golden rule: mock client.responses.create, never call
# the real API.
# ---------------------------------------------------------------------

from services.vision_tool import match_photo_to_candidates  # noqa: E402


def _two_candidates():
    return [
        ("Gye Nyame White Necklace", b"candidate-1-bytes", "image/jpeg"),
        ("Custom Adinkra Chains Gold Necklace", b"candidate-2-bytes", "image/jpeg"),
    ]


def test_match_photo_to_candidates_returns_index_of_confident_match(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_vision_response("2")
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    result = match_photo_to_candidates(b"customer-photo-bytes", "image/jpeg", _two_candidates())

    # Assert: "2" (1-indexed, matching the prompt's own numbering) maps
    # to index 1 in the candidates list
    assert result == 1


def test_match_photo_to_candidates_returns_none_for_none_response(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_vision_response("NONE")
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    result = match_photo_to_candidates(b"customer-photo-bytes", "image/jpeg", _two_candidates())

    # Assert
    assert result is None


def test_match_photo_to_candidates_returns_none_for_unparseable_response(monkeypatch):
    # Arrange: model didn't follow instructions -- must not crash or
    # guess, just treat it the same as no match
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_vision_response("I think it's the second one")
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    result = match_photo_to_candidates(b"customer-photo-bytes", "image/jpeg", _two_candidates())

    # Assert
    assert result is None


def test_match_photo_to_candidates_returns_none_for_out_of_range_index(monkeypatch):
    # Arrange: only 2 candidates sent, model returned "5"
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_vision_response("5")
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    result = match_photo_to_candidates(b"customer-photo-bytes", "image/jpeg", _two_candidates())

    # Assert
    assert result is None


def test_match_photo_to_candidates_returns_none_for_empty_candidate_list(monkeypatch):
    # Arrange: nothing to compare against -- must not call the API at all
    fake_client = MagicMock()
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    result = match_photo_to_candidates(b"customer-photo-bytes", "image/jpeg", [])

    # Assert
    assert result is None
    fake_client.responses.create.assert_not_called()


def test_match_photo_to_candidates_sends_one_image_per_candidate_plus_the_customer_photo(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_vision_response("1")
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    match_photo_to_candidates(b"customer-photo-bytes", "image/jpeg", _two_candidates())

    # Assert: 1 customer photo + 2 candidate photos = 3 input_image blocks
    _, kwargs = fake_client.responses.create.call_args
    content = kwargs["input"][0]["content"]
    image_blocks = [c for c in content if c["type"] == "input_image"]
    assert len(image_blocks) == 3


def test_match_photo_to_candidates_uses_temperature_zero(monkeypatch):
    # Arrange: consistency matters more than variation for a
    # pick-one-of-few decision -- see the module's own comment on this,
    # confirmed live, 2026-08-17, the same photo matched differently on
    # two separate attempts.
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _mock_vision_response("1")
    monkeypatch.setattr(vision_tool, "client", fake_client)

    # Act
    match_photo_to_candidates(b"customer-photo-bytes", "image/jpeg", _two_candidates())

    # Assert
    _, kwargs = fake_client.responses.create.call_args
    assert kwargs["temperature"] == 0


def test_vision_client_disables_sdk_level_retries():
    # The SDK retries transient failures internally by default, on top
    # of this module's own manual retry/backoff loop -- confirmed live,
    # 2026-08-17, a single failed request took close to 2 minutes to
    # surface, far longer than the app-level retry math alone explains.
    assert vision_tool.client.max_retries == 0


def test_match_photo_to_candidates_raises_after_exhausting_retries(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.responses.create.side_effect = APIConnectionError(request=MagicMock())
    monkeypatch.setattr(vision_tool, "client", fake_client)
    monkeypatch.setattr(vision_tool.time, "sleep", lambda seconds: None)

    # Act / Assert
    with pytest.raises(VisionServiceError):
        match_photo_to_candidates(b"customer-photo-bytes", "image/jpeg", _two_candidates())
