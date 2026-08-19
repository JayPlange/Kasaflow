"""
Unit tests for services/image_embed_tool.py

Same golden rule as test_vision_tool.py: never call the real Cohere API
here. We mock cohere.ClientV2 and check our own request-shape and
error-handling logic, not Cohere's actual behaviour -- that's what a
live regression check is for, and this module hasn't had one yet (see
its module docstring).
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services import image_embed_tool
from services.image_embed_tool import ImageEmbedError, embed_images


def _mock_embed_response(vectors: list[list[float]]):
    return SimpleNamespace(embeddings=SimpleNamespace(float=vectors))


def test_embed_images_returns_empty_list_for_empty_input(monkeypatch):
    # No client call should happen at all -- nothing to embed.
    fake_client_factory = MagicMock()
    monkeypatch.setattr(image_embed_tool, "_client", fake_client_factory)

    result = embed_images([])

    assert result == []
    fake_client_factory.assert_not_called()


def test_embed_images_returns_one_vector_per_input(monkeypatch):
    fake_client = MagicMock()
    fake_client.embed.return_value = _mock_embed_response([[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(image_embed_tool, "_client", lambda: fake_client)

    result = embed_images(["data:image/jpeg;base64,aaa", "data:image/jpeg;base64,bbb"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_images_sends_the_documented_request_shape(monkeypatch):
    fake_client = MagicMock()
    fake_client.embed.return_value = _mock_embed_response([[0.1, 0.2]])
    monkeypatch.setattr(image_embed_tool, "_client", lambda: fake_client)

    embed_images(["data:image/jpeg;base64,aaa"])

    _, kwargs = fake_client.embed.call_args
    assert kwargs["input_type"] == "image"
    assert kwargs["images"] == ["data:image/jpeg;base64,aaa"]
    assert kwargs["model"] == "embed-v4.0"


class _DictOnlyEmbeddings:
    """Stands in for a response.embeddings object that only supports
    dict-style access (`response.embeddings["float"]`), not attribute
    access -- exercises embed_images()'s fallback path, since which
    style the real Cohere SDK actually returns hasn't been confirmed
    against a live response yet (see module docstring)."""

    def __getitem__(self, key):
        assert key == "float"
        return [[0.5, 0.6]]


def test_embed_images_falls_back_to_dict_style_response_access(monkeypatch):
    fake_response = SimpleNamespace(embeddings=_DictOnlyEmbeddings())
    fake_client = MagicMock()
    fake_client.embed.return_value = fake_response
    monkeypatch.setattr(image_embed_tool, "_client", lambda: fake_client)

    result = embed_images(["data:image/jpeg;base64,aaa"])

    assert result == [[0.5, 0.6]]


def test_embed_images_wraps_a_client_exception(monkeypatch):
    fake_client = MagicMock()
    fake_client.embed.side_effect = RuntimeError("network down")
    monkeypatch.setattr(image_embed_tool, "_client", lambda: fake_client)

    with pytest.raises(ImageEmbedError, match="Cohere embed request failed"):
        embed_images(["data:image/jpeg;base64,aaa"])


def test_client_raises_when_cohere_api_key_not_configured(monkeypatch):
    monkeypatch.setattr(image_embed_tool, "settings", replace(image_embed_tool.settings, cohere_api_key=None))

    with pytest.raises(ImageEmbedError, match="COHERE_API_KEY is not configured"):
        embed_images(["data:image/jpeg;base64,aaa"])
