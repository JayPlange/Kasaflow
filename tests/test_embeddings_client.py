"""
Unit tests for services/embeddings_client.py

Same golden rule as test_llm.py: never call the real OpenAI API here.
We mock the client's embeddings.create call and check our own retry
and error-handling logic, not OpenAI's actual behaviour.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError

from services import embeddings_client
from services.embeddings_client import EmbeddingError


def _mock_embeddings_response(vectors: list[list[float]]):
    """Builds a fake response object shaped like the real OpenAI SDK's."""
    return SimpleNamespace(data=[SimpleNamespace(embedding=vector) for vector in vectors])


def test_embed_texts_returns_empty_list_for_empty_input():
    # Arrange / Act: no client call should happen at all
    result = embeddings_client.embed_texts([])

    # Assert
    assert result == []


def test_embed_texts_returns_vectors_in_order(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = _mock_embeddings_response(
        [[0.1, 0.2], [0.3, 0.4]]
    )
    monkeypatch.setattr(embeddings_client, "client", fake_client)

    # Act
    result = embeddings_client.embed_texts(["returns policy", "warranty"])

    # Assert
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    fake_client.embeddings.create.assert_called_once()


def test_embed_texts_retries_on_transient_error_then_succeeds(monkeypatch):
    # Arrange: first call raises a connection error, second succeeds
    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = [
        APIConnectionError(request=MagicMock()),
        _mock_embeddings_response([[0.5, 0.5]]),
    ]
    monkeypatch.setattr(embeddings_client, "client", fake_client)
    monkeypatch.setattr(embeddings_client.time, "sleep", lambda seconds: None)

    # Act
    result = embeddings_client.embed_texts(["returns policy"])

    # Assert
    assert result == [[0.5, 0.5]]
    assert fake_client.embeddings.create.call_count == 2


def test_embed_texts_raises_embedding_error_after_exhausting_retries(monkeypatch):
    # Arrange: every attempt fails
    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = APIConnectionError(request=MagicMock())
    monkeypatch.setattr(embeddings_client, "client", fake_client)
    monkeypatch.setattr(embeddings_client.time, "sleep", lambda seconds: None)

    # Act / Assert
    with pytest.raises(EmbeddingError):
        embeddings_client.embed_texts(["returns policy"])
