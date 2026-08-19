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


def test_embeddings_client_disables_sdk_level_retries():
    # Same fix as vision_tool.py and llm.py's clients -- this module
    # already implements its own retry/backoff loop, so the SDK's own
    # default internal retries only add compounding delay on top of it.
    assert embeddings_client.client.max_retries == 0


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


def test_embed_texts_batches_requests_over_the_limit(monkeypatch):
    """OpenAI hard-rejects an `input` array over 2048 items in one
    request -- the real adomdejeweller.com catalogue (3,918 products)
    exceeds that, so embed_texts must chunk transparently. Using a
    monkeypatched, much smaller batch size here so the test stays fast
    without needing an actual 2048+ item list to exercise the same logic."""
    # Arrange
    monkeypatch.setattr(embeddings_client, "_MAX_BATCH_SIZE", 2)
    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = [
        _mock_embeddings_response([[1.0], [2.0]]),  # batch 1: items 0-1
        _mock_embeddings_response([[3.0], [4.0]]),  # batch 2: items 2-3
        _mock_embeddings_response([[5.0]]),          # batch 3: item 4
    ]
    monkeypatch.setattr(embeddings_client, "client", fake_client)

    texts = ["a", "b", "c", "d", "e"]

    # Act
    result = embeddings_client.embed_texts(texts)

    # Assert: three requests made (2+2+1), results concatenated in the
    # original order, and each request only ever saw its own slice
    assert fake_client.embeddings.create.call_count == 3
    assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    call_inputs = [call.kwargs["input"] for call in fake_client.embeddings.create.call_args_list]
    assert call_inputs == [["a", "b"], ["c", "d"], ["e"]]


def test_embed_texts_single_batch_under_the_limit_makes_one_request(monkeypatch):
    # Arrange
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = _mock_embeddings_response([[0.1], [0.2], [0.3]])
    monkeypatch.setattr(embeddings_client, "client", fake_client)

    # Act
    result = embeddings_client.embed_texts(["a", "b", "c"])

    # Assert: well under _MAX_BATCH_SIZE (2048) -- exactly one request, no chunking
    assert fake_client.embeddings.create.call_count == 1
    assert result == [[0.1], [0.2], [0.3]]


def test_embed_texts_raises_embedding_error_after_exhausting_retries(monkeypatch):
    # Arrange: every attempt fails
    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = APIConnectionError(request=MagicMock())
    monkeypatch.setattr(embeddings_client, "client", fake_client)
    monkeypatch.setattr(embeddings_client.time, "sleep", lambda seconds: None)

    # Act / Assert
    with pytest.raises(EmbeddingError):
        embeddings_client.embed_texts(["returns policy"])
