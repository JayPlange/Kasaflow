"""
Thin wrapper around the OpenAI embeddings endpoint.

Kept separate from llm.py on purpose: llm.py's job is tool *selection*
(one JSON decision per message), this module's job is turning text into
vectors for retrieval. Different endpoint, different failure shape,
different caller (knowledge_base.py, not router.py).
"""

import logging
import time

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)


class EmbeddingError(Exception):
    """Raised when the embeddings API cannot be reached or returns something unusable."""


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in one request, preserving input order.

    Same retry philosophy as llm.py's _call_llm: transient network/timeout
    errors are worth retrying, auth or bad-request errors are not -- they
    will fail identically on the next attempt, so retrying just burns
    time and money.
    """
    if not texts:
        return []

    last_error: Exception | None = None
    total_attempts = settings.llm_max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            response = client.embeddings.create(model=settings.embedding_model, input=texts)
            return [item.embedding for item in response.data]

        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            logger.warning(
                "Embeddings call failed (attempt %s/%s): %s", attempt, total_attempts, e
            )
            if attempt < total_attempts:
                time.sleep(min(2**attempt, 8))  # 2s, 4s, 8s...

        except APIError as e:
            logger.error("Non-retryable embeddings API error: %s", e)
            raise EmbeddingError(f"Embeddings request failed: {e}") from e

    raise EmbeddingError(f"Embeddings API unreachable after {total_attempts} attempts: {last_error}")
