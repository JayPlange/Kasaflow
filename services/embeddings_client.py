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

# max_retries=0: see vision_tool.py's client for why -- this module
# already implements its own retry/backoff loop, so the SDK's own
# default internal retries only add compounding delay on top of it
# without adding any real reliability.
client = OpenAI(api_key=settings.openai_api_key, max_retries=0)

# OpenAI's embeddings endpoint hard-rejects an `input` array longer than
# 2048 entries (a single request, regardless of token count). Fine
# against the placeholder catalogue (4 entries) or a small policy doc
# set, but the real adomdejeweller.com catalogue is 3,918 products --
# comfortably over the limit -- so a single unbatched call to this
# function against the real catalogue fails outright with a 400. Chunk
# every call so this function stays correct regardless of how large the
# caller's input list is, rather than pushing that limit onto every caller.
_MAX_BATCH_SIZE = 2048


class EmbeddingError(Exception):
    """Raised when the embeddings API cannot be reached or returns something unusable."""


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, preserving input order.

    Transparently splits into <=2048-item batches (OpenAI's per-request
    array limit) and concatenates the results in the original order, so
    callers never need to know or care how large their input list is.
    """
    if not texts:
        return []

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), _MAX_BATCH_SIZE):
        embeddings.extend(_embed_batch(texts[start : start + _MAX_BATCH_SIZE]))
    return embeddings


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a single batch (already <=2048 items) in one request.

    Same retry philosophy as llm.py's _call_llm: transient network/timeout
    errors are worth retrying, auth or bad-request errors are not -- they
    will fail identically on the next attempt, so retrying just burns
    time and money.
    """
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
