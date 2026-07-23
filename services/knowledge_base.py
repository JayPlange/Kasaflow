"""
Retrieval over KasaFlow's policy/FAQ documents -- the RAG piece.

Why this exists: get_product_price and friends answer questions with a
single deterministic fact (a price, a delivery window). Policy questions
("what's your returns policy", "how do I clean a ring") don't fit that
shape -- the honest answer is a paragraph of written policy, not a
number. Two bad ways to handle that: let the LLM answer from its own
idea of what the policy probably is (it will occasionally invent one),
or stuff every policy document into every prompt (expensive, and still
requires the model to pick the right paragraph out of an unrelated
pile). RAG is the middle path: embed the documents once, embed the
customer's question the same way, and retrieve only the document(s)
that actually match by cosine similarity. The model never sees -- and
therefore can't hallucinate past -- the policies it wasn't asked about.

Documents are embedded lazily on first use and cached in memory for the
life of the process, not re-embedded on every question. The document
set is small and changes rarely, so re-embedding per request would be
pure waste.
"""

import json
import logging
import math
from pathlib import Path

from app.config import settings
from services.embeddings_client import embed_texts

logger = logging.getLogger(__name__)

# Below this cosine similarity, treat it as "no real match" rather than
# force-returning the least-bad document. A generic or off-topic
# question should get "I don't have a policy for that", not whichever
# document happened to score highest.
_DEFAULT_MIN_SCORE = 0.15


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class KnowledgeBase:
    """Loads policy documents and answers similarity queries against them."""

    def __init__(self, documents_path: Path):
        self._documents_path = documents_path
        self._documents: list[dict] | None = None
        self._embeddings: list[list[float]] | None = None

    def _ensure_loaded(self) -> None:
        if self._documents is not None:
            return

        try:
            with open(self._documents_path, "r") as file:
                self._documents = json.load(file)
        except FileNotFoundError:
            logger.error("Policy documents file not found at %s", self._documents_path)
            self._documents = []
        except json.JSONDecodeError as e:
            logger.error(
                "Policy documents file at %s is not valid JSON: %s", self._documents_path, e
            )
            self._documents = []

        self._embeddings = embed_texts([doc["text"] for doc in self._documents]) if self._documents else []

    def retrieve(self, query: str, top_k: int = 2, min_score: float = _DEFAULT_MIN_SCORE) -> list[dict]:
        """Return up to top_k documents whose embedding is closest to the query's,
        each annotated with its similarity score, filtered to min_score and above.
        """
        self._ensure_loaded()
        if not self._documents:
            return []

        query_embedding = embed_texts([query])[0]
        scored = [
            {**doc, "score": _cosine_similarity(query_embedding, doc_embedding)}
            for doc, doc_embedding in zip(self._documents, self._embeddings)
        ]
        scored.sort(key=lambda d: d["score"], reverse=True)
        return [doc for doc in scored[:top_k] if doc["score"] >= min_score]

    def reload(self) -> None:
        """Force re-reading the documents file and re-embedding on next use.

        Not called in normal operation -- the document set doesn't change
        while the process is running -- but useful for tests, and for an
        eventual admin endpoint if policies get edited without a restart.
        """
        self._documents = None
        self._embeddings = None


_kb = KnowledgeBase(settings.policies_path)


def get_knowledge_base() -> KnowledgeBase:
    """Exposed so tests (and callers) can reach the module-level instance
    without importing the private `_kb` directly -- same pattern as
    services.memory.get_session_store().
    """
    return _kb
