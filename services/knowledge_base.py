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

# Second, independent gate on top of the similarity score -- 2026-08-20
# architecture audit, failure #6: with only ~6 short documents and a
# 0.15 threshold, a coincidentally above-threshold match against the
# WRONG document is a structural risk, not a one-off (this is exactly
# what happened live: a customer disputing which karat their order used
# scored above 0.15 against warranty_policy and got warranty text back).
# A document is only actually returned if the query also contains a
# plausible keyword for that document's real topic -- "I found a
# document" is not the same claim as "I found an answer to this
# question". Deliberately hand-written and short rather than derived
# from the document text automatically: the failure mode here is
# semantic near-misses, so the second gate needs to be a different kind
# of signal (literal topic words), not another embedding comparison that
# would share the same blind spot. A document with no entry here fails
# open (see _topic_matches()) rather than silently becoming unreachable
# the moment it's added to policies.json without this table being
# updated too -- add an entry here whenever a new policy document is added.
_TOPIC_KEYWORDS: dict[str, set[str]] = {
    "returns_policy": {"return", "returns", "refund", "refunds", "money back", "send back", "exchange"},
    "warranty_policy": {
        "warranty", "warrantee", "guarantee", "defect", "defective", "broken", "repair",
        "replace", "replacement", "faulty", "malfunction",
    },
    "ring_sizing": {"size", "sizes", "sizing", "resize", "resizing", "fit", "too big", "too small", "finger"},
    "jewellery_care": {
        "clean", "cleaning", "care", "store", "storage", "tarnish", "polish", "shower",
        "swim", "swimming", "chemical", "lotion", "perfume",
    },
    "custom_engraving": {"engrave", "engraving", "engraved", "inscription", "personalise", "personalize", "personalised", "personalized"},
    "payment_methods": {
        "pay", "payment", "payments", "card", "cards", "mobile money", "momo",
        "instalment", "instalments", "installment", "installments", "afterpay",
        "buy now pay later", "checkout",
    },
}


def _topic_matches(doc_id: str, query_lower: str) -> bool:
    keywords = _TOPIC_KEYWORDS.get(doc_id)
    if not keywords:
        return True
    return any(keyword in query_lower for keyword in keywords)


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
        each annotated with its similarity score, filtered to min_score and above,
        AND filtered again by _topic_matches() -- see that function's comment
        for why a similarity score alone isn't trusted as "this is the answer".
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
        query_lower = query.lower()
        return [
            doc for doc in scored[:top_k]
            if doc["score"] >= min_score and _topic_matches(doc["id"], query_lower)
        ]

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
