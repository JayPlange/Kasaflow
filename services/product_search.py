"""
Semantic search over the product catalogue.

Why this exists: get_product_price's original exact-match lookup was
written against placeholder data ("ring"/"gold", "chain"/"gold") where
exact match was the whole problem. The real catalogue from
adomdejeweller.com has specific, individual product names like "Gye
Nyame White Necklace with Earrings, 30g" -- a customer asking for "a
gold necklace" will never string-match that, no matter how the LLM
normalises their message. This mirrors knowledge_base.py's approach
exactly: embed the catalogue once, embed the query the same way,
retrieve by cosine similarity. Same reasoning, different documents.
"""

import json
import logging
import math
import re
from pathlib import Path

from app.config import settings
from services.embeddings_client import embed_texts

logger = logging.getLogger(__name__)

# Below this similarity, treat it as "nothing in the catalogue matches"
# rather than confidently returning the least-bad guess. Getting this
# wrong in the direction of "too eager" means a customer asking for a
# watch could get quoted a ring's price -- worse than saying "I don't
# have that."
_DEFAULT_MIN_SCORE = 0.3

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _searchable_text(entry: dict) -> str:
    # Category folded in on purpose: "necklace" needs to match products
    # whose name doesn't literally contain the word "necklace" but whose
    # category does (e.g. a named piece like "Buzzing Solid Chain...").
    parts = [entry.get("product", ""), entry.get("category") or "", entry.get("material", "")]
    return " ".join(p for p in parts if p)


# _WORD_RE only matches letters, so a query like "14k Big Stone Yellow
# Gold Ring, 15g" (product_tool.py builds queries as "{material}
# {product_name}", and karats/weights are almost always written as
# digits+letter) tokenizes the digits away entirely, leaving bare
# single-letter fragments like "k" and "g". Below this length, the
# substring-tolerant check two lines down becomes actively wrong rather
# than lenient: "g" is a substring of nearly every product name in a
# jewellery catalogue (weights are almost always "...Ng"), so a query
# for one specific ring was scoring a real, unrelated ring *higher* than
# the correct one purely because its name happened to contain a "g" or
# a "k" somewhere -- confirmed live, 2026-08-13: a customer asked for
# "Big Stone Yellow Gold Ring, 15g" in 14k, an exact match this store
# genuinely has, and got quoted and nearly ordered a completely
# different item ("Sparkling Small Stone Yellow Gold Ring, 16g")
# instead, because "big" spuriously matched via a bare "g" fragment in
# the wrong product's own "16g" weight suffix. Filtering out anything
# shorter than 3 letters removes the fragments without weakening the
# real substring tolerance this function still needs for genuine partial
# words like "bracelet"/"Bracelets".
_MIN_KEYWORD_LENGTH = 3


def _keyword_overlap(query: str, product_name: str) -> int:
    """How many of the query's words literally appear in this product's
    name (substring-tolerant both ways, so "bracelet" still counts
    against "Bracelets" without needing real stemming).

    Why this exists: cosine similarity alone put a generic product
    ("Golden Necklace, 10g") ahead of a literal, more specific match
    ("Chain Gold Necklace, 50g") for the query "gold chain" -- a real,
    measured near-tie (0.5977 vs 0.5867) that the wrong side won. A
    customer who says "chain" and means it should out-rank a product
    that merely sounds similar in embedding space but doesn't actually
    say "chain" anywhere in its name.
    """
    query_words = {w for w in _WORD_RE.findall(query.lower()) if len(w) >= _MIN_KEYWORD_LENGTH}
    name_words = {w for w in _WORD_RE.findall(product_name.lower()) if len(w) >= _MIN_KEYWORD_LENGTH}
    return sum(
        1
        for qw in query_words
        if any(qw in nw or nw in qw for nw in name_words)
    )


class ProductIndex:
    def __init__(self, products_path: Path):
        self._products_path = products_path
        self._entries: list[dict] | None = None
        self._embeddings: list[list[float]] | None = None

    def _ensure_loaded(self) -> None:
        if self._entries is not None:
            return

        try:
            with open(self._products_path, "r") as file:
                self._entries = json.load(file)
        except FileNotFoundError:
            logger.error("Products file not found at %s", self._products_path)
            self._entries = []
        except json.JSONDecodeError as e:
            logger.error("Products file at %s is not valid JSON: %s", self._products_path, e)
            self._entries = []

        self._embeddings = (
            embed_texts([_searchable_text(entry) for entry in self._entries])
            if self._entries
            else []
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = _DEFAULT_MIN_SCORE,
        min_keyword_overlap: int = 0,
    ) -> list[dict]:
        self._ensure_loaded()
        if not self._entries:
            return []

        query_embedding = embed_texts([query])[0]
        scored = [
            {
                **entry,
                "score": _cosine_similarity(query_embedding, entry_embedding),
                "_keyword_overlap": _keyword_overlap(query, entry.get("product", "")),
            }
            for entry, entry_embedding in zip(self._entries, self._embeddings)
        ]
        # Literal keyword overlap decides first, cosine similarity breaks
        # ties within the same overlap count. This deliberately overrides
        # embedding rank when a candidate actually contains a word the
        # customer used and another, higher-cosine candidate doesn't --
        # see _keyword_overlap's docstring for the measured case that
        # motivated this.
        scored.sort(key=lambda e: (e["_keyword_overlap"], e["score"]), reverse=True)
        filtered = [
            entry
            for entry in scored[:top_k]
            if entry["score"] >= min_score and entry["_keyword_overlap"] >= min_keyword_overlap
        ]
        for entry in filtered:
            entry.pop("_keyword_overlap", None)
        return filtered

    def reload(self) -> None:
        """Call after woocommerce_sync.py rewrites products.json, so a
        running process picks up new prices without a restart."""
        self._entries = None
        self._embeddings = None


_index = ProductIndex(settings.products_path)


def get_product_index() -> ProductIndex:
    return _index
