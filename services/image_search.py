"""
Fast, precomputed image-similarity search over the catalogue's product
photos.

Mirrors product_search.py's ProductIndex exactly on purpose -- same
reasoning, different embedding source: embed the catalogue once
(offline, see image_embeddings_sync.py), embed the query the same way,
retrieve by cosine similarity.

Why this exists, and why it replaced photo_match_tool.py's original
text-mediated narrowing: describing a photo in words first, then
searching the catalogue by that description, cascades badly when the
description is wrong -- confirmed live, 2026-08-18, a photo of loose
components was described as "cuff bracelet", and that one wrong word
meant the real matching product never reached the candidate shortlist
at all. Embedding the photo directly and comparing it to catalogue
photo embeddings removes that failure mode: there's no text description
in the narrowing step to go wrong, the comparison is photo-to-photo
throughout.

Unlike product_search.py's ProductIndex, this index's embeddings are
NOT computed lazily on first use -- image_embeddings_sync.py must be
run ahead of time (a real Cohere API call per catalogue photo, same
reasoning as woocommerce_sync.py being a separate offline step rather
than a live per-request call). This index only ever reads the
already-computed file; a missing or stale file means photo
identification silently has nothing to search, not a slow first
request.
"""

import base64
import json
import logging
import math
from pathlib import Path

from app.config import settings
from services.image_embed_tool import ImageEmbedError, embed_images

logger = logging.getLogger(__name__)

# Same reasoning as product_search.py's _DEFAULT_MIN_SCORE: below this
# similarity, treat it as "nothing in the catalogue looks like this"
# rather than confidently returning the least-bad guess. Cosine
# similarity on CLIP-style image embeddings for two genuinely different
# jewellery photos still tends to score moderately (they're all photos
# of gold/silver items against similar backgrounds), so this threshold
# is a starting point, not a measured one -- has NOT been tuned against
# real catalogue photos and a real customer photo yet (see this
# module's own docstring on live verification).
_DEFAULT_MIN_SCORE = 0.5


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ImageIndex:
    def __init__(self, embeddings_path: Path):
        self._embeddings_path = embeddings_path
        self._entries: list[dict] | None = None  # [{"image_url": ..., "embedding": [...]}]

    def _ensure_loaded(self) -> None:
        if self._entries is not None:
            return
        try:
            with open(self._embeddings_path, "r") as file:
                self._entries = json.load(file)
        except FileNotFoundError:
            logger.error(
                "Image embeddings file not found at %s -- run "
                "`python -m services.image_embeddings_sync` first.",
                self._embeddings_path,
            )
            self._entries = []
        except json.JSONDecodeError as e:
            logger.error("Image embeddings file at %s is not valid JSON: %s", self._embeddings_path, e)
            self._entries = []

    def search(
        self,
        image_bytes: bytes,
        mime_type: str,
        top_k: int = 6,
        min_score: float = _DEFAULT_MIN_SCORE,
    ) -> list[dict]:
        """Returns up to top_k {"image_url", "score"} entries, best
        match first, for catalogue photos most visually similar to the
        given image. Raises ImageEmbedError if the query photo itself
        can't be embedded (a genuine failure, distinct from "loaded
        fine, nothing matched well enough") -- callers should treat
        that as "image search unavailable this request", same as any
        other external API failure elsewhere in this codebase, not
        silently return an empty result indistinguishable from a real
        no-match."""
        self._ensure_loaded()
        if not self._entries:
            return []

        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        [query_embedding] = embed_images([data_url])

        scored = [
            {"image_url": entry["image_url"], "score": _cosine_similarity(query_embedding, entry["embedding"])}
            for entry in self._entries
        ]
        scored.sort(key=lambda e: e["score"], reverse=True)
        return [entry for entry in scored[:top_k] if entry["score"] >= min_score]

    def reload(self) -> None:
        """Call after image_embeddings_sync.py rewrites the embeddings
        file, so a running process picks up new/changed catalogue
        photos without a restart."""
        self._entries = None


_index = ImageIndex(settings.image_embeddings_path)


def get_image_index() -> ImageIndex:
    return _index
