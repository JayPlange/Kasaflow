"""
Thin wrapper around Cohere's Embed v4 API for image embeddings.

Why Cohere, not OpenAI: as of this writing, OpenAI has no hosted
image-embedding endpoint at all -- text-embedding-3-small/large are
text-only, and GPT-4o/vision calls (see vision_tool.py) return a text
description of an image, not a reusable vector. Confirmed against
OpenAI's own API reference before reaching for a second vendor, not
assumed (2026-08-18).

Why this exists: services/photo_match_tool.py's original design
narrowed candidates by text (describe the photo, then search the
catalogue by that description) before comparing photos. That cascades
badly when the initial text description is wrong -- confirmed live,
2026-08-18: a photo of loose components was described as "cuff
bracelet", and that wrong word steered the search away from the real
matching product entirely, so it never even reached the shortlist.
Image embeddings let services/image_search.py narrow candidates by
directly comparing photos, with no text description in the loop to go
wrong.

Confirmed against Cohere's real v2 embed API reference
(docs.cohere.com/v2/reference/embed) at the time this was written --
request shape (model, input_type="image", images=[data URIs]) and
response shape (embeddings.float, one vector per input image, same
order) both come from that page, not assumption. NOT yet exercised
against a real, authenticated Cohere call from this environment (no
outbound access to api.cohere.com here, and no API key configured) --
flagging that explicitly rather than presenting this as verified, the
same discipline vision_tool.py followed before its own first live
test. Needs one real check (a real COHERE_API_KEY, one real image)
before this is trusted in front of a customer.
"""

import logging

import cohere

from app.config import settings

logger = logging.getLogger(__name__)

# embed-v4.0: the only Cohere Embed model version confirmed (per the
# same API reference above) to support more than one image per call --
# earlier v3.x models cap at 1 image/call, which would mean one HTTP
# round-trip per catalogue photo when precomputing embeddings for the
# whole catalogue, not just per customer query.
_MODEL = "embed-v4.0"

# Cohere's default output dimension for embed-v4 is 1536. Kept explicit
# rather than left to the API default -- the offline sync
# (image_embeddings_sync.py) and the live per-request search
# (image_search.py) both call embed_images(), and if Cohere's own
# default ever changes, a silent dimension mismatch between
# already-stored catalogue embeddings and a freshly-embedded query
# photo would break cosine similarity in a confusing way, not a loud
# one. 512 is the smallest supported size and plenty for a same-item
# visual match (not a fine-grained classification task) -- smaller
# also means less to store per catalogue photo and a faster search.
_OUTPUT_DIMENSION = 512


class ImageEmbedError(Exception):
    """Raised when the Cohere embed API can't be reached, isn't configured, or returns something unusable."""


def _client() -> cohere.ClientV2:
    if not settings.cohere_api_key:
        raise ImageEmbedError("COHERE_API_KEY is not configured.")
    return cohere.ClientV2(api_key=settings.cohere_api_key)


def embed_images(image_data_urls: list[str]) -> list[list[float]]:
    """Returns one embedding vector per input data URI, in the same
    order as the input list. Raises ImageEmbedError on any failure --
    callers decide whether that's fatal (image_embeddings_sync.py, the
    offline catalogue build) or something to fall back gracefully from
    (image_search.py, the live per-request path).

    image_data_urls: each a data URI (e.g. "data:image/jpeg;base64,...")
    -- Cohere's documented image formats are jpeg, png, webp, and gif.
    """
    if not image_data_urls:
        return []

    try:
        response = _client().embed(
            model=_MODEL,
            input_type="image",
            images=image_data_urls,
            embedding_types=["float"],
            output_dimension=_OUTPUT_DIMENSION,
        )
    except ImageEmbedError:
        raise
    except Exception as e:
        # Cohere's SDK exceptions aren't imported/narrowed by type here
        # -- unlike vision_tool.py's use of the openai package (whose
        # exact exception hierarchy is well-documented and already
        # relied on elsewhere in this codebase), this integration is
        # brand new and its real failure shapes haven't been observed
        # yet. Catching broadly and wrapping is the honest choice until
        # a live run shows what actually needs distinguishing (a
        # timeout vs. a bad request vs. an auth failure).
        raise ImageEmbedError(f"Cohere embed request failed: {e}") from e

    try:
        embeddings = response.embeddings.float
    except AttributeError:
        # The documented response body is {"embeddings": {"float": [...]}}
        # -- some SDK versions may expose this as dict access rather
        # than attribute access. Handling both since this hasn't been
        # exercised against a real response from this environment (see
        # module docstring).
        embeddings = response.embeddings["float"]

    return list(embeddings)
