"""
Identifies the EXACT catalogue product in a customer's photo, not just
its category.

Why two stages, not one: a single vision call can describe a photo
("gold pendant necklace with chain link design") but can't reliably
guess a specific marketing name ("Custom Adinkra Chains Gold Necklace
with Earrings, 30g") from visual features alone -- no catalogue name is
inferable purely by looking at a chain pattern. So this narrows to a
handful of visually-plausible candidates first, then asks the vision
model to compare the customer's photo against just those candidates'
real catalogue photos -- "which of these five, or none" is a much
narrower and more answerable question than "which of 3,918 products is
this".

Narrowing is now IMAGE-native (services/image_search.py's precomputed
Cohere embeddings), not text-mediated. It was originally built as a
text search over a vision-generated description of the photo, but that
cascades badly when the description is wrong: confirmed live,
2026-08-18, a photo of loose components was described as "cuff
bracelet with teardrop ends", and that one wrong word steered the text
search away from the real matching product entirely -- it never
reached the shortlist for the visual comparison step to even consider.
Comparing photo embeddings directly removes that failure mode: there's
no text description in the narrowing step to go wrong.

The vision comparison step (services/vision_tool.py's
match_photo_to_candidates) is kept even with image-native narrowing --
image embeddings are good at "roughly the same style/category" but can
still conflate genuinely different, visually similar products (two
different cross-necklace-and-earring sets, confirmed live, 2026-08-18);
the vision model's finer-grained comparison is the best tool available
for that last distinction.

Returns None, not a guess, when nothing is confidently identified --
callers should fall back to the ordinary text-driven pipeline rather
than presenting a low-confidence match as a real identification (same
no-fabrication principle response_formatter.py's module docstring
states).
"""

import json
import logging
from urllib.parse import urlparse

import requests

from app.config import settings
from services.image_embed_tool import ImageEmbedError
from services.image_search import get_image_index
from services.vision_tool import match_photo_to_candidates

logger = logging.getLogger(__name__)

# How many distinct products to show the vision model per photo. Kept
# small deliberately: each candidate is a whole extra image in the same
# API call.
_MAX_CANDIDATES = 6

# Same hotlinking workaround as app/demo_routes.py's image_proxy: this
# WordPress host rejects cross-origin image requests without the
# store's own origin as Referer (confirmed there, images silently
# failed to render without it). Duplicated rather than imported from
# demo_routes.py on purpose -- importing a route module from a service
# module would invert the app -> services dependency direction the rest
# of this codebase keeps.
_ALLOWED_IMAGE_HOST = urlparse(settings.woocommerce_url).hostname if settings.woocommerce_url else None


def _fetch_candidate_image(url: str) -> bytes | None:
    """Best-effort fetch of one candidate's catalogue photo. Returns
    None on any failure (disallowed host, network error, non-2xx) --
    a single bad candidate photo shouldn't take down the whole match
    attempt, it should just be skipped."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    if _ALLOWED_IMAGE_HOST and parsed.hostname != _ALLOWED_IMAGE_HOST:
        logger.warning("Refused to fetch candidate image from disallowed host: %s", parsed.hostname)
        return None
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; KasaFlowDemo/1.0)",
                "Referer": f"https://{parsed.hostname}/",
            },
        )
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logger.warning("Candidate image fetch failed for %s: %s", url, e)
        return None


def _product_name_for_image_url(image_url: str) -> str | None:
    """image_search.py's ImageIndex only knows about image_url<->embedding
    (deliberately -- it has no reason to know about products, karats, or
    prices, see its own module docstring). This is the one place that
    maps a matched photo back to the product name it belongs to, by
    reading the same catalogue file every other tool in this codebase
    reads independently (see product_tool.py's get_product_price for
    the same pattern)."""
    try:
        with open(settings.products_path, "r") as file:
            products = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Could not read products file to resolve image match: %s", e)
        return None
    return next((p["product"] for p in products if p.get("image_url") == image_url), None)


def identify_product_from_photo(image_bytes: bytes, mime_type: str) -> str | None:
    """Returns the matched product name, or None when there's no
    confident match."""
    try:
        image_matches = get_image_index().search(image_bytes, mime_type, top_k=_MAX_CANDIDATES)
    except ImageEmbedError as e:
        logger.warning("Image search unavailable, cannot attempt a photo match: %s", e)
        return None

    if not image_matches:
        return None

    candidates: list[tuple[str, bytes, str]] = []
    for match in image_matches:
        image_url = match["image_url"]
        name = _product_name_for_image_url(image_url)
        if name is None:
            continue
        candidate_bytes = _fetch_candidate_image(image_url)
        if candidate_bytes is None:
            continue
        candidates.append((name, candidate_bytes, "image/jpeg"))

    if not candidates:
        logger.info("No candidate photos could be resolved/fetched -- skipping visual match")
        return None

    matched_index = match_photo_to_candidates(image_bytes, mime_type, candidates)
    if matched_index is None:
        return None
    return candidates[matched_index][0]
