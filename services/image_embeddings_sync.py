"""
Pre-computes an image embedding for every distinct product photo in the
catalogue via Cohere's Embed v4 API, and writes the result to
data/image_embeddings.json (services/image_search.py reads this file at
runtime).

Why offline, not live: identify_product_from_photo() needs to compare a
customer's photo against every catalogue photo on each request --
embedding the whole catalogue on every request would be slow and
needlessly expensive. Embedding each photo once here, on a schedule
(same reasoning and cadence as woocommerce_sync.py), means the live
request path only ever embeds the ONE new customer photo against
already-computed catalogue vectors.

One embedding per distinct image_url, not per catalogue row -- most
rows are karat/size variants of the same product sharing one photo
(see product_search.py's own module docstring for the same pattern),
so embedding every row would mean paying for and storing the same
photo's embedding dozens of times over for nothing.

Usage:
    python -m services.image_embeddings_sync

Requires COHERE_API_KEY in .env. Also requires data/products.json to
already be up to date -- run `python -m services.woocommerce_sync`
first if catalogue photos may have changed since the last sync.

NOT yet run against the real catalogue or a real Cohere account from
this environment (no outbound access to either adomdejeweller.com's
images or api.cohere.com here, and no COHERE_API_KEY configured) --
this is the one place that first real run needs to happen, the same
way woocommerce_sync.py's first real run was what actually proved that
integration out.
"""

import base64
import json
import logging
import sys
import time
from urllib.parse import urlparse

import requests

from app.config import settings
from services.image_embed_tool import ImageEmbedError, embed_images

logger = logging.getLogger(__name__)

# Cohere's embed-v4.0 has no documented per-call image count limit, but
# does cap the combined request size at 20MB -- batching a modest,
# fixed number of photos per call rather than either "one at a time"
# (many more HTTP round-trips than necessary) or "everything in one
# call" (an unverified assumption about how large real catalogue photos
# are) is the conservative middle ground until a real run shows the
# actual numbers.
_BATCH_SIZE = 10

# Same hotlinking workaround already established in
# services/photo_match_tool.py and app/demo_routes.py's image_proxy --
# this WordPress host rejects cross-origin image requests without the
# store's own origin as Referer.
_ALLOWED_IMAGE_HOST = urlparse(settings.woocommerce_url).hostname if settings.woocommerce_url else None


def _require_cohere_config() -> None:
    if not settings.cohere_api_key:
        raise RuntimeError("COHERE_API_KEY is not set. Add it to your .env file before running this.")


def _distinct_image_urls(products: list[dict]) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for product in products:
        url = product.get("image_url")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _fetch_image_bytes(url: str) -> bytes | None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    if _ALLOWED_IMAGE_HOST and parsed.hostname != _ALLOWED_IMAGE_HOST:
        logger.warning("Refused to fetch image from disallowed host: %s", parsed.hostname)
        return None
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; KasaFlowImageSync/1.0)",
                "Referer": f"https://{parsed.hostname}/",
            },
        )
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logger.warning("Image fetch failed for %s: %s", url, e)
        return None


def _mime_type_for(url: str) -> str:
    lower = url.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def build_image_embeddings() -> list[dict]:
    _require_cohere_config()

    with open(settings.products_path, "r") as file:
        products = json.load(file)

    urls = _distinct_image_urls(products)
    logger.info("Found %d distinct catalogue photos to embed", len(urls))

    results: list[dict] = []
    batch_urls: list[str] = []
    batch_data_urls: list[str] = []

    def _flush() -> None:
        if not batch_data_urls:
            return
        try:
            embeddings = embed_images(batch_data_urls)
        except ImageEmbedError as e:
            logger.warning("Embedding batch failed (%d photos skipped): %s", len(batch_data_urls), e)
            return
        for url, embedding in zip(batch_urls, embeddings):
            results.append({"image_url": url, "embedding": embedding})

    for url in urls:
        image_bytes = _fetch_image_bytes(url)
        if image_bytes is None:
            logger.warning("Skipping %s -- could not fetch", url)
            continue
        data_url = f"data:{_mime_type_for(url)};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        batch_urls.append(url)
        batch_data_urls.append(data_url)

        if len(batch_data_urls) >= _BATCH_SIZE:
            _flush()
            batch_urls, batch_data_urls = [], []
            time.sleep(0.2)  # light pacing, not a documented rate limit -- just cheap insurance

    _flush()
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    embeddings = build_image_embeddings()

    if not embeddings:
        logger.error("Embedded 0 photos -- refusing to overwrite an existing embeddings file with nothing.")
        sys.exit(1)

    with open(settings.image_embeddings_path, "w") as file:
        json.dump(embeddings, file)

    logger.info("Wrote %d image embeddings to %s", len(embeddings), settings.image_embeddings_path)


if __name__ == "__main__":
    main()
