"""
Pulls the real product catalogue from the jeweller's WooCommerce store and
writes it to data/products.json, replacing the hand-written placeholder
data that shipped with the engine.

Why a separate sync step rather than calling WooCommerce live on every
customer message: the catalogue changes maybe a few times a week, not per
request, and a live call on the hot path adds a third-party dependency
(and its latency/failure modes) to every single customer interaction for
no benefit. Run this on a schedule instead -- a cron job, a GitHub Action,
or just by hand after a known price change -- and the request path stays
exactly as fast and as simple as it already is.

Usage:
    python -m services.woocommerce_sync

Requires WOOCOMMERCE_URL, WOOCOMMERCE_CONSUMER_KEY, and
WOOCOMMERCE_CONSUMER_SECRET in .env. Generate the key/secret in
WordPress admin: WooCommerce -> Settings -> Advanced -> REST API ->
Add key (Read permissions are enough, this script never writes back).
"""

import json
import logging
import sys
import time

import requests

from app.config import settings

logger = logging.getLogger(__name__)

PER_PAGE = 100
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 3  # 3s, 6s, 12s between retries


def _get_with_retry(url: str, **kwargs) -> requests.Response:
    """adomdejeweller.com is on shared WordPress hosting -- it occasionally
    stalls past our 30s timeout on a single request, not because anything
    is actually wrong, just a slow response. Retrying the same request a
    few times with backoff clears that without needing a human to re-run
    the whole sync by hand."""
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, **kwargs)
            response.raise_for_status()
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt == MAX_ATTEMPTS:
                break
            wait = BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Request to %s timed out (attempt %d/%d) -- retrying in %ds",
                url, attempt, MAX_ATTEMPTS, wait,
            )
            time.sleep(wait)
    raise last_error


def _require_woocommerce_config() -> None:
    missing = [
        name
        for name, value in [
            ("WOOCOMMERCE_URL", settings.woocommerce_url),
            ("WOOCOMMERCE_CONSUMER_KEY", settings.woocommerce_consumer_key),
            ("WOOCOMMERCE_CONSUMER_SECRET", settings.woocommerce_consumer_secret),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing WooCommerce config: {', '.join(missing)}. Add them to .env."
        )


def _fetch_all_products() -> list[dict]:
    """Paginate through every product in the store. WooCommerce caps
    per_page at 100, so a store with more than 100 products needs more
    than one request -- this loops until a page comes back short."""
    products = []
    page = 1
    auth = (settings.woocommerce_consumer_key, settings.woocommerce_consumer_secret)

    while True:
        response = _get_with_retry(
            f"{settings.woocommerce_url.rstrip('/')}/wp-json/wc/v3/products",
            params={"per_page": PER_PAGE, "page": page, "status": "publish"},
            auth=auth,
            timeout=30,
        )
        batch = response.json()
        if not batch:
            break
        products.extend(batch)
        if len(batch) < PER_PAGE:
            break
        page += 1

    return products


def _fetch_variations(product_id: int) -> list[dict]:
    auth = (settings.woocommerce_consumer_key, settings.woocommerce_consumer_secret)
    response = _get_with_retry(
        f"{settings.woocommerce_url.rstrip('/')}/wp-json/wc/v3/products/{product_id}/variations",
        params={"per_page": PER_PAGE},
        auth=auth,
        timeout=30,
    )
    return response.json()


def _variation_label(variation: dict) -> str | None:
    """WooCommerce variations carry their distinguishing attribute (e.g.
    Karat: 18k) in `attributes`, not as a plain string. Join every
    attribute this variation sets so a two-attribute product (Karat +
    Silver Alloy Option, as adomdejeweller.com's necklaces have) doesn't
    silently collapse to just one of them."""
    attrs = variation.get("attributes", [])
    labels = [a["option"] for a in attrs if a.get("option")]
    return " / ".join(labels) if labels else None


def _product_image_url(product: dict) -> str | None:
    """First image in WooCommerce's `images` array -- that's the product's
    main/featured photo, same one shown first on the site."""
    images = product.get("images") or []
    return images[0]["src"] if images and images[0].get("src") else None


def _variation_image_url(variation: dict, fallback: str | None) -> str | None:
    """A variation carries its own `image` (singular) only when it looks
    different from the parent product (e.g. rose gold vs yellow gold of
    the same design) -- fall back to the parent's image when it doesn't."""
    image = variation.get("image")
    if image and image.get("src"):
        return image["src"]
    return fallback


def build_catalogue() -> list[dict]:
    _require_woocommerce_config()
    raw_products = _fetch_all_products()
    catalogue = []

    for product in raw_products:
        image_url = _product_image_url(product)
        base_entry = {
            "product": product["name"],
            "category": product["categories"][0]["name"] if product.get("categories") else None,
            "in_stock": product.get("stock_status") == "instock",
            "url": product.get("permalink"),
            "image_url": image_url,
        }

        if product.get("type") == "variable" and product.get("variations"):
            for variation in _fetch_variations(product["id"]):
                price = variation.get("price")
                if price in (None, ""):
                    continue
                catalogue.append(
                    {
                        **base_entry,
                        "material": _variation_label(variation) or "standard",
                        "price": float(price),
                        "in_stock": variation.get("stock_status", base_entry["in_stock"]) == "instock"
                        if isinstance(variation.get("stock_status"), str)
                        else base_entry["in_stock"],
                        "image_url": _variation_image_url(variation, image_url),
                    }
                )
        else:
            price = product.get("price")
            if price in (None, ""):
                logger.warning("Skipping %r -- no price set", product.get("name"))
                continue
            catalogue.append(
                {**base_entry, "material": "standard", "price": float(price)}
            )

    return catalogue


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    catalogue = build_catalogue()

    if not catalogue:
        logger.error("Fetched 0 products -- refusing to overwrite an existing catalogue with nothing.")
        sys.exit(1)

    with open(settings.products_path, "w") as file:
        json.dump(catalogue, file, indent=2)

    logger.info("Wrote %d product entries to %s", len(catalogue), settings.products_path)


if __name__ == "__main__":
    main()
