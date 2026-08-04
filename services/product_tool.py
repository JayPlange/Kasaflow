"""
Deterministic lookup of a product's price. Intentionally dumb: the LLM
decided *that* a price lookup should happen, this function just does it.

Tries an exact product+material match first (fast, no API call, and
exactly right for the placeholder catalogue this shipped with). Falls
back to semantic search (services/product_search.py) when nothing
matches exactly -- necessary once data/products.json holds real
WooCommerce product names instead of generic categories, since "gold
ring" will never exact-match "Heart Twin Gold Ring, 16g".
"""

import json
import logging

from app.config import settings
from services.product_search import get_product_index

logger = logging.getLogger(__name__)


def get_product_price(product_name: str, material: str):
    try:
        with open(settings.products_path, "r") as file:
            products = json.load(file)
    except FileNotFoundError:
        logger.error("Products file not found at %s", settings.products_path)
        return None
    except json.JSONDecodeError as e:
        logger.error("Products file at %s is not valid JSON: %s", settings.products_path, e)
        return None

    for product in products:
        if product["product"] == product_name and product["material"] == material:
            return product

    logger.info(
        "No exact match for product_name=%s material=%s -- falling back to semantic search",
        product_name,
        material,
    )
    # Same "fail gracefully, not raise" rule already applied to the file
    # errors above -- a price lookup should never take down the whole
    # customer request just because the embeddings call had a bad day
    # (network issue, invalid key, OpenAI outage). Worth knowing in logs,
    # not worth an unhandled exception reaching the customer.
    try:
        matches = get_product_index().search(f"{material} {product_name}", top_k=1)
    except Exception as e:
        logger.error("Semantic search failed, falling back to no match: %s", e)
        return None

    if matches:
        best = matches[0]
        logger.info(
            "Semantic match: %r (score=%.3f) for query %r",
            best["product"],
            best["score"],
            f"{material} {product_name}",
        )
        return {k: v for k, v in best.items() if k != "score"}

    logger.info("No product match at all for product_name=%s material=%s", product_name, material)
    return None
