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
import re

from app.config import settings
from services.product_search import get_product_index

logger = logging.getLogger(__name__)

# Same pattern as recommendation_service.py's own _KARAT_RE, duplicated
# rather than imported for the same standalone-file reason documented in
# response_formatter.py's module docstring: matches the karat digits at
# the start of any of the catalogue's real formats, "18k", "18", or the
# Rings compound "18 / Women US 12 (21.4 mm)".
_KARAT_RE = re.compile(r"^\s*(\d+)\s*k?\b", re.IGNORECASE)


def _extract_karat(value: str | None) -> str | None:
    if not value:
        return None
    match = _KARAT_RE.match(value)
    return match.group(1) if match else None


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

    # A sized/varianted product (e.g. a ring with 33 karat+size
    # combinations) stores material as "{karat} / {size} ({mm})", never
    # a bare "18k" -- so the literal exact match above can never succeed
    # for these, and used to fall straight through to semantic search
    # below, which silently loses the requested karat entirely (see
    # product_search.py's _keyword_overlap docstring: digits get
    # stripped during keyword filtering, so "18k" becomes a filtered-out
    # single-letter "k"), returning whichever variant the embedding
    # happened to rank first. Confirmed live, 2026-08-16: asked for the
    # Set Multi Stone Golf Ring in 18k, quoted GH₵8,824.20 -- the 12k
    # price -- not the real 18k price of GH₵12,033.00. Matching on the
    # exact product name plus extracted karat catches this
    # deterministically, before semantic search ever runs. Safe to
    # return the first match without asking for size: every variant of
    # the same product at the same karat shares one price regardless of
    # size in this catalogue (confirmed against the real data), size
    # only matters for confirm_order()'s variation_id downstream, not
    # for pricing.
    target_karat = _extract_karat(material)
    if target_karat:
        karat_matches = [
            product for product in products
            if product["product"] == product_name and _extract_karat(product.get("material")) == target_karat
        ]
        if karat_matches:
            return karat_matches[0]

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
        # min_keyword_overlap=1: cosine similarity alone isn't a strict
        # enough bar for a made-up or unstocked product -- "unicorn
        # pendant" scored above the default 0.3 similarity threshold
        # against a real necklace (confirmed live, 2026-08-12) purely
        # because both sit in the same "gold jewellery" embedding
        # neighbourhood, with zero words in common. Requiring the query
        # to share at least one literal word with the matched product's
        # name is a much stronger signal that this is a genuine match,
        # not two unrelated things that happen to embed nearby.
        matches = get_product_index().search(f"{material} {product_name}", top_k=1, min_keyword_overlap=1)
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
