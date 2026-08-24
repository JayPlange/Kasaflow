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
    else:
        # No karat stated or extractable at all (material is "unknown",
        # or something that isn't a karat, e.g. a stray "white gold" --
        # see router.py's material-context-override fix, 2026-08-24, for
        # a live case of the latter). The two deterministic paths above
        # only ever activate once a karat is present, so a plain,
        # karat-less "how much is X" for a real catalogue product used to
        # fall straight through to semantic search below -- seeded with
        # the literal string "unknown {product_name}" -- instead of ever
        # checking the exact name on its own. Confirmed live, 2026-08-24
        # (Webb/GPT 50-turn test): "how much is the Big White Crown Stone
        # Gold Ring, 14g" (no karat stated) returned "couldn't find that
        # one", while the exact same product with a karat stated ("...in
        # 12k") priced correctly in the same session -- the product was
        # never actually missing from the catalogue, only unreachable by
        # this specific phrasing.
        #
        # Reuses get_product_karat_options() rather than duplicating its
        # exact-name-match/dedupe logic here. A single real option means
        # there's nothing to ask -- answer directly, same as any other
        # exact match. More than one real option means a karat genuinely
        # has to be chosen: this store's own rule (see llm.py's
        # propose_order guidance) is never to assume 18k or any other
        # karat, so this returns the same {"product", "karat_options"}
        # shape list_karat_options() does -- response_formatter.py
        # already renders that as "comes in: ..." rather than silently
        # picking one or claiming the product doesn't exist.
        options = get_product_karat_options(product_name)
        if len(options) == 1:
            return options[0]
        if options:
            return {"product": product_name, "karat_options": options}

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
        # A karat was explicitly stated but semantic search (not the
        # exact/karat-match paths above, both of which already require an
        # exact product_name) is reached only when product_name itself
        # didn't literally match anything in the catalogue -- a mangled
        # or slightly-off name from the model, most likely. In that case
        # the embedding match can easily land on the RIGHT product at the
        # WRONG karat (its nearest neighbour by meaning, not by the
        # specific variant asked for), and nothing before this point ever
        # checks that. Confirmed live, 2026-08-20: a customer's explicit
        # "change the karat to 18" produced a correction_note claiming
        # "updated the karat to 18k" while the actual proposal silently
        # priced at 12k -- the requested karat and the returned product's
        # karat had quietly diverged. Refusing a karat-mismatched
        # semantic match turns that into an honest "couldn't find that
        # product" instead of a silently wrong price.
        if target_karat and _extract_karat(best.get("material")) != target_karat:
            logger.warning(
                "Semantic match %r (material=%s) doesn't match the requested karat=%s for "
                "product_name=%s -- refusing the mismatch rather than silently pricing at "
                "the wrong karat.",
                best["product"], best.get("material"), target_karat, product_name,
            )
            return None
        logger.info(
            "Semantic match: %r (score=%.3f) for query %r",
            best["product"],
            best["score"],
            f"{material} {product_name}",
        )
        return {k: v for k, v in best.items() if k != "score"}

    logger.info("No product match at all for product_name=%s material=%s", product_name, material)
    return None


def get_product_price_by_id(product_id, material):
    """Same karat-aware lookup as get_product_price(), but keyed on the
    catalogue's own numeric `id` instead of a restated product_name
    string.

    Exists for order_tool.propose_order()'s correction-recovery fallback
    (see that function's docstring): `id` is stable across every
    karat/size variant of one named product in this catalogue (confirmed
    against the real data, 2026-08-21 -- "Big White Crown Stone Gold
    Ring, 14g" has 33 rows, one per karat+size combination, every one of
    them sharing id=5892; only variation_id changes per row), so a
    session that already has a verified id for its active product can
    reprice a karat/material correction against that id directly,
    without ever depending on the model correctly restating the full
    product_name string again.

    Deliberately does NOT fall back to semantic search, same reasoning
    as get_product_karat_options() above: the caller already has a
    specific, previously-verified id in hand, so a fuzzy fallback here
    would risk quietly mixing in a different product's variant instead
    of just returning nothing. A caller with only a name and no id
    should use get_product_price() instead."""
    try:
        with open(settings.products_path, "r") as file:
            products = json.load(file)
    except FileNotFoundError:
        logger.error("Products file not found at %s", settings.products_path)
        return None
    except json.JSONDecodeError as e:
        logger.error("Products file at %s is not valid JSON: %s", settings.products_path, e)
        return None

    matches = [product for product in products if product.get("id") == product_id]
    if not matches:
        return None

    # Same two-tier match as get_product_price(): a plain exact material
    # match first (covers non-sized products, whose material is a bare
    # "18k" rather than the sized "{karat} / {size} (mm)" format), then
    # karat-extraction for sized products.
    for product in matches:
        if product.get("material") == material:
            return product

    target_karat = _extract_karat(material)
    if target_karat:
        karat_matches = [
            product for product in matches
            if _extract_karat(product.get("material")) == target_karat
        ]
        if karat_matches:
            return karat_matches[0]

    logger.info(
        "No karat match for product_id=%s material=%s among %d known variant(s)",
        product_id, material, len(matches),
    )
    return None


def get_product_karat_options(product_name: str) -> list[dict]:
    """All karat variants of one exact product name, sorted highest
    karat first -- used once a specific product has already been
    identified with confidence (e.g. a matched customer photo, see
    services/photo_match_tool.py) and the customer needs the full
    price-by-karat picture, not just one karat's price.

    Deliberately does NOT fall back to semantic search the way
    get_product_price() does: the caller already has an exact
    product_name in hand from a confirmed match, so a fuzzy fallback
    here would risk quietly mixing in a different, similarly-named
    product's variants instead of just returning nothing.
    """
    try:
        with open(settings.products_path, "r") as file:
            products = json.load(file)
    except FileNotFoundError:
        logger.error("Products file not found at %s", settings.products_path)
        return []
    except json.JSONDecodeError as e:
        logger.error("Products file at %s is not valid JSON: %s", settings.products_path, e)
        return []

    matches = [p for p in products if p["product"] == product_name]

    # Dedupe by karat -- a sized product (rings) repeats the same karat
    # across several size rows, all sharing one price (same "size
    # doesn't affect price" fact already established in
    # get_product_price()'s comment above). Showing "18k: X" three times
    # over for three sizes would just repeat the same line.
    seen_karats: set[str] = set()
    deduped: list[dict] = []
    for product in matches:
        karat = _extract_karat(product.get("material"))
        key = karat or (product.get("material") or "")
        if key in seen_karats:
            continue
        seen_karats.add(key)
        deduped.append(product)

    def _sort_key(product: dict) -> int:
        karat = _extract_karat(product.get("material"))
        return -int(karat) if karat and karat.isdigit() else 0

    deduped.sort(key=_sort_key)
    return deduped


def list_karat_options(product_name: str) -> dict:
    """Tool-registry entry point wrapping get_product_karat_options() above
    into the {"product": ..., "karat_options": [...]} dict shape every
    other registered tool returns.

    get_product_karat_options() itself returns a bare list -- correct for
    its original caller (demo_routes.py's photo-match flow, which builds
    its own result dict around it), but execute_tool()/response_formatter.
    py are both written for dict results (see tool_executor.py's own type
    hint and response_formatter.py's duck-typed "key in result" dispatch,
    which silently mis-evaluates against a bare list). Wired in here,
    2026-08-24, as this text-conversation tool's own registered entry
    point specifically so get_product_price() keeps returning a raw
    catalogue row unchanged -- nothing about that existing shape moves.
    """
    return {"product": product_name, "karat_options": get_product_karat_options(product_name)}
