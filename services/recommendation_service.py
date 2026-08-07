"""
Recommend products by category and/or karat. Same failure modes as
product_tool.py (missing/corrupt data file) so it gets the same
defensive handling.

Why this isn't a plain `material == material` filter: the real
WooCommerce catalogue's `material` field is inconsistent by category --
Necklaces store a bare karat ("18k" or, for some entries, just "18"),
Rings store a compound "{karat} / {gender} US {size} ({mm})" string.
Checked against the live 3918-entry sync (2026-08-07): every single
entry is gold at some karat, none are silver or any other metal, so a
customer saying "gold" has nothing narrower to match against than "all
of it" -- only a stated karat ("18k", "14 karat") actually narrows
anything. `category` ("Rings"/"Necklaces") is the clean, reliable field
and the one customers are far more likely to browse by anyway ("what
rings do you have").
"""

import json
import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)

# Matches the karat digits at the start of any of the catalogue's real
# formats: "18k", "18", or the Rings compound "18 / Women US 12 (21.4 mm)".
_KARAT_RE = re.compile(r"^\s*(\d+)\s*k?\b", re.IGNORECASE)


def _extract_karat(value: str | None) -> str | None:
    if not value:
        return None
    match = _KARAT_RE.match(value)
    return match.group(1) if match else None


def _is_unset(value: str | None) -> bool:
    return not value or value.strip().lower() == "unknown"


def recommend_products(material: str = "unknown", category: str = "unknown") -> dict:
    try:
        with open(settings.products_path, "r") as file:
            products = json.load(file)
    except FileNotFoundError:
        logger.error("Products file not found at %s", settings.products_path)
        return {"recommendations": []}
    except json.JSONDecodeError as e:
        logger.error("Products file at %s is not valid JSON: %s", settings.products_path, e)
        return {"recommendations": []}

    recommendations = products

    if not _is_unset(category):
        target_category = category.strip().lower()
        recommendations = [
            p for p in recommendations
            if (p.get("category") or "").strip().lower() == target_category
        ]

    # A generic metal word ("gold") has no karat digits to extract --
    # correctly falls through to "no karat filter" rather than matching
    # nothing, since the whole catalogue is gold anyway.
    target_karat = _extract_karat(material) if not _is_unset(material) else None
    if target_karat:
        recommendations = [
            p for p in recommendations
            if _extract_karat(p.get("material")) == target_karat
        ]

    if not recommendations:
        logger.info("No recommendations found for material=%s category=%s", material, category)

    return {"recommendations": recommendations}
