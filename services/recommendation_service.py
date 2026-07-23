"""
Recommend products by material. Same failure modes as product_tool.py
(missing/corrupt data file) so it gets the same defensive handling.
"""

import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def recommend_products(material: str) -> dict:
    try:
        with open(settings.products_path, "r") as file:
            products = json.load(file)
    except FileNotFoundError:
        logger.error("Products file not found at %s", settings.products_path)
        return {"recommendations": []}
    except json.JSONDecodeError as e:
        logger.error("Products file at %s is not valid JSON: %s", settings.products_path, e)
        return {"recommendations": []}

    recommendations = [p for p in products if p.get("material") == material]

    if not recommendations:
        logger.info("No recommendations found for material=%s", material)

    return {"recommendations": recommendations}
