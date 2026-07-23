"""
Deterministic lookup of a product's price. Intentionally dumb: the LLM
decided *that* a price lookup should happen, this function just does it.
"""

import json
import logging

from app.config import settings

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

    logger.info("No product match for product_name=%s material=%s", product_name, material)
    return None
