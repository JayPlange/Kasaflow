"""
Composes two existing tools (price lookup + delivery info) into a single
quote. This is a good example of a "composite tool" -- it doesn't touch
the products file or delivery data directly, it reuses the tools that
already own that logic.
"""

import logging

from services.delivery_tool import get_delivery_information
from services.product_tool import get_product_price

logger = logging.getLogger(__name__)


def generate_quote(product_name: str, material: str) -> dict:
    product = get_product_price(product_name, material)

    if not product:
        logger.info("No quote generated: no match for %s / %s", product_name, material)
        return {"message": "Sorry, we couldn't find that product."}

    return {
        "product": product["product"],
        "material": product["material"],
        "price": product["price"],
        "delivery": get_delivery_information(),
    }
