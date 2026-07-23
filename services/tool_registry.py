from services.delivery_tool import get_delivery_information
from services.product_tool import get_product_price
from services.quote_service import generate_quote
from services.recommendation_service import recommend_products

TOOLS = {
    "get_product_price": get_product_price,
    "get_delivery_information": get_delivery_information,
    "generate_quote": generate_quote,
    "recommend_products": recommend_products,
}
