from services.delivery_tool import get_delivery_information
from services.order_tool import confirm_order, propose_order
from services.policy_tool import answer_policy_question
from services.product_tool import get_product_price
from services.quote_service import generate_quote
from services.recommendation_service import recommend_products

TOOLS = {
    "get_product_price": get_product_price,
    "get_delivery_information": get_delivery_information,
    "generate_quote": generate_quote,
    "recommend_products": recommend_products,
    "answer_policy_question": answer_policy_question,
    # propose_order/confirm_order both require a session_id argument that
    # customers never provide and the LLM is never asked for -- router.py
    # injects it for these two tools specifically before calling
    # execute_tool(). See router.py's _SESSION_AWARE_TOOLS.
    "propose_order": propose_order,
    "confirm_order": confirm_order,
}
