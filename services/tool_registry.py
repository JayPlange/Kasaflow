from services.delivery_tool import get_delivery_information
from services.order_tool import cancel_order, confirm_order, get_order_status, propose_order
from services.policy_tool import answer_policy_question
from services.product_tool import get_product_price, list_karat_options
from services.quote_service import generate_quote
from services.recommendation_service import recommend_products

TOOLS = {
    "get_product_price": get_product_price,
    # list_karat_options() wraps get_product_karat_options() into the
    # dict shape this registry/response_formatter.py expect -- see that
    # function's docstring in product_tool.py.
    "get_product_karat_options": list_karat_options,
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
    # cancel_order is session-aware for the same reason -- see
    # router.py's _SESSION_AWARE_TOOLS -- it needs to know which
    # session's last_confirmed_order to fall back to when the customer
    # doesn't state an order number explicitly.
    "cancel_order": cancel_order,
    # get_order_status is session-aware for the identical reason as
    # cancel_order immediately above -- same fallback-to-last-order need,
    # just read-only instead of a cancellation.
    "get_order_status": get_order_status,
}
