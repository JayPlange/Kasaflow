"""
Turns a tool's raw result dict into a sentence a customer can actually
read over WhatsApp.

Why this exists and router.py doesn't just do it: route_customer()'s
JSON contract is already a real, presumably-already-integrated-against
API response shape (the README is explicit that this contract has
stayed stable through an internal rewrite). Reshaping that into prose
for WhatsApp specifically, rather than changing what /process returns
for everyone, keeps that promise intact.

Written as duck-typed shape matching (which keys are present) rather
than router.py telling it which tool ran, on purpose -- it means this
file can be deleted and rebuilt without touching router.py or the tools
themselves at all.
"""


def format_for_customer(result: dict | None) -> str:
    if result is not None and "results" in result:
        # router.py's multi-request shape -- a message that contained
        # more than one distinct ask. Format each sub-result exactly
        # the same way a single-result reply would be, then present
        # them as a numbered list so the customer can tell which
        # answer belongs to which part of what they asked.
        parts = [format_for_customer(sub_result) for sub_result in result["results"]]
        if len(parts) == 1:
            return parts[0]
        return "\n\n".join(f"{i}. {part}" for i, part in enumerate(parts, start=1))

    if result is None:
        # get_product_price returns None directly (not an exception) when
        # nothing matches, including the no-match case after the semantic
        # search fallback comes back empty. generate_quote already wraps
        # that into a message for its own callers, but a bare
        # get_product_price call reaches this function with nothing in
        # front of it -- same customer-facing message either way, rather
        # than crashing on `"error" in None`.
        return "Sorry, we couldn't find that product."

    if "error" in result:
        return result["error"]

    if "answer" in result:
        # answer_policy_question's shape -- already a sentence.
        return result["answer"]

    if "message" in result and "product" not in result:
        # generate_quote's "couldn't find that product" case.
        return result["message"]

    if "recommendations" in result:
        items = result["recommendations"]
        if not items:
            return "I don't have anything matching that right now -- want me to check something else?"
        lines = [f"- {p['product']} ({p['material']}): GH₵{p['price']:,.2f}" for p in items[:5]]
        return "Here's what I found:\n" + "\n".join(lines)

    if "price" in result and "delivery" in result:
        delivery = result["delivery"]
        return (
            f"The {result['material']} {result['product']} is GH₵{result['price']:,.2f}. "
            f"Delivery is {delivery['delivery_time']}, GH₵{delivery['shipping_cost']} shipping."
        )

    if "price" in result:
        return f"The {result['material']} {result['product']} is GH₵{result['price']:,.2f}."

    if "delivery_time" in result:
        return f"Delivery takes {result['delivery_time']}, shipping is GH₵{result['shipping_cost']}."

    # Unknown shape -- surface something rather than send nothing, but
    # this branch being hit means a new tool was added without updating
    # this file, worth noticing in logs, not just in a customer's chat.
    return "Let me get back to you on that."
