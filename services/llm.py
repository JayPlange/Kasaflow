"""
LLM tool-selection layer.

Responsible for exactly one thing: turning a customer message into
either a single {"tool": ..., "arguments": {...}} request, or -- when
the message genuinely contains more than one distinct ask -- a
{"requests": [{"tool": ..., "arguments": {...}}, ...]} list of them.
It does not execute tools and it does not talk to the customer
directly (see router.py).

The multi-request shape is additive, not a replacement: single-intent
messages (still the overwhelming majority) return exactly the same
{"tool", "arguments"} shape this always returned, so router.py's
existing contract-stable single-result path, and everything that
depends on it, is untouched. "requests" only appears when a message
actually needs it.
"""

import json
import logging
import time

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)

# A customer message could in principle list many items -- this caps
# how many distinct tool calls one message can trigger, so a single
# WhatsApp message can't fan out into an unbounded number of catalogue
# lookups / LLM-adjacent tool calls. Matches the existing 5-item cap
# response_formatter.py already applies when listing recommendations.
MAX_REQUESTS_PER_MESSAGE = 5


class ToolSelectionError(Exception):
    """Raised when the LLM fails to return a usable tool request."""


_PROMPT_TEMPLATE = """
You are an AI tool selector for a jewellery store.

Your job is NOT to answer the customer.

Your job is ONLY to decide:

1. Which tool should be called.
2. What arguments that tool needs.

Customer messages may arrive in English, Twi, or a mix of both -- many
come from a transcribed voice note, not typed text. The product
catalogue itself is English-only, so any product, material, or category
word you extract must be translated/normalised into its English
equivalent before you return it, never passed through in the original
language. For example, a Twi word for a product like "kyɛn" means
"chain" -- extract `product_name: "chain"`, not `product_name: "kyɛn"`.
This applies to every argument below, not just product_name.

Available tools:

1. get_product_price
Arguments:
- product_name
- material
Use this when the customer names ONE specific product -- whether they're asking its price, asking to see a photo/picture of it, or both. Its result always includes the product's image when one exists, so it's also the right tool for "send me a photo of the [named item]" / "can I see that necklace", not just price questions. Only use recommend_products instead when the customer is browsing a type/category rather than naming one specific item.

2. get_delivery_information
Arguments:
- none
Use this when the customer asks generically about delivery/shipping ("how does delivery work", "do you ship abroad") without asking about a specific product's price. This store doesn't have a fixed delivery price or time -- it lists the real delivery arrangements (rider delivery within Accra, rider delivery within Kumasi, or shipping outside Ghana) for the customer to choose from, not a cost.

3. generate_quote
Arguments:
- product_name
- material
Use this when the customer wants a product's price AND to know about delivery together, not just a price. Same as get_delivery_information above, this returns the real delivery choices, not a fixed cost/time.

4. recommend_products
Arguments:
- material
- category
Use this when the customer is browsing rather than asking about one named product -- "what rings do you have", "what's available in gold", "show me your necklaces in 18k".
- `category` must be one of this store's actual two catalogue categories: "Rings" or "Necklaces". Map whatever word the customer used onto one of those two -- "chain(s)", "necklace", and "pendant" all mean `category: "Necklaces"`; "band", "wedding ring", and "engagement ring" all mean `category: "Rings"`. Set to "unknown" if the customer didn't mention a type, or if what they asked for is clearly neither (e.g. bracelets, earrings, watches -- this store doesn't stock those, so pass their word through as-is rather than forcing it into Rings or Necklaces).
- `material` is the karat purity, e.g. "18k" or "14k", ONLY if the customer stated a specific karat. Every item this store sells is gold, so if the customer just says "gold" without a karat number, set material to "unknown" -- there's nothing to narrow, it would only exclude items that shouldn't be excluded.

5. answer_policy_question
Arguments:
- question
Use this when the customer asks about store policy rather than a specific product or delivery time -- for example returns, warranty, ring sizing, jewellery care, custom engraving, or payment methods. Pass their question through in their own words as `question`.

6. propose_order
Arguments:
- product_name
- material
- quantity
- delivery_address
- delivery_option
Use this when the customer clearly wants to PLACE an order, not just ask a price or a quote, and has given enough detail to price it. If they haven't stated how many they want, set quantity to "unknown" rather than assuming 1. If they haven't given a delivery address yet, set delivery_address to "unknown" -- never invent either one.
`delivery_option` must be one of this store's three real delivery arrangements: "accra_rider" (rider delivery within Accra), "kumasi_rider" (rider delivery within Kumasi), or "international" (shipping outside Ghana). Infer it from the delivery address or an explicit statement -- "deliver to Accra"/an Accra address means "accra_rider"; "Kumasi" means "kumasi_rider"; anywhere outside Ghana, or an explicit "ship it"/"I'm not in Ghana", means "international". Set to "unknown" if you genuinely can't tell. This store doesn't price delivery automatically -- a human arranges the actual rider/shipping after the order is placed -- so this only needs to capture which of the three arrangements the customer wants, not a cost or time. This never actually creates the order; it only prices the product and shows the customer a proposal to confirm.

7. confirm_order
Arguments:
- none
Use this ONLY when the customer is clearly confirming an order that was already proposed to them earlier in this conversation -- for example "yes", "confirm", "go ahead", "place the order". Never call this from a message that hasn't already seen a proposed order; there is nothing to confirm without one.

8. cancel_order
Arguments:
- order_id
Use this when the customer wants to cancel an order they already placed -- "cancel my order", "cancel order 6846", "I don't want it anymore". If they stated an order number, pass it as `order_id`. If they didn't (e.g. "cancel my order" with no number), set `order_id` to "unknown" -- the system will look up their most recent confirmed order for you; never invent a number. This is only for cancelling an order that was already confirmed -- if nothing has been confirmed yet in this conversation, a "cancel" style message more likely means they want to stop what they were doing, not this tool; use converse for that instead.

9. converse
Arguments:
- reply
Use this for messages that are purely conversational and need no business tool at all: greetings ("hey", "hi", "good morning"), farewells, thanks ("medaase"), casual acknowledgements ("okay", "nice", "haha"), reactions and emojis, light humour, and simple social questions ("how are you?"). This is the ONLY tool where you write the actual customer-facing reply yourself, as the `reply` argument -- there is no deterministic tool behind it, because there is no business fact to look up. Keep it natural, warm, and concise -- a short, casual sentence or two, not a list of everything you can do. Match the customer's language (English, Twi, or a natural mix of both). Never mention tools, JSON, databases, or system/error states in the reply.

Do NOT use converse when:
- the customer asks about a product, price, availability, delivery, or policy -- those need the matching tool above, even if the message is short or casual in tone.
- the customer names, repeats, or otherwise identifies a specific catalogue item in any way -- even with no question mark, no "how much"/"show me", just the item's name on its own (e.g. a customer typing back a product name from something you showed them a moment ago). That is them telling you which item they mean, not making conversation -- call get_product_price with that product_name immediately. Do not respond by asking what they'd like to do with it; ask that only if it is genuinely still unclear after using this turn's context (see the pending-intent context below, if present).
- the customer's message is a vague product-adjacent ask with nothing named yet -- "can I see a photo", "show me pictures", "how much is it" -- and there is no clearer signal in the message. These ARE business requests, just ones missing a detail. If the message names or clearly implies a catalogue CATEGORY (e.g. "necklace images", "pictures of your rings", "chain photos"), call recommend_products with that category instead -- a category is enough to act on, it is not a missing detail the way a single unnamed product is. Only when there is truly no product AND no category to go on (e.g. bare "show me pictures", "how much is it") should you call get_product_price with product_name "unknown" and let the system's own missing-product handling ask which one -- either way, do not improvise your own clarifying question through converse.
- the customer wants a recommendation, wants to place/change/confirm/cancel an order, or is answering a question this assistant itself just asked while an order was being collected (see any pending-order / partial-order context below, if present) -- those must go to the matching business tool, not converse, no matter how short the reply looks.
- answering would require inventing or implying any catalogue, pricing, stock, delivery, or order information converse itself has no access to. If in doubt whether something is a real business question, prefer the business tool over converse.

Exception: if the last-action-outcome context further below tells you to use converse to explain a recent failure (a customer asking "why?" or reacting to something that just went wrong), do that -- that instruction overrides the business-question rules above for that one reply, since explaining what already happened is not the same as answering a new business question.

Examples:

Customer: "hey"
{{"tool": "converse", "arguments": {{"reply": "Hey! 👋 How can I help you today?"}}}}

Customer: "how are you?"
{{"tool": "converse", "arguments": {{"reply": "I'm doing great, thanks for asking! How about you?"}}}}

Customer: "medaase"
{{"tool": "converse", "arguments": {{"reply": "You're very welcome!"}}}}

Customer: "ei that's expensive oo"
{{"tool": "converse", "arguments": {{"reply": "Haha I hear you! Want me to show you a few more affordable options?"}}}}

Customer: "urm what do you do?"
{{"tool": "converse", "arguments": {{"reply": "I help with pretty much everything here 😊 Ask me about our jewellery, prices, photos, delivery, or place an order any time."}}}}

Customer: "how much is the gold chain?"
-> NOT converse. Use get_product_price.

Customer: "do you have bracelets?"
-> NOT converse. Use recommend_products.

Customer: "this Set Multi Stone Golf Ring, 7g" (just naming a product, no question)
-> NOT converse. Use get_product_price with product_name "Set Multi Stone Golf Ring, 7g".

Customer: "yeah i wanna see pictures" (nothing specific named yet)
-> NOT converse. Use get_product_price with product_name "unknown" -- let the system's own missing-product handling ask which one, don't improvise that question yourself.

Customer: "necklace images" (a category, not a specific product)
-> NOT converse, and NOT get_product_price. Use recommend_products with category "Necklaces" -- a category is enough to act on immediately.

Customer: "why?" (right after being told an order/action failed)
-> converse, using the last-action-outcome context below to explain the real reason -- do not say you haven't seen anything, and do not guess a different reason.

Customer: "cancel my order" (no number given)
{{"tool": "cancel_order", "arguments": {{"order_id": "unknown"}}}}

Customer: "please cancel order 6846"
{{"tool": "cancel_order", "arguments": {{"order_id": "6846"}}}}

Rules:

- If the customer asks only for a price, use get_product_price.
- If the customer asks to see a photo/picture of one specific, named product, use get_product_price (its result carries the image) -- even if they never mention price and even if that product was only just recommended to them a moment ago. Do not use recommend_products for this; that returns a browse-style list, not one item's photo.
- If the customer asks for delivery/shipping info only, use get_delivery_information.
- If the customer wants a full quote (price + delivery), use generate_quote.
- If the customer is browsing by product type and/or karat rather than asking about one item, use recommend_products.
- If the customer is asking about returns, warranty, sizing, care, engraving, or payment methods, use answer_policy_question.
- If the customer wants to actually place an order and has given enough detail, use propose_order.
- If the customer is confirming an order proposed earlier in this conversation, use confirm_order.
- If the customer wants to cancel an order they already placed, use cancel_order. If they gave an order number, pass it; otherwise set order_id to "unknown" and let the system find their most recent order.
- If the customer's message is purely social (a greeting, thanks, a reaction, small talk) with no business question in it, use converse. If there's ANY pending order or order-in-progress context below and the message plausibly answers it (a bare number, an address, a delivery choice), that takes priority over converse -- continue the order instead, exactly as instructed in that context.
- If the customer mentions "this", "that one", or similar references, infer the product, material, or category from earlier in THIS message if possible.
- If a tool needs product_name, material, or category and you genuinely cannot determine it from this message alone, set that argument to the literal string "unknown" rather than guessing. The system remembers what the customer discussed earlier in the conversation and will fill "unknown" in for you -- inventing a value yourself would override that and risk quoting the wrong product.

Multiple requests in one message:

A single customer message can genuinely contain more than one distinct
ask -- two different products ("how much is a gold ring and a silver
chain"), the same product in more than one karat ("do you have this
ring in 14k and 18k"), or a product question plus an unrelated policy
question ("how much is a gold chain, and what's your returns policy").
When that happens, do NOT pick only one and drop the rest -- return
EVERY distinct ask as its own entry in a "requests" list instead of a
single "tool"/"arguments" pair:

{{
  "requests": [
    {{"tool": "...", "arguments": {{...}}}},
    {{"tool": "...", "arguments": {{...}}}}
  ]
}}

Only do this when the message actually contains more than one distinct
ask. A single product with a single price question is still just one
request in the normal shape -- don't split something that doesn't need
splitting.

Return ONLY valid JSON. Do not include markdown formatting or commentary.

Example:

{{
  "tool": "get_product_price",
  "arguments": {{
    "product_name": "ring",
    "material": "gold"
  }}
}}

Example (Twi transcript, translated into English before being returned):

Customer said: "sika kyɛn no bo yɛ sɛn" ("how much is the gold chain")

{{
  "tool": "get_product_price",
  "arguments": {{
    "product_name": "chain",
    "material": "gold"
  }}
}}

Example (message contains two distinct asks):

Customer said: "how much is a gold ring and a silver chain"

{{
  "requests": [
    {{
      "tool": "get_product_price",
      "arguments": {{
        "product_name": "ring",
        "material": "gold"
      }}
    }},
    {{
      "tool": "get_product_price",
      "arguments": {{
        "product_name": "chain",
        "material": "silver"
      }}
    }}
  ]
}}

{pending_order_state}

{order_draft_state}

{pending_intent_state}

{last_action_outcome_state}

Customer:
{message}
"""


def _pending_order_state_line(pending_order: dict | None) -> str:
    """Describes whether this session currently has an order awaiting
    confirmation, for the confirm_order guidance above.

    Exists because understand_customer() otherwise only ever sees the
    customer's current message in isolation -- no conversation history,
    no memory of what this assistant itself last asked. A bare "yh"/
    "yeah" is genuinely ambiguous with zero context: it could confirm a
    real pending order, or it could just as easily be agreeing to "want
    me to show you a photo?" from a browsing reply that never proposed
    an order at all. Without this line the model has no way to tell
    those apart and would sometimes guess confirm_order regardless of
    whether anything was actually pending (confirmed live -- see
    services/order_tool.py's confirm_order() decline message, which
    exists specifically to handle that guess coming back wrong)."""
    if pending_order:
        return (
            f"This customer currently has a pending order awaiting confirmation: "
            f"{pending_order['quantity']} x {pending_order['material']} {pending_order['product']}, "
            f"total GH₵{pending_order['total']:,.2f}. If their message clearly confirms this "
            f"(\"yes\", \"confirm\", \"go ahead\", \"yh\", \"ok place it\"), use confirm_order."
        )
    return (
        "This customer does NOT currently have any pending order awaiting confirmation. "
        "Do not use confirm_order for this message, even if it sounds like a bare "
        "agreement (\"yh\", \"yeah\", \"ok\", \"sure\") -- there is nothing for them to be "
        "confirming right now, so treat a bare agreement as being about whatever was "
        "actually offered (seeing a photo, narrowing down a choice, or similar), not "
        "as placing an order."
    )


_ORDER_DRAFT_LABELS = {
    "product_name": "product",
    "material": "material/karat",
    "quantity": "quantity",
    "delivery_address": "delivery address",
    "delivery_option": "delivery option (accra_rider / kumasi_rider / international)",
}


def _order_draft_state_line(order_draft: dict | None) -> str:
    """Describes an order already in progress for this session, so a
    short reply like a bare number or a bare address can be recognised
    as continuing it.

    Exists for the same reason as _pending_order_state_line() above, one
    step earlier in the flow: once propose_order has asked "How many
    would you like?", the customer's next message is often just "2" --
    with no conversation history, the model has no way to know that's
    an answer to a question it can't see, and would sometimes route it
    to a completely different tool instead of continuing the order
    (confirmed live, 2026-08-12). get_order_draft() in memory.py builds
    this from whatever's already been remembered."""
    if not order_draft:
        return ""

    known = [f"{_ORDER_DRAFT_LABELS[key]}={value}" for key, value in order_draft.items() if value]
    missing = [_ORDER_DRAFT_LABELS[key] for key, value in order_draft.items() if not value]
    if not missing:
        # Everything's already known -- nothing left for a short reply
        # to be answering, so there's nothing useful to say here.
        return ""

    known_text = ", ".join(known) if known else "nothing yet"
    missing_text = ", ".join(missing)
    return (
        f"This customer already has an order in progress. Known so far: {known_text}. "
        f"Still missing: {missing_text}. If their current message is short and only makes "
        f"sense as answering one of the missing pieces (a bare number for quantity, a bare "
        f"address, \"Accra\"/\"Kumasi\"/\"ship it\"/\"outside Ghana\" for delivery option), "
        f"treat it as continuing this order: use propose_order, fill in the new piece from "
        f"their message, and keep every OTHER already-known value exactly as given above. "
        f"Exception: if their current message clearly states a different value for one of "
        f"the already-known fields instead -- a correction, e.g. \"sorry, Kumasi not Accra\", "
        f"or a different quantity/address than what's listed above -- use their new value for "
        f"that field, not the old one; do not silently keep a value they just corrected. Don't "
        f"ask them to repeat anything they haven't changed, and don't restart with a different "
        f"tool."
    )


_PENDING_INTENT_TOOL_DESCRIPTIONS = {
    "get_product_price": "see its price and photo",
    "generate_quote": "get a full quote (price and delivery options)",
}


def _pending_intent_state_line(pending_intent: str | None) -> str:
    """Describes an unresolved product-lookup attempt from a previous
    turn, so a customer who then names/identifies the product on their
    very next message gets it resolved immediately instead of being
    asked yet another clarifying question.

    Exists because a message like "yeah i wanna see pictures" or "how
    much is it" with no product named is a genuine business ask, just
    missing one detail -- get_product_price runs with product_name
    "unknown", finds nothing, and this session is left with an
    unresolved intent. Without this line, the customer's next message
    naming the product (e.g. "this Set Multi Stone Golf Ring, 7g") looks,
    in isolation, like they're just repeating a name with no clear ask --
    and converse (see llm.py's tool 9) or another clarifying question can
    swallow it instead of actually completing the lookup the customer
    already asked for (confirmed live, 2026-08-13: the customer had to
    name the product twice, then got told "I couldn't find that one",
    then had to explain that it had literally just been shown to them).
    router.py's set_pending_intent()/get_pending_intent() track this."""
    if not pending_intent:
        return ""
    action = _PENDING_INTENT_TOOL_DESCRIPTIONS.get(pending_intent, "look something up")
    return (
        f"This customer recently tried to {action} but hadn't named a specific product, "
        f"so nothing could be found yet. If their current message now names, repeats, or "
        f"otherwise identifies a specific product -- even just the name on its own, with no "
        f"question -- treat that as completing this request: call {pending_intent} with that "
        f"product name right away. Do not ask them what they'd like to do with it and do not "
        f"use converse; they already told you what they wanted, they're now just telling you "
        f"which item."
    )


def _last_action_outcome_state_line(last_action_outcome: dict | None) -> str:
    """Describes a genuine, unrecoverable failure of a real business
    action the customer just tried -- not a "still missing a detail"
    prompt (those are self-explanatory and covered by
    _order_draft_state_line() above), but something that failed for a
    real reason with nothing the customer could have done differently.

    Exists because a customer's very next message after a failure like
    that is very often "why?", or confusion/frustration expressed some
    other way -- and without this, understand_customer() has no idea a
    failure even happened a moment ago (confirmed live, 2026-08-13: "why?"
    got "could you clarify what you mean", and the very next message got
    told "I haven't seen any order from you yet" -- an outright false
    claim, since an order attempt had just failed, not never happened).
    See memory.set_last_action_outcome()/get_last_action_outcome()."""
    if not last_action_outcome:
        return ""
    explanation = last_action_outcome.get("customer_safe_explanation")
    if not explanation:
        return ""
    return (
        f"The last thing this customer tried just failed, for a real reason: {explanation} "
        f"If their current message is asking why, expressing confusion, or reacting to that "
        f"failure in any way (\"why?\", \"why not?\", \"oh wow\", \"sigh\", or similar), use "
        f"converse and explain using that exact reason. Do not claim nothing happened, do not "
        f"say you haven't seen anything from them, and do not invent a different reason. If "
        f"their current message is clearly about something else entirely, ignore this and "
        f"proceed normally."
    )


def _build_prompt(
    message: str,
    pending_order: dict | None,
    order_draft: dict | None,
    pending_intent: str | None = None,
    last_action_outcome: dict | None = None,
) -> str:
    return _PROMPT_TEMPLATE.format(
        message=message,
        pending_order_state=_pending_order_state_line(pending_order),
        order_draft_state=_order_draft_state_line(order_draft),
        pending_intent_state=_pending_intent_state_line(pending_intent),
        last_action_outcome_state=_last_action_outcome_state_line(last_action_outcome),
    )


def _call_llm(
    message: str,
    pending_order: dict | None,
    order_draft: dict | None,
    pending_intent: str | None = None,
    last_action_outcome: dict | None = None,
) -> str:
    """Call the model with retries on transient failures only.

    Auth errors, bad requests, etc. (APIError) are not retried — retrying
    those just burns time and money for a call that will never succeed.
    """
    last_error: Exception | None = None
    total_attempts = settings.llm_max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            response = client.responses.create(
                model=settings.openai_model,
                input=_build_prompt(message, pending_order, order_draft, pending_intent, last_action_outcome),
                timeout=settings.llm_timeout_seconds,
            )
            return response.output_text

        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            logger.warning(
                "LLM call failed (attempt %s/%s): %s", attempt, total_attempts, e
            )
            if attempt < total_attempts:
                time.sleep(min(2**attempt, 8))  # 2s, 4s, 8s...

        except APIError as e:
            logger.error("Non-retryable LLM API error: %s", e)
            raise ToolSelectionError(f"LLM request failed: {e}") from e

    raise ToolSelectionError(f"LLM unreachable after {total_attempts} attempts: {last_error}")


def _parse_tool_request(raw_text: str) -> dict:
    cleaned = raw_text.strip()

    # Defensive: models occasionally wrap JSON in markdown fences even
    # when told not to. Strip them rather than crashing on it.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("LLM output was not valid JSON: %s | raw=%r", e, raw_text)
        raise ToolSelectionError("LLM did not return valid JSON") from e

    if "requests" in data:
        return _validate_multi_request(data)

    if "tool" not in data or "arguments" not in data:
        logger.error("LLM output missing required keys: %s", data)
        raise ToolSelectionError(f"LLM response missing 'tool' or 'arguments': {data}")

    return data


def _validate_multi_request(data: dict) -> dict:
    requests = data["requests"]

    if not isinstance(requests, list) or not requests:
        logger.error("LLM 'requests' was not a non-empty list: %s", data)
        raise ToolSelectionError(f"LLM response's 'requests' must be a non-empty list: {data}")

    for entry in requests:
        if not isinstance(entry, dict) or "tool" not in entry or "arguments" not in entry:
            logger.error("LLM 'requests' entry missing 'tool' or 'arguments': %s", entry)
            raise ToolSelectionError(f"LLM 'requests' entry missing 'tool' or 'arguments': {entry}")

    if len(requests) > MAX_REQUESTS_PER_MESSAGE:
        logger.warning(
            "LLM returned %d requests for one message -- truncating to %d",
            len(requests), MAX_REQUESTS_PER_MESSAGE,
        )
        requests = requests[:MAX_REQUESTS_PER_MESSAGE]

    return {"requests": requests}


def understand_customer(
    message: str,
    pending_order: dict | None = None,
    order_draft: dict | None = None,
    pending_intent: str | None = None,
    last_action_outcome: dict | None = None,
) -> dict:
    """pending_order, when provided, is router.py's read of this
    session's pending proposal (see order_tool.get_pending_order_summary)
    -- see _pending_order_state_line()'s docstring for why this needs to
    be passed in explicitly rather than left for the model to infer.

    order_draft is the equivalent one step earlier -- an order that's
    been started but not yet fully priced (see memory.get_order_draft()
    and _order_draft_state_line()).

    pending_intent is one step earlier again -- a product lookup the
    customer asked for but hadn't named a product for yet (see
    memory.get_pending_intent() and _pending_intent_state_line()).

    last_action_outcome is a different axis entirely -- not "still
    missing something", but a real business action that was already
    fully specified and still failed (see memory.get_last_action_outcome()
    and _last_action_outcome_state_line())."""
    if not message or not message.strip():
        raise ValueError("message must not be empty")

    raw_text = _call_llm(message, pending_order, order_draft, pending_intent, last_action_outcome)
    logger.info("Raw LLM output: %s", raw_text)

    return _parse_tool_request(raw_text)
