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

2. get_delivery_information
Arguments:
- none

3. generate_quote
Arguments:
- product_name
- material
Use this when the customer wants a full quote (price AND delivery together), not just a price.

4. recommend_products
Arguments:
- material
- category
Use this when the customer is browsing rather than asking about one named product -- "what rings do you have", "what's available in gold", "show me your necklaces in 18k".
- `category` is the product type, e.g. "Rings" or "Necklaces". Set to "unknown" if the customer didn't mention a type.
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
Use this when the customer clearly wants to PLACE an order, not just ask a price or a quote, and has given enough detail to price it. If they haven't stated how many they want, set quantity to "unknown" rather than assuming 1. If they haven't given a delivery address yet, set delivery_address to "unknown" -- never invent either one. This never actually creates the order; it only prices it and shows the customer a proposal to confirm.

7. confirm_order
Arguments:
- none
Use this ONLY when the customer is clearly confirming an order that was already proposed to them earlier in this conversation -- for example "yes", "confirm", "go ahead", "place the order". Never call this from a message that hasn't already seen a proposed order; there is nothing to confirm without one.

Rules:

- If the customer asks only for a price, use get_product_price.
- If the customer asks for delivery/shipping info only, use get_delivery_information.
- If the customer wants a full quote (price + delivery), use generate_quote.
- If the customer is browsing by product type and/or karat rather than asking about one item, use recommend_products.
- If the customer is asking about returns, warranty, sizing, care, engraving, or payment methods, use answer_policy_question.
- If the customer wants to actually place an order and has given enough detail, use propose_order.
- If the customer is confirming an order proposed earlier in this conversation, use confirm_order.
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

Customer:
{message}
"""


def _build_prompt(message: str) -> str:
    return _PROMPT_TEMPLATE.format(message=message)


def _call_llm(message: str) -> str:
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
                input=_build_prompt(message),
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


def understand_customer(message: str) -> dict:
    if not message or not message.strip():
        raise ValueError("message must not be empty")

    raw_text = _call_llm(message)
    logger.info("Raw LLM output: %s", raw_text)

    return _parse_tool_request(raw_text)
