"""
LLM tool-selection layer.

Responsible for exactly one thing: turning a customer message into a
{"tool": ..., "arguments": {...}} request. It does not execute tools
and it does not talk to the customer directly (see router.py).
"""

import json
import logging
import time

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)


class ToolSelectionError(Exception):
    """Raised when the LLM fails to return a usable tool request."""


_PROMPT_TEMPLATE = """
You are an AI tool selector for a jewellery store.

Your job is NOT to answer the customer.

Your job is ONLY to decide:

1. Which tool should be called.
2. What arguments that tool needs.

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
Use this when the customer asks what's available in a given material, without naming a specific product.

5. answer_policy_question
Arguments:
- question
Use this when the customer asks about store policy rather than a specific product or delivery time -- for example returns, warranty, ring sizing, jewellery care, custom engraving, or payment methods. Pass their question through in their own words as `question`.

Rules:

- If the customer asks only for a price, use get_product_price.
- If the customer asks for delivery/shipping info only, use get_delivery_information.
- If the customer wants a full quote (price + delivery), use generate_quote.
- If the customer is browsing by material rather than asking about one item, use recommend_products.
- If the customer is asking about returns, warranty, sizing, care, engraving, or payment methods, use answer_policy_question.
- If the customer mentions "this", "that one", or similar references, infer the product or material from earlier in THIS message if possible.
- If a tool needs product_name or material and you genuinely cannot determine it from this message alone, set that argument to the literal string "unknown" rather than guessing. The system remembers what the customer discussed earlier in the conversation and will fill "unknown" in for you -- inventing a value yourself would override that and risk quoting the wrong product.

Return ONLY valid JSON. Do not include markdown formatting or commentary.

Example:

{{
  "tool": "get_product_price",
  "arguments": {{
    "product_name": "ring",
    "material": "gold"
  }}
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

    if "tool" not in data or "arguments" not in data:
        logger.error("LLM output missing required keys: %s", data)
        raise ToolSelectionError(f"LLM response missing 'tool' or 'arguments': {data}")

    return data


def understand_customer(message: str) -> dict:
    if not message or not message.strip():
        raise ValueError("message must not be empty")

    raw_text = _call_llm(message)
    logger.info("Raw LLM output: %s", raw_text)

    return _parse_tool_request(raw_text)
