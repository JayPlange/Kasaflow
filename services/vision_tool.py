"""
Thin wrapper around OpenAI's vision-capable Responses API.

Turns a customer's product photo into a short text description, which
then gets routed through the exact same pipeline as typed or
transcribed text (see whatsapp_routes.py) -- a photo becomes just
another way to produce customer_text, no separate matching logic
needed. This mirrors how voice_tool.py's transcribe_audio() feeds a
transcript into the same router; describe_product_image() feeds a
description in the same way.

Not yet confirmed against a real request/response the way
voice_tool.py and whatsapp_client.py were -- flagging that explicitly
rather than presenting this as verified. The Responses API's
multimodal input shape (input_image content blocks) is the documented
shape, but hasn't been exercised against the real API from this
environment (outbound access to api.openai.com isn't available here).
Worth a live check before this goes to production, the same way the
Twi-translation and multi-request prompt changes got one.
"""

import base64
import logging
import time

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)

_TIMEOUT_SECONDS = 20
_MAX_RETRIES = 2

# The sentinel the model is told to return for a photo that isn't
# jewellery at all, so the caller can tell "genuinely nothing there"
# apart from "described it, just didn't match the catalogue" -- those
# need different replies to the customer.
_NOT_JEWELLERY = "NOT_JEWELLERY"

_PROMPT = f"""You are looking at a photo a customer sent to a jewellery store on WhatsApp.

Describe the jewellery item in this photo as a short search query, the
way a customer might type it -- item type, metal/colour, and any
distinguishing style, nothing else. For example: "gold twist ring",
"silver chain necklace", "white stone gold earrings".

If the photo does not clearly show a piece of jewellery, respond with
exactly: {_NOT_JEWELLERY}

Respond with ONLY the description (or {_NOT_JEWELLERY}), no other text,
no markdown, no punctuation beyond what's naturally in the description."""


class VisionServiceError(Exception):
    """Raised when the vision API can't be reached or returns something unusable."""


def describe_product_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Returns a short text description of the jewellery item in the
    photo, suitable for feeding straight into the same catalogue search
    a typed/transcribed message would use (get_product_price,
    recommend_products, etc. via route_customer).

    Returns "" when the photo doesn't appear to show a piece of
    jewellery at all, so the caller can give the customer an honest
    "couldn't tell what that was" reply instead of routing a
    nonsense description into a catalogue match.
    """
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"

    last_error: Exception | None = None
    total_attempts = _MAX_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        try:
            response = client.responses.create(
                model=settings.openai_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": _PROMPT},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                timeout=_TIMEOUT_SECONDS,
            )
            text = response.output_text.strip()
            return "" if text == _NOT_JEWELLERY else text

        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            logger.warning(
                "Vision call failed (attempt %s/%s): %s", attempt, total_attempts, e
            )
            if attempt < total_attempts:
                time.sleep(min(2**attempt, 8))  # 2s, 4s...

        except APIError as e:
            logger.error("Non-retryable vision API error: %s", e)
            raise VisionServiceError(f"Vision request failed: {e}") from e

    raise VisionServiceError(f"Vision API unreachable after {total_attempts} attempts: {last_error}")
