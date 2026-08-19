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

# max_retries=0: the openai SDK retries transient failures internally
# by default (2 retries), on top of the manual retry/backoff loop both
# functions below already implement. Left at the SDK default, a
# connection error can be retried up to (our attempts) x (SDK retries)
# times before either loop gives up -- confirmed live, 2026-08-17: a
# single failed photo request took close to 2 minutes to finally
# surface "Vision API unreachable after 3 attempts", far longer than
# 3 x _TIMEOUT_SECONDS (60s) alone would explain. Disabling the SDK's
# own retries leaves exactly the one retry policy defined here.
client = OpenAI(api_key=settings.openai_api_key, max_retries=0)

_TIMEOUT_SECONDS = 20
_MAX_RETRIES = 2

# The sentinel the model is told to return for a photo that isn't
# jewellery at all, so the caller can tell "genuinely nothing there"
# apart from "described it, just didn't match the catalogue" -- those
# need different replies to the customer.
_NOT_JEWELLERY = "NOT_JEWELLERY"

# Confirmed live, 2026-08-18: a real catalogue photo showing loose
# teardrop-shaped charms and a curved wire on a workbench (not a
# finished, worn piece) was confidently described as "gold cuff
# bracelet with teardrop ends" -- wrong item type entirely. That wrong
# category word then steered the downstream candidate search (services/
# photo_match_tool.py) toward completely unrelated products, so the
# correct match never even reached the shortlist for the visual
# comparison step to consider. The instruction below now allows
# describing shape/material honestly instead of forcing a specific
# item-type guess when the photo itself doesn't clearly show one --
# better to search on "gold teardrop charms" than to confidently guess
# the wrong category.
_PROMPT = f"""You are looking at a photo a customer sent to a jewellery store on WhatsApp.

Describe the item in this photo as a short search query, the way a
customer might type it -- item type, metal/colour, and any
distinguishing style, nothing else. For example: "gold twist ring",
"silver chain necklace", "white stone gold earrings".

If the photo doesn't clearly show what TYPE of finished piece it is
(ring, bracelet, necklace, earrings) -- for example loose components,
raw material, or an unusual angle -- do not guess a category. Instead
describe just the visible shapes, materials, and colours, e.g. "gold
teardrop-shaped charms" rather than confidently guessing "bracelet" or
"necklace" when the photo doesn't actually show one assembled.

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


# Confirmed live, 2026-08-17: two visually similar plain gold chain
# necklaces in the catalogue ("Chain Gold Necklace, 50g" and "Rounded
# Chains Goold Necklace, 12g") were genuinely confusable -- the same
# customer photo was matched to the wrong one once, then correctly the
# next time. A wrong-but-confident answer here is worse than an honest
# "can't tell", the same no-fabrication principle response_formatter.py
# already follows for pricing -- the prompt below leans explicitly
# conservative, and temperature=0 (see the call below) removes one
# source of that inconsistency, though vision comparisons like this
# aren't perfectly deterministic even at temperature=0.
_MATCH_PROMPT = """You are comparing a customer's photo of a jewellery item (the first image) against several candidate product photos from a jewellery store's own catalogue (the images that follow, each preceded by a number and its product name).

Determine whether the customer's photo shows the SAME physical piece of jewellery as one of the candidates -- not just a similar style or type, the same specific product. Catalogue photos are often studio shots while a customer's photo may be taken in different lighting or against a different background, so allow for that, but the design, stones, chain style, and any charm/pendant details must genuinely match.

Look closely before deciding: plain gold chains in particular can look almost identical at a glance but differ in link shape, thickness, or length once you compare closely -- check these details specifically before picking one, rather than matching on general impression alone.

Being wrong is worse than saying you're not sure. If you have any real doubt, or if two candidates both look plausible, respond NONE rather than guessing between them.

Respond with ONLY the candidate's number if you are confident it is the same item, or exactly NONE if you are not confident any candidate matches. No other text, no punctuation, no explanation."""


def _data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def match_photo_to_candidates(
    customer_image_bytes: bytes,
    customer_mime_type: str,
    candidates: list[tuple[str, bytes, str]],
) -> int | None:
    """Given a customer's photo and a short list of
    (product_name, image_bytes, mime_type) candidates from the
    catalogue, asks the vision model which candidate, if any, is the
    same physical item.

    Returns the 0-based index into `candidates` of a confident match,
    or None. Deliberately a narrow "which of these few, or none"
    question rather than "identify this from the whole catalogue" --
    the candidates are expected to already be narrowed by text search
    first (see services/photo_match_tool.py's module docstring for why
    a single description-then-search step can't reliably guess a
    specific catalogue name from visual features alone).

    A second, distinct use of the vision API from describe_product_image
    above -- multiple images in one call, rather than one. Not yet
    confirmed against a real request/response any more than that
    function was before its own first live test (this environment has
    no outbound access to api.openai.com) -- flagging that explicitly
    rather than presenting this as verified. Needs a live check the
    same way describe_product_image got one before this is trusted in
    front of a customer.
    """
    if not candidates:
        return None

    content = [
        {"type": "input_text", "text": _MATCH_PROMPT},
        {"type": "input_text", "text": "Customer's photo:"},
        {"type": "input_image", "image_url": _data_url(customer_image_bytes, customer_mime_type)},
    ]
    for i, (name, image_bytes, mime_type) in enumerate(candidates, start=1):
        content.append({"type": "input_text", "text": f"Candidate {i}: {name}"})
        content.append({"type": "input_image", "image_url": _data_url(image_bytes, mime_type)})

    last_error: Exception | None = None
    total_attempts = _MAX_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        try:
            response = client.responses.create(
                model=settings.openai_model,
                input=[{"role": "user", "content": content}],
                timeout=_TIMEOUT_SECONDS,
                # 0, not left at the API default -- this is a
                # pick-one-of-few decision where consistency matters
                # more than creative variation. Doesn't make this fully
                # deterministic (see docstring above), but removes
                # sampling randomness as one source of the inconsistency
                # confirmed live, 2026-08-17.
                temperature=0,
            )
            text = response.output_text.strip()
            if text.upper() == "NONE":
                return None
            try:
                index = int(text)
            except ValueError:
                logger.warning("Vision match returned an unparseable response: %r", text)
                return None
            if 1 <= index <= len(candidates):
                return index - 1
            logger.warning("Vision match returned an out-of-range candidate number: %r", text)
            return None

        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            logger.warning(
                "Vision match call failed (attempt %s/%s): %s", attempt, total_attempts, e
            )
            if attempt < total_attempts:
                time.sleep(min(2**attempt, 8))

        except APIError as e:
            logger.error("Non-retryable vision API error during photo match: %s", e)
            raise VisionServiceError(f"Vision match request failed: {e}") from e

    raise VisionServiceError(f"Vision API unreachable after {total_attempts} attempts: {last_error}")
