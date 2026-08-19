"""
WhatsApp Cloud API webhook.

Design choice worth being explicit about: the customer's WhatsApp phone
number is used directly as the session_id passed to route_customer(). It
already exists, it's stable across a conversation, and it's exactly the
identifier the engine needs to remember "the ring we discussed
earlier" -- no separate session-mapping layer required.

Fast-ACK pattern: WhatsApp expects a 200 within a few seconds or it
retries the whole delivery. Real work (downloading audio, calling Khaya,
calling the LLM) happens in a BackgroundTask, added after the handler
already has everything it needs from the payload -- the response to
Meta goes out immediately, the customer's actual reply follows once
processing finishes.
"""

import logging
import os

from fastapi import APIRouter, BackgroundTasks, Request, Response

from app.config import settings
from services.response_formatter import format_for_customer
from services.router import route_customer
from services.vision_tool import VisionServiceError, describe_product_image
from services.voice_tool import VoiceServiceError, synthesize_speech, transcribe_audio
from services.whatsapp_client import (
    WhatsAppError,
    download_media,
    send_audio_message,
    send_image_message,
    send_text_message,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Not yet measured which language dominates this jeweller's real voice
# notes -- per tonight's plan, that's exactly what this week's visit is
# for. English is the safer default until the actual mix is known;
# override with DEFAULT_VOICE_LANGUAGE in .env once it is.
_DEFAULT_VOICE_LANGUAGE = os.getenv("DEFAULT_VOICE_LANGUAGE", "english")


@router.get("/webhook/whatsapp")
def verify_webhook(request: Request):
    """Meta calls this once, when the webhook URL is first registered in
    the developer console, to prove you control the endpoint."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.whatsapp_verify_token:
        logger.info("WhatsApp webhook verified successfully")
        return Response(content=challenge, media_type="text/plain")

    logger.warning("WhatsApp webhook verification failed -- token mismatch")
    return Response(status_code=403)


@router.post("/webhook/whatsapp")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()

    try:
        messages = (
            payload["entry"][0]["changes"][0]["value"].get("messages", [])
        )
    except (KeyError, IndexError):
        # Meta also posts non-message events (delivery receipts, etc.)
        # to this same endpoint -- not an error, just nothing to do.
        return {"status": "ignored"}

    for message in messages:
        background_tasks.add_task(_handle_message, message)

    # ACK immediately. Meta doesn't want to wait for an LLM round-trip.
    return {"status": "received"}


def _handle_message(message: dict) -> None:
    from_number = message["from"]
    message_type = message["type"]

    try:
        if message_type == "text":
            customer_text = message["text"]["body"]

        elif message_type == "audio":
            audio_bytes = download_media(message["audio"]["id"])
            customer_text = transcribe_audio(audio_bytes, language=_DEFAULT_VOICE_LANGUAGE)
            if not customer_text.strip():
                send_text_message(
                    from_number,
                    "I couldn't quite make that out -- mind typing it instead, or trying the voice note again?",
                )
                return

        elif message_type == "image":
            image_bytes = download_media(message["image"]["id"])
            # describe_product_image() turns the photo into the same kind
            # of short text a typed/transcribed message would produce --
            # from here on it goes through the exact same
            # route_customer()/format_for_customer() pipeline as text and
            # audio, no separate image-matching logic needed.
            customer_text = describe_product_image(image_bytes)
            if not customer_text.strip():
                send_text_message(
                    from_number,
                    "I couldn't quite tell what that was from the photo -- mind describing it in words instead?",
                )
                return

            # A caption sent alongside the photo ("is this the Adinkra
            # necklace") is a real identifying signal -- previously
            # dropped entirely, since only message["image"]["id"] was
            # ever read here. WhatsApp's Cloud API image object supports
            # an optional "caption" field (see whatsapp_client.py's own
            # send_image_message, which sends one on the way out); the
            # inbound payload mirrors that shape. Confirmed a matching
            # gap existed in app/demo_routes.py's own image branch,
            # 2026-08-17, fixed there the same way.
            caption = (message["image"].get("caption") or "").strip()
            if caption:
                customer_text = f"{caption} {customer_text}"

        else:
            logger.info("Unhandled WhatsApp message type: %s", message_type)
            return

        result = route_customer(customer_text, session_id=from_number)
        reply_text = format_for_customer(result)
        # get_product_price/generate_quote's shapes carry image_url when
        # the matched product has one; result can also be a bare list
        # (recommendations) or None (no match at all), neither of which
        # has a single photo to attach, so only single-product results
        # get an image reply.
        image_url = result.get("image_url") if isinstance(result, dict) else None

        if message_type == "audio":
            try:
                audio_reply = synthesize_speech(reply_text, language=_DEFAULT_VOICE_LANGUAGE)
                send_audio_message(from_number, audio_reply)
            except VoiceServiceError as e:
                logger.warning("TTS reply failed, falling back to text: %s", e)
                send_text_message(from_number, reply_text)
        elif image_url:
            try:
                send_image_message(from_number, image_url, caption=reply_text)
            except WhatsAppError as e:
                logger.warning("Image reply failed, falling back to text: %s", e)
                send_text_message(from_number, reply_text)
        else:
            send_text_message(from_number, reply_text)

    except (VoiceServiceError, WhatsAppError, VisionServiceError) as e:
        logger.error("Failed to handle message from %s: %s", from_number, e)
    except Exception:
        logger.exception("Unexpected error handling message from %s", from_number)
