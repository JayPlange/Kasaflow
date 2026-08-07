"""
Thin wrapper around Meta's WhatsApp Cloud API (Graph API). Confirmed
against Meta's own public API reference, not a guess like Khaya's exact
paths in voice_tool.py -- this is a stable, well-documented API.

Requires a Meta developer app with WhatsApp added, a permanent (or
long-lived) access token, and the phone number ID for the jeweller's
WhatsApp Business number. All three come from Meta's developer console,
not something this code can provision.
"""

import logging
import time

import requests

from app.config import settings

logger = logging.getLogger(__name__)

_GRAPH_API_VERSION = "v21.0"
_TIMEOUT_SECONDS = 30
_MAX_RETRIES = 2


class WhatsAppError(Exception):
    """Raised when the Graph API can't be reached or rejects a request."""


def _require_whatsapp_config() -> None:
    missing = [
        name
        for name, value in [
            ("WHATSAPP_ACCESS_TOKEN", settings.whatsapp_access_token),
            ("WHATSAPP_PHONE_NUMBER_ID", settings.whatsapp_phone_number_id),
        ]
        if not value
    ]
    if missing:
        raise WhatsAppError(f"Missing WhatsApp config: {', '.join(missing)}.")


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.whatsapp_access_token}"}


def _base_url() -> str:
    return f"https://graph.facebook.com/{_GRAPH_API_VERSION}/{settings.whatsapp_phone_number_id}"


def _post_with_retry(url: str, **kwargs) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            response = requests.post(url, timeout=_TIMEOUT_SECONDS, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning("WhatsApp API timeout (attempt %s): %s", attempt, e)
            time.sleep(min(2**attempt, 8))
        except requests.exceptions.HTTPError as e:
            # 4xx from Meta (bad token, invalid recipient) won't succeed
            # on retry -- fail immediately rather than burn attempts.
            logger.error("WhatsApp API rejected the request: %s -- %s", e, response.text)
            raise WhatsAppError(f"WhatsApp API error: {e}") from e
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning("WhatsApp API connection error (attempt %s): %s", attempt, e)
            time.sleep(min(2**attempt, 8))

    raise WhatsAppError(f"WhatsApp API unreachable after retries: {last_error}")


def send_text_message(to: str, body: str) -> None:
    _require_whatsapp_config()
    _post_with_retry(
        f"{_base_url()}/messages",
        headers=_headers(),
        json={
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": body},
        },
    )


def send_image_message(to: str, image_url: str, caption: str | None = None) -> None:
    """Sends an image by URL directly -- unlike send_audio_message, this
    doesn't need the upload-then-send two-step, because WhatsApp will
    fetch a publicly reachable `link` itself. Every image URL this gets
    called with comes from the synced WooCommerce catalogue (real
    product photos already hosted on adomdejeweller.com), so a public
    link is always what's available here."""
    _require_whatsapp_config()

    image_payload: dict = {"link": image_url}
    if caption:
        image_payload["caption"] = caption

    _post_with_retry(
        f"{_base_url()}/messages",
        headers=_headers(),
        json={
            "messaging_product": "whatsapp",
            "to": to,
            "type": "image",
            "image": image_payload,
        },
    )


def send_audio_message(to: str, audio_bytes: bytes, mime_type: str = "audio/mpeg") -> None:
    """Uploads the audio as media first (WhatsApp requires a media ID,
    not a raw attachment, for outbound messages), then sends it."""
    _require_whatsapp_config()

    upload = requests.post(
        f"{_base_url()}/media",
        headers=_headers(),
        data={"messaging_product": "whatsapp", "type": mime_type},
        files={"file": ("reply.mp3", audio_bytes, mime_type)},
        timeout=_TIMEOUT_SECONDS,
    )
    upload.raise_for_status()
    media_id = upload.json()["id"]

    _post_with_retry(
        f"{_base_url()}/messages",
        headers=_headers(),
        json={
            "messaging_product": "whatsapp",
            "to": to,
            "type": "audio",
            "audio": {"id": media_id},
        },
    )


def download_media(media_id: str) -> bytes:
    """Two-step fetch, per Meta's API: resolve the media ID to a
    short-lived URL, then download from that URL, both authenticated."""
    _require_whatsapp_config()

    meta_response = requests.get(
        f"https://graph.facebook.com/{_GRAPH_API_VERSION}/{media_id}",
        headers=_headers(),
        timeout=_TIMEOUT_SECONDS,
    )
    meta_response.raise_for_status()
    media_url = meta_response.json()["url"]

    file_response = requests.get(media_url, headers=_headers(), timeout=_TIMEOUT_SECONDS)
    file_response.raise_for_status()
    return file_response.content
