"""
Browser-based demo dashboard -- not part of the WhatsApp product path.

Exists to demo KasaFlow to the pilot business owner without depending
on Meta's WhatsApp webhook restrictions (an unpublished app never
receives real inbound webhooks, see whatsapp_routes.py's neighbouring
manual check scripts) or passing a phone around the room. Hits the
exact same route_customer()/tool_executor() pipeline a real WhatsApp
message would, so what the owner sees here is the real system, not a
mockup.

Local/dev only, deliberately unauthenticated (unlike /process in
main.py) -- do not expose this route publicly as-is.

Browsers record voice notes as WebM/Opus, which Khaya's ASR does not
accept (see voice_tool.py's confirmed format list: ogg, mp3, wav,
flac). ffmpeg must be installed and on PATH to transcode; this is a
demo-only dependency, not something the production WhatsApp path
needs, since real WhatsApp voice notes already arrive as OGG/Opus.
"""

import base64
import logging
import subprocess
import uuid
from pathlib import Path
from urllib.parse import quote, urlparse

import requests
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, Response

from app.config import settings
from services.memory import remember_context
from services.photo_match_tool import identify_product_from_photo
from services.product_tool import get_product_karat_options
from services.response_formatter import _group_by_product, _select_diverse_groups, format_for_customer
from services.router import route_customer
from services.vision_tool import VisionServiceError, describe_product_image
from services.voice_tool import VoiceServiceError, synthesize_speech, transcribe_audio

# How many product photos to attach for a recommendations reply -- kept
# small deliberately, same reasoning as response_formatter.py's own
# _MAX_VARIANTS_LISTED: a "what rings do you have" query can match
# thousands of raw catalogue rows, and nobody wants a wall of photos any
# more than a wall of text.
_MAX_RECOMMENDATION_CARDS = 4

logger = logging.getLogger(__name__)
router = APIRouter()

_DASHBOARD_HTML_PATH = Path(__file__).parent / "demo_dashboard.html"

# Product photos come straight from the live WooCommerce site
# (data/products.json's image_url, see woocommerce_sync.py). Browsers
# embedding those URLs directly send this dashboard's localhost origin
# as the Referer -- many WordPress hosts (this one included, confirmed
# by images silently failing to render in the dashboard while the same
# URLs load fine when the site itself is browsed directly) reject
# cross-origin image requests as hotlinking. Fetching server-side and
# streaming the bytes back sidesteps that: the request to
# adomdejeweller.com now comes from this server, not the customer's
# browser, so there's no foreign Referer to reject.
_ALLOWED_IMAGE_HOST = urlparse(settings.woocommerce_url).hostname if settings.woocommerce_url else None


def _proxied_image_url(original_url: str | None) -> str | None:
    if not original_url:
        return None
    return f"/demo/image-proxy?url={quote(original_url, safe='')}"


@router.get("/demo/image-proxy")
def image_proxy(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return Response(status_code=400)
    # Restrict to the store's own image host -- this route is
    # unauthenticated (see module docstring), so without this check it
    # would be an open image-fetching proxy for anyone who can reach the
    # dev server, not just a hotlink workaround for our own catalogue.
    if _ALLOWED_IMAGE_HOST and parsed.hostname != _ALLOWED_IMAGE_HOST:
        logger.warning("Refused to proxy image from disallowed host: %s", parsed.hostname)
        return Response(status_code=400)

    try:
        upstream = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; KasaFlowDemo/1.0)",
                # Send the store's own origin as the referer, the one
                # thing a genuine on-site image request would always
                # have and a hotlinked one wouldn't.
                "Referer": f"https://{parsed.hostname}/",
            },
        )
        upstream.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Image proxy fetch failed for %s: %s", url, e)
        return Response(status_code=502)

    return Response(
        content=upstream.content,
        media_type=upstream.headers.get("content-type", "image/jpeg"),
    )


def _convert_to_wav(raw_bytes: bytes) -> bytes:
    process = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-y", "pipe:1"],
        input=raw_bytes,
        capture_output=True,
        timeout=30,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed -- is ffmpeg installed and on PATH? "
            f"stderr: {process.stderr.decode(errors='ignore')[:500]}"
        )
    return process.stdout


@router.get("/demo", response_class=HTMLResponse)
def demo_dashboard() -> str:
    return _DASHBOARD_HTML_PATH.read_text(encoding="utf-8")


def _build_recommendation_cards(result: dict) -> list[dict]:
    """One photo card per distinct product in a recommendations reply --
    answers "can I see pictures" for a browse-style query directly,
    rather than needing the assistant to remember what it just listed
    and handle a separate follow-up request for photos."""
    items = result.get("recommendations") or []
    # Same category-diversified selection as the text reply (see
    # response_formatter._select_diverse_groups) -- without this, an
    # unfiltered browse's photo cards would show 4 necklaces and no
    # rings purely because of catalogue file order, while the text
    # above them correctly shows a mix. Cards and text must agree.
    groups = _select_diverse_groups(_group_by_product(items), max_groups=_MAX_RECOMMENDATION_CARDS)
    cards = []
    for name, variants in groups:
        with_photo = next((v for v in variants if v.get("image_url")), variants[0])
        prices = [v["price"] for v in variants]
        low, high = min(prices), max(prices)
        price_label = f"GH₵{low:,.2f}" if low == high else f"GH₵{low:,.2f}-GH₵{high:,.2f}"
        cards.append({
            "product": name,
            "image_url": _proxied_image_url(with_photo.get("image_url")),
            "price_label": price_label,
        })
    return [c for c in cards if c["image_url"]]


@router.post("/demo/message")
async def demo_message(
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    session_id: str | None = Form(None),
    language: str = Form("twi"),
    voice_reply: bool = Form(False),
):
    session_id = session_id or str(uuid.uuid4())
    transcript = None
    result = None
    matched_tool = None
    photo_unconfirmed = False

    if audio is not None:
        raw_bytes = await audio.read()
        try:
            wav_bytes = _convert_to_wav(raw_bytes)
        except Exception as e:
            logger.exception("Audio conversion failed")
            return {"error": str(e)}

        try:
            transcript = transcribe_audio(wav_bytes, language=language, audio_format="wav")
        except VoiceServiceError as e:
            logger.error("Transcription failed: %s", e)
            return {"error": f"Transcription failed: {e}"}

        if not transcript.strip():
            return {"error": "Couldn't make that out -- try again or type it instead."}
        customer_text = transcript
    elif image is not None:
        # Mirrors whatsapp_routes.py's image branch -- a photo becomes
        # customer_text the same way a voice note does. describe_product_image()
        # itself was flagged in vision_tool.py as not yet confirmed
        # against a real OpenAI response before this dashboard existed;
        # confirmed working live, 2026-08-17.
        image_bytes = await image.read()
        image_mime_type = image.content_type or "image/jpeg"
        try:
            transcript = describe_product_image(image_bytes, mime_type=image_mime_type)
        except VisionServiceError as e:
            logger.error("Image description failed: %s", e)
            return {"error": f"Image description failed: {e}"}

        if not transcript.strip():
            return {"error": "Couldn't quite tell what that was -- mind describing it in words instead?"}

        # A caption sent alongside the photo ("is this the Adinkra
        # necklace") is a real identifying signal -- previously dropped
        # entirely, since this branch only ever looked at `image` and
        # never `text`. No longer used for candidate narrowing itself
        # (see photo_match_tool.py's module docstring for why that
        # moved to image embeddings), but still folded into
        # customer_text: it's what gets routed if no confident photo
        # match is found, and what's shown as the "Read as" caption in
        # the UI either way.
        caption = text.strip() if text and text.strip() else None
        customer_text = f"{caption} {transcript}" if caption else transcript

        # Confirmed live, 2026-08-17: a generic vision description
        # ("gold pendant necklace with chain link design") routed
        # through the ordinary text pipeline only ever produces a
        # category-level browse (see llm.py's recommend_products
        # guidance), never a specific-item match -- there's no literal
        # product name to exact-match against. identify_product_from_photo()
        # attempts an actual visual match (image embeddings, then a
        # vision comparison against the narrowed shortlist -- see
        # photo_match_tool.py) before falling back to that same generic
        # browse.
        try:
            matched_name = identify_product_from_photo(image_bytes, image_mime_type)
        except VisionServiceError as e:
            logger.warning("Photo identification failed, falling back to text pipeline: %s", e)
            matched_name = None

        if matched_name:
            variants = get_product_karat_options(matched_name)
            matched_image_url = next((v.get("image_url") for v in variants if v.get("image_url")), None)
            result = {
                "identified_product": {"product": matched_name, "image_url": matched_image_url, "variants": variants},
                # Duplicated at top level too -- the image_url/cards
                # extraction below (shared with every other result
                # shape) reads result.get("image_url") directly, same
                # as get_product_price()'s shape does.
                "image_url": matched_image_url,
            }
            # A confident match bypasses route_customer() entirely (it's
            # answered deterministically, no LLM call needed), which
            # means remember_context() -- normally called inside
            # router.py's _execute_single() for every LLM-driven tool
            # call -- never runs for this turn. Without this, a
            # follow-up "I'll take it" or "2 please" right after a photo
            # match has nothing to resolve "it"/an implicit product
            # against, since nothing was ever remembered.
            #
            # material and category explicitly cleared (set to None,
            # not just omitted): remember_context() only overwrites keys
            # actually present in the dict passed to it, so if an
            # earlier, unrelated product in this same session had
            # remembered a karat (e.g. "the 18k ring" from three turns
            # ago), leaving material out here would let that stale
            # karat silently attach itself to this newly-identified,
            # different product on the very next turn.
            remember_context(session_id, {"product_name": matched_name, "material": None, "category": None})
            matched_tool = "identify_product_from_photo"
        else:
            # No confident single-item match -- fall back to the
            # ordinary text-driven pipeline, same behaviour as before
            # this feature existed. Flagged with photo_unconfirmed so
            # the reply below is honest about being a nearest-match
            # browse, not a confirmed identification.
            photo_unconfirmed = True
    elif text and text.strip():
        customer_text = text.strip()
    else:
        return {"error": "Send text, a voice note, or a photo."}

    if result is None:
        result = route_customer(customer_text, session_id=session_id)

    reply_text = format_for_customer(result)
    if photo_unconfirmed:
        reply_text = (
            "I couldn't confirm that's one exact item in our catalogue, but here's what looks closest:\n\n"
            + reply_text
        )
    image_url = _proxied_image_url(result.get("image_url")) if isinstance(result, dict) else None
    cards = _build_recommendation_cards(result) if isinstance(result, dict) and "recommendations" in result else []

    reply_audio_base64 = None
    if voice_reply:
        try:
            audio_bytes = synthesize_speech(reply_text, language=language)
            reply_audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        except VoiceServiceError as e:
            logger.warning("TTS reply failed, returning text only: %s", e)

    return {
        "session_id": session_id,
        "transcript": transcript,
        "reply_text": reply_text,
        "image_url": image_url,
        "cards": cards,
        "reply_audio_base64": reply_audio_base64,
        "tool": matched_tool,
    }
