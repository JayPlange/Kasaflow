"""
Thin wrapper around Khaya AI's ASR v3 and TTS v2 APIs.

Confirmed against real requests and real responses (2026-08-05), not
guessed, everything in the earlier version of this file that was
flagged as unconfirmed has now been checked directly against Khaya's
live API:

  - Real API host is translation-api.ghananlp.org, NOT
    translation.ghananlp.org (that's the docs/developer-portal site
    only -- confirmed by a live request returning a proper response
    from the -api host and nothing usable from the docs one).
  - Auth: `Ocp-Apim-Subscription-Key` header, confirmed via a real 401
    (wrong/missing key) then a real 200 (correct key).
  - Language codes are ISO 639-3 (three letters), confirmed via
    GET /asr/v3/languages: "twi" for Twi, "eng" for English, "pcm" for
    Naija Pidgin. The two-letter codes this file used to send ("tw",
    "en") are explicitly rejected by the real API -- its own error
    message for the legacy code is "Invalid language provided. Use
    'twi' instead."
  - ASR: POST /asr/v3/transcribe?language={code}, raw audio bytes as
    the request body (not a multipart file upload), response is
    {"text": "..."}.
  - TTS: POST /tts/v2/synthesize, a JSON body (not raw bytes, not
    multipart), response is raw audio bytes in the requested format.

Isolated behind transcribe_audio() / synthesize_speech() so swapping
providers later is still a one-file change, same reasoning as before,
just against confirmed shapes now instead of assumed ones.
"""

import logging
import time

import requests

from app.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30
_MAX_RETRIES = 2

# Confirmed directly against GET /asr/v3/languages -- this is the real
# response, not a guess. Twi has two dialect codes ("twi" = Asante Twi,
# "atw" = Akuapem Twi); "twi" is used here as the default since Asante
# Twi is the more widely spoken dialect and a reasonable default absent
# a reason to pick the other one for a specific customer base.
LANGUAGE_CODES = {
    "twi": "twi",
    "akuapem_twi": "atw",
    "english": "eng",
    "pidgin": "pcm",
}

# TTS speaker IDs, confirmed via a real 400 response listing the valid
# set: "Speaker 'invalid_speaker' not found. Available speakers are
# ['male_low', 'male_high', 'female']". Not necessarily the complete
# list for every language, GET /tts/v2/speakers is the authoritative
# source if a specific language ever 400s on one of these.
DEFAULT_SPEAKER = None  # omit to let Khaya choose the default voice for the language


class VoiceServiceError(Exception):
    """Raised when Khaya can't be reached or returns something unusable."""


def _require_khaya_config() -> None:
    if not settings.khaya_api_key:
        raise VoiceServiceError(
            "KHAYA_API_KEY is not set. Add it to .env -- generate one at "
            "translation.ghananlp.org after subscribing to the free Developer tier."
        )


def _headers() -> dict:
    return {"Ocp-Apim-Subscription-Key": settings.khaya_api_key}


def _raise_for_khaya_error(response: requests.Response, context: str) -> None:
    """Khaya's 400 responses carry a real, specific reason in JSON, not
    just a status code -- surface that instead of a generic HTTP error,
    it's usually the exact fix needed (e.g. "Use 'twi' instead")."""
    try:
        body = response.json()
        detail = body.get("error", {}).get("message", response.text)
    except ValueError:
        detail = response.text
    raise VoiceServiceError(f"{context}: {response.status_code} -- {detail}")


def transcribe_audio(audio_bytes: bytes, language: str = "twi", audio_format: str = "ogg") -> str:
    """Send a voice note's raw audio bytes to Khaya's ASR endpoint,
    return the transcript as plain text.

    language: one of LANGUAGE_CODES' keys, or a raw ISO 639-3 code directly.
    audio_format: "ogg", "mp3"/"mpeg", "wav", or "flac" -- confirmed
    supported formats. WhatsApp voice notes arrive as OGG/Opus, which
    Khaya accepts directly, no conversion needed.
    """
    _require_khaya_config()
    lang_code = LANGUAGE_CODES.get(language, language)
    content_type = "audio/mpeg" if audio_format in ("mp3", "mpeg") else f"audio/{audio_format}"

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            response = requests.post(
                f"{settings.khaya_api_base}/asr/v3/transcribe",
                headers={**_headers(), "Content-Type": content_type},
                params={"language": lang_code},
                data=audio_bytes,  # raw bytes, confirmed -- not files=
                timeout=_TIMEOUT_SECONDS,
            )
            if response.status_code == 400:
                _raise_for_khaya_error(response, "Khaya ASR rejected the request")
            response.raise_for_status()
            return response.json().get("text", "")

        except VoiceServiceError:
            raise  # already a clean, specific message from _raise_for_khaya_error
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning("Khaya ASR timeout (attempt %s): %s", attempt, e)
            time.sleep(min(2**attempt, 8))
        except requests.exceptions.HTTPError as e:
            logger.error("Khaya ASR rejected the request: %s", e)
            raise VoiceServiceError(f"Transcription failed: {e}") from e
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning("Khaya ASR connection error (attempt %s): %s", attempt, e)
            time.sleep(min(2**attempt, 8))

    raise VoiceServiceError(f"Khaya ASR unreachable after retries: {last_error}")


def synthesize_speech(
    text: str,
    language: str = "twi",
    speaker_id: str | None = DEFAULT_SPEAKER,
    audio_format: str = "mp3",
) -> bytes:
    """Turn a reply into spoken audio for sending back as a WhatsApp
    voice note. Defaults to mp3 to match whatsapp_client.py's
    send_audio_message default mime type -- change both together if
    either changes."""
    _require_khaya_config()
    lang_code = LANGUAGE_CODES.get(language, language)

    payload = {"text": text, "language": lang_code, "format": audio_format}
    if speaker_id:
        payload["speaker_id"] = speaker_id

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            response = requests.post(
                f"{settings.khaya_api_base}/tts/v2/synthesize",
                headers=_headers(),
                json=payload,
                timeout=_TIMEOUT_SECONDS,
            )
            if response.status_code == 400:
                _raise_for_khaya_error(response, "Khaya TTS rejected the request")
            response.raise_for_status()
            return response.content

        except VoiceServiceError:
            raise
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning("Khaya TTS timeout (attempt %s): %s", attempt, e)
            time.sleep(min(2**attempt, 8))
        except requests.exceptions.HTTPError as e:
            logger.error("Khaya TTS rejected the request: %s", e)
            raise VoiceServiceError(f"Speech synthesis failed: {e}") from e
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning("Khaya TTS connection error (attempt %s): %s", attempt, e)
            time.sleep(min(2**attempt, 8))

    raise VoiceServiceError(f"Khaya TTS unreachable after retries: {last_error}")
