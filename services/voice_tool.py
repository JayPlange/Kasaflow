"""
Thin wrapper around Khaya AI's ASR and TTS APIs (translation.ghananlp.org).

CONFIRM BEFORE RELYING ON THIS: Khaya's developer portal is built on
Azure API Management, which confirms the auth header shape
(Ocp-Apim-Subscription-Key) and the exact request shape for /translate:

    POST /translate
    {"in": "Kofi is going to school", "lang": "en-tw"}

The ASR and TTS endpoint paths below (/asr, /tts) are the conventional
Azure APIM pattern, NOT confirmed against Khaya's actual operation list --
that page requires a signed-in account to render. Sign up at
translation.ghananlp.org, grab a subscription key, and check the real
paths/parameter names in the portal (or just hit these with a real audio
file and see what comes back) before this goes anywhere near production.
Everything else here -- retry policy, error handling, isolation behind
one interface -- is safe to build against now regardless of whether the
exact path turns out to be /asr or /speech-to-text.

Isolated behind transcribe_audio() / synthesize_speech() specifically so
swapping providers later (Abena, a fine-tuned Whisper, whichever) is a
one-file change, same reasoning as the original roadmap's plan for this
module, just against a provider that already has real Twi ASR instead of
one that doesn't yet.
"""

import logging
import time

import requests

from app.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30
_MAX_RETRIES = 2

# Khaya's playground UI listed "English (Ghana)" as en, standard ISO
# codes elsewhere. tw = Twi. Confirm the exact codes Khaya expects
# against the real docs -- this is the same "verify before trusting"
# flag as the endpoint paths above.
LANGUAGE_CODES = {
    "twi": "tw",
    "english": "en",
    "pidgin": "pcm",
}


class VoiceServiceError(Exception):
    """Raised when Khaya can't be reached or returns something unusable."""


def _require_khaya_config() -> None:
    if not settings.khaya_api_key:
        raise VoiceServiceError(
            "KHAYA_API_KEY is not set. Add it to .env -- generate one at "
            "translation.ghananlp.org after signing up (free tier: 100 calls)."
        )


def _headers() -> dict:
    return {"Ocp-Apim-Subscription-Key": settings.khaya_api_key}


def transcribe_audio(audio_bytes: bytes, language: str = "twi") -> str:
    """Send a voice note's raw audio bytes to Khaya's ASR endpoint,
    return the transcript as plain text.

    language: one of LANGUAGE_CODES' keys ("twi", "english", "pidgin").
    WhatsApp voice notes arrive as OGG/Opus -- confirm Khaya accepts that
    directly (Abena's playground does; Khaya's format support isn't
    confirmed here) or convert with ffmpeg first if it rejects the format.
    """
    _require_khaya_config()
    lang_code = LANGUAGE_CODES.get(language, language)

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            response = requests.post(
                f"{settings.khaya_api_base}/asr",  # PATH UNCONFIRMED, see module docstring
                headers=_headers(),
                params={"lang": lang_code},
                files={"audio": ("voice_note.ogg", audio_bytes, "audio/ogg")},
                timeout=_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            # Key name (`transcript` vs `text` vs `out`) unconfirmed --
            # log the raw shape once against a real response and fix this
            # line rather than guess further.
            return data.get("transcript") or data.get("text") or ""

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


def synthesize_speech(text: str, language: str = "twi") -> bytes:
    """Turn a reply into spoken audio (bytes, expected MP3/OGG -- confirm
    which against a real response) for sending back as a WhatsApp voice
    note. Twi TTS is confirmed live on Khaya's playground; this is the
    API-shaped version of the same thing."""
    _require_khaya_config()
    lang_code = LANGUAGE_CODES.get(language, language)

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            response = requests.post(
                f"{settings.khaya_api_base}/tts",  # PATH UNCONFIRMED, see module docstring
                headers=_headers(),
                json={"text": text, "lang": lang_code},
                timeout=_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.content

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
