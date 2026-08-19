"""
Google Geocoding API wrapper, used to check a customer's real delivery
address against this business's actual rider zones (Accra, Kumasi) --
and, separately, whether it's inside Ghana at all.

Why this exists, not just a bigger hardcoded city list: this file's
predecessor (delivery_tool.py's _GHANA_PLACE_NAMES) only recognised
Ghana's 16 regional capitals. Real customers don't give region names --
they give neighbourhoods ("East Legon", "Suame", "Mankessim"), which is
exactly how people actually talk about where they live (2026-08-19,
Webb: "most customers would mention... East Legon... Suame... Mankessim
rather than call out the name of that region"). No fixed list can cover
every neighbourhood in Ghana; a real geocoder resolves "East Legon" to
locality "Accra" the same way it resolves "Suame" to "Kumasi", without
this codebase needing to know either name in advance.

Confirmed against Google's own Geocoding API reference
(developers.google.com/maps/documentation/geocoding) at the time this
was written -- the request shape (GET .../geocode/json?address=...&key=...)
and response shape (top-level "status" and "results", each result's
"address_components" list of {"long_name", "short_name", "types": [...]})
both come from that page and from Google's own published example JSON,
not assumption. NOT yet exercised against a real, authenticated Google
Maps call from this environment (no GOOGLE_MAPS_API_KEY configured here,
and no outbound access to maps.googleapis.com from this sandbox) --
flagging that explicitly, the same discipline image_embed_tool.py
followed for Cohere before its own first live run. Needs one real check
(a real GOOGLE_MAPS_API_KEY, a handful of real Ghanaian addresses -- East
Legon, Suame, Mankessim are the exact three Webb asked about) before
this is trusted in front of a customer.

Never a hard dependency: resolve_delivery_match() below is the only
function anything else in this codebase should call. It falls back to
delivery_tool.delivery_option_matches_address()'s offline heuristic
whenever geocoding isn't configured or fails for any reason (network
error, bad response, no API key) -- the same "best-effort, never block
the core flow on an external service" pattern already used for staff
WhatsApp notifications elsewhere in this codebase (see order_tool.py's
_notify_staff_of_new_order()).
"""

import logging

import requests

from app.config import settings
from services.delivery_tool import delivery_option_matches_address

logger = logging.getLogger(__name__)

_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Types that plausibly identify the city/town a customer means, from
# most to least specific. Checked broadly (any of these mentioning
# "accra"/"kumasi") rather than picking exactly one type, since which
# type carries the city name varies by address -- a landmark-only
# address might only have it in "locality", while a well-known
# neighbourhood might carry it in "administrative_area_level_2" instead.
_LOCALITY_TYPES = {
    "locality",
    "sublocality",
    "sublocality_level_1",
    "administrative_area_level_2",
}


class GeocodingError(Exception):
    """Raised when the Google Geocoding API can't be reached, isn't configured, or returns something unusable."""


def _require_config() -> None:
    if not settings.google_maps_api_key:
        raise GeocodingError("GOOGLE_MAPS_API_KEY is not configured.")


def classify_ghana_address(address: str) -> dict | None:
    """Resolves a free-text address to {"in_ghana": bool, "matches_accra":
    bool, "matches_kumasi": bool}.

    Returns None when Google genuinely couldn't geocode the address at
    all (status ZERO_RESULTS, or an empty results list) -- distinct from
    GeocodingError, which means the *lookup itself* failed (network,
    auth, quota), not that the address was checked and found nowhere.
    Callers treat both the same way today (fall back to the offline
    heuristic, see resolve_delivery_match() below), but they're kept
    distinct in case that ever needs to change -- a genuinely
    unresolvable address is a different problem from a broken API call.

    matches_accra/matches_kumasi look for "accra"/"kumasi" (case
    insensitive) anywhere across the locality-shaped address components
    (see _LOCALITY_TYPES) -- not just the top-level "locality" type,
    since Google doesn't always put a neighbourhood's parent city there.
    """
    _require_config()

    try:
        response = requests.get(
            _GEOCODE_URL,
            params={"address": address, "key": settings.google_maps_api_key},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise GeocodingError(f"Google geocoding request failed: {e}") from e
    except ValueError as e:
        # response.json() failing to parse -- not documented as a real
        # possibility for this API, but not worth trusting blindly
        # either, on the same "don't guess" principle as everything
        # else in this file.
        raise GeocodingError(f"Google geocoding returned an unparsable response: {e}") from e

    status = data.get("status")
    if status == "ZERO_RESULTS":
        return None
    if status != "OK":
        raise GeocodingError(
            f"Google geocoding returned status {status!r}: {data.get('error_message', '')}"
        )

    results = data.get("results") or []
    if not results:
        return None

    components = results[0].get("address_components", [])

    country_code = None
    locality_names = []
    for component in components:
        types = component.get("types", [])
        if "country" in types:
            country_code = component.get("short_name")
        if any(t in _LOCALITY_TYPES for t in types):
            locality_names.append((component.get("long_name") or "").lower())

    locality_text = " ".join(locality_names)
    return {
        "in_ghana": country_code == "GH",
        "matches_accra": "accra" in locality_text,
        "matches_kumasi": "kumasi" in locality_text,
    }


def resolve_delivery_match(key: str | None, address: str | None) -> bool:
    """Drop-in, geocoding-backed replacement for
    delivery_tool.delivery_option_matches_address() -- same signature,
    same meaning (True = the chosen delivery arrangement plausibly
    matches the stated address), but resolves neighbourhood-level
    addresses correctly instead of only recognising a fixed list of
    city names.

    This is the only function order_tool.py should call for this check.
    Falls back to the offline heuristic whenever geocoding isn't
    configured (no GOOGLE_MAPS_API_KEY) or fails for any reason --
    logged, not raised, so a Google outage or quota issue degrades to
    today's behaviour rather than blocking a customer from placing an
    order at all."""
    if key not in ("accra_rider", "kumasi_rider", "international"):
        return True

    if not settings.google_maps_api_key:
        return delivery_option_matches_address(key, address)

    try:
        classification = classify_ghana_address(address or "")
    except GeocodingError as e:
        logger.warning("Geocoding failed for address %r, falling back to offline check: %s", address, e)
        return delivery_option_matches_address(key, address)

    if classification is None:
        # Google couldn't geocode this at all (gibberish, too vague,
        # ...) -- the offline substring/city-name check is no less
        # informed in this specific case, so fall back to it rather
        # than assuming a match or a mismatch either way.
        return delivery_option_matches_address(key, address)

    if key == "accra_rider":
        return classification["matches_accra"]
    if key == "kumasi_rider":
        return classification["matches_kumasi"]
    return not classification["in_ghana"]
