"""
Delivery is offered, not priced.

Adom De Jeweller delivers by rider from its Accra and Kumasi branches,
or ships internationally -- the real cost and timing for any of those
depends on the specific pickup/drop-off and isn't something this system
can compute on its own (no per-zone rate table exists, and inventing
one would violate this codebase's own no-fabrication rule -- see
response_formatter.py's module docstring). So this tool doesn't quote a
delivery price or time at all. It only ever offers the customer a
choice between the real delivery options; a human (the rider
coordinator) finalises the actual delivery once the order is placed --
see order_tool.py's confirm_order() for that handoff.
"""

import re

# The only three real delivery arrangements this business offers.
# `key` is what propose_order()/the LLM deal with internally; `label`
# is what gets shown to a customer.
DELIVERY_OPTIONS = [
    {"key": "accra_rider", "label": "rider delivery within Accra"},
    {"key": "kumasi_rider", "label": "rider delivery within Kumasi"},
    {"key": "international", "label": "shipping outside Ghana"},
]

_VALID_KEYS = {option["key"] for option in DELIVERY_OPTIONS}

# Ghana's 16 regional capitals, plus the country name itself. Exists
# solely so delivery_option_matches_address() can catch "international"
# wrongly picked for a real Ghanaian address outside Accra/Kumasi --
# confirmed live, 2026-08-18: a customer who said they lived in Cape
# Coast (a real Ghanaian city, Central Region's capital) was told their
# order would ship "via shipping outside Ghana", which is simply false.
# This business only actually offers three arrangements (see
# DELIVERY_OPTIONS above), and Cape Coast is neither of the two rider
# zones -- there's no real arrangement this system can assert with
# confidence, so the honest answer is "let our team confirm", the same
# "say you don't know rather than guess" principle
# delivery_option_matches_address() already applies to accra_rider/
# kumasi_rider below, not a guess dressed up as a real delivery option.
_GHANA_PLACE_NAMES = {
    "ghana",
    "accra",
    "kumasi",
    "sekondi",
    "takoradi",
    "cape coast",
    "koforidua",
    "ho",
    "tamale",
    "bolgatanga",
    "wa",
    "sunyani",
    "techiman",
    "goaso",
    "sefwi wiawso",
    "dambai",
    "nalerigu",
    "damongo",
    "mankessim",
}

# Real Accra/Kumasi neighbourhoods and suburbs -- what customers
# actually type (Webb, 2026-08-19: "most customers would mention Accra
# but will type East Legon... Suame is a smaller town in Kumasi").
# Google Geocoding (services/geocoding_tool.py) resolves any of these
# properly and is preferred whenever GOOGLE_MAPS_API_KEY is configured;
# this curated list exists so accra_rider/kumasi_rider matching still
# works for the common cases while that key is unavailable (Webb's
# Google Cloud billing card needs sorting first, 2026-08-19), and it
# keeps working as a zero-cost baseline afterwards too. Not
# exhaustive -- an unlisted neighbourhood still falls through to a
# "let our team confirm" mismatch rather than a wrong guess, same
# conservative bias as everywhere else in this file.
_ACCRA_NEIGHBOURHOODS = {
    "east legon", "west legon", "north legon", "south legon",
    "airport residential", "cantonments", "spintex", "dzorwulu",
    "adenta", "madina", "tema", "achimota", "dansoman", "labone",
    "osu", "labadi", "teshie", "nungua", "abelemkpe", "roman ridge",
    "kaneshie", "circle", "lapaz", "haatso", "ashongman", "taifa",
    "sowutuom", "weija", "abeka", "kwashieman", "odorkor",
    # Kasoa: a real, populous Greater Accra town, previously absent from
    # every list in this file (including _GHANA_PLACE_NAMES), so it fell
    # through to the generic three-way delivery question instead of
    # getting the same neighbourhood-matching treatment Tema/Madina
    # already get. See the 2026-08-20 architecture audit, failure #7.
    "kasoa",
    # Ashaiman: same gap as Kasoa above, same fix -- a real, populous
    # Greater Accra industrial town, not a Kasoa-style edge case.
    # Webb/GPT 50-turn live test, 2026-08-24.
    "ashaiman",
}

_KUMASI_NEIGHBOURHOODS = {
    "suame", "adum", "asokwa", "bantama", "ahodwo", "nhyiaeso",
    "asafo", "santasi", "kwadaso", "atonsu", "ayigya", "bomso",
    "asawase", "tafo", "oforikrom", "kotei",
    # Kejetia: Kumasi's central market and lorry station -- arguably the
    # single most-referenced landmark in the city, and a real live-test
    # customer named it right after already being on a Kumasi delivery
    # ("no, Kejetia"), which reset to the generic three-way question
    # instead of confirming Kumasi coverage. Same gap as Kasoa/Ashaiman
    # above. Confirmed live, 2026-08-24 (Webb/GPT 50-turn test).
    "kejetia",
    # Manhyia: the Asantehene's palace, similarly landmark-level known
    # in Kumasi and similarly absent -- added alongside Kejetia rather
    # than waiting for it to surface as its own separate live miss.
    "manhyia",
}

# Full-word/phrase membership only -- see _contains_place()'s docstring
# for why a bare substring check isn't safe for short place names.
_ACCRA_PLACE_NAMES = {"accra"} | _ACCRA_NEIGHBOURHOODS
_KUMASI_PLACE_NAMES = {"kumasi"} | _KUMASI_NEIGHBOURHOODS


def _contains_place(text_lower: str, places) -> bool:
    """True if any of `places` appears in `text_lower` as a whole word
    or phrase, not merely as a substring.

    Matters most for short, common place names -- "Ho" and "Wa" (both
    real Ghanaian regional capitals in _GHANA_PLACE_NAMES) are also
    ordinary two-letter sequences that a plain `in` check would
    false-positive on inside unrelated words ("show", "workshop",
    "wardrobe", ...). See the 2026-08-20 architecture audit, failure #7.
    `\\b` word boundaries handle multi-word phrases ("east legon") the
    same way, since whitespace is itself a boundary."""
    return any(re.search(rf"\b{re.escape(place)}\b", text_lower) for place in places)


def get_delivery_information(address: str = "unknown") -> dict:
    """Lists the real delivery options rather than quoting a price/time
    -- see module docstring for why.

    `address` is optional, for a customer who names a specific place
    while asking about delivery ("what of Bolgatanga, I'm in the
    northern region") rather than asking generically ("how does
    delivery work"). Confirmed live, 2026-08-22: a named-place question
    like that got the exact same generic three-way list as a bare
    "how does delivery work", with the actual place name silently
    ignored -- this tool had no way to say anything about a specific
    location at all, only ever recite the fixed menu.

    When a real address is given, resolves it the same way
    order_tool.propose_order() already resolves a delivery address
    (geocoding_tool.infer_delivery_option(), geocoding-backed when
    configured, the offline curated-list classifier otherwise) and
    returns the match as `matched_zone` alongside the same
    `delivery_options` list, so response_formatter.py can give a
    location-aware answer instead of the generic one. `matched_zone` is
    one of "accra_rider"/"kumasi_rider"/"international" (confident
    enough to confirm outright), "ghana_other" (a real Ghanaian place,
    just not a rider zone -- team confirms, same as propose_order's own
    handling of this case), or None (genuinely unclear, same "don't
    guess" fallback everywhere else in this file). Omitted entirely
    (falls back to the plain shape below) when no real address was
    given -- the bare "how does delivery work" case is unchanged.

    The import below is local, not at module level, because
    geocoding_tool.py already imports classify_zone_offline from THIS
    module (its own offline fallback) -- importing infer_delivery_option
    back at module level here would be a circular import. Safe as a
    local import: it only needs resolving once this function actually
    runs, by which point both modules are already fully loaded."""
    address_stripped = (address or "").strip()
    if not address_stripped or address_stripped.lower() == "unknown":
        return {"delivery_options": DELIVERY_OPTIONS}

    from services.geocoding_tool import infer_delivery_option

    return {
        "delivery_options": DELIVERY_OPTIONS,
        "matched_zone": infer_delivery_option(address_stripped),
        "queried_address": address_stripped,
    }


def is_valid_delivery_option(key: str | None) -> bool:
    return key in _VALID_KEYS


def delivery_option_label(key: str | None) -> str | None:
    return next((option["label"] for option in DELIVERY_OPTIONS if option["key"] == key), None)


def delivery_option_matches_address(key: str | None, address: str | None) -> bool:
    """Whether a rider delivery option plausibly matches the stated
    delivery address. A simple substring check, not geocoding -- good
    enough to catch the case where a customer's chosen delivery
    arrangement and stated address name two different cities (e.g.
    address "Tamale", delivery option "kumasi_rider"), confirmed live,
    2026-08-14: a customer got a confirmation claiming delivery "to
    Tamale via rider delivery within Kumasi", which is not a real
    arrangement this business offers.

    Deliberately conservative in one direction only: this can still
    produce a false "mismatch" for a real Accra/Kumasi address that
    names neither the city nor one of the curated neighbourhoods in
    _ACCRA_NEIGHBOURHOODS/_KUMASI_NEIGHBOURHOODS above (e.g. an
    unlisted suburb, a landmark with no place name at all). That
    failure mode routes the order to manual staff confirmation
    instead of asserting a wrong delivery arrangement -- the same
    "say you don't know rather than guess" principle the rest of this
    file already follows for pricing delivery itself. It never produces
    a false match, which is the direction that actually matters: it
    can't quietly clear a real mismatch.

    "international" has no address-region constraint in the sense that
    it isn't tied to one specific city the way the two rider options
    are -- but it still mismatches when the address is recognisably a
    real Ghanaian place (see _GHANA_PLACE_NAMES above): "international"
    given for an address that names an actual Ghanaian city is exactly
    as wrong as "kumasi_rider" given for Tamale, and deserves the same
    "let our team confirm" softening rather than asserting a shipping
    arrangement that isn't real (confirmed live, 2026-08-18: a Cape
    Coast address was told it would ship "outside Ghana"). This can
    still miss a genuine Ghanaian address that isn't in the curated list
    (a small town, a landmark with no city name) -- same conservative
    bias as the rider checks below: it never produces a false "mismatch"
    for an address that's actually outside Ghana, only a possible false
    "match" for an unlisted Ghanaian one, which is the safer direction
    to be wrong in.

    An unrecognised key isn't this function's job to judge and also
    matches, so an already-invalid delivery_option doesn't get flagged
    twice over by two different checks."""
    if key not in ("accra_rider", "kumasi_rider", "international"):
        return True
    address_lower = (address or "").lower()
    if key == "accra_rider":
        return _contains_place(address_lower, _ACCRA_PLACE_NAMES)
    if key == "kumasi_rider":
        return _contains_place(address_lower, _KUMASI_PLACE_NAMES)
    return not _contains_place(address_lower, _GHANA_PLACE_NAMES)


def classify_zone_offline(address: str | None) -> str | None:
    """Best-effort classification of a free-text address into one of
    four buckets: "accra_rider", "kumasi_rider", "ghana_other" (a real
    Ghanaian place, just not one of the two rider zones), or None
    (genuinely unclear -- could be an unlisted Ghanaian town, could be
    international, this function doesn't guess).

    Exists so a customer who names a real neighbourhood ("East Legon",
    "Suame") never has to pick a delivery arrangement from a menu they
    already effectively answered by saying where they live (Webb,
    2026-08-19, live: told "east legon", the assistant still asked
    "Would you like rider delivery within Accra, rider delivery within
    Kumasi, or shipping outside Ghana?" -- clunky and redundant when the
    address alone already resolves it). geocoding_tool.infer_delivery_option()
    is the geocoding-backed version of this same idea, and is what
    order_tool.py actually calls; this offline version is its fallback,
    same relationship as delivery_option_matches_address() has with
    resolve_delivery_match().

    Deliberately never returns "international" -- unlike the other two
    zones, there's no positive offline signal for "this is genuinely
    outside Ghana" (an unrecognised address is just as likely to be an
    unlisted Ghanaian town as a foreign one). Same conservative bias as
    delivery_option_matches_address(): only ever asserts a zone it has
    real evidence for."""
    address_lower = (address or "").lower().strip()
    if not address_lower:
        return None
    if _contains_place(address_lower, _ACCRA_PLACE_NAMES):
        return "accra_rider"
    if _contains_place(address_lower, _KUMASI_PLACE_NAMES):
        return "kumasi_rider"
    if _contains_place(address_lower, _GHANA_PLACE_NAMES):
        return "ghana_other"
    return None


def delivery_options_phrase(options: list[dict] | None = None) -> str:
    """"rider delivery within Accra, rider delivery within Kumasi, or
    shipping outside Ghana" -- the one place this phrasing is built, so
    order_tool.py's clarifying question and response_formatter.py's
    customer-facing replies never drift out of sync with each other or
    with the real option labels above."""
    labels = [option["label"] for option in (options if options is not None else DELIVERY_OPTIONS)]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"
