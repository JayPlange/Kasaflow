from services.delivery_tool import (
    DELIVERY_OPTIONS,
    delivery_option_label,
    delivery_option_matches_address,
    delivery_options_phrase,
    get_delivery_information,
    is_valid_delivery_option,
)


def test_get_delivery_information_lists_the_real_options():
    # Arrange: nothing to set up, this tool takes no arguments

    # Act
    result = get_delivery_information()

    # Assert: no invented price/time -- just the real choices a customer
    # actually has (see delivery_tool.py's module docstring for why
    # there's no fixed rate to quote)
    assert result == {"delivery_options": DELIVERY_OPTIONS}
    keys = {option["key"] for option in result["delivery_options"]}
    assert keys == {"accra_rider", "kumasi_rider", "international"}


def test_is_valid_delivery_option_accepts_real_keys():
    for option in DELIVERY_OPTIONS:
        assert is_valid_delivery_option(option["key"])


def test_is_valid_delivery_option_rejects_anything_else():
    assert not is_valid_delivery_option("unknown")
    assert not is_valid_delivery_option(None)
    assert not is_valid_delivery_option("accra")  # close, but not the real key


def test_delivery_option_label_returns_the_display_text():
    assert delivery_option_label("accra_rider") == "rider delivery within Accra"


def test_delivery_option_label_returns_none_for_an_invalid_key():
    assert delivery_option_label("nonsense") is None


def test_delivery_options_phrase_uses_an_oxford_comma_and_or():
    # The one place this exact wording is built -- order_tool.py's
    # clarifying question and response_formatter.py's replies both use
    # this, so a missing "or" here would show up in both places at once.
    phrase = delivery_options_phrase()
    assert phrase == (
        "rider delivery within Accra, rider delivery within Kumasi, "
        "or shipping outside Ghana"
    )


def test_delivery_options_phrase_handles_a_single_option():
    phrase = delivery_options_phrase([{"key": "accra_rider", "label": "rider delivery within Accra"}])
    assert phrase == "rider delivery within Accra"


# ---------------------------------------------------------------------
# delivery_option_matches_address -- catches the case where the chosen
# rider option and the stated address name two different cities (the
# live Tamale/kumasi_rider case, 2026-08-14). Conservative in one
# direction only: never false-positive-matches a real mismatch, but can
# false-negative (report "mismatch") on a real Accra/Kumasi address that
# doesn't literally contain the city name.
# ---------------------------------------------------------------------

def test_matches_when_accra_rider_and_address_contains_accra():
    assert delivery_option_matches_address("accra_rider", "12 Cantonments Road, Accra")


def test_mismatches_when_accra_rider_but_address_is_elsewhere():
    assert not delivery_option_matches_address("accra_rider", "Tamale")


def test_matches_when_kumasi_rider_and_address_contains_kumasi():
    assert delivery_option_matches_address("kumasi_rider", "Adum, Kumasi")


def test_mismatches_when_kumasi_rider_but_address_is_elsewhere():
    assert not delivery_option_matches_address("kumasi_rider", "Tamale")


def test_matches_is_case_insensitive():
    assert delivery_option_matches_address("accra_rider", "ACCRA")
    assert delivery_option_matches_address("kumasi_rider", "kumasi central market")


def test_matches_accra_rider_for_a_real_accra_neighbourhood_without_the_word_accra():
    # Webb, 2026-08-19: "most customers would mention Accra but will
    # type East Legon, which is a smaller town in Accra." This is the
    # curated no-API-key fallback for that exact case.
    assert delivery_option_matches_address("accra_rider", "East Legon")
    assert delivery_option_matches_address("accra_rider", "east legon")


def test_matches_kumasi_rider_for_a_real_kumasi_neighbourhood_without_the_word_kumasi():
    # Webb, 2026-08-19: "Suame is a smaller town in Kumasi."
    assert delivery_option_matches_address("kumasi_rider", "Suame")


def test_mismatches_kumasi_rider_for_an_accra_neighbourhood():
    # A curated Accra neighbourhood must not also satisfy kumasi_rider
    assert not delivery_option_matches_address("kumasi_rider", "East Legon")


def test_international_matches_an_address_with_no_ghanaian_place_name():
    # A real international address -- no reason to flag it
    assert delivery_option_matches_address("international", "221B Baker Street, London")
    assert delivery_option_matches_address("international", "")
    assert delivery_option_matches_address("international", None)


def test_international_mismatches_a_real_ghanaian_address():
    # "international" picked for an address that's actually a real
    # Ghanaian city is just as wrong as kumasi_rider picked for Tamale --
    # confirmed live, 2026-08-18: a Cape Coast address was told it would
    # ship "outside Ghana", which is false. See _GHANA_PLACE_NAMES.
    assert not delivery_option_matches_address("international", "Cape Coast")
    assert not delivery_option_matches_address("international", "Tamale")
    assert not delivery_option_matches_address("international", "Adum, Kumasi")


def test_international_mismatch_is_case_insensitive():
    assert not delivery_option_matches_address("international", "CAPE COAST")


def test_international_mismatches_mankessim():
    # Webb, 2026-08-19: "Mankessim is a smaller town in Cape Coast."
    # A real Ghanaian town, just not one of the two rider zones.
    assert not delivery_option_matches_address("international", "Mankessim")


def test_invalid_key_always_matches_rather_than_double_flagging():
    # An already-invalid delivery_option is is_valid_delivery_option()'s
    # job to catch, not this function's -- it shouldn't also report a
    # mismatch for a key that's wrong in the first place.
    assert delivery_option_matches_address("nonsense", "Accra")
    assert delivery_option_matches_address(None, "Accra")


def test_mismatch_with_no_address_at_all():
    assert not delivery_option_matches_address("accra_rider", "")
    assert not delivery_option_matches_address("accra_rider", None)
