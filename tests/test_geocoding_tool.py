"""
Unit tests for services/geocoding_tool.py

Same golden rule as test_image_embed_tool.py: never call the real Google
Geocoding API here. We mock requests.get and check our own request-shape,
parsing, and fallback logic, not Google's actual behaviour -- that's what
a live regression check is for, and this module hasn't had one yet (see
its module docstring).
"""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from services import geocoding_tool
from services.geocoding_tool import GeocodingError, classify_ghana_address, resolve_delivery_match


def _mock_geocode_response(status="OK", results=None):
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"status": status, "results": results or []}
    return fake_response


def _address_component(long_name, short_name, types):
    return {"long_name": long_name, "short_name": short_name, "types": types}


_EAST_LEGON_RESULT = {
    "address_components": [
        _address_component("East Legon", "East Legon", ["sublocality", "sublocality_level_1", "political"]),
        _address_component("Accra", "Accra", ["locality", "political"]),
        _address_component("Greater Accra Region", "Greater Accra Region", ["administrative_area_level_1", "political"]),
        _address_component("Ghana", "GH", ["country", "political"]),
    ]
}

_SUAME_RESULT = {
    "address_components": [
        _address_component("Suame", "Suame", ["sublocality", "political"]),
        _address_component("Kumasi", "Kumasi", ["locality", "political"]),
        _address_component("Ashanti Region", "Ashanti Region", ["administrative_area_level_1", "political"]),
        _address_component("Ghana", "GH", ["country", "political"]),
    ]
}

_CAPE_COAST_RESULT = {
    "address_components": [
        _address_component("Cape Coast", "Cape Coast", ["locality", "political"]),
        _address_component("Central Region", "Central Region", ["administrative_area_level_1", "political"]),
        _address_component("Ghana", "GH", ["country", "political"]),
    ]
}

_LONDON_RESULT = {
    "address_components": [
        _address_component("Baker Street", "Baker Street", ["route"]),
        _address_component("London", "London", ["postal_town", "political"]),
        _address_component("United Kingdom", "GB", ["country", "political"]),
    ]
}


def _settings_with(monkeypatch, **overrides):
    monkeypatch.setattr(geocoding_tool, "settings", replace(geocoding_tool.settings, **overrides))


# ---------------------------------------------------------------------
# classify_ghana_address
# ---------------------------------------------------------------------

def test_classify_raises_when_not_configured(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key=None)

    with pytest.raises(GeocodingError, match="GOOGLE_MAPS_API_KEY"):
        classify_ghana_address("East Legon")


def test_classify_resolves_a_neighbourhood_to_its_parent_city(monkeypatch):
    # The exact case this module exists for: "East Legon" is a real
    # Accra neighbourhood, not a name any hardcoded city list would
    # recognise -- confirmed live, 2026-08-19 (Webb).
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    fake_get = MagicMock(return_value=_mock_geocode_response(results=[_EAST_LEGON_RESULT]))
    monkeypatch.setattr(geocoding_tool.requests, "get", fake_get)

    result = classify_ghana_address("East Legon, Accra")

    assert result == {"in_ghana": True, "matches_accra": True, "matches_kumasi": False}


def test_classify_resolves_suame_to_kumasi(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(geocoding_tool.requests, "get", MagicMock(return_value=_mock_geocode_response(results=[_SUAME_RESULT])))

    result = classify_ghana_address("Suame")

    assert result == {"in_ghana": True, "matches_accra": False, "matches_kumasi": True}


def test_classify_flags_cape_coast_as_ghana_but_neither_zone(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(geocoding_tool.requests, "get", MagicMock(return_value=_mock_geocode_response(results=[_CAPE_COAST_RESULT])))

    result = classify_ghana_address("Cape Coast")

    assert result == {"in_ghana": True, "matches_accra": False, "matches_kumasi": False}


def test_classify_flags_a_genuine_international_address(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(geocoding_tool.requests, "get", MagicMock(return_value=_mock_geocode_response(results=[_LONDON_RESULT])))

    result = classify_ghana_address("221B Baker Street, London")

    assert result == {"in_ghana": False, "matches_accra": False, "matches_kumasi": False}


def test_classify_returns_none_for_zero_results(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(geocoding_tool.requests, "get", MagicMock(return_value=_mock_geocode_response(status="ZERO_RESULTS")))

    assert classify_ghana_address("asdkjaslkdj") is None


def test_classify_raises_on_a_non_ok_non_zero_results_status(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"status": "REQUEST_DENIED", "error_message": "bad key"}
    monkeypatch.setattr(geocoding_tool.requests, "get", MagicMock(return_value=fake_response))

    with pytest.raises(GeocodingError, match="REQUEST_DENIED"):
        classify_ghana_address("East Legon")


def test_classify_wraps_a_network_failure(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(
        geocoding_tool.requests, "get",
        MagicMock(side_effect=geocoding_tool.requests.exceptions.ConnectionError("no route")),
    )

    with pytest.raises(GeocodingError, match="Google geocoding request failed"):
        classify_ghana_address("East Legon")


def test_classify_sends_the_documented_request_shape(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    fake_get = MagicMock(return_value=_mock_geocode_response(results=[_EAST_LEGON_RESULT]))
    monkeypatch.setattr(geocoding_tool.requests, "get", fake_get)

    classify_ghana_address("East Legon, Accra")

    args, kwargs = fake_get.call_args
    assert args[0] == geocoding_tool._GEOCODE_URL
    assert kwargs["params"]["address"] == "East Legon, Accra"
    assert kwargs["params"]["key"] == "fake-key"


# ---------------------------------------------------------------------
# resolve_delivery_match -- the wrapper order_tool.py actually calls
# ---------------------------------------------------------------------

def test_resolve_delivery_match_falls_back_when_not_configured(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key=None)
    fake_get = MagicMock()
    monkeypatch.setattr(geocoding_tool.requests, "get", fake_get)

    # Falls back to the offline heuristic entirely -- no network call at all
    assert resolve_delivery_match("accra_rider", "12 Cantonments Road, Accra") is True
    assert resolve_delivery_match("kumasi_rider", "Tamale") is False
    fake_get.assert_not_called()


def test_resolve_delivery_match_uses_geocoding_for_a_real_neighbourhood(monkeypatch):
    # The exact case the offline heuristic gets wrong: "East Legon"
    # contains neither "accra" nor any of the curated Ghana place names,
    # so the offline check alone would report a mismatch even though
    # it's genuinely inside the Accra rider zone.
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(geocoding_tool.requests, "get", MagicMock(return_value=_mock_geocode_response(results=[_EAST_LEGON_RESULT])))

    assert resolve_delivery_match("accra_rider", "East Legon") is True


def test_resolve_delivery_match_still_flags_cape_coast_as_international_mismatch(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(geocoding_tool.requests, "get", MagicMock(return_value=_mock_geocode_response(results=[_CAPE_COAST_RESULT])))

    assert resolve_delivery_match("international", "Cape Coast") is False


def test_resolve_delivery_match_does_not_flag_a_genuine_international_address(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(geocoding_tool.requests, "get", MagicMock(return_value=_mock_geocode_response(results=[_LONDON_RESULT])))

    assert resolve_delivery_match("international", "221B Baker Street, London") is True


def test_resolve_delivery_match_falls_back_on_geocoding_error(monkeypatch):
    # Google is configured but the call fails (quota, network, ...) --
    # must not block the order, and must not silently claim a match
    # either. Falls back to the same offline heuristic used when
    # unconfigured.
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(
        geocoding_tool.requests, "get",
        MagicMock(side_effect=geocoding_tool.requests.exceptions.Timeout("slow")),
    )

    assert resolve_delivery_match("kumasi_rider", "Adum, Kumasi") is True  # offline heuristic still catches this
    assert resolve_delivery_match("kumasi_rider", "Tamale") is False


def test_resolve_delivery_match_falls_back_when_address_cannot_be_geocoded_at_all(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    monkeypatch.setattr(geocoding_tool.requests, "get", MagicMock(return_value=_mock_geocode_response(status="ZERO_RESULTS")))

    # Offline heuristic: "accra" not in "asdkjaslkdj" -> mismatch
    assert resolve_delivery_match("accra_rider", "asdkjaslkdj") is False


def test_resolve_delivery_match_returns_true_for_an_invalid_key_without_geocoding(monkeypatch):
    _settings_with(monkeypatch, google_maps_api_key="fake-key")
    fake_get = MagicMock()
    monkeypatch.setattr(geocoding_tool.requests, "get", fake_get)

    assert resolve_delivery_match("nonsense", "Accra") is True
    assert resolve_delivery_match(None, "Accra") is True
    fake_get.assert_not_called()
