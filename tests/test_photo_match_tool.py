"""
Unit tests for services/photo_match_tool.py's identify_product_from_photo().

Every dependency (the image index, image fetching, product-name lookup,
and the vision comparison call) is mocked -- this tests the
orchestration logic (narrow via image embeddings, resolve product
names, fetch photos, ask vision, map back to a product name), not real
search ranking, a real Cohere call, or a real network fetch.
"""

import json
from unittest.mock import MagicMock

from services import photo_match_tool
from services.image_embed_tool import ImageEmbedError
from services.photo_match_tool import identify_product_from_photo


def _image_matches():
    return [
        {"image_url": "https://x/a.jpg", "score": 0.91},
        {"image_url": "https://x/b.jpg", "score": 0.77},
    ]


def _fake_products_file(tmp_path):
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps([
        {"product": "Gye Nyame White Necklace", "material": "18k", "image_url": "https://x/a.jpg"},
        {"product": "Custom Adinkra Chains Gold Necklace", "material": "18k", "image_url": "https://x/b.jpg"},
    ]))
    return fake_file


def _patch_products_path(monkeypatch, tmp_path):
    from dataclasses import replace
    fake_file = _fake_products_file(tmp_path)
    monkeypatch.setattr(photo_match_tool, "settings", replace(photo_match_tool.settings, products_path=fake_file))


def test_identify_product_from_photo_returns_matched_name(monkeypatch, tmp_path):
    _patch_products_path(monkeypatch, tmp_path)
    fake_index = MagicMock()
    fake_index.search.return_value = _image_matches()
    monkeypatch.setattr(photo_match_tool, "get_image_index", lambda: fake_index)
    monkeypatch.setattr(photo_match_tool, "_fetch_candidate_image", lambda url: b"fake-jpeg-bytes")
    # Two candidates resolved -- index 1 is "Custom Adinkra Chains Gold Necklace"
    monkeypatch.setattr(photo_match_tool, "match_photo_to_candidates", lambda *a, **k: 1)

    result = identify_product_from_photo(b"customer-photo-bytes", "image/jpeg")

    assert result == "Custom Adinkra Chains Gold Necklace"
    fake_index.search.assert_called_once_with(b"customer-photo-bytes", "image/jpeg", top_k=photo_match_tool._MAX_CANDIDATES)


def test_identify_product_from_photo_returns_none_when_vision_finds_no_match(monkeypatch, tmp_path):
    _patch_products_path(monkeypatch, tmp_path)
    fake_index = MagicMock()
    fake_index.search.return_value = _image_matches()
    monkeypatch.setattr(photo_match_tool, "get_image_index", lambda: fake_index)
    monkeypatch.setattr(photo_match_tool, "_fetch_candidate_image", lambda url: b"fake-jpeg-bytes")
    monkeypatch.setattr(photo_match_tool, "match_photo_to_candidates", lambda *a, **k: None)

    result = identify_product_from_photo(b"customer-photo-bytes", "image/jpeg")

    assert result is None


def test_identify_product_from_photo_returns_none_when_image_search_finds_nothing(monkeypatch, tmp_path):
    _patch_products_path(monkeypatch, tmp_path)
    fake_index = MagicMock()
    fake_index.search.return_value = []
    monkeypatch.setattr(photo_match_tool, "get_image_index", lambda: fake_index)
    match_mock = MagicMock()
    monkeypatch.setattr(photo_match_tool, "match_photo_to_candidates", match_mock)

    result = identify_product_from_photo(b"customer-photo-bytes", "image/jpeg")

    assert result is None
    match_mock.assert_not_called()


def test_identify_product_from_photo_returns_none_when_image_search_is_unavailable(monkeypatch, tmp_path):
    # COHERE_API_KEY not configured, or the API call itself failed --
    # either way this must not crash the whole photo request, just fall
    # back (photo_match_tool's caller treats None the same as "no
    # confident match").
    _patch_products_path(monkeypatch, tmp_path)
    fake_index = MagicMock()
    fake_index.search.side_effect = ImageEmbedError("COHERE_API_KEY is not configured.")
    monkeypatch.setattr(photo_match_tool, "get_image_index", lambda: fake_index)

    assert identify_product_from_photo(b"customer-photo-bytes", "image/jpeg") is None


def test_identify_product_from_photo_skips_a_match_whose_product_name_cannot_be_resolved(monkeypatch, tmp_path):
    # A matched image_url that isn't in products.json (stale embeddings
    # file after a catalogue change, say) must be skipped, not crash --
    # the second, resolvable candidate should still be offered to vision.
    from dataclasses import replace
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps([
        {"product": "Custom Adinkra Chains Gold Necklace", "material": "18k", "image_url": "https://x/b.jpg"},
    ]))
    monkeypatch.setattr(photo_match_tool, "settings", replace(photo_match_tool.settings, products_path=fake_file))

    fake_index = MagicMock()
    fake_index.search.return_value = _image_matches()  # includes unresolvable https://x/a.jpg
    monkeypatch.setattr(photo_match_tool, "get_image_index", lambda: fake_index)
    monkeypatch.setattr(photo_match_tool, "_fetch_candidate_image", lambda url: b"fake-jpeg-bytes")
    match_mock = MagicMock(return_value=0)
    monkeypatch.setattr(photo_match_tool, "match_photo_to_candidates", match_mock)

    result = identify_product_from_photo(b"customer-photo-bytes", "image/jpeg")

    candidates = match_mock.call_args[0][2]
    assert len(candidates) == 1
    assert candidates[0][0] == "Custom Adinkra Chains Gold Necklace"
    assert result == "Custom Adinkra Chains Gold Necklace"


def test_identify_product_from_photo_skips_candidates_whose_photo_fails_to_fetch(monkeypatch, tmp_path):
    _patch_products_path(monkeypatch, tmp_path)
    fake_index = MagicMock()
    fake_index.search.return_value = _image_matches()
    monkeypatch.setattr(photo_match_tool, "get_image_index", lambda: fake_index)

    def _fetch(url):
        return None if url == "https://x/a.jpg" else b"fake-jpeg-bytes"

    monkeypatch.setattr(photo_match_tool, "_fetch_candidate_image", _fetch)
    match_mock = MagicMock(return_value=0)
    monkeypatch.setattr(photo_match_tool, "match_photo_to_candidates", match_mock)

    result = identify_product_from_photo(b"customer-photo-bytes", "image/jpeg")

    candidates = match_mock.call_args[0][2]
    assert len(candidates) == 1
    assert candidates[0][0] == "Custom Adinkra Chains Gold Necklace"
    assert result == "Custom Adinkra Chains Gold Necklace"


def test_identify_product_from_photo_returns_none_when_no_candidate_photos_fetchable(monkeypatch, tmp_path):
    _patch_products_path(monkeypatch, tmp_path)
    fake_index = MagicMock()
    fake_index.search.return_value = _image_matches()
    monkeypatch.setattr(photo_match_tool, "get_image_index", lambda: fake_index)
    monkeypatch.setattr(photo_match_tool, "_fetch_candidate_image", lambda url: None)
    match_mock = MagicMock()
    monkeypatch.setattr(photo_match_tool, "match_photo_to_candidates", match_mock)

    result = identify_product_from_photo(b"customer-photo-bytes", "image/jpeg")

    assert result is None
    match_mock.assert_not_called()


def test_fetch_candidate_image_refuses_disallowed_host(monkeypatch):
    monkeypatch.setattr(photo_match_tool, "_ALLOWED_IMAGE_HOST", "adomdejeweller.com")
    monkeypatch.setattr(
        photo_match_tool.requests, "get", MagicMock(side_effect=AssertionError("should never be called"))
    )

    result = photo_match_tool._fetch_candidate_image("https://evil.example.com/x.jpg")

    assert result is None


def test_fetch_candidate_image_returns_none_on_request_failure(monkeypatch):
    import requests as requests_module

    monkeypatch.setattr(photo_match_tool, "_ALLOWED_IMAGE_HOST", "adomdejeweller.com")

    def _raise(*a, **k):
        raise requests_module.exceptions.ConnectionError("boom")

    monkeypatch.setattr(photo_match_tool.requests, "get", _raise)

    result = photo_match_tool._fetch_candidate_image("https://adomdejeweller.com/x.jpg")

    assert result is None


def test_product_name_for_image_url_returns_none_when_file_missing(monkeypatch, tmp_path):
    from dataclasses import replace
    missing_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(photo_match_tool, "settings", replace(photo_match_tool.settings, products_path=missing_file))

    assert photo_match_tool._product_name_for_image_url("https://x/a.jpg") is None
