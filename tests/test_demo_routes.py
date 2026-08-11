"""
Unit tests for app/demo_routes.py's image proxy and card-diversification
wiring.

Calling the route functions directly (image_proxy, _build_recommendation_cards)
rather than through a full TestClient + FastAPI app, matching
test_whatsapp_routes.py's approach -- a fast, focused unit test, no real
network call.
"""

from unittest.mock import MagicMock

import pytest

from app import demo_routes
from app.demo_routes import _build_recommendation_cards, _proxied_image_url, image_proxy


# ---------------------------------------------------------------------
# _proxied_image_url
# ---------------------------------------------------------------------

def test_proxied_image_url_returns_none_for_none():
    assert _proxied_image_url(None) is None


def test_proxied_image_url_rewrites_through_the_proxy_route():
    result = _proxied_image_url("https://adomdejeweller.com/wp-content/uploads/ring.jpg")
    assert result.startswith("/demo/image-proxy?url=")
    assert "adomdejeweller.com" in result


# ---------------------------------------------------------------------
# image_proxy
# ---------------------------------------------------------------------

def test_image_proxy_fetches_and_streams_an_allowed_host(monkeypatch):
    monkeypatch.setattr(demo_routes, "_ALLOWED_IMAGE_HOST", "adomdejeweller.com")

    fake_response = MagicMock()
    fake_response.content = b"fake-jpeg-bytes"
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.raise_for_status = MagicMock()
    captured = {}

    def fake_get(url, timeout, headers):
        captured["url"] = url
        captured["headers"] = headers
        return fake_response

    monkeypatch.setattr(demo_routes.requests, "get", fake_get)

    response = image_proxy("https://adomdejeweller.com/wp-content/uploads/ring.jpg")

    assert response.status_code == 200
    assert response.body == b"fake-jpeg-bytes"
    # The upstream request must carry the store's own origin as Referer --
    # that's the entire point of proxying (see module docstring).
    assert captured["headers"]["Referer"] == "https://adomdejeweller.com/"


def test_image_proxy_refuses_a_disallowed_host(monkeypatch):
    monkeypatch.setattr(demo_routes, "_ALLOWED_IMAGE_HOST", "adomdejeweller.com")
    monkeypatch.setattr(demo_routes.requests, "get", MagicMock(side_effect=AssertionError("should never be called")))

    response = image_proxy("https://evil.example.com/whatever.jpg")

    assert response.status_code == 400


def test_image_proxy_returns_502_when_upstream_fetch_fails(monkeypatch):
    monkeypatch.setattr(demo_routes, "_ALLOWED_IMAGE_HOST", "adomdejeweller.com")

    import requests as requests_module

    def fake_get(url, timeout, headers):
        raise requests_module.exceptions.ConnectionError("boom")

    monkeypatch.setattr(demo_routes.requests, "get", fake_get)

    response = image_proxy("https://adomdejeweller.com/wp-content/uploads/ring.jpg")

    assert response.status_code == 502


def test_image_proxy_rejects_a_non_http_scheme(monkeypatch):
    monkeypatch.setattr(demo_routes, "_ALLOWED_IMAGE_HOST", "adomdejeweller.com")
    monkeypatch.setattr(demo_routes.requests, "get", MagicMock(side_effect=AssertionError("should never be called")))

    response = image_proxy("file:///etc/passwd")

    assert response.status_code == 400


# ---------------------------------------------------------------------
# _build_recommendation_cards
# ---------------------------------------------------------------------

def test_build_recommendation_cards_routes_images_through_the_proxy():
    result = {
        "recommendations": [
            {"product": "Ring A", "material": "18k", "price": 100.0, "category": "Rings", "image_url": "https://adomdejeweller.com/a.jpg"},
        ]
    }
    cards = _build_recommendation_cards(result)
    assert len(cards) == 1
    assert cards[0]["image_url"].startswith("/demo/image-proxy?url=")


def test_build_recommendation_cards_drops_products_with_no_photo_at_all():
    result = {
        "recommendations": [
            {"product": "Ring A", "material": "18k", "price": 100.0, "category": "Rings", "image_url": None},
        ]
    }
    cards = _build_recommendation_cards(result)
    assert cards == []
