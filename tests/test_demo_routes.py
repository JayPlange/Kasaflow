"""
Unit tests for app/demo_routes.py's image proxy and card-diversification
wiring.

Calling the route functions directly (image_proxy, _build_recommendation_cards)
rather than through a full TestClient + FastAPI app, matching
test_whatsapp_routes.py's approach -- a fast, focused unit test, no real
network call.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from app import demo_routes
from app.demo_routes import _build_recommendation_cards, _proxied_image_url, image_proxy


class _FakeUploadFile:
    """Minimal stand-in for FastAPI's UploadFile -- just enough for
    demo_message()'s `await image.read()` and `.content_type` access.
    No test framework (starlette's TestClient, httpx) needed for this,
    same "call the function directly" approach the rest of this file
    already uses for image_proxy()."""

    def __init__(self, data: bytes, content_type: str = "image/jpeg", name: str = "photo.jpg"):
        self._data = data
        self.content_type = content_type
        self.filename = name

    async def read(self) -> bytes:
        return self._data


def _run_demo_message(**overrides):
    """demo_message()'s parameters default to FastAPI's Form(...)/File(...)
    marker objects, not real None -- those only resolve to the actual
    None default when the framework itself calls the route. Calling the
    coroutine directly (no asyncio test plugin configured in this repo,
    so asyncio.run() rather than an async test function) means every
    parameter must be passed explicitly, or a marker object leaks
    through and the None-checks inside will not behave the way a real
    request's absent field would."""
    kwargs = dict(text=None, audio=None, image=None, session_id=None, language="english", voice_reply=False)
    kwargs.update(overrides)
    return asyncio.run(demo_routes.demo_message(**kwargs))


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


# ---------------------------------------------------------------------
# demo_message -- image branch. Mirrors whatsapp_routes.py's image
# handling: describe_product_image() turns the photo into customer_text,
# which then goes through the exact same route_customer()/
# format_for_customer() pipeline as typed text.
# ---------------------------------------------------------------------

def test_demo_message_routes_a_photo_through_vision_and_the_normal_pipeline(monkeypatch):
    monkeypatch.setattr(demo_routes, "describe_product_image", lambda image_bytes, mime_type=None: "gold twist ring")
    monkeypatch.setattr(
        demo_routes, "route_customer",
        lambda text, session_id: {"product": "Twist Ring", "material": "18k", "price": 1200.0},
    )

    result = _run_demo_message(image=_FakeUploadFile(b"fake-jpeg-bytes"))

    assert result["transcript"] == "gold twist ring"
    assert "1,200.00" in result["reply_text"]
    assert result["session_id"]


def test_demo_message_gives_a_friendly_error_when_photo_is_not_jewellery(monkeypatch):
    # describe_product_image() returns "" for a photo that doesn't show
    # jewellery at all (see vision_tool.py's _NOT_JEWELLERY sentinel) --
    # must not route an empty string into catalogue matching.
    monkeypatch.setattr(demo_routes, "describe_product_image", lambda image_bytes, mime_type=None: "")
    route_customer = MagicMock()
    monkeypatch.setattr(demo_routes, "route_customer", route_customer)

    result = _run_demo_message(image=_FakeUploadFile(b"fake-jpeg-bytes"))

    assert "error" in result
    route_customer.assert_not_called()


def test_demo_message_surfaces_a_clean_error_when_vision_service_fails(monkeypatch):
    def _raise(*a, **k):
        raise demo_routes.VisionServiceError("vision API unreachable")

    monkeypatch.setattr(demo_routes, "describe_product_image", _raise)

    result = _run_demo_message(image=_FakeUploadFile(b"fake-jpeg-bytes"))

    assert "error" in result
    assert "vision API unreachable" in result["error"]
