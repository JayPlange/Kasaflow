"""
Unit tests for services/image_embeddings_sync.py

No real HTTP fetches or Cohere calls -- both are mocked. This tests the
dedup-by-image_url, skip-on-failure, and batching orchestration only.
"""

import json
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from services import image_embeddings_sync
from services.image_embed_tool import ImageEmbedError


def _settings_with(monkeypatch, **overrides):
    monkeypatch.setattr(
        image_embeddings_sync, "settings", replace(image_embeddings_sync.settings, **overrides)
    )


def _write_products(tmp_path, rows):
    path = tmp_path / "products.json"
    path.write_text(json.dumps(rows))
    return path


def test_build_image_embeddings_raises_when_cohere_not_configured(monkeypatch, tmp_path):
    _settings_with(monkeypatch, cohere_api_key=None, products_path=_write_products(tmp_path, []))

    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        image_embeddings_sync.build_image_embeddings()


def test_build_image_embeddings_dedupes_by_image_url(monkeypatch, tmp_path):
    products_path = _write_products(tmp_path, [
        {"product": "Ring A", "material": "18k", "image_url": "https://x/a.jpg"},
        {"product": "Ring A", "material": "14k", "image_url": "https://x/a.jpg"},  # same photo, different karat
        {"product": "Ring B", "material": "18k", "image_url": "https://x/b.jpg"},
    ])
    _settings_with(monkeypatch, cohere_api_key="fake-key", products_path=products_path)
    monkeypatch.setattr(image_embeddings_sync, "_fetch_image_bytes", lambda url: b"fake-jpeg-bytes")
    embed_mock = MagicMock(side_effect=lambda urls: [[0.1, 0.2] for _ in urls])
    monkeypatch.setattr(image_embeddings_sync, "embed_images", embed_mock)
    monkeypatch.setattr(image_embeddings_sync.time, "sleep", lambda s: None)

    results = image_embeddings_sync.build_image_embeddings()

    # Only 2 distinct photos, not 3 rows
    assert {r["image_url"] for r in results} == {"https://x/a.jpg", "https://x/b.jpg"}
    assert len(results) == 2


def test_build_image_embeddings_skips_photos_that_fail_to_fetch(monkeypatch, tmp_path):
    products_path = _write_products(tmp_path, [
        {"product": "Ring A", "material": "18k", "image_url": "https://x/a.jpg"},
        {"product": "Ring B", "material": "18k", "image_url": "https://x/b.jpg"},
    ])
    _settings_with(monkeypatch, cohere_api_key="fake-key", products_path=products_path)

    def _fetch(url):
        return None if url == "https://x/a.jpg" else b"fake-jpeg-bytes"

    monkeypatch.setattr(image_embeddings_sync, "_fetch_image_bytes", _fetch)
    monkeypatch.setattr(image_embeddings_sync, "embed_images", lambda urls: [[0.1, 0.2] for _ in urls])
    monkeypatch.setattr(image_embeddings_sync.time, "sleep", lambda s: None)

    results = image_embeddings_sync.build_image_embeddings()

    assert len(results) == 1
    assert results[0]["image_url"] == "https://x/b.jpg"


def test_build_image_embeddings_skips_a_batch_that_fails_to_embed(monkeypatch, tmp_path):
    products_path = _write_products(tmp_path, [
        {"product": "Ring A", "material": "18k", "image_url": "https://x/a.jpg"},
    ])
    _settings_with(monkeypatch, cohere_api_key="fake-key", products_path=products_path)
    monkeypatch.setattr(image_embeddings_sync, "_fetch_image_bytes", lambda url: b"fake-jpeg-bytes")
    monkeypatch.setattr(
        image_embeddings_sync, "embed_images",
        MagicMock(side_effect=ImageEmbedError("Cohere embed request failed: boom")),
    )
    monkeypatch.setattr(image_embeddings_sync.time, "sleep", lambda s: None)

    # Must not raise -- a failed batch is logged and skipped, same "one
    # bad item shouldn't sink the whole sync" principle as
    # woocommerce_sync.py's per-product handling.
    results = image_embeddings_sync.build_image_embeddings()

    assert results == []


def test_build_image_embeddings_batches_photos_by_batch_size(monkeypatch, tmp_path):
    rows = [
        {"product": f"Ring {i}", "material": "18k", "image_url": f"https://x/{i}.jpg"}
        for i in range(25)
    ]
    products_path = _write_products(tmp_path, rows)
    _settings_with(monkeypatch, cohere_api_key="fake-key", products_path=products_path)
    monkeypatch.setattr(image_embeddings_sync, "_fetch_image_bytes", lambda url: b"fake-jpeg-bytes")
    monkeypatch.setattr(image_embeddings_sync, "_BATCH_SIZE", 10)
    embed_mock = MagicMock(side_effect=lambda urls: [[0.1, 0.2] for _ in urls])
    monkeypatch.setattr(image_embeddings_sync, "embed_images", embed_mock)
    monkeypatch.setattr(image_embeddings_sync.time, "sleep", lambda s: None)

    results = image_embeddings_sync.build_image_embeddings()

    assert len(results) == 25
    # 25 photos at batch size 10 -> 3 calls (10, 10, 5)
    assert embed_mock.call_count == 3


def test_fetch_image_bytes_refuses_disallowed_host(monkeypatch):
    monkeypatch.setattr(image_embeddings_sync, "_ALLOWED_IMAGE_HOST", "adomdejeweller.com")
    monkeypatch.setattr(
        image_embeddings_sync.requests, "get", MagicMock(side_effect=AssertionError("should never be called"))
    )

    assert image_embeddings_sync._fetch_image_bytes("https://evil.example.com/x.jpg") is None


def test_mime_type_for_infers_from_extension():
    assert image_embeddings_sync._mime_type_for("https://x/a.png") == "image/png"
    assert image_embeddings_sync._mime_type_for("https://x/a.webp") == "image/webp"
    assert image_embeddings_sync._mime_type_for("https://x/a.gif") == "image/gif"
    assert image_embeddings_sync._mime_type_for("https://x/a.jpg") == "image/jpeg"
    assert image_embeddings_sync._mime_type_for("https://x/a") == "image/jpeg"
