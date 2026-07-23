"""
Unit tests for services/product_tool.py

These are the "wide base of the pyramid" tests: no server, no network,
no AI. Just calling one Python function directly and checking what
comes back. Every test follows Arrange -> Act -> Assert.
"""

import json
from dataclasses import replace

from services import product_tool


def _settings_with_products_path(path):
    # Settings is a frozen dataclass (deliberately immutable, see
    # config.py) so we can't do `settings.products_path = path`.
    # dataclasses.replace() builds a new instance with one field swapped.
    return replace(product_tool.settings, products_path=path)


def test_get_product_price_returns_matching_product(monkeypatch, tmp_path):
    # Arrange: write a small fake products file and point the tool at it,
    # instead of relying on the real data/products.json. This keeps the
    # test isolated -- it can't be broken by someone editing real data.
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps([
        {"product": "ring", "material": "gold", "price": 1200},
        {"product": "ring", "material": "silver", "price": 350},
    ]))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act
    result = product_tool.get_product_price("ring", "gold")

    # Assert
    assert result == {"product": "ring", "material": "gold", "price": 1200}


def test_get_product_price_returns_none_when_no_match(monkeypatch, tmp_path):
    # Arrange
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps([
        {"product": "ring", "material": "gold", "price": 1200},
    ]))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act
    result = product_tool.get_product_price("bracelet", "platinum")

    # Assert
    assert result is None


def test_get_product_price_returns_none_when_file_missing(monkeypatch, tmp_path):
    # Arrange: point at a file that doesn't exist
    missing_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(missing_file))

    # Act
    result = product_tool.get_product_price("ring", "gold")

    # Assert: should fail gracefully, not raise
    assert result is None


def test_get_product_price_returns_none_when_file_is_corrupt(monkeypatch, tmp_path):
    # Arrange: a products file with broken JSON
    bad_file = tmp_path / "products.json"
    bad_file.write_text("{ this is not valid json ]")
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(bad_file))

    # Act
    result = product_tool.get_product_price("ring", "gold")

    # Assert: should fail gracefully, not raise
    assert result is None
