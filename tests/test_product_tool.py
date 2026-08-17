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


# ---------------------------------------------------------------------
# Karat-aware matching for sized/varianted products -- a ring with
# several karat+size combinations stores material as
# "{karat} / {size} (mm)", never a bare "18k", so the literal exact
# match above can never succeed for these on its own. Confirmed live,
# 2026-08-16: asked for the Set Multi Stone Golf Ring in 18k, got quoted
# the 12k price instead, because it fell through to semantic search,
# which silently drops numeric karat digits during keyword filtering
# (see product_search.py's _keyword_overlap docstring).
# ---------------------------------------------------------------------

def _variant_catalogue():
    return [
        {"product": "Set Multi Stone Golf Ring, 7g", "material": "12 / Women US 8 (18.2 mm)", "price": 8824.2},
        {"product": "Set Multi Stone Golf Ring, 7g", "material": "12 / Women US 9 (19.0 mm)", "price": 8824.2},
        {"product": "Set Multi Stone Golf Ring, 7g", "material": "18 / Women US 8 (18.2 mm)", "price": 12033.0},
        {"product": "Set Multi Stone Golf Ring, 7g", "material": "18 / Women US 9 (19.0 mm)", "price": 12033.0},
    ]


def test_get_product_price_matches_the_right_karat_for_a_sized_variant(monkeypatch, tmp_path):
    # Arrange
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_variant_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act
    result = product_tool.get_product_price("Set Multi Stone Golf Ring, 7g", "18k")

    # Assert: the real 18k price, not the 12k price a plain exact/semantic
    # match would have silently returned instead
    assert result["price"] == 12033.0
    assert result["material"].startswith("18")


def test_get_product_price_karat_match_is_size_indifferent_since_price_is_uniform(monkeypatch, tmp_path):
    # Arrange: every size at a given karat shares one price in this
    # catalogue -- size only matters for confirm_order()'s variation_id
    # downstream, not for pricing, so returning any matching size is
    # correct, not just convenient
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_variant_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act
    result = product_tool.get_product_price("Set Multi Stone Golf Ring, 7g", "12k")

    # Assert
    assert result["price"] == 8824.2
    assert result["material"].startswith("12")


def test_get_product_price_falls_through_to_semantic_search_when_no_karat_match(monkeypatch, tmp_path):
    # Arrange: a karat is stated, but no product in the file has that
    # exact product name at all -- must not error, must still reach the
    # existing semantic-search fallback unchanged
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_variant_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    fake_index = type("FakeIndex", (), {"search": lambda self, *a, **k: []})()
    monkeypatch.setattr(product_tool, "get_product_index", lambda: fake_index)

    # Act
    result = product_tool.get_product_price("Completely Different Ring", "18k")

    # Assert: no crash, correctly falls through to "no match"
    assert result is None
