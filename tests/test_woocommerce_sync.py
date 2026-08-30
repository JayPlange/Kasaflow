"""
Unit tests for services/woocommerce_sync.py's build_catalogue().

Scoped to the "id"/"variation_id" capture added alongside order_tool.py
-- this file had no test coverage before that change. Mocks the private
_fetch_all_products/_fetch_variations helpers rather than requests
directly, since build_catalogue()'s own job (the thing worth testing) is
shaping WooCommerce's response into a catalogue entry, not the HTTP call
itself.
"""

from dataclasses import replace

import pytest

from services import woocommerce_sync


@pytest.fixture(autouse=True)
def woocommerce_config(monkeypatch):
    """build_catalogue() calls _require_woocommerce_config() before
    fetching anything, same gate order_tool.py has for its own settings
    (see tests/test_order_tool.py's _woocommerce_settings()). CI has no
    real WooCommerce credentials in its env, so without this every test
    here fails on that check before it ever reaches the mocked fetch --
    caught by CI, not by a local run against a .env that already has
    real sync credentials set."""
    monkeypatch.setattr(
        woocommerce_sync,
        "settings",
        replace(
            woocommerce_sync.settings,
            woocommerce_url="https://adomdejeweller.com",
            woocommerce_consumer_key="ck_test",
            woocommerce_consumer_secret="cs_test",
        ),
    )


def test_build_catalogue_captures_simple_product_id(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        woocommerce_sync,
        "_fetch_all_products",
        lambda: [
            {
                "id": 101,
                "name": "Signet Ring",
                "type": "simple",
                "categories": [{"name": "Rings"}],
                "stock_status": "instock",
                "permalink": "https://adomdejeweller.com/product/signet-ring",
                "images": [],
                "price": "850",
            }
        ],
    )

    # Act
    catalogue = woocommerce_sync.build_catalogue()

    # Assert: a simple product needs its own id to be orderable later --
    # see order_tool.py, which requires this field to create a real order
    assert catalogue[0]["id"] == 101
    assert "variation_id" not in catalogue[0]


def test_build_catalogue_captures_variation_id_and_parent_id(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        woocommerce_sync,
        "_fetch_all_products",
        lambda: [
            {
                "id": 200,
                "name": "Heart Twin Ring",
                "type": "variable",
                "variations": [301, 302],
                "categories": [{"name": "Rings"}],
                "stock_status": "instock",
                "permalink": "https://adomdejeweller.com/product/heart-twin-ring",
                "images": [],
            }
        ],
    )
    monkeypatch.setattr(
        woocommerce_sync,
        "_fetch_variations",
        lambda product_id: [
            {"id": 301, "price": "1200", "attributes": [{"option": "18k"}], "stock_status": "instock"},
            {"id": 302, "price": "900", "attributes": [{"option": "14k"}], "stock_status": "instock"},
        ],
    )

    # Act
    catalogue = woocommerce_sync.build_catalogue()

    # Assert: each variation carries the WooCommerce ids order_tool.py
    # needs -- the *parent* product id (not orderable alone) plus its own
    # variation id (what actually gets ordered)
    assert len(catalogue) == 2
    assert catalogue[0]["id"] == 200
    assert catalogue[0]["variation_id"] == 301
    assert catalogue[1]["variation_id"] == 302


# ---------------------------------------------------------------------
# material -- task #126, confirmed live 2026-08-30 (Webb): a real
# WooCommerce variations fetch for product id 6417 ("Minimal White Stone
# Gold Ring, 1g") showed a "Karat" attribute (bare option "18", not
# "18k") alongside a genuinely separate "Ring Sizes" attribute. Joining
# both indiscriminately (the old behaviour) produced material="18 /
# Women US 9.5 (19.4 mm)" instead of "18k".
# ---------------------------------------------------------------------

def test_build_catalogue_drops_ring_size_and_normalises_bare_karat(monkeypatch):
    # Exact shape confirmed against the real store, not guessed.
    monkeypatch.setattr(
        woocommerce_sync,
        "_fetch_all_products",
        lambda: [
            {
                "id": 6417,
                "name": "Minimal White Stone Gold Ring, 1g",
                "type": "variable",
                "variations": [6452],
                "categories": [{"name": "Rings"}],
                "stock_status": "instock",
                "permalink": "https://adomdejeweller.com/product/minimal-white-stone-gold-ring-1g",
                "images": [],
            }
        ],
    )
    monkeypatch.setattr(
        woocommerce_sync,
        "_fetch_variations",
        lambda product_id: [
            {
                "id": 6452,
                "price": "20628",
                "stock_status": "instock",
                "attributes": [
                    {"id": 8, "name": "Karat", "slug": "pa_karat", "option": "18"},
                    {"id": 6, "name": "Ring Sizes", "slug": "pa_ring-sizes", "option": "Women US 9.5 (19.4 mm)"},
                ],
            },
        ],
    )

    catalogue = woocommerce_sync.build_catalogue()

    assert len(catalogue) == 1
    assert catalogue[0]["material"] == "18k"


def test_build_catalogue_still_joins_a_genuinely_different_second_attribute(monkeypatch):
    # The necklace case this join was originally built for -- a real
    # SECOND distinguishing attribute (not a size) must still be kept,
    # not swept up by the new size-only exclusion.
    monkeypatch.setattr(
        woocommerce_sync,
        "_fetch_all_products",
        lambda: [
            {
                "id": 500,
                "name": "Custom Necklace",
                "type": "variable",
                "variations": [601],
                "categories": [{"name": "Necklaces"}],
                "stock_status": "instock",
                "permalink": "https://adomdejeweller.com/product/custom-necklace",
                "images": [],
            }
        ],
    )
    monkeypatch.setattr(
        woocommerce_sync,
        "_fetch_variations",
        lambda product_id: [
            {
                "id": 601,
                "price": "45000",
                "stock_status": "instock",
                "attributes": [
                    {"id": 1, "name": "Karat", "slug": "pa_karat", "option": "18k"},
                    {"id": 2, "name": "Silver Alloy Option", "slug": "pa_alloy", "option": "Sterling"},
                ],
            },
        ],
    )

    catalogue = woocommerce_sync.build_catalogue()

    assert catalogue[0]["material"] == "18k / Sterling"


def test_build_catalogue_leaves_an_already_suffixed_karat_untouched(monkeypatch):
    # A karat option that already says "18k" (not the bare "18" form)
    # must not become "18kk" or otherwise get double-normalised.
    monkeypatch.setattr(
        woocommerce_sync,
        "_fetch_all_products",
        lambda: [
            {
                "id": 700,
                "name": "Simple Ring",
                "type": "variable",
                "variations": [701],
                "categories": [{"name": "Rings"}],
                "stock_status": "instock",
                "permalink": "https://adomdejeweller.com/product/simple-ring",
                "images": [],
            }
        ],
    )
    monkeypatch.setattr(
        woocommerce_sync,
        "_fetch_variations",
        lambda product_id: [
            {
                "id": 701,
                "price": "1000",
                "stock_status": "instock",
                "attributes": [{"id": 8, "name": "Karat", "slug": "pa_karat", "option": "18k"}],
            },
        ],
    )

    catalogue = woocommerce_sync.build_catalogue()

    assert catalogue[0]["material"] == "18k"
