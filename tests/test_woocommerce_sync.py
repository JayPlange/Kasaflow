"""
Unit tests for services/woocommerce_sync.py's build_catalogue().

Scoped to the "id"/"variation_id" capture added alongside order_tool.py
-- this file had no test coverage before that change. Mocks the private
_fetch_all_products/_fetch_variations helpers rather than requests
directly, since build_catalogue()'s own job (the thing worth testing) is
shaping WooCommerce's response into a catalogue entry, not the HTTP call
itself.
"""

from services import woocommerce_sync


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
