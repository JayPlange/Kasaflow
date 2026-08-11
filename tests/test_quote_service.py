"""
Unit tests for services/quote_service.py

generate_quote() composes get_product_price() and
get_delivery_information(). Rather than re-testing the products file
here too, we mock those two lower-level functions and just check that
quote_service wires their outputs together correctly. Each layer tests
its own responsibility, not the layer below it again.
"""

from unittest.mock import MagicMock

from services import quote_service

_DELIVERY_OPTIONS = [
    {"key": "accra_rider", "label": "rider delivery within Accra"},
    {"key": "kumasi_rider", "label": "rider delivery within Kumasi"},
    {"key": "international", "label": "shipping outside Ghana"},
]


def test_generate_quote_returns_combined_quote_when_product_found(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        quote_service,
        "get_product_price",
        MagicMock(return_value={"product": "ring", "material": "gold", "price": 1200}),
    )
    monkeypatch.setattr(
        quote_service,
        "get_delivery_information",
        MagicMock(return_value={"delivery_options": _DELIVERY_OPTIONS}),
    )

    # Act
    result = quote_service.generate_quote("ring", "gold")

    # Assert: real delivery choices, not an invented cost/time (see
    # delivery_tool.py's module docstring)
    assert result == {
        "product": "ring",
        "material": "gold",
        "price": 1200,
        "delivery_options": _DELIVERY_OPTIONS,
    }


def test_generate_quote_includes_image_url_when_product_has_one(monkeypatch):
    # Arrange: real WooCommerce products carry a photo, the placeholder
    # catalogue never did -- this is the shape a real product looks like
    monkeypatch.setattr(
        quote_service,
        "get_product_price",
        MagicMock(return_value={
            "product": "ring",
            "material": "gold",
            "price": 1200,
            "image_url": "https://adomdejeweller.com/wp-content/uploads/ring.jpg",
        }),
    )
    monkeypatch.setattr(
        quote_service,
        "get_delivery_information",
        MagicMock(return_value={"delivery_options": _DELIVERY_OPTIONS}),
    )

    # Act
    result = quote_service.generate_quote("ring", "gold")

    # Assert
    assert result["image_url"] == "https://adomdejeweller.com/wp-content/uploads/ring.jpg"


def test_generate_quote_omits_image_url_when_product_has_none(monkeypatch):
    # Arrange: matches the old placeholder catalogue's shape -- no
    # image_url key at all, not even an empty string
    monkeypatch.setattr(
        quote_service,
        "get_product_price",
        MagicMock(return_value={"product": "ring", "material": "gold", "price": 1200}),
    )
    monkeypatch.setattr(
        quote_service,
        "get_delivery_information",
        MagicMock(return_value={"delivery_options": _DELIVERY_OPTIONS}),
    )

    # Act
    result = quote_service.generate_quote("ring", "gold")

    # Assert: the dict's shape is exactly what it was before image_url existed
    assert "image_url" not in result


def test_generate_quote_returns_friendly_message_when_product_missing(monkeypatch):
    # Arrange: simulate no matching product, like the old bug used to crash on
    monkeypatch.setattr(quote_service, "get_product_price", MagicMock(return_value=None))
    delivery_mock = MagicMock()
    monkeypatch.setattr(quote_service, "get_delivery_information", delivery_mock)

    # Act
    result = quote_service.generate_quote("bracelet", "platinum")

    # Assert: no crash, friendly message, and delivery info was never even fetched
    assert result == {"message": "Sorry, we couldn't find that product."}
    delivery_mock.assert_not_called()
