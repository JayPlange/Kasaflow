import json
from dataclasses import replace

from services import recommendation_service


def _settings_with_products_path(path):
    return replace(recommendation_service.settings, products_path=path)


def test_recommend_products_filters_by_material(monkeypatch, tmp_path):
    # Arrange
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps([
        {"product": "ring", "material": "gold", "price": 1200},
        {"product": "chain", "material": "gold", "price": 1800},
        {"product": "ring", "material": "silver", "price": 350},
    ]))
    monkeypatch.setattr(
        recommendation_service, "settings", _settings_with_products_path(fake_file)
    )

    # Act
    result = recommendation_service.recommend_products("gold")

    # Assert
    assert len(result["recommendations"]) == 2
    assert all(p["material"] == "gold" for p in result["recommendations"])


def test_recommend_products_returns_empty_list_when_no_match(monkeypatch, tmp_path):
    # Arrange
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps([
        {"product": "ring", "material": "gold", "price": 1200},
    ]))
    monkeypatch.setattr(
        recommendation_service, "settings", _settings_with_products_path(fake_file)
    )

    # Act
    result = recommendation_service.recommend_products("platinum")

    # Assert: empty list, not an error
    assert result == {"recommendations": []}


def test_recommend_products_returns_empty_list_when_file_missing(monkeypatch, tmp_path):
    # Arrange
    missing_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(
        recommendation_service, "settings", _settings_with_products_path(missing_file)
    )

    # Act
    result = recommendation_service.recommend_products("gold")

    # Assert: fails gracefully, matching product_tool.py's pattern
    assert result == {"recommendations": []}
