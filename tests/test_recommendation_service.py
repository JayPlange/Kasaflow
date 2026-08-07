import json
from dataclasses import replace

from services import recommendation_service


def _settings_with_products_path(path):
    return replace(recommendation_service.settings, products_path=path)


# Real catalogue shapes (from the live 2026-08-07 WooCommerce sync):
# Necklaces carry a bare karat ("18k" or, inconsistently, just "18"),
# Rings carry a compound "{karat} / {gender} US {size} ({mm})" string,
# and nothing in the catalogue is anything other than gold.
_CATALOGUE = [
    {"product": "Gye Nyame Necklace, 30g", "category": "Necklaces", "material": "18k", "price": 51000},
    {"product": "Plain Necklace, 20g", "category": "Necklaces", "material": "18", "price": 34000},
    {"product": "Twin Chain, 15g", "category": "Necklaces", "material": "14k", "price": 22000},
    {"product": "Square Stone Ring, 16g", "category": "Rings", "material": "18 / Women US 12 (21.4 mm)", "price": 27504},
    {"product": "Small Stone Ring, 6g", "category": "Rings", "material": "14 / Women US 12.5 (21.8 mm)", "price": 8938},
]


def _write_catalogue(tmp_path, monkeypatch, catalogue=_CATALOGUE):
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(catalogue))
    monkeypatch.setattr(
        recommendation_service, "settings", _settings_with_products_path(fake_file)
    )


def test_recommend_products_filters_by_category(monkeypatch, tmp_path):
    _write_catalogue(tmp_path, monkeypatch)

    result = recommendation_service.recommend_products(category="Rings")

    assert len(result["recommendations"]) == 2
    assert all(p["category"] == "Rings" for p in result["recommendations"])


def test_recommend_products_filters_by_karat_across_formats(monkeypatch, tmp_path):
    """"18k" (Necklaces), "18" (Necklaces), and the Rings compound string
    all carry karat 18 -- a karat filter of "18k" must match all three,
    despite none of them sharing the same literal material string."""
    _write_catalogue(tmp_path, monkeypatch)

    result = recommendation_service.recommend_products(material="18k")

    assert {p["product"] for p in result["recommendations"]} == {
        "Gye Nyame Necklace, 30g",
        "Plain Necklace, 20g",
        "Square Stone Ring, 16g",
    }


def test_recommend_products_combines_category_and_karat(monkeypatch, tmp_path):
    _write_catalogue(tmp_path, monkeypatch)

    result = recommendation_service.recommend_products(material="14k", category="Rings")

    assert [p["product"] for p in result["recommendations"]] == ["Small Stone Ring, 6g"]


def test_recommend_products_generic_gold_with_no_karat_matches_everything(monkeypatch, tmp_path):
    """The whole catalogue is gold -- a bare "gold" (no karat digits)
    has nothing to narrow against, so it must not exclude anything."""
    _write_catalogue(tmp_path, monkeypatch)

    result = recommendation_service.recommend_products(material="gold")

    assert len(result["recommendations"]) == len(_CATALOGUE)


def test_recommend_products_unknown_arguments_match_everything(monkeypatch, tmp_path):
    _write_catalogue(tmp_path, monkeypatch)

    result = recommendation_service.recommend_products()

    assert len(result["recommendations"]) == len(_CATALOGUE)


def test_recommend_products_returns_empty_list_when_category_has_no_match(monkeypatch, tmp_path):
    _write_catalogue(tmp_path, monkeypatch)

    result = recommendation_service.recommend_products(category="Bracelets")

    assert result == {"recommendations": []}


def test_recommend_products_returns_empty_list_when_file_missing(monkeypatch, tmp_path):
    missing_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(
        recommendation_service, "settings", _settings_with_products_path(missing_file)
    )

    result = recommendation_service.recommend_products(material="gold")

    # Fails gracefully, matching product_tool.py's pattern.
    assert result == {"recommendations": []}


def test_recommend_products_returns_empty_list_when_file_is_bad_json(monkeypatch, tmp_path):
    bad_file = tmp_path / "products.json"
    bad_file.write_text("not valid json")
    monkeypatch.setattr(
        recommendation_service, "settings", _settings_with_products_path(bad_file)
    )

    result = recommendation_service.recommend_products(material="gold")

    assert result == {"recommendations": []}
