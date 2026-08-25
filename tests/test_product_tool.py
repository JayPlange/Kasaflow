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


def test_get_product_price_returns_karat_breakdown_when_name_matches_and_no_karat_stated(monkeypatch, tmp_path):
    # Confirmed live, 2026-08-24 (Webb/GPT 50-turn test): "how much is
    # the Big White Crown Stone Gold Ring, 14g" (no karat stated)
    # returned "couldn't find that one" even though the product exists
    # verbatim -- the deterministic karat-match path above only ever
    # activates once a karat is present. This must find the exact
    # product and, since more than one real karat exists, name them
    # rather than silently guessing one or claiming no match at all.
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_variant_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act
    result = product_tool.get_product_price("Set Multi Stone Golf Ring, 7g", "unknown")

    # Assert: the same {"product", "karat_options"} shape
    # list_karat_options() returns, deduped and sorted highest-first --
    # response_formatter.py already knows how to render this.
    assert result["product"] == "Set Multi Stone Golf Ring, 7g"
    assert [o["material"][:2] for o in result["karat_options"]] == ["18", "12"]


def test_get_product_price_answers_directly_when_name_matches_and_only_one_karat_exists(monkeypatch, tmp_path):
    # A product with only one real karat has nothing to ask about --
    # same "no-variant product" precedent as scenario 3 in this
    # engagement's own test history, just reached via a karat-less
    # query instead of an already-resolved one.
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_necklace_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act: "Other Necklace" only has one row/karat in _necklace_catalogue()
    result = product_tool.get_product_price("Other Necklace", "unknown")

    # Assert: a normal, single priced product, not a karat_options shape
    assert result == {"product": "Other Necklace", "material": "18k", "price": 20000.0, "image_url": "https://x/b.jpg"}


def test_get_product_price_still_falls_back_to_semantic_search_when_name_matches_nothing_at_all(monkeypatch, tmp_path):
    # The new karat-less exact-name path above must not swallow the
    # existing semantic-search fallback for a genuinely mangled/off
    # product_name -- get_product_karat_options() correctly returns []
    # for that, and this must fall through unchanged.
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_necklace_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    matching = {"product": "Gye Nyame White Necklace", "material": "18k", "price": 51000.0, "score": 0.91}
    fake_index = type("FakeIndex", (), {"search": lambda self, *a, **k: [matching]})()
    monkeypatch.setattr(product_tool, "get_product_index", lambda: fake_index)

    # Act
    result = product_tool.get_product_price("Gye Nyame Necklace", "unknown")

    # Assert: reached semantic search and returned its match, exactly as
    # before this fix
    assert result["price"] == 51000.0


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


def test_get_product_price_refuses_a_semantic_match_at_the_wrong_karat(monkeypatch, tmp_path):
    # Confirmed live, 2026-08-20: an explicit "change the karat to 18"
    # produced a correction_note claiming the karat was updated to 18k,
    # while the actual proposal underneath silently priced at 12k --
    # semantic search's top match shared the product but not the karat,
    # and nothing before this guard ever checked that. This must come
    # back as no match at all, not a silently wrong price.
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_variant_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # The exact/karat-match paths both require an exact product_name, so
    # this only reaches semantic search when the name itself is slightly
    # off -- simulate that, and have the fake index return the 12k
    # variant even though 18k was requested.
    mismatched = {
        "product": "Set Multi Stone Golf Ring, 7g",
        "material": "12 / Women US 8 (18.2 mm)",
        "price": 8824.2,
        "score": 0.91,
    }
    fake_index = type("FakeIndex", (), {"search": lambda self, *a, **k: [mismatched]})()
    monkeypatch.setattr(product_tool, "get_product_index", lambda: fake_index)

    # Act
    result = product_tool.get_product_price("Set Multi Stone Golf Ring 7g", "18k")

    # Assert: refused, not silently returned at the wrong karat
    assert result is None


def test_get_product_price_accepts_a_semantic_match_at_the_right_karat(monkeypatch, tmp_path):
    # The guard above must not become overzealous -- a semantic match
    # that DOES share the requested karat is exactly what this fallback
    # exists for, and must still work.
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_variant_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    matching = {
        "product": "Set Multi Stone Golf Ring, 7g",
        "material": "18 / Women US 8 (18.2 mm)",
        "price": 12033.0,
        "score": 0.91,
    }
    fake_index = type("FakeIndex", (), {"search": lambda self, *a, **k: [matching]})()
    monkeypatch.setattr(product_tool, "get_product_index", lambda: fake_index)

    # Act
    result = product_tool.get_product_price("Set Multi Stone Golf Ring 7g", "18k")

    # Assert
    assert result["price"] == 12033.0


# ---------------------------------------------------------------------
# get_product_karat_options -- the full price-by-karat breakdown for
# one already-identified product (see photo_match_tool.py), not just
# the single karat get_product_price() returns.
# ---------------------------------------------------------------------

def _necklace_catalogue():
    return [
        {"product": "Gye Nyame White Necklace", "material": "18k", "price": 51000.0, "image_url": "https://x/a.jpg"},
        {"product": "Gye Nyame White Necklace", "material": "14k", "price": 45000.0, "image_url": "https://x/a.jpg"},
        {"product": "Gye Nyame White Necklace", "material": "12k", "price": 39000.0, "image_url": "https://x/a.jpg"},
        {"product": "Other Necklace", "material": "18k", "price": 20000.0, "image_url": "https://x/b.jpg"},
    ]


def test_get_product_karat_options_returns_only_matching_product_sorted_by_karat(monkeypatch, tmp_path):
    # Arrange
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_necklace_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act
    result = product_tool.get_product_karat_options("Gye Nyame White Necklace")

    # Assert: highest karat first, only this product's rows, none of the
    # unrelated "Other Necklace" rows leaked in
    assert [r["material"] for r in result] == ["18k", "14k", "12k"]
    assert all(r["product"] == "Gye Nyame White Necklace" for r in result)


def test_get_product_karat_options_dedupes_sized_variants_sharing_a_karat(monkeypatch, tmp_path):
    # Arrange: a ring with several sizes at the same karat -- same
    # "size doesn't affect price" fact get_product_price() already
    # relies on, so only one row per karat should come back
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_variant_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act
    result = product_tool.get_product_karat_options("Set Multi Stone Golf Ring, 7g")

    # Assert: 4 rows in, 2 distinct karats out
    assert len(result) == 2
    assert {r["material"][:2] for r in result} == {"12", "18"}


def test_get_product_karat_options_returns_empty_list_for_no_match(monkeypatch, tmp_path):
    # Arrange
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_necklace_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act
    result = product_tool.get_product_karat_options("Completely Unknown Product")

    # Assert: no fabricated fallback -- see module docstring, this
    # deliberately doesn't fall through to semantic search
    assert result == []


def test_get_product_karat_options_returns_empty_list_when_file_missing(monkeypatch, tmp_path):
    # Arrange
    missing_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(missing_file))

    # Act / Assert: fails gracefully, not with an exception
    assert product_tool.get_product_karat_options("Anything") == []


# ---------------------------------------------------------------------
# list_karat_options -- the tool-registry entry point wrapping
# get_product_karat_options() above into the dict shape execute_tool()/
# response_formatter.py expect (see that function's docstring).
# ---------------------------------------------------------------------

def test_list_karat_options_wraps_the_bare_list_in_a_dict_shape(monkeypatch, tmp_path):
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_necklace_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    result = product_tool.list_karat_options("Gye Nyame White Necklace")

    assert result["product"] == "Gye Nyame White Necklace"
    assert [r["material"] for r in result["karat_options"]] == ["18k", "14k", "12k"]


def test_list_karat_options_returns_empty_karat_options_for_no_match(monkeypatch, tmp_path):
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_necklace_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    result = product_tool.list_karat_options("Completely Unknown Product")

    assert result == {"product": "Completely Unknown Product", "karat_options": []}


# ---------------------------------------------------------------------
# get_product_weight -- reads weight only from the resolved catalogue
# product name (Webb, 2026-08-25: never inferred from the customer's
# own raw text), same deterministic-lookup shape as get_product_price().
# ---------------------------------------------------------------------

def test_get_product_weight_extracts_weight_from_the_canonical_product_name(monkeypatch, tmp_path):
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_variant_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    result = product_tool.get_product_weight("Set Multi Stone Golf Ring, 7g")

    assert result == {"product": "Set Multi Stone Golf Ring, 7g", "weight": "7g"}


def test_get_product_weight_returns_none_weight_for_a_real_product_with_no_weight_in_its_name(monkeypatch, tmp_path):
    # "Gye Nyame White Necklace" (this fixture's name, no weight suffix)
    # is a real, matched catalogue product -- weight genuinely isn't
    # parseable from it, which is a different, honest situation from the
    # product not existing at all (see the no-match test below).
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_necklace_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    result = product_tool.get_product_weight("Gye Nyame White Necklace")

    assert result == {"product": "Gye Nyame White Necklace", "weight": None}


def test_get_product_weight_returns_none_for_no_match_at_all(monkeypatch, tmp_path):
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_necklace_catalogue()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # Act / Assert: bare None (not a dict), same convention as
    # get_product_price() -- response_formatter.py's existing "couldn't
    # find that one" no-match message already handles this.
    assert product_tool.get_product_weight("Completely Unknown Product") is None


def test_get_product_weight_returns_none_when_file_missing(monkeypatch, tmp_path):
    missing_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(missing_file))

    assert product_tool.get_product_weight("Anything") is None


def test_get_product_weight_does_not_infer_from_a_decimal_gram_value():
    # A sanity check on the regex itself, not the file lookup -- confirms
    # it also handles a non-integer weight cleanly (real catalogue
    # weights are all whole grams, but nothing about the regex assumes
    # that).
    assert product_tool._extract_weight("Some Ring, 7.5g") == "7.5g"
    assert product_tool._extract_weight("Custom Butterfly Gold Ring") is None


# ---------------------------------------------------------------------
# get_product_price_by_id -- order_tool.propose_order()'s
# correction-recovery fallback, 2026-08-21 (Webb, real
# OpenAI-backed /demo run: a correction's restated product_name dropped
# the ", 14g" weight suffix, and get_product_price()'s exact-match-first
# design can never recover from that on its own). id is stable across
# every karat/size variant of one named product in this catalogue
# (confirmed against the real data), so a session that already knows a
# product's id can reprice a correction against it directly, without
# ever depending on the model restating the name correctly again.
# ---------------------------------------------------------------------

def _ringed_catalogue_with_ids():
    return [
        {"id": 5892, "variation_id": 5920, "product": "Big White Crown Stone Gold Ring, 14g",
         "material": "12 / Women US 9.5 (19.4 mm)", "price": 88242.0},
        {"id": 5892, "variation_id": 5921, "product": "Big White Crown Stone Gold Ring, 14g",
         "material": "18 / Women US 9.5 (19.4 mm)", "price": 132000.0},
        # A different catalogue entry that happens to share a name
        # prefix with a different weight suffix -- real, confirmed data
        # shape (data/products.json, 2026-08-21): id=6800, distinct from
        # the 6520 entry it's a near-duplicate of.
        {"id": 6800, "variation_id": 7001, "product": "Custom Gye Nyame Gold Necklace with Earrings, 20g",
         "material": "18k", "price": 200000.0},
    ]


def test_get_product_price_by_id_returns_the_matching_karat(monkeypatch, tmp_path):
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_ringed_catalogue_with_ids()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    result = product_tool.get_product_price_by_id(5892, "18k")

    assert result["price"] == 132000.0
    assert result["material"].startswith("18")


def test_get_product_price_by_id_returns_none_for_an_unknown_id(monkeypatch, tmp_path):
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_ringed_catalogue_with_ids()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    assert product_tool.get_product_price_by_id(999999, "18k") is None


def test_get_product_price_by_id_returns_none_when_the_karat_has_no_variant(monkeypatch, tmp_path):
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_ringed_catalogue_with_ids()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    # 5892 only has 12k/18k rows in this fake catalogue -- 22k doesn't
    # exist, and this must refuse rather than guess.
    assert product_tool.get_product_price_by_id(5892, "22k") is None


def test_get_product_price_by_id_matches_a_bare_exact_material_first(monkeypatch, tmp_path):
    # Non-sized products store material as a bare "18k", not the sized
    # "{karat} / {size}" format -- the exact-match tier must still work
    # for those, same as get_product_price()'s own first tier.
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_ringed_catalogue_with_ids()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    result = product_tool.get_product_price_by_id(6800, "18k")

    assert result["price"] == 200000.0


def test_get_product_price_by_id_never_crosses_into_a_different_id(monkeypatch, tmp_path):
    # The near-duplicate-name collision this whole fallback exists to
    # stay safe against: id=6800's own material must never be returned
    # for a query against a completely different id, regardless of any
    # name similarity -- this function never even looks at `product`
    # (the name), only `id`.
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(_ringed_catalogue_with_ids()))
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(fake_file))

    result = product_tool.get_product_price_by_id(5892, "18k")

    assert result["id"] == 5892
    assert result["price"] != 200000.0


def test_get_product_price_by_id_returns_none_when_file_missing(monkeypatch, tmp_path):
    missing_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(product_tool, "settings", _settings_with_products_path(missing_file))

    assert product_tool.get_product_price_by_id(5892, "18k") is None
