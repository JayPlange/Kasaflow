"""
Unit tests for services/product_search.py

No test file existed for this module before -- it shipped with the
real WooCommerce catalogue sync and the keyword-overlap tie-break fix,
untested. embed_texts is mocked throughout (never call the real OpenAI
API in a unit test, same rule as every other test file here), with
controlled, hand-picked vectors so cosine similarity is deterministic
and the tie-break behaviour can be asserted precisely.
"""

import json
from dataclasses import replace

from services import product_search
from services.product_search import ProductIndex, _keyword_overlap


def _settings_with_products_path(path):
    return replace(product_search.settings, products_path=path)


# ---------------------------------------------------------------------
# _keyword_overlap: pure function, no mocking needed
# ---------------------------------------------------------------------

def test_keyword_overlap_counts_shared_words():
    assert _keyword_overlap("gold chain", "Chain Gold Necklace, 50g") == 2


def test_keyword_overlap_is_substring_tolerant_both_ways():
    # "bracelet" (query) should count against "Bracelets" (product name)
    # without needing real stemming, in either direction
    assert _keyword_overlap("bracelet", "Bracelets") == 1
    assert _keyword_overlap("bracelets", "bracelet chain") == 1


def test_keyword_overlap_is_case_insensitive():
    assert _keyword_overlap("GOLD Chain", "chain gold necklace") == 2


def test_keyword_overlap_zero_when_nothing_shared():
    assert _keyword_overlap("platinum watch", "Gold Ring, 5g") == 0


# ---------------------------------------------------------------------
# ProductIndex.search: the real behaviour that motivated this file --
# see _keyword_overlap's docstring for the actual measured case
# (cosine 0.5977 vs 0.5867) this reproduces with controlled vectors.
# ---------------------------------------------------------------------

def _fake_embed_from(vectors: dict[str, list[float]]):
    def _fake_embed(texts: list[str]) -> list[list[float]]:
        return [vectors[t] for t in texts]
    return _fake_embed


def test_search_keyword_overlap_outranks_a_higher_cosine_score(monkeypatch, tmp_path):
    # Arrange: "Golden Necklace" has a *higher* raw cosine score to the
    # query than "Chain Gold Necklace" does, but doesn't actually
    # contain the word "chain" -- the customer said "chain" and meant
    # it, so the literal match must win despite the worse cosine score.
    entries = [
        {"product": "Golden Necklace, 10g", "category": "Necklaces", "material": "18k", "price": 500},
        {"product": "Chain Gold Necklace, 50g", "category": "Necklaces", "material": "18k", "price": 2000},
    ]
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(entries))
    monkeypatch.setattr(product_search, "settings", _settings_with_products_path(fake_file))

    vectors = {
        "Golden Necklace, 10g Necklaces 18k": [1.0, 0.0],       # cosine 1.0 with the query -- "wins" on similarity alone
        "Chain Gold Necklace, 50g Necklaces 18k": [0.5, 0.8660254],  # cosine 0.5 -- worse, but the real keyword match
        "gold chain": [1.0, 0.0],
    }
    monkeypatch.setattr(product_search, "embed_texts", _fake_embed_from(vectors))

    index = ProductIndex(fake_file)

    # Act
    results = index.search("gold chain")

    # Assert: the literal "chain" match comes first despite the lower cosine score
    assert results[0]["product"] == "Chain Gold Necklace, 50g"
    assert results[1]["product"] == "Golden Necklace, 10g"
    # Internal ranking field never leaks into what a caller sees
    assert "_keyword_overlap" not in results[0]


def test_search_filters_out_scores_below_min_score(monkeypatch, tmp_path):
    # Arrange: totally unrelated product, near-zero similarity
    entries = [{"product": "Platinum Watch", "category": "Watches", "material": "platinum", "price": 9000}]
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps(entries))
    monkeypatch.setattr(product_search, "settings", _settings_with_products_path(fake_file))

    vectors = {
        "Platinum Watch Watches platinum": [1.0, 0.0],
        "gold ring": [0.0, 1.0],  # orthogonal -- cosine 0.0
    }
    monkeypatch.setattr(product_search, "embed_texts", _fake_embed_from(vectors))

    index = ProductIndex(fake_file)

    # Act
    results = index.search("gold ring")

    # Assert: below the default 0.3 threshold -- correctly nothing, not the least-bad guess
    assert results == []


def test_search_respects_top_k(monkeypatch):
    entries = [
        {"product": f"Ring {i}", "category": "Rings", "material": "18k", "price": 100}
        for i in range(5)
    ]
    vectors = {f"Ring {i} Rings 18k": [1.0, 0.0] for i in range(5)}
    vectors["ring"] = [1.0, 0.0]
    monkeypatch.setattr(product_search, "embed_texts", _fake_embed_from(vectors))

    index = ProductIndex(products_path=None)  # never touches disk -- _entries set directly below
    index._entries = entries
    index._embeddings = [[1.0, 0.0]] * 5

    # Act
    results = index.search("ring", top_k=2)

    # Assert
    assert len(results) == 2


def test_search_returns_empty_list_when_file_missing(tmp_path, monkeypatch):
    missing_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(product_search, "settings", _settings_with_products_path(missing_file))

    index = ProductIndex(missing_file)

    # Act
    results = index.search("anything")

    # Assert: fails gracefully, matching every other tool's pattern
    assert results == []


def test_reload_clears_cache_so_the_next_search_reloads_from_disk(monkeypatch, tmp_path):
    # Arrange
    fake_file = tmp_path / "products.json"
    fake_file.write_text(json.dumps([{"product": "Ring", "category": "Rings", "material": "18k", "price": 100}]))
    monkeypatch.setattr(product_search, "settings", _settings_with_products_path(fake_file))
    monkeypatch.setattr(
        product_search, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts]
    )

    index = ProductIndex(fake_file)
    index.search("ring")  # loads and caches
    assert index._entries is not None

    # Act: woocommerce_sync.py rewrote the file; reload() should drop the cache
    index.reload()

    # Assert
    assert index._entries is None
    assert index._embeddings is None
