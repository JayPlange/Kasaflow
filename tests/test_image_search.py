"""
Unit tests for services/image_search.py's ImageIndex.

No real Cohere calls -- embed_images() is mocked throughout, this
tests the cosine-ranking and file-loading logic only.
"""

import json

from services import image_search
from services.image_search import ImageIndex, _cosine_similarity


def test_cosine_similarity_identical_vectors_scores_one():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_scores_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_handles_a_zero_vector_without_dividing_by_zero():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_search_returns_empty_list_when_embeddings_file_missing(monkeypatch, tmp_path):
    index = ImageIndex(tmp_path / "does_not_exist.json")
    monkeypatch.setattr(image_search, "embed_images", lambda urls: (_ for _ in ()).throw(AssertionError("should not be called")))

    assert index.search(b"photo-bytes", "image/jpeg") == []


def test_search_returns_empty_list_for_corrupt_embeddings_file(tmp_path):
    bad_file = tmp_path / "image_embeddings.json"
    bad_file.write_text("{ not valid json ]")
    index = ImageIndex(bad_file)

    assert index.search(b"photo-bytes", "image/jpeg") == []


def test_search_ranks_by_cosine_similarity_best_first(monkeypatch, tmp_path):
    embeddings_file = tmp_path / "image_embeddings.json"
    embeddings_file.write_text(json.dumps([
        {"image_url": "https://x/close.jpg", "embedding": [1.0, 0.0]},
        {"image_url": "https://x/far.jpg", "embedding": [0.0, 1.0]},
        {"image_url": "https://x/medium.jpg", "embedding": [0.7, 0.7]},
    ]))
    index = ImageIndex(embeddings_file)
    monkeypatch.setattr(image_search, "embed_images", lambda urls: [[1.0, 0.0]])

    result = index.search(b"photo-bytes", "image/jpeg", top_k=3, min_score=0.0)

    assert [r["image_url"] for r in result] == ["https://x/close.jpg", "https://x/medium.jpg", "https://x/far.jpg"]


def test_search_filters_out_results_below_min_score(monkeypatch, tmp_path):
    embeddings_file = tmp_path / "image_embeddings.json"
    embeddings_file.write_text(json.dumps([
        {"image_url": "https://x/close.jpg", "embedding": [1.0, 0.0]},
        {"image_url": "https://x/far.jpg", "embedding": [0.0, 1.0]},
    ]))
    index = ImageIndex(embeddings_file)
    monkeypatch.setattr(image_search, "embed_images", lambda urls: [[1.0, 0.0]])

    result = index.search(b"photo-bytes", "image/jpeg", top_k=5, min_score=0.5)

    assert len(result) == 1
    assert result[0]["image_url"] == "https://x/close.jpg"


def test_search_respects_top_k(monkeypatch, tmp_path):
    embeddings_file = tmp_path / "image_embeddings.json"
    embeddings_file.write_text(json.dumps([
        {"image_url": f"https://x/{i}.jpg", "embedding": [1.0, 0.0]} for i in range(10)
    ]))
    index = ImageIndex(embeddings_file)
    monkeypatch.setattr(image_search, "embed_images", lambda urls: [[1.0, 0.0]])

    result = index.search(b"photo-bytes", "image/jpeg", top_k=3, min_score=0.0)

    assert len(result) == 3


def test_reload_forces_the_embeddings_file_to_be_re_read(monkeypatch, tmp_path):
    embeddings_file = tmp_path / "image_embeddings.json"
    embeddings_file.write_text(json.dumps([{"image_url": "https://x/a.jpg", "embedding": [1.0, 0.0]}]))
    index = ImageIndex(embeddings_file)
    monkeypatch.setattr(image_search, "embed_images", lambda urls: [[1.0, 0.0]])

    first = index.search(b"photo-bytes", "image/jpeg", min_score=0.0)
    assert len(first) == 1

    embeddings_file.write_text(json.dumps([
        {"image_url": "https://x/a.jpg", "embedding": [1.0, 0.0]},
        {"image_url": "https://x/b.jpg", "embedding": [1.0, 0.0]},
    ]))
    index.reload()

    second = index.search(b"photo-bytes", "image/jpeg", min_score=0.0)
    assert len(second) == 2
