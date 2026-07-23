"""
Unit tests for services/knowledge_base.py -- the RAG retrieval piece.

We never call the real embeddings API here: embed_texts is monkeypatched
to a tiny deterministic stand-in that maps a handful of known keywords to
orthogonal basis vectors, so cosine similarity behaves predictably
without needing real semantic embeddings. That's enough to prove the
retrieval logic (ranking, thresholding, caching, reload) is correct --
whether real embeddings actually capture meaning well is what the
regression tests (opt-in, real API) are for.
"""

import json

from services import knowledge_base as kb_module
from services.knowledge_base import KnowledgeBase


def _fake_embed_texts(calls):
    """Returns a fake embed_texts that records how many times it's called
    and maps text containing "return" / "warranty" to basis vectors,
    anything else to the zero vector.
    """

    def _embed(texts):
        calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "return" in lowered:
                vectors.append([1.0, 0.0])
            elif "warranty" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.0, 0.0])
        return vectors

    return _embed


def _write_documents(tmp_path, documents):
    path = tmp_path / "policies.json"
    path.write_text(json.dumps(documents))
    return path


def test_retrieve_returns_the_closest_matching_document(monkeypatch, tmp_path):
    # Arrange
    documents = [
        {"id": "returns_policy", "title": "Returns", "text": "Our return policy is 30 days."},
        {"id": "warranty_policy", "title": "Warranty", "text": "Our warranty covers defects."},
    ]
    path = _write_documents(tmp_path, documents)
    calls = []
    monkeypatch.setattr(kb_module, "embed_texts", _fake_embed_texts(calls))
    kb = KnowledgeBase(path)

    # Act: a question about returns should match the returns document, not warranty
    results = kb.retrieve("What is your return policy?")

    # Assert
    assert len(results) == 1
    assert results[0]["id"] == "returns_policy"
    assert results[0]["score"] == 1.0


def test_retrieve_returns_empty_list_when_nothing_scores_above_threshold(monkeypatch, tmp_path):
    # Arrange: a query that doesn't match either known keyword embeds as
    # the zero vector, so cosine similarity against every document is 0
    documents = [
        {"id": "returns_policy", "title": "Returns", "text": "Our return policy is 30 days."},
    ]
    path = _write_documents(tmp_path, documents)
    calls = []
    monkeypatch.setattr(kb_module, "embed_texts", _fake_embed_texts(calls))
    kb = KnowledgeBase(path)

    # Act
    results = kb.retrieve("what's the weather like today?")

    # Assert
    assert results == []


def test_retrieve_returns_empty_list_when_file_missing(monkeypatch, tmp_path):
    # Arrange
    missing_path = tmp_path / "does_not_exist.json"
    calls = []
    monkeypatch.setattr(kb_module, "embed_texts", _fake_embed_texts(calls))
    kb = KnowledgeBase(missing_path)

    # Act
    results = kb.retrieve("what is your return policy?")

    # Assert: fails gracefully, doesn't raise, and never calls embed_texts
    # since there's nothing to embed
    assert results == []
    assert calls == []


def test_retrieve_returns_empty_list_when_file_is_corrupt(monkeypatch, tmp_path):
    # Arrange
    bad_path = tmp_path / "policies.json"
    bad_path.write_text("{ not valid json ]")
    calls = []
    monkeypatch.setattr(kb_module, "embed_texts", _fake_embed_texts(calls))
    kb = KnowledgeBase(bad_path)

    # Act
    results = kb.retrieve("what is your return policy?")

    # Assert
    assert results == []


def test_documents_are_embedded_once_and_cached(monkeypatch, tmp_path):
    # Arrange
    documents = [
        {"id": "returns_policy", "title": "Returns", "text": "Our return policy is 30 days."},
    ]
    path = _write_documents(tmp_path, documents)
    calls = []
    monkeypatch.setattr(kb_module, "embed_texts", _fake_embed_texts(calls))
    kb = KnowledgeBase(path)

    # Act: two separate queries against the same instance
    kb.retrieve("what is your return policy?")
    kb.retrieve("can I return this ring?")

    # Assert: the documents batch (["Our return policy is 30 days."]) was
    # only embedded once -- the second retrieve() call only embeds the
    # new query, not the whole document set again
    document_embed_calls = [c for c in calls if c == ["Our return policy is 30 days."]]
    assert len(document_embed_calls) == 1
    assert len(calls) == 3  # 1 document batch + 2 query embeds


def test_reload_forces_documents_to_be_re_read_and_re_embedded(monkeypatch, tmp_path):
    # Arrange
    documents = [
        {"id": "returns_policy", "title": "Returns", "text": "Our return policy is 30 days."},
    ]
    path = _write_documents(tmp_path, documents)
    calls = []
    monkeypatch.setattr(kb_module, "embed_texts", _fake_embed_texts(calls))
    kb = KnowledgeBase(path)
    kb.retrieve("what is your return policy?")

    # Act
    kb.reload()
    kb.retrieve("what is your return policy?")

    # Assert: the document batch was embedded again after reload()
    document_embed_calls = [c for c in calls if c == ["Our return policy is 30 days."]]
    assert len(document_embed_calls) == 2


def test_get_knowledge_base_returns_the_same_module_level_instance():
    # Arrange / Act
    first = kb_module.get_knowledge_base()
    second = kb_module.get_knowledge_base()

    # Assert
    assert first is second
