"""
Unit tests for services/policy_tool.py

Mocks the knowledge base entirely -- these tests are about the tool's
own logic (empty question, no match, formatting the response), not
about retrieval quality, which test_knowledge_base.py already covers.
"""

from unittest.mock import MagicMock

from services import policy_tool


def test_answer_policy_question_rejects_empty_question():
    # Arrange / Act
    result = policy_tool.answer_policy_question("")

    # Assert
    assert result["sources"] == []
    assert "answer" in result


def test_answer_policy_question_rejects_blank_question():
    # Arrange / Act
    result = policy_tool.answer_policy_question("   ")

    # Assert
    assert result["sources"] == []


def test_answer_policy_question_returns_friendly_message_when_no_match(monkeypatch):
    # Arrange
    fake_kb = MagicMock()
    fake_kb.retrieve.return_value = []
    monkeypatch.setattr(policy_tool, "get_knowledge_base", lambda: fake_kb)

    # Act
    result = policy_tool.answer_policy_question("do you sell insurance for spacecraft?")

    # Assert
    assert result["sources"] == []
    assert "don't have a written policy" in result["answer"]


def test_answer_policy_question_returns_best_match_and_sources(monkeypatch):
    # Arrange
    fake_kb = MagicMock()
    fake_kb.retrieve.return_value = [
        {"id": "returns_policy", "title": "Returns", "text": "30 day returns.", "score": 0.9},
        {"id": "warranty_policy", "title": "Warranty", "text": "12 month warranty.", "score": 0.2},
    ]
    monkeypatch.setattr(policy_tool, "get_knowledge_base", lambda: fake_kb)

    # Act
    result = policy_tool.answer_policy_question("what's your returns policy?")

    # Assert: the top match's text is the answer, all matches show up as sources
    assert result["answer"] == "30 day returns."
    assert result["sources"] == [
        {"id": "returns_policy", "title": "Returns", "score": 0.9},
        {"id": "warranty_policy", "title": "Warranty", "score": 0.2},
    ]
