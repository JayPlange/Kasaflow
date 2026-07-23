"""
answer_policy_question -- the RAG-backed tool the LLM calls when a
customer asks about policy (returns, warranty, sizing, care, engraving,
payment) rather than a specific product's price or availability.
"""

import logging

from services.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


def answer_policy_question(question: str) -> dict:
    if not question or not question.strip():
        return {
            "answer": "Could you tell me a bit more about what you'd like to know?",
            "sources": [],
        }

    matches = get_knowledge_base().retrieve(question)

    if not matches:
        logger.info("No policy match found for question=%r", question)
        return {
            "answer": (
                "I don't have a written policy that covers that -- "
                "I'll flag it for a team member to confirm."
            ),
            "sources": [],
        }

    best = matches[0]
    return {
        "answer": best["text"],
        "sources": [
            {"id": match["id"], "title": match["title"], "score": round(match["score"], 3)}
            for match in matches
        ],
    }
