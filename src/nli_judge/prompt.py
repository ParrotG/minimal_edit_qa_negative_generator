from __future__ import annotations


def build_qa_premise(knowledge: str, question: str) -> str:
    """Build canonical QA premise text used by NLI checks."""

    return f"{knowledge}\nQuestion: {question}"
