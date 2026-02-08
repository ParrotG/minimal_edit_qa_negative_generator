from __future__ import annotations


def build_qa_premise(knowledge: str, question: str) -> str:
    """Build a canonical QA premise string shared across NLI and DPO stages."""
    return f"{knowledge}\nQuestion: {question}"


def build_qa_answer_prefix(knowledge: str, question: str) -> str:
    """Build a canonical QA answer prefix for conditional log-probability scoring."""
    return f"{build_qa_premise(knowledge, question)}\nAnswer: "
