from __future__ import annotations

from typing import Optional


def build_qa_premise(knowledge: str, question: str) -> str:
    """Build canonical QA premise text for NLI-style checks."""

    knowledge_text = str(knowledge or "").strip()
    question_text = str(question or "").strip()
    if knowledge_text:
        return f"{knowledge_text}\nQuestion: {question_text}"
    return f"Question: {question_text}"


def build_qa_answer_prefix(knowledge: str, question: str) -> str:
    """Build canonical QA answer-generation prefix."""

    return build_qa_answer_prefix_from_premise(build_qa_premise(knowledge=knowledge, question=question))


def build_qa_answer_prefix_from_premise(premise: str) -> str:
    """Build canonical answer prefix from an existing QA premise string."""

    return f"{str(premise or '').strip()}\nAnswer: "


def build_qa_question_answer_text(question: str, answer: str) -> str:
    """Build canonical question-answer text for semantic consistency checks."""

    question_text = str(question or "").strip()
    answer_text = str(answer or "").strip()
    return f"Question: {question_text}\nAnswer: {answer_text}"


def build_repair_prompt(
    knowledge: str,
    question: str,
    rejected_answer: str,
    reference_answer: Optional[str] = None,
) -> str:
    """Build a minimal-edit repair prompt for hard-sample correction."""

    ref_block = ""
    if reference_answer:
        ref_block = f"Reference answer (for calibration only):\n{reference_answer}\n\n"

    return (
        "You are a strict factual editor.\n"
        "Task: minimally edit the candidate answer so it is fully supported by the knowledge and directly answers the question.\n"
        "Constraints:\n"
        "1) Keep wording, tone, and format as close as possible to the candidate answer.\n"
        "2) Do not add any unsupported facts.\n"
        "3) Return only the revised answer text, without explanation.\n\n"
        f"Knowledge:\n{knowledge}\n\n"
        f"Question: {question}\n\n"
        f"Candidate answer:\n{rejected_answer}\n\n"
        f"{ref_block}"
        "Revised answer:\n"
    )
