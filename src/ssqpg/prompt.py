from __future__ import annotations

from typing import Optional

from nli_judge.prompt import build_qa_premise as _build_nli_qa_premise


def build_qa_premise(knowledge: str, question: str) -> str:
    """Build the canonical QA premise used by NLI verification and DPO prompt."""

    return _build_nli_qa_premise(knowledge=knowledge, question=question)


def build_qa_answer_prefix(knowledge: str, question: str) -> str:
    """Build the canonical answer-generation prefix."""

    return f"{build_qa_premise(knowledge, question)}\nAnswer: "


def build_generation_prompt(knowledge: str, question: str) -> str:
    """Build the prompt used for self-sampling answers."""

    return build_qa_answer_prefix(knowledge, question)


def build_repair_prompt(
    knowledge: str,
    question: str,
    rejected_answer: str,
    reference_answer: Optional[str] = None,
) -> str:
    """Build a minimal-edit repair prompt for hard samples."""

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
