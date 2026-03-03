from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from qa_judge.judge import AnswerJudge


@dataclass
class UnanswerablePrefilterReport:
    """NLI prefilter result for one unanswerable raw sample."""

    keep: bool
    full_binary_decision: str
    margin: float | None
    issues: list[str] = field(default_factory=list)
    judge_payload: dict[str, Any] = field(default_factory=dict)


def build_reference_answer_hypothesis(reference_answer: str) -> str:
    """Build the fixed statement used to test whether the original answer is still supported."""

    answer = str(reference_answer or "").strip()
    return f"The answer is {answer}."


def check_reference_answer_unsupported(
    knowledge: str,
    question: str,
    reference_answer: str,
    judge: AnswerJudge,
) -> UnanswerablePrefilterReport:
    """Keep one sample only if the original reference answer is no longer supported."""

    hypothesis = build_reference_answer_hypothesis(reference_answer)
    judged_rows, _ = judge.judge(
        [
            {
                "knowledge": str(knowledge or "").strip(),
                "question": str(question or "").strip(),
                "answer": hypothesis,
            }
        ]
    )
    judge_payload: Dict[str, Any] = dict((judged_rows[0].get("judge") or {})) if judged_rows else {}
    full_binary = dict(judge_payload.get("full_binary") or {})
    decision = str(full_binary.get("decision") or "no")
    margin = judge_payload.get("margin")
    keep = decision == "no"
    issues: list[str] = []
    if not keep:
        issues.append("Original reference answer is still supported by the constructed knowledge.")
    return UnanswerablePrefilterReport(
        keep=keep,
        full_binary_decision=decision,
        margin=None if margin is None else float(margin),
        issues=issues,
        judge_payload=judge_payload,
    )
