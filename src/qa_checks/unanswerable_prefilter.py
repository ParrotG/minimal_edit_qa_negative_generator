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
    score: float | None
    flipped_full_binary_decision: str | None = None
    flipped_margin: float | None = None
    used_flipped_check: bool = False
    issues: list[str] = field(default_factory=list)
    judge_payload: dict[str, Any] = field(default_factory=dict)


def build_reference_answer_hypothesis(reference_answer: str) -> str:
    """Build the fixed statement used to test whether the original answer is still supported."""

    answer = str(reference_answer or "").strip()
    return f"The answer is {answer}."


def flip_yes_no_answer(reference_answer: str) -> str | None:
    """Flip a yes/no answer when applicable."""

    answer = str(reference_answer or "").strip().lower()
    if answer == "yes":
        return "no"
    if answer == "no":
        return "yes"
    return None


def check_reference_answer_unsupported(
    knowledge: str,
    question: str,
    reference_answer: str,
    judge: AnswerJudge,
) -> UnanswerablePrefilterReport:
    """Keep one sample only if the original reference answer is no longer supported."""

    def _judge_one(answer_text: str) -> tuple[str, float | None, Dict[str, Any]]:
        hypothesis = build_reference_answer_hypothesis(answer_text)
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
        return decision, (None if margin is None else float(margin)), judge_payload

    decision, margin, judge_payload = _judge_one(reference_answer)
    flipped_answer = flip_yes_no_answer(reference_answer)
    flipped_decision: str | None = None
    flipped_margin: float | None = None
    flipped_payload: Dict[str, Any] | None = None
    used_flipped_check = flipped_answer is not None
    keep = decision == "no"

    if flipped_answer is not None:
        flipped_decision, flipped_margin, flipped_payload = _judge_one(flipped_answer)
        keep = keep and flipped_decision == "no"

    margins = [value for value in (margin, flipped_margin) if value is not None]
    score = float(sum(margins) / len(margins)) if margins else None
    issues: list[str] = []
    if decision != "no":
        issues.append("Original reference answer is still supported by the constructed knowledge.")
    if used_flipped_check and flipped_decision != "no":
        issues.append("Flipped yes/no answer is still supported by the constructed knowledge.")

    payload: Dict[str, Any] = {
        "original": judge_payload,
    }
    if flipped_payload is not None:
        payload["flipped"] = flipped_payload

    return UnanswerablePrefilterReport(
        keep=keep,
        full_binary_decision=decision,
        margin=margin,
        score=score,
        flipped_full_binary_decision=flipped_decision,
        flipped_margin=flipped_margin,
        used_flipped_check=used_flipped_check,
        issues=issues,
        judge_payload=payload,
    )
