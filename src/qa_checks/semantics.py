from __future__ import annotations

from typing import Optional

from qa_judge.structured import StructuredAnswerJudge
from qa_protocol.schema import StructuredQaOutput

from .report import SemanticCheckReport


def evaluate_structured_semantics(
    *,
    question: str,
    knowledge: str,
    output: StructuredQaOutput,
    judge: Optional[StructuredAnswerJudge],
    support_window_knowledge: Optional[str] = None,
) -> SemanticCheckReport:
    """Evaluate semantic support for one structured output."""

    if judge is None:
        return SemanticCheckReport(
            ok=None,
            supported=None,
            answer_type_ok=None,
            refusal_ok=None,
            details={},
            issues=[],
        )

    judged_rows, _ = judge.judge_rows(
        [
            {
                "question": question,
                "knowledge": knowledge,
                "support_window_knowledge": support_window_knowledge or knowledge,
                "parsed_output": output.model_dump(mode="json"),
            }
        ]
    )
    payload = judged_rows[0].get("structured_judge") or {}
    answer_judge = payload.get("answer_judge") or {}
    issues = list(payload.get("issues") or [])
    return SemanticCheckReport(
        ok=payload.get("ok"),
        supported=answer_judge.get("is_correct"),
        answer_type_ok=answer_judge.get("qa_consistent"),
        refusal_ok=payload.get("refusal_ok"),
        details=payload,
        issues=issues,
    )
