from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List

from qa_judge.structured import StructuredAnswerJudge
from qa_protocol.schema import StructuredQaOutput

from .report import SemanticCheckReport


def _empty_semantic_report() -> SemanticCheckReport:
    return SemanticCheckReport(
        ok=None,
        supported=None,
        answer_type_ok=None,
        refusal_ok=None,
        decision=None,
        margin=None,
        details={},
        issues=[],
    )


def evaluate_structured_semantics_batch(
    *,
    rows: Sequence[Dict[str, Any]],
    judge: Optional[StructuredAnswerJudge],
    decision_source: str = "full_binary",
) -> List[SemanticCheckReport]:
    """Evaluate semantic support for many structured outputs in one batch."""

    if not rows:
        return []
    if judge is None:
        return [_empty_semantic_report() for _ in rows]

    judged_rows, _ = judge.judge_rows(rows, decision_source=decision_source)
    reports: List[SemanticCheckReport] = []
    for judged_row in judged_rows:
        payload = judged_row.get("structured_judge") or {}
        answer_judge = payload.get("answer_judge") or {}
        issues = list(payload.get("issues") or [])
        reports.append(
            SemanticCheckReport(
                ok=payload.get("ok"),
                supported=payload.get("supported"),
                answer_type_ok=answer_judge.get("qa_consistent"),
                refusal_ok=payload.get("refusal_ok"),
                decision=payload.get("decision"),
                margin=payload.get("margin"),
                details=payload,
                issues=issues,
            )
        )
    return reports


def evaluate_structured_semantics(
    *,
    question: str,
    knowledge: str,
    output: StructuredQaOutput,
    judge: Optional[StructuredAnswerJudge],
    decision_source: str = "full_binary",
) -> SemanticCheckReport:
    """Evaluate semantic support for one structured output."""

    reports = evaluate_structured_semantics_batch(
        rows=[
            {
                "question": question,
                "knowledge": knowledge,
                "parsed_output": output.model_dump(mode="json"),
            }
        ],
        judge=judge,
        decision_source=decision_source,
    )
    return reports[0] if reports else _empty_semantic_report()
