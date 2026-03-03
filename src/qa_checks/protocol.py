from __future__ import annotations

from .report import ProtocolCheckReport
from qa_protocol.normalize import normalize_text
from qa_protocol.refusal import is_refusal_template
from qa_protocol.schema import Answerability, StructuredQaOutput
from qa_protocol.spec import DEFAULT_PROTOCOL_SPEC, ProtocolSpec


def check_protocol_constraints(
    output: StructuredQaOutput,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> ProtocolCheckReport:
    """Check protocol constraints beyond schema validity."""

    issues: list[str] = []
    answer_length = len(output.answer)
    evidence_count = len(output.evidence)
    if normalize_text(output.rationale) == normalize_text(output.answer):
        issues.append("Rationale must not be identical to the final answer.")

    if output.answerability == Answerability.UNANSWERABLE:
        if evidence_count != 0:
            issues.append("Unanswerable output must use an empty evidence list.")
        if not is_refusal_template(output.answer, spec=spec):
            issues.append("Unanswerable output must use an allowed refusal template.")
    else:
        if evidence_count < 1:
            issues.append("Answerable output must include at least one evidence quote.")
        if evidence_count > spec.max_evidence_count:
            issues.append(f"Evidence count exceeds max_evidence_count={spec.max_evidence_count}.")

    return ProtocolCheckReport(
        ok=not issues,
        issues=issues,
        answer_length=answer_length,
        evidence_count=evidence_count,
    )
