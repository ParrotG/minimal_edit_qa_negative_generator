from __future__ import annotations

from typing import Any, Sequence

from qa_protocol.normalize import normalize_quote
from qa_protocol.schema import Answerability, StructuredQaOutput
from qa_protocol.spec import DEFAULT_PROTOCOL_SPEC, ProtocolSpec

from .report import EvidenceCheckReport


def check_evidence_quotes(
    *,
    output: StructuredQaOutput,
    knowledge: str,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> EvidenceCheckReport:
    """Check evidence quote quality and substring validity."""

    issues: list[str] = []
    normalized_knowledge = normalize_quote(knowledge)
    normalized_quotes: list[str] = []
    matched = 0

    for idx, item in enumerate(output.evidence):
        quote = str(item.quote or "").strip()
        normalized_quote = normalize_quote(quote)
        normalized_quotes.append(normalized_quote)

        if len(quote) < spec.min_quote_chars:
            issues.append(f"Evidence quote #{idx} is shorter than min_quote_chars={spec.min_quote_chars}.")
        if len(quote) > spec.max_quote_chars:
            issues.append(f"Evidence quote #{idx} exceeds max_quote_chars={spec.max_quote_chars}.")
        if normalized_quote and normalized_quote in normalized_knowledge:
            matched += 1
        else:
            issues.append(f"Evidence quote #{idx} is not a substring of the knowledge.")

    if len(set(normalized_quotes)) != len(normalized_quotes):
        issues.append("Evidence quotes must be unique after normalization.")

    if output.answerability == Answerability.UNANSWERABLE and output.evidence:
        issues.append("Unanswerable output must not include evidence quotes.")

    total = max(1, len(output.evidence))
    return EvidenceCheckReport(
        ok=not issues,
        issues=issues,
        quote_match_rate=float(matched / total),
        unique_quotes=len(set(normalized_quotes)),
    )


def check_evidence_against_supporting_facts(
    *,
    output: StructuredQaOutput,
    supporting_sentences: Sequence[dict[str, Any]],
) -> EvidenceCheckReport:
    """Check whether answerable evidence is grounded within gold supporting facts."""

    if output.answerability == Answerability.UNANSWERABLE:
        return EvidenceCheckReport(ok=True, issues=[], quote_match_rate=1.0, unique_quotes=0)

    normalized_supports = [
        normalize_quote(str(item.get("sentence") or ""))
        for item in supporting_sentences
        if str(item.get("sentence") or "").strip()
    ]
    issues: list[str] = []
    matched = 0
    normalized_quotes: list[str] = []

    for idx, item in enumerate(output.evidence):
        normalized_quote = normalize_quote(str(item.quote or ""))
        normalized_quotes.append(normalized_quote)
        if any(normalized_quote and normalized_quote in support for support in normalized_supports):
            matched += 1
        else:
            issues.append(f"Evidence quote #{idx} is outside the gold supporting facts.")

    total = max(1, len(output.evidence))
    return EvidenceCheckReport(
        ok=not issues,
        issues=issues,
        quote_match_rate=float(matched / total),
        unique_quotes=len(set(normalized_quotes)),
    )
