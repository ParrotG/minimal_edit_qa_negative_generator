from __future__ import annotations

from typing import Any, Dict, List

from .report import ValidationReport


def derive_confidence_label(report: ValidationReport) -> str | None:
    """Derive a confidence label for one validated candidate."""

    if not report.parse_ok:
        return None
    if not report.overall_ok:
        return "low"

    correctness = report.correctness
    semantics = report.semantics
    if correctness is not None and correctness.exact_match and (semantics is None or semantics.ok is True):
        return "high"
    if correctness is not None and correctness.semantic_match:
        return "medium"
    if semantics is not None and semantics.ok is True:
        return "medium"
    return "low"


def score_validation_report(report: ValidationReport) -> float:
    """Score a validation report for candidate selection."""

    score = 0.0
    if report.parse_ok:
        score += 1.0
    if report.protocol.ok:
        score += 2.0
    if report.evidence.ok:
        score += 2.0
    if report.answerability_match is True:
        score += 1.0
    if report.correctness is not None:
        score += 2.0 * float(report.correctness.token_f1)
        if report.correctness.exact_match:
            score += 1.0
    if report.semantics is not None and report.semantics.ok is True:
        score += 1.5
    if report.overall_ok:
        score += 2.0
    return float(score)


def select_best_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select one best candidate for each example id."""

    grouped: dict[str, list[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("id") or ""), []).append(row)

    selected: List[Dict[str, Any]] = []
    for _, items in grouped.items():
        best = max(items, key=lambda item: float((item.get("validation_report") or {}).get("selection_score", -1e9)))
        selected.append(best)
    return selected
