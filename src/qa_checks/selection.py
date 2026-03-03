from __future__ import annotations

import math
from typing import Any, Dict, List


def score_validation_report(validation_report: Dict[str, Any]) -> float:
    """Score a validation report using only soft ranking signals."""

    soft_metrics = dict(validation_report.get("soft_metrics") or {})
    semantic_margin = soft_metrics.get("semantic_margin")
    supporting_match = float(soft_metrics.get("supporting_fact_match_rate") or 0.0)
    token_f1 = float(soft_metrics.get("correctness_token_f1") or 0.0)
    evidence_count = float(soft_metrics.get("evidence_count") or 0.0)

    semantic_term = -1e6
    if semantic_margin is not None:
        semantic_term = float(semantic_margin) * 100.0

    return float(semantic_term + 10.0 * supporting_match + 5.0 * token_f1 - 0.1 * evidence_count)


def _candidate_rank_key(row: Dict[str, Any]) -> tuple[float, float, float, float, int]:
    validation_report = dict(row.get("validation_report") or {})
    soft_metrics = dict(validation_report.get("soft_metrics") or {})
    semantic_margin = soft_metrics.get("semantic_margin")
    semantic_rank = float(semantic_margin) if semantic_margin is not None else float("-inf")
    supporting_match = float(soft_metrics.get("supporting_fact_match_rate") or 0.0)
    token_f1 = float(soft_metrics.get("correctness_token_f1") or 0.0)
    evidence_count = float(soft_metrics.get("evidence_count") or math.inf)
    candidate_id = int(row.get("candidate_id") or 0)
    return (
        semantic_rank,
        supporting_match,
        token_f1,
        -evidence_count,
        -candidate_id,
    )


def select_best_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select one hard-pass candidate for each example id."""

    grouped: dict[str, list[Dict[str, Any]]] = {}
    for row in rows:
        if not bool((row.get("validation_report") or {}).get("hard_pass")):
            continue
        grouped.setdefault(str(row.get("id") or ""), []).append(row)

    selected: List[Dict[str, Any]] = []
    for _, items in grouped.items():
        best = max(items, key=_candidate_rank_key)
        selected.append(best)
    return selected


def _resolve_confidence_labels(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["medium"]
    if count == 2:
        return ["low", "high"]

    labels: list[str] = []
    low_cut = count // 3
    high_start = count - low_cut
    middle_count = count - low_cut - (count - high_start)
    labels.extend(["low"] * low_cut)
    labels.extend(["medium"] * middle_count)
    labels.extend(["high"] * (count - len(labels)))
    return labels


def assign_quantile_confidence_labels(rows: List[Dict[str, Any]]) -> None:
    """Assign derived confidence on selected answerable rows using semantic-margin quantiles."""

    candidates: list[tuple[float, Dict[str, Any]]] = []
    for row in rows:
        validation_report = dict(row.get("validation_report") or {})
        parsed_output = dict(row.get("parsed_output") or {})
        if not bool(validation_report.get("hard_pass")):
            continue
        if str(parsed_output.get("answerability") or "") != "answerable":
            continue
        semantic_margin = validation_report.get("semantic_margin")
        if semantic_margin is None:
            continue
        candidates.append((float(semantic_margin), row))

    if not candidates:
        return

    candidates.sort(key=lambda item: item[0])
    labels = _resolve_confidence_labels(len(candidates))
    for label, (_, row) in zip(labels, candidates):
        validation_report = dict(row.get("validation_report") or {})
        validation_report["derived_confidence"] = label
        row["validation_report"] = validation_report
