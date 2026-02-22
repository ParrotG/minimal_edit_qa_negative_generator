from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


def _row_decision(row: Dict[str, Any]) -> str:
    judge = row.get("judge") or {}
    reject = judge.get("reject_aware") or {}
    decision = str(reject.get("decision") or "").strip().lower()
    if decision in {"yes", "no", "abstain"}:
        return decision
    return "yes" if bool(judge.get("is_correct", False)) else "no"


def _group_label(correct_count: int, wrong_count: int) -> str:
    if correct_count > 0 and wrong_count == 0:
        return "easy"
    if correct_count > 0 and wrong_count > 0:
        return "medium"
    if correct_count == 0 and wrong_count > 0:
        return "hard"
    return "empty"


def summarize_question_groups(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Summarize per-question positive/negative distribution after answer filtering."""

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)

    out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "empty": 0}

    for rid, q_rows in grouped.items():
        positives = [r for r in q_rows if _row_decision(r) == "yes"]
        negatives = [r for r in q_rows if _row_decision(r) == "no"]
        abstains = [r for r in q_rows if _row_decision(r) == "abstain"]
        label = _group_label(correct_count=len(positives), wrong_count=len(negatives))
        counts[label] = counts.get(label, 0) + 1

        first = q_rows[0]
        trial_count = len(q_rows)
        correct_count = len(positives)
        out.append(
            {
                "id": rid,
                "knowledge": first["knowledge"],
                "question": first["question"],
                "reference_answer": first.get("reference_answer"),
                "trial_count": trial_count,
                "correct_count": correct_count,
                "wrong_count": len(negatives),
                "abstain_count": len(abstains),
                "accuracy": float(correct_count / max(1, trial_count)),
                "group_label": label,
                "positives": positives,
                "negatives": negatives,
                "abstains": abstains,
            }
        )

    return out, counts
