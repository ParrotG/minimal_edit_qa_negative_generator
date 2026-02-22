from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from .config import PairSelectConfig
from .text import length_ratio, normalized_edit_distance


def _support_score(row: Dict[str, Any]) -> float:
    judge = row.get("judge") or {}
    return float(judge.get("margin", 0.0))


def _judge_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep a compact judge snapshot instead of copying the whole judge object."""

    judge = row.get("judge") or {}
    return {
        "margin": judge.get("margin"),
        "full_binary": (judge.get("full_binary") or {}).get("decision"),
        "reject_aware": (judge.get("reject_aware") or {}).get("decision"),
        "qa_consistent": judge.get("qa_consistent"),
        "is_correct": judge.get("is_correct"),
    }


def _choose_positive(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return max(candidates, key=_support_score)


def _low_margin_score(margin: float) -> float:
    # Lower margin should be ranked higher for negative selection.
    return 1.0 / (1.0 + math.exp(margin))


def _negative_pair_score(chosen_answer: str, negative_row: Dict[str, Any], cfg: PairSelectConfig) -> Tuple[float, Dict[str, float]]:
    margin = _support_score(negative_row)
    low_margin = _low_margin_score(margin)
    edit_proximity = max(0.0, min(1.0, 1.0 - normalized_edit_distance(chosen_answer, str(negative_row.get("answer") or ""))))
    ratio = length_ratio(chosen_answer, str(negative_row.get("answer") or ""))
    length_proximity = max(0.0, min(1.0, 1.0 - abs(1.0 - ratio)))

    score = (
        cfg.hard_negative_low_margin_weight * low_margin
        + cfg.hard_negative_edit_proximity_weight * edit_proximity
        + cfg.hard_negative_length_proximity_weight * length_proximity
    )
    parts = {
        "margin": float(margin),
        "low_margin": float(low_margin),
        "edit_proximity": float(edit_proximity),
        "length_proximity": float(length_proximity),
    }
    return float(score), parts


def _choose_negative(chosen_answer: str, candidates: Sequence[Dict[str, Any]], cfg: PairSelectConfig) -> Tuple[Dict[str, Any], Dict[str, float]]:
    scored: List[Tuple[Dict[str, Any], float, Dict[str, float]]] = []
    for candidate in candidates:
        score, parts = _negative_pair_score(chosen_answer=chosen_answer, negative_row=candidate, cfg=cfg)
        scored.append((candidate, score, parts))
    best = max(scored, key=lambda x: x[1])
    return best[0], {"pair_score": float(best[1]), **best[2]}


def _status_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in records:
        counts[str(row.get("status") or "unknown")] += 1
    return dict(counts)


def build_pairs(groups: Sequence[Dict[str, Any]], cfg: PairSelectConfig) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build one pair per question from already-grouped question records."""

    out: List[Dict[str, Any]] = []

    for group in groups:
        positives = list(group.get("positives") or [])
        negatives = list(group.get("negatives") or [])
        label = str(group.get("group_label") or "empty")
        first = positives[0] if positives else (negatives[0] if negatives else None)
        if first is None:
            continue

        rec: Dict[str, Any] = {
            "id": group["id"],
            "task": "qa",
            "knowledge": first["knowledge"],
            "question": first["question"],
            "trial_count": int(group.get("trial_count", 0)),
            "correct_count": int(group.get("correct_count", 0)),
            "accuracy": float(group.get("accuracy", 0.0)),
            "difficulty_label": label,
            "status": "drop",
            "chosen": "",
            "rejected": "",
            "chosen_origin": "none",
            "pair_meta": {},
        }
        reference_answer = str(first.get("reference_answer") or "").strip()
        if reference_answer:
            rec["reference_answer"] = reference_answer

        if label == "medium":
            chosen_row = _choose_positive(positives)
            rejected_row, neg_parts = _choose_negative(chosen_answer=chosen_row["answer"], candidates=negatives, cfg=cfg)
            rec["status"] = "ready"
            rec["chosen"] = chosen_row["answer"]
            rec["rejected"] = rejected_row["answer"]
            rec["chosen_origin"] = "self_sample"
            rec["pair_meta"] = {
                "chosen_sample_id": int(chosen_row.get("sample_id", -1)),
                "rejected_sample_id": int(rejected_row.get("sample_id", -1)),
                "chosen_judge_summary": _judge_summary(chosen_row),
                "rejected_judge_summary": _judge_summary(rejected_row),
                "negative_pair_score": neg_parts,
            }
        elif label == "easy":
            if cfg.drop_easy:
                rec["status"] = "drop_easy"
            else:
                rec["status"] = "needs_negative"
                chosen_row = _choose_positive(positives)
                rec["chosen"] = chosen_row["answer"]
                rec["chosen_origin"] = "self_sample"
                rec["pair_meta"] = {
                    "chosen_sample_id": int(chosen_row.get("sample_id", -1)),
                    "chosen_judge_summary": _judge_summary(chosen_row),
                }
        elif label == "hard":
            if cfg.keep_unresolved_hard:
                rec["status"] = "needs_positive"
                rejected_row = max(negatives, key=_support_score)
                rec["rejected"] = rejected_row["answer"]
                rec["pair_meta"] = {
                    "rejected_sample_id": int(rejected_row.get("sample_id", -1)),
                    "rejected_judge_summary": _judge_summary(rejected_row),
                }
            else:
                rec["status"] = "drop_hard"
        else:
            rec["status"] = "drop_empty"

        out.append(rec)

    return out, _status_counts(out)
