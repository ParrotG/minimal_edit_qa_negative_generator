from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from .config import PairSelectConfig
from .prompt import build_qa_premise
from .text import length_ratio, normalized_edit_distance


def _support_score(row: Dict[str, Any]) -> float:
    judge = row.get("judge") or {}
    return float(judge.get("candidate_entail_primary", 0.0))


def _group_label(correct_count: int, wrong_count: int) -> str:
    if correct_count > 0 and wrong_count == 0:
        return "easy"
    if correct_count > 0 and wrong_count > 0:
        return "medium"
    if correct_count == 0 and wrong_count > 0:
        return "hard"
    return "empty"


def summarize_question_groups(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Summarize per-question correctness distribution after single-answer filtering."""

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)

    out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "empty": 0}

    for rid, q_rows in grouped.items():
        pos = [r for r in q_rows if bool((r.get("judge") or {}).get("is_correct", False))]
        neg = [r for r in q_rows if not bool((r.get("judge") or {}).get("is_correct", False))]
        label = _group_label(correct_count=len(pos), wrong_count=len(neg))
        counts[label] = counts.get(label, 0) + 1

        first = q_rows[0]
        trial_count = len(q_rows)
        correct_count = len(pos)
        out.append(
            {
                "id": rid,
                "knowledge": first["knowledge"],
                "question": first["question"],
                "reference_answer": first.get("reference_answer"),
                "trial_count": trial_count,
                "correct_count": correct_count,
                "wrong_count": len(neg),
                "accuracy": float(correct_count / max(1, trial_count)),
                "group_label": label,
                "positives": pos,
                "negatives": neg,
            }
        )

    return out, counts


def _choose_positive(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return max(candidates, key=_support_score)


def _negative_pair_score(chosen_answer: str, negative_row: Dict[str, Any], cfg: PairSelectConfig) -> Tuple[float, Dict[str, float]]:
    entail = _support_score(negative_row)
    edit_proximity = max(0.0, min(1.0, 1.0 - normalized_edit_distance(chosen_answer, str(negative_row.get("answer") or ""))))
    lr = length_ratio(chosen_answer, str(negative_row.get("answer") or ""))
    length_proximity = max(0.0, min(1.0, 1.0 - abs(1.0 - lr)))
    score = (
        cfg.hard_negative_entail_weight * entail
        + cfg.hard_negative_edit_proximity_weight * edit_proximity
        + cfg.hard_negative_length_proximity_weight * length_proximity
    )
    parts = {
        "entail": float(entail),
        "edit_proximity": float(edit_proximity),
        "length_proximity": float(length_proximity),
    }
    return float(score), parts


def _choose_negative(chosen_answer: str, candidates: Sequence[Dict[str, Any]], cfg: PairSelectConfig) -> Tuple[Dict[str, Any], Dict[str, float]]:
    scored: List[Tuple[Dict[str, Any], float, Dict[str, float]]] = []
    for cand in candidates:
        score, parts = _negative_pair_score(chosen_answer=chosen_answer, negative_row=cand, cfg=cfg)
        scored.append((cand, score, parts))
    best = max(scored, key=lambda x: x[1])
    return best[0], {"pair_score": float(best[1]), **best[2]}


def build_pairs_v2(rows: Sequence[Dict[str, Any]], cfg: PairSelectConfig) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    """Build pair records after single-answer filtering and optional hard augmentation."""

    q_groups, group_counts = summarize_question_groups(rows)
    status_counts: Dict[str, int] = defaultdict(int)
    out: List[Dict[str, Any]] = []

    for group in q_groups:
        rid = group["id"]
        positives = list(group["positives"])
        negatives = list(group["negatives"])
        label = str(group["group_label"])
        first = positives[0] if positives else (negatives[0] if negatives else None)
        if first is None:
            continue

        rec: Dict[str, Any] = {
            "id": rid,
            "task": "qa",
            "knowledge": first["knowledge"],
            "question": first["question"],
            "prompt": build_qa_premise(first["knowledge"], first["question"]),
            "eval_question": first["question"],
            "eval_contexts": [first["knowledge"]],
            "trial_count": int(group["trial_count"]),
            "correct_count": int(group["correct_count"]),
            "accuracy": float(group["accuracy"]),
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
                "chosen_judge": chosen_row.get("judge", {}),
                "rejected_judge": rejected_row.get("judge", {}),
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
                    "chosen_judge": chosen_row.get("judge", {}),
                }
        elif label == "hard":
            if cfg.keep_unresolved_hard:
                rec["status"] = "needs_positive"
                rejected_row = max(negatives, key=_support_score)
                rec["rejected"] = rejected_row["answer"]
                rec["pair_meta"] = {
                    "rejected_sample_id": int(rejected_row.get("sample_id", -1)),
                    "rejected_judge": rejected_row.get("judge", {}),
                }
            else:
                rec["status"] = "drop_hard"
        else:
            rec["status"] = "drop_empty"

        status_counts[rec["status"]] += 1
        out.append(rec)

    return out, dict(status_counts), group_counts
