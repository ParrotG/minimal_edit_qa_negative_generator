from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from .config import PairBuildConfig
from .prompt import build_qa_premise


def _support_score(row: Dict[str, Any]) -> float:
    judge = row.get("judge") or {}
    return float(judge.get("candidate_entail_primary", 0.0))


def _difficulty_label(accuracy: float, trials: int, min_trials_for_very: int) -> str:
    if accuracy >= 1.0:
        return "very_easy" if trials >= min_trials_for_very else "easy"
    if accuracy <= 0.0:
        return "very_hard" if trials >= min_trials_for_very else "hard"
    return "medium"


def build_pairs(judged_rows: Sequence[Dict[str, Any]], cfg: PairBuildConfig) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Aggregate judged rows by question and construct pair-level records."""

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in judged_rows:
        grouped[str(row["id"])].append(row)

    out: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = defaultdict(int)

    for rid, rows in grouped.items():
        rows = list(rows)
        correct_rows = [r for r in rows if bool((r.get("judge") or {}).get("is_correct", False))]
        wrong_rows = [r for r in rows if not bool((r.get("judge") or {}).get("is_correct", False))]

        trials = len(rows)
        correct = len(correct_rows)
        accuracy = float(correct / max(1, trials))
        label = _difficulty_label(accuracy, trials, cfg.min_trials_for_very_label)

        first = rows[0]
        reference_answer = str(first.get("reference_answer") or "").strip()

        rec: Dict[str, Any] = {
            "id": rid,
            "knowledge": first["knowledge"],
            "question": first["question"],
            "prompt": build_qa_premise(first["knowledge"], first["question"]),
            "eval_question": first["question"],
            "eval_contexts": [first["knowledge"]],
            "task": "qa",
            "trial_count": trials,
            "correct_count": correct,
            "accuracy": accuracy,
            "difficulty_label": label,
            "status": "drop",
            "chosen": "",
            "rejected": "",
            "chosen_origin": "none",
            "pair_meta": {},
        }
        if reference_answer:
            rec["reference_answer"] = reference_answer

        if correct_rows and wrong_rows:
            chosen_row = max(correct_rows, key=_support_score)
            rejected_row = max(wrong_rows, key=_support_score)
            rec["status"] = "ready"
            rec["chosen"] = chosen_row["answer"]
            rec["rejected"] = rejected_row["answer"]
            rec["chosen_origin"] = "self_sample"
            rec["pair_meta"] = {
                "chosen_sample_id": int(chosen_row.get("sample_id", -1)),
                "rejected_sample_id": int(rejected_row.get("sample_id", -1)),
                "chosen_judge": chosen_row.get("judge", {}),
                "rejected_judge": rejected_row.get("judge", {}),
            }

        elif correct_rows and not wrong_rows:
            chosen_row = max(correct_rows, key=_support_score)
            rec["chosen"] = chosen_row["answer"]
            rec["chosen_origin"] = "self_sample"
            rec["pair_meta"] = {
                "chosen_sample_id": int(chosen_row.get("sample_id", -1)),
                "chosen_judge": chosen_row.get("judge", {}),
            }
            if cfg.keep_easy_without_negative:
                rec["status"] = "needs_more_sampling"
            else:
                rec["status"] = "drop"

        elif wrong_rows and not correct_rows:
            # Keep a hard negative candidate and defer chosen-answer creation to repair stage.
            rejected_row = max(wrong_rows, key=_support_score)
            rec["status"] = "needs_repair"
            rec["rejected"] = rejected_row["answer"]
            rec["pair_meta"] = {
                "rejected_sample_id": int(rejected_row.get("sample_id", -1)),
                "rejected_judge": rejected_row.get("judge", {}),
            }

        status_counts[rec["status"]] += 1
        out.append(rec)

    return out, dict(status_counts)
