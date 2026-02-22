from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .config import PairFilterConfig
from prompt import build_qa_premise
from .text import length_ratio, normalized_edit_distance


def _extract_pair_margins(row: Dict[str, Any]) -> Tuple[float, float]:
    meta = row.get("pair_meta") or {}
    chosen = (meta.get("chosen_judge_summary") or {}).get("margin")
    rejected = (meta.get("rejected_judge_summary") or {}).get("margin")
    if chosen is None or rejected is None:
        raise ValueError("pair_meta.*_judge_summary.margin is required for margin-gap filtering")
    return float(chosen), float(rejected)


def apply_pair_filters(
    rows: Sequence[Dict[str, Any]],
    cfg: PairFilterConfig,
    include_prompt: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Apply quality filters and export final DPO-ready pairs."""

    kept: List[Dict[str, Any]] = []
    counts = {
        "num_input": 0,
        "num_kept": 0,
        "drop_not_ready": 0,
        "drop_empty": 0,
        "drop_length": 0,
        "drop_edit": 0,
        "drop_missing_margin": 0,
        "drop_margin_gap": 0,
    }

    for row in rows:
        counts["num_input"] += 1

        if row.get("status") != "ready":
            counts["drop_not_ready"] += 1
            continue

        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        if not chosen or not rejected:
            counts["drop_empty"] += 1
            continue

        trace: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}

        ratio = length_ratio(chosen, rejected)
        keep_len = cfg.min_length_ratio <= ratio <= cfg.max_length_ratio
        trace.append({"filter": "length_ratio", "keep": keep_len, "reason": "ok" if keep_len else "ratio_out_of_range"})
        if not keep_len:
            counts["drop_length"] += 1
            continue
        meta["length_ratio"] = {"ratio": ratio}

        norm = normalized_edit_distance(chosen, rejected)
        keep_edit = cfg.min_norm_edit <= norm <= cfg.max_norm_edit
        trace.append({"filter": "edit_distance", "keep": keep_edit, "reason": "ok" if keep_edit else "edit_out_of_range"})
        if not keep_edit:
            counts["drop_edit"] += 1
            continue
        meta["edit_distance"] = {"norm": norm}

        try:
            chosen_margin, rejected_margin = _extract_pair_margins(row)
        except ValueError:
            counts["drop_missing_margin"] += 1
            continue

        margin_gap = chosen_margin - rejected_margin
        keep_margin = margin_gap >= cfg.min_margin_gap
        trace.append(
            {
                "filter": "margin_gap",
                "keep": keep_margin,
                "reason": "ok" if keep_margin else "margin_gap_too_small",
            }
        )
        if not keep_margin:
            counts["drop_margin_gap"] += 1
            continue
        meta["margin_gap"] = {
            "chosen_margin": chosen_margin,
            "rejected_margin": rejected_margin,
            "gap": margin_gap,
            "min_gap": cfg.min_margin_gap,
        }

        out = {
            "id": row["id"],
            "task": "qa",
            "knowledge": row["knowledge"],
            "question": row["question"],
            "chosen": chosen,
            "rejected": rejected,
            "difficulty_label": row.get("difficulty_label"),
            "trial_count": row.get("trial_count"),
            "correct_count": row.get("correct_count"),
            "accuracy": row.get("accuracy"),
            "chosen_origin": row.get("chosen_origin"),
            "pair_meta": row.get("pair_meta", {}),
            "filter_meta": meta,
            "filter_trace": trace,
        }
        if include_prompt:
            out["prompt"] = build_qa_premise(row["knowledge"], row["question"])
        kept.append(out)

    counts["num_kept"] = len(kept)
    return kept, counts
