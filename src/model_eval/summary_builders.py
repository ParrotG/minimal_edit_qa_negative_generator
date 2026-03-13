from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dataio import read_jsonl_list


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    """Read a CSV file into a list of dictionaries."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _model_key(row: Dict[str, Any]) -> Tuple[str, int, str, str, str]:
    return (
        str(row.get("model_tag") or ""),
        int(row.get("model_step") or 0),
        str(row.get("model_path") or ""),
        str(row.get("eval_track") or ""),
        str(row.get("eval_variant") or ""),
    )


def _sample_key(row: Dict[str, Any]) -> Tuple[str, str]:
    source_id = str(row.get("source_id") or "").strip()
    sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
    return source_id, sample_id


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _safe_mean(values: Iterable[float]) -> float:
    seq = [float(value) for value in values]
    if not seq:
        return float("nan")
    return float(sum(seq) / len(seq))


def constraint_pass(
    row: Dict[str, Any],
    *,
    parse_ok_threshold: float,
    protocol_ok_threshold: float,
    evidence_ok_threshold: float,
) -> bool:
    """Evaluate hard validation constraints for one structured checkpoint row."""

    parse_ok = _safe_float(row.get("parse_ok_rate"))
    protocol_ok = _safe_float(row.get("protocol_ok_rate_given_parse_ok"))
    evidence_ok = _safe_float(row.get("evidence_substring_rate_on_usable_pred_answerable"))
    return (
        not math.isnan(parse_ok)
        and not math.isnan(protocol_ok)
        and not math.isnan(evidence_ok)
        and parse_ok >= float(parse_ok_threshold)
        and protocol_ok >= float(protocol_ok_threshold)
        and evidence_ok >= float(evidence_ok_threshold)
    )


def select_best_checkpoint(
    *,
    eval_rows: Sequence[Dict[str, Any]],
    loss_rows: Sequence[Dict[str, Any]],
    parse_ok_threshold: float,
    protocol_ok_threshold: float,
    evidence_ok_threshold: float,
) -> Dict[str, Any]:
    """Select the best checkpoint from validation evaluation rows."""

    loss_by_key = {_model_key(row): dict(row) for row in loss_rows}
    candidate_rows = [dict(row) for row in eval_rows if str(row.get("model_tag") or "").strip() != "base"]
    if not candidate_rows:
        raise RuntimeError("No checkpoint rows were found in the validation structured evaluation curve.")

    passed = [
        row
        for row in candidate_rows
        if constraint_pass(
            row,
            parse_ok_threshold=parse_ok_threshold,
            protocol_ok_threshold=protocol_ok_threshold,
            evidence_ok_threshold=evidence_ok_threshold,
        )
    ]
    active = passed if passed else candidate_rows

    def sort_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
        loss_row = loss_by_key.get(_model_key(row), {})
        correctness = _safe_float(row.get("e2e_reviewed_correctness_success_rate"))
        answerability = _safe_float(row.get("e2e_answerability_accuracy"))
        semantic = _safe_float(row.get("e2e_semantic_success_rate"))
        mean_loss = _safe_float(loss_row.get("mean_loss"))
        if math.isnan(correctness):
            correctness = float("-inf")
        if math.isnan(answerability):
            answerability = float("-inf")
        if math.isnan(semantic):
            semantic = float("-inf")
        if math.isnan(mean_loss):
            mean_loss = float("inf")
        return (
            correctness,
            answerability,
            semantic,
            -mean_loss,
            -int(row.get("model_step") or 0),
        )

    selected = max(active, key=sort_key)
    selected_key = _model_key(selected)
    return {
        "constraint_satisfied": bool(passed),
        "selected_model_tag": selected["model_tag"],
        "selected_model_step": int(selected["model_step"]),
        "selected_model_path": selected["model_path"],
        "selected_eval_track": selected.get("eval_track"),
        "selected_eval_variant": selected.get("eval_variant"),
        "selection_metrics": {
            "e2e_reviewed_correctness_success_rate": selected.get("e2e_reviewed_correctness_success_rate"),
            "e2e_answerability_accuracy": selected.get("e2e_answerability_accuracy"),
            "e2e_semantic_success_rate": selected.get("e2e_semantic_success_rate"),
            "mean_loss": (loss_by_key.get(selected_key) or {}).get("mean_loss"),
        },
        "num_candidates": len(candidate_rows),
        "num_constraint_pass": len(passed),
        "candidate_rows": candidate_rows,
    }


def build_validation_summary_rows(
    *,
    eval_rows: Sequence[Dict[str, Any]],
    loss_rows: Sequence[Dict[str, Any]],
    selection: Dict[str, Any],
    parse_ok_threshold: float,
    protocol_ok_threshold: float,
    evidence_ok_threshold: float,
) -> List[Dict[str, Any]]:
    """Build the validation-only checkpoint comparison table."""

    loss_by_key = {_model_key(row): dict(row) for row in loss_rows if str(row.get("model_tag") or "").strip() != "base"}
    selected_path = str(selection.get("selected_model_path") or "")
    out_rows: List[Dict[str, Any]] = []
    for row in sorted(
        [dict(row) for row in eval_rows if str(row.get("model_tag") or "").strip() != "base"],
        key=lambda item: int(item.get("model_step") or 0),
    ):
        loss_row = loss_by_key.get(_model_key(row), {})
        out_rows.append(
            {
                **row,
                "mean_loss": loss_row.get("mean_loss"),
                "num_used_rows": loss_row.get("num_used_rows"),
                "constraint_satisfied": constraint_pass(
                    row,
                    parse_ok_threshold=parse_ok_threshold,
                    protocol_ok_threshold=protocol_ok_threshold,
                    evidence_ok_threshold=evidence_ok_threshold,
                ),
                "selected_best": str(row.get("model_path") or "") == selected_path,
            }
        )
    return out_rows


def build_test_summary_rows(
    *,
    track_rows: Sequence[Dict[str, Any]],
    detail_rows: Sequence[Dict[str, Any]],
    deepeval_detail_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build the final test summary with e2e DeepEval metrics."""

    track_details_by_model: Dict[Tuple[str, int, str, str, str], List[Dict[str, Any]]] = {}
    for row in detail_rows:
        track_details_by_model.setdefault(_model_key(row), []).append(dict(row))

    deepeval_details_by_model: Dict[Tuple[str, int, str, str, str], List[Dict[str, Any]]] = {}
    for row in deepeval_detail_rows:
        deepeval_details_by_model.setdefault(_model_key(row), []).append(dict(row))

    out_rows: List[Dict[str, Any]] = []
    for row in sorted(
        [dict(item) for item in track_rows],
        key=lambda item: (str(item.get("eval_track") or ""), int(item.get("model_step") or 0), str(item.get("eval_variant") or "")),
    ):
        model_key = _model_key(row)
        track_detail_rows = track_details_by_model.get(model_key, [])
        deepeval_model_rows = deepeval_details_by_model.get(model_key, [])
        deepeval_by_sample = {_sample_key(item): dict(item) for item in deepeval_model_rows}

        answerable_gate_total = 0
        deepeval_scored_on_answerable_gate = 0
        num_e2e_deepeval_success = 0
        num_answerable_deepeval_success = 0
        for detail in track_detail_rows:
            source_is_answerable = bool(detail.get("source_is_answerable"))
            source_is_unanswerable = bool(detail.get("source_is_unanswerable"))
            e2e_answerability_success = bool(detail.get("e2e_answerability_success"))
            answerable_gate = bool(source_is_answerable and e2e_answerability_success)
            deepeval_detail = deepeval_by_sample.get(_sample_key(detail))
            deepeval_success = None if deepeval_detail is None else deepeval_detail.get("success")
            if answerable_gate:
                answerable_gate_total += 1
                if deepeval_success in {True, False}:
                    deepeval_scored_on_answerable_gate += 1

            e2e_deepeval_success = False
            if source_is_unanswerable:
                e2e_deepeval_success = e2e_answerability_success
            elif answerable_gate:
                e2e_deepeval_success = bool(deepeval_success is True)
                num_answerable_deepeval_success += int(e2e_deepeval_success)
            num_e2e_deepeval_success += int(e2e_deepeval_success)

        scores = [
            float(item["score"])
            for item in deepeval_model_rows
            if item.get("score") is not None and not math.isnan(_safe_float(item.get("score")))
        ]
        num_scored = sum(1 for item in deepeval_model_rows if item.get("success") in {True, False})
        deepeval_pass_count = sum(1 for item in deepeval_model_rows if item.get("success") is True)
        num_errors = sum(1 for item in deepeval_model_rows if item.get("error"))
        num_source_answerable = _safe_float(row.get("num_source_answerable"))
        answerable_denominator = 0 if math.isnan(num_source_answerable) else int(num_source_answerable)

        out_rows.append(
            {
                **row,
                "deepeval_num_cases": len(deepeval_model_rows),
                "deepeval_num_scored": num_scored,
                "deepeval_scored_rate": _safe_rate(num_scored, len(deepeval_model_rows)),
                "deepeval_pass_count": deepeval_pass_count,
                "deepeval_pass_rate_on_scored": _safe_rate(deepeval_pass_count, num_scored),
                "deepeval_num_errors": num_errors,
                "deepeval_hallucination_score_mean": _safe_mean(scores),
                "deepeval_scored_rate_on_answerable_gate": _safe_rate(
                    deepeval_scored_on_answerable_gate,
                    answerable_gate_total,
                ),
                "num_e2e_deepeval_success": num_e2e_deepeval_success,
                "e2e_deepeval_pass_rate": _safe_rate(num_e2e_deepeval_success, len(track_detail_rows)),
                "num_answerable_deepeval_success": num_answerable_deepeval_success,
                "answerable_deepeval_pass_rate": _safe_rate(
                    num_answerable_deepeval_success,
                    answerable_denominator,
                ),
            }
        )
    return out_rows


def load_jsonl_rows(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Read and concatenate multiple JSONL files."""

    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl_list(path))
    return rows
