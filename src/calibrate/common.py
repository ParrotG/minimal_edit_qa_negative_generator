from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

try:
    from src.dataio import read_jsonl_list
    from src.model_eval.common import (
        extract_structured_payload,
        is_flat_answerable_for_content_eval,
        is_flat_matcher_applicable,
        is_structured_answerable_for_semantic_eval,
        is_structured_matcher_applicable,
    )
    from src.prompt import build_qa_premise
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from dataio import read_jsonl_list
    from model_eval.common import (
        extract_structured_payload,
        is_flat_answerable_for_content_eval,
        is_flat_matcher_applicable,
        is_structured_answerable_for_semantic_eval,
        is_structured_matcher_applicable,
    )
    from prompt import build_qa_premise


def normalize_list(value: Any) -> List[str]:
    """Normalize value into a list of non-empty strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def safe_int(value: Any, default: int) -> int:
    """Best-effort integer parsing with fallback."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort float parsing with optional fallback."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_dataset_split(data_path: str, split: str) -> Dataset:
    """Load a dataset split from save_to_disk directory or JSON/JSONL file."""

    if os.path.isdir(data_path):
        ds_obj = load_from_disk(data_path)
        if isinstance(ds_obj, DatasetDict):
            if split not in ds_obj:
                raise ValueError(f"Split '{split}' not found in dataset dict. Available: {list(ds_obj.keys())}")
            return ds_obj[split]
        if isinstance(ds_obj, Dataset):
            return ds_obj
        raise ValueError(f"Unsupported dataset object from {data_path}: {type(ds_obj)}")
    lower_path = str(data_path).lower()
    if lower_path.endswith(".jsonl"):
        return Dataset.from_list(read_jsonl_list(data_path))
    if lower_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return Dataset.from_list(payload)
        raise ValueError(f"JSON calibration input must be a list of rows: {data_path}")
    return load_dataset("json", data_files=data_path, split="train")


def parse_value_set(text: str) -> Set[str]:
    """Parse comma-separated values into normalized lowercase set."""

    return {token.strip().lower() for token in text.split(",") if token.strip()}


def parse_label_to_bool(value: Any, positive_values: Set[str], negative_values: Set[str]) -> Optional[bool]:
    """Parse arbitrary label value to boolean according to provided vocab."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    label = str(value).strip().lower()
    if not label:
        return None
    if label in positive_values:
        return True
    if label in negative_values:
        return False
    return None


def _extract_question(row: Dict[str, Any], question_field: str) -> str:
    candidates = [
        row.get(question_field),
        row.get("question"),
        row.get("input"),
        row.get("eval_question"),
    ]
    for item in candidates:
        text = str(item or "").strip()
        if text:
            return text
    return ""


def _extract_answer(row: Dict[str, Any], answer_field: str) -> str:
    candidates = [
        row.get(answer_field),
        row.get("raw_answer"),
        row.get("answer"),
        row.get("actual_output"),
    ]
    for item in candidates:
        text = str(item or "").strip()
        if text:
            return text
    return ""


def _extract_knowledge(row: Dict[str, Any], knowledge_field: str, context_field: str) -> str:
    direct = str(row.get(knowledge_field) or "").strip()
    if direct:
        return direct

    contexts = normalize_list(row.get(context_field))
    if contexts:
        return "\n\n".join(contexts)

    for key in ("knowledge", "context", "contexts", "eval_contexts", "retrieval_context"):
        values = normalize_list(row.get(key))
        if values:
            return "\n\n".join(values)
    return ""


def load_labeled_rows(
    *,
    data_path: str,
    split: str,
    seed: int,
    max_samples: int,
    question_field: str,
    answer_field: str,
    knowledge_field: str,
    context_field: str,
    label_field: str,
    positive_values: Set[str],
    negative_values: Set[str],
    allow_unlabeled: bool,
    model_tag_field: str,
    model_step_field: str,
    model_path_field: str,
    sample_id_field: str,
    source_id_field: str,
) -> List[Dict[str, Any]]:
    """Load and normalize external labeled QA rows."""

    ds = load_dataset_split(data_path=data_path, split=split).shuffle(seed=seed)
    rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(ds):
        question = _extract_question(row, question_field=question_field)
        answer = _extract_answer(row, answer_field=answer_field)
        knowledge = _extract_knowledge(row, knowledge_field=knowledge_field, context_field=context_field)
        if not question or not answer or not knowledge:
            continue

        label_bool = parse_label_to_bool(row.get(label_field), positive_values=positive_values, negative_values=negative_values)
        if label_bool is None and not allow_unlabeled:
            continue

        sample_id = safe_int(row.get(sample_id_field), idx)
        source_id = str(row.get(source_id_field) or sample_id)
        model_tag = str(row.get(model_tag_field) or "model")
        model_step = safe_int(row.get(model_step_field), 0)
        model_path = str(row.get(model_path_field) or model_tag)

        rows.append(
            {
                "model_tag": model_tag,
                "model_step": model_step,
                "model_path": model_path,
                "sample_id": sample_id,
                "source_id": source_id,
                "question": question,
                "knowledge": knowledge,
                "answer": answer,
                "gold_supported": label_bool,
                "raw_label": row.get(label_field),
            }
        )

        if max_samples > 0 and len(rows) >= max_samples:
            break

    return rows


def group_rows_by_model(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int, str], List[Dict[str, Any]]]:
    """Group rows by model identifiers."""

    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model_tag"]), int(row["model_step"]), str(row["model_path"]))
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    """Write rows to CSV with union fieldnames."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    """Mean with NaN fallback for empty input."""

    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def binary_stats(preds: Sequence[bool], refs: Sequence[bool]) -> Dict[str, float]:
    """Compute binary classification metrics."""

    n = len(preds)
    if n == 0:
        return {
            "accuracy": float("nan"),
            "cohen_kappa": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    tp = sum(1 for p, r in zip(preds, refs) if p and r)
    tn = sum(1 for p, r in zip(preds, refs) if (not p) and (not r))
    fp = sum(1 for p, r in zip(preds, refs) if p and (not r))
    fn = sum(1 for p, r in zip(preds, refs) if (not p) and r)

    accuracy = (tp + tn) / n
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

    p_yes_pred = sum(1 for x in preds if x) / n
    p_yes_ref = sum(1 for x in refs if x) / n
    p_e = p_yes_pred * p_yes_ref + (1.0 - p_yes_pred) * (1.0 - p_yes_ref)
    kappa = float("nan") if abs(1.0 - p_e) < 1e-12 else (accuracy - p_e) / (1.0 - p_e)

    return {
        "accuracy": float(accuracy),
        "cohen_kappa": float(kappa),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def build_values(explicit: str, v_min: float, v_max: float, v_step: float, name: str) -> List[float]:
    """Build grid values from explicit list or range triplet."""

    if explicit.strip():
        values = sorted(set(float(token.strip()) for token in explicit.split(",") if token.strip()))
        if not values:
            raise ValueError(f"{name} explicit values are empty.")
        return values

    if v_step <= 0.0:
        raise ValueError(f"{name} step must be > 0, got {v_step}")
    if v_max < v_min:
        raise ValueError(f"{name} max must be >= min, got min={v_min}, max={v_max}")

    values: List[float] = []
    cur = v_min
    while cur <= v_max + 1e-12:
        values.append(round(cur, 10))
        cur += v_step
    return values


def iter_grid(a_values: Iterable[float], b_values: Iterable[float]) -> Iterable[Tuple[float, float]]:
    """Cartesian product helper."""

    for a in a_values:
        for b in b_values:
            yield float(a), float(b)


def parse_human_label(value: Any) -> Optional[bool]:
    """Parse human annotation label to a boolean value."""

    return parse_label_to_bool(
        value,
        positive_values={"1", "true", "yes", "y", "supported", "correct", "equivalent", "match"},
        negative_values={"0", "false", "no", "n", "unsupported", "incorrect", "not_equivalent", "mismatch"},
    )


def _extract_reference_answer(row: Dict[str, Any]) -> str:
    direct = str(row.get("reference_answer") or "").strip()
    if direct:
        return direct
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("reference_answer") or "").strip()
    return ""


def extract_final_answer_text(row: Dict[str, Any]) -> str:
    """Extract the final answer text from either structured or flat rows."""

    structured = extract_structured_payload(row)
    if structured:
        return str(structured.get("answer") or "").strip()
    return _extract_answer(row, answer_field="answer")


def extract_rationale_text(row: Dict[str, Any]) -> str:
    """Extract rationale text from a structured payload when available."""

    structured = extract_structured_payload(row)
    if not structured:
        return ""
    return str(structured.get("rationale") or "").strip()


def extract_evidence_text(row: Dict[str, Any]) -> str:
    """Extract a human-readable evidence string from structured output."""

    structured = extract_structured_payload(row)
    if not structured:
        return ""
    evidence = structured.get("evidence")
    if not isinstance(evidence, list):
        return ""
    quotes: List[str] = []
    for item in evidence:
        if isinstance(item, dict):
            quote = str(item.get("quote") or "").strip()
            if quote:
                quotes.append(quote)
    return "\n".join(quotes).strip()


def is_flat_detail_row(row: Dict[str, Any]) -> bool:
    """Return whether a row comes from flat evaluation with extraction state."""

    if extract_structured_payload(row) is not None:
        return False
    return any(
        key in row
        for key in ("extraction_parse_ok", "refusal_detected", "extracted_answer", "pred_answerability")
    ) or (
        bool(str(row.get("eval_track") or "").strip() == "base_task")
        or bool(str(row.get("raw_answer") or "").strip())
        or (
            bool(str(row.get("question") or "").strip())
            and bool(str(row.get("knowledge") or "").strip())
            and bool(str(row.get("answer") or "").strip())
        )
    )


def is_structured_detail_row(row: Dict[str, Any]) -> bool:
    """Return whether a row comes from structured evaluation."""

    if extract_structured_payload(row) is not None:
        return True
    return any(key in row for key in ("parse_ok", "protocol_ok", "protocol_report", "parsed_output", "raw_output"))


def has_flat_eval_state(row: Dict[str, Any]) -> bool:
    """Return whether a flat row already passed through extraction/refusal evaluation."""

    return any(
        key in row
        for key in ("extraction_parse_ok", "refusal_detected", "extracted_answer", "pred_answerability")
    )


def build_pack_row(*, row: Dict[str, Any], task_type: str, pack_id: str, idx: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build one annotation-pack row for a task type from a mixed evaluation row."""

    question = _extract_question(row, question_field="question")
    knowledge = _extract_knowledge(row, knowledge_field="knowledge", context_field="context")
    reference_answer = _extract_reference_answer(row)
    rationale = extract_rationale_text(row)
    evidence_text = extract_evidence_text(row)
    structured_payload = extract_structured_payload(row)
    is_flat_row = is_flat_detail_row(row)
    is_structured_row = is_structured_detail_row(row)

    answer = ""

    premise_text = ""
    hypothesis_text = ""
    if task_type == "nli_flat":
        if not is_flat_row:
            return None, "missing_eval_state"
        if not has_flat_eval_state(row):
            return None, "missing_eval_state"
        if not is_flat_answerable_for_content_eval(row):
            skipped = "refusal_or_unanswerable" if row.get("refusal_detected") is True or str(row.get("pred_answerability") or "") == "unanswerable" else "missing_fields"
            return None, skipped
        answer = _extract_answer(row, answer_field="raw_answer")
        if not answer:
            answer = str(row.get("extracted_answer") or "").strip()
        if not question or not knowledge or not answer:
            return None, "missing_fields"
        premise_text = build_qa_premise(knowledge=knowledge, question=question)
        hypothesis_text = answer
    elif task_type == "nli_structured":
        if not is_structured_row:
            return None, "missing_fields"
        if not is_structured_answerable_for_semantic_eval(row):
            return None, "refusal_or_unanswerable"
        answer = str((structured_payload or {}).get("answer") or "").strip()
        if not question or not evidence_text or not rationale or not answer:
            return None, "missing_fields"
        premise_text = build_qa_premise(knowledge=evidence_text, question=question)
        hypothesis_text = f"{rationale}\nTherefore the answer is {answer}"
    elif task_type == "matcher":
        if not is_flat_row and not is_structured_row:
            return None, "missing_eval_state"
        if is_flat_row and not is_structured_row and not has_flat_eval_state(row):
            return None, "missing_eval_state"
        flat_applicable = is_flat_row and is_flat_matcher_applicable(row)
        structured_applicable = is_structured_row and is_structured_matcher_applicable(row)
        if not flat_applicable and not structured_applicable:
            skipped = "matcher_not_applicable" if row.get("correctness_ok") is not None else "refusal_or_unanswerable"
            return None, skipped
        answer = str(row.get("extracted_answer") or "").strip() if flat_applicable else str((structured_payload or {}).get("answer") or "").strip()
        if not answer:
            answer = extract_final_answer_text(row)
        if not question or not knowledge or not reference_answer or not answer:
            return None, "missing_fields"
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    sample_id = safe_int(row.get("sample_id"), idx)
    source_id = str(row.get("source_id") or sample_id)
    model_tag = str(row.get("model_tag") or "model")
    model_step = safe_int(row.get("model_step"), 0)
    model_path = str(row.get("model_path") or model_tag)
    input_source_path = str(row.get("input_source_path") or "").strip()
    return ({
        "task_type": task_type,
        "pack_id": pack_id,
        "sample_id": sample_id,
        "source_id": source_id,
        "model_tag": model_tag,
        "model_step": model_step,
        "model_path": model_path,
        "input_source_path": input_source_path,
        "question": question,
        "knowledge": knowledge,
        "reference_answer": reference_answer,
        "answer": answer,
        "evidence_text": evidence_text,
        "rationale": rationale,
        "premise_text": premise_text,
        "hypothesis_text": hypothesis_text,
        "human_label": None,
        "human_notes": "",
    }, None)


def _build_task_candidate_rows(
    *,
    rows: Sequence[Dict[str, Any]],
    task_type: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pre-filter rows for a calibration task before deterministic sampling."""

    candidates: List[Dict[str, Any]] = []
    counts = {
        "num_candidate_rows": 0,
        "num_skipped_wrong_row_type": 0,
    }
    for row in rows:
        is_flat_row = is_flat_detail_row(row)
        is_structured_row = is_structured_detail_row(row)
        keep = False
        if task_type == "nli_flat":
            keep = is_flat_row
        elif task_type == "nli_structured":
            keep = is_structured_row
        elif task_type == "matcher":
            keep = is_flat_row or is_structured_row
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        if keep:
            candidates.append(dict(row))
            counts["num_candidate_rows"] += 1
        else:
            counts["num_skipped_wrong_row_type"] += 1
    return candidates, counts


def build_annotation_pack_rows(
    *,
    rows: Sequence[Dict[str, Any]],
    task_types: Sequence[str],
    max_samples_per_task: int,
    seed: int,
    pack_id: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build a mixed annotation pack with deterministic per-task sampling."""

    out_rows: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "pack_id": pack_id,
        "num_input_rows": len(rows),
        "tasks": {},
        "input_sources": {
            str(row.get("input_source_path") or "<unknown>"): {
                "num_input_rows": sum(
                    1
                    for candidate in rows
                    if str(candidate.get("input_source_path") or "<unknown>") == str(row.get("input_source_path") or "<unknown>")
                ),
                "tasks": {},
            }
            for row in rows
        },
    }

    for task_type in task_types:
        candidate_rows, candidate_metrics = _build_task_candidate_rows(rows=rows, task_type=task_type)
        if candidate_rows:
            shuffled_rows = [dict(row) for row in candidate_rows]
            random.Random(seed).shuffle(shuffled_rows)
        else:
            shuffled_rows = []
        kept = 0
        skipped_missing = 0
        skipped_missing_eval_state = 0
        skipped_refusal_or_unanswerable = 0
        skipped_matcher_not_applicable = 0
        for idx, row in enumerate(shuffled_rows):
            source_key = str(row.get("input_source_path") or "<unknown>")
            source_metrics = metrics["input_sources"][source_key]
            source_task_metrics = source_metrics["tasks"].setdefault(
                task_type,
                {
                    "num_kept": 0,
                    "num_skipped_missing_fields": 0,
                    "num_skipped_missing_eval_state": 0,
                    "num_skipped_refusal_or_unanswerable": 0,
                    "num_skipped_matcher_not_applicable": 0,
                },
            )
            pack_row, skipped_reason = build_pack_row(row=row, task_type=task_type, pack_id=pack_id, idx=idx)
            if pack_row is None:
                if skipped_reason == "missing_eval_state":
                    skipped_missing_eval_state += 1
                    source_task_metrics["num_skipped_missing_eval_state"] += 1
                elif skipped_reason == "refusal_or_unanswerable":
                    skipped_refusal_or_unanswerable += 1
                    source_task_metrics["num_skipped_refusal_or_unanswerable"] += 1
                elif skipped_reason == "matcher_not_applicable":
                    skipped_matcher_not_applicable += 1
                    source_task_metrics["num_skipped_matcher_not_applicable"] += 1
                else:
                    skipped_missing += 1
                    source_task_metrics["num_skipped_missing_fields"] += 1
                continue
            out_rows.append(pack_row)
            kept += 1
            source_task_metrics["num_kept"] += 1
            if max_samples_per_task > 0 and kept >= max_samples_per_task:
                break
        metrics["tasks"][task_type] = {
            **candidate_metrics,
            "num_kept": kept,
            "num_skipped_missing_fields": skipped_missing,
            "num_skipped_missing_eval_state": skipped_missing_eval_state,
            "num_skipped_refusal_or_unanswerable": skipped_refusal_or_unanswerable,
            "num_skipped_matcher_not_applicable": skipped_matcher_not_applicable,
        }

    return out_rows, metrics


def load_annotation_rows(data_path: str, split: str, seed: int, max_samples: int) -> List[Dict[str, Any]]:
    """Load human-annotated pack rows and normalize the binary human label."""

    ds = load_dataset_split(data_path=data_path, split=split).shuffle(seed=seed)
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        task_type = str(row.get("task_type") or "").strip()
        if not task_type:
            continue
        normalized = dict(row)
        normalized["task_type"] = task_type
        normalized["sample_id"] = safe_int(row.get("sample_id"), idx)
        normalized["source_id"] = str(row.get("source_id") or normalized["sample_id"])
        normalized["model_tag"] = str(row.get("model_tag") or "model")
        normalized["model_step"] = safe_int(row.get("model_step"), 0)
        normalized["model_path"] = str(row.get("model_path") or normalized["model_tag"])
        normalized["human_label_bool"] = parse_human_label(row.get("human_label"))
        rows.append(normalized)
        if max_samples > 0 and len(rows) >= max_samples:
            break
    return rows


def group_rows_by_task(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str], List[Dict[str, Any]]]:
    """Group rows by task type only."""

    grouped: Dict[Tuple[str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("task_type") or ""),)
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def group_rows_by_task_and_model(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, int, str], List[Dict[str, Any]]]:
    """Group rows by task type and model identifiers."""

    grouped: Dict[Tuple[str, str, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("task_type") or ""),
            str(row.get("model_tag") or "model"),
            safe_int(row.get("model_step"), 0),
            str(row.get("model_path") or row.get("model_tag") or "model"),
        )
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def group_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    group_by: str,
) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    """Group annotation rows according to the requested aggregation level."""

    if group_by == "task":
        return group_rows_by_task(rows)
    if group_by == "task_and_model":
        return group_rows_by_task_and_model(rows)
    raise ValueError(f"Unsupported calibration group_by: {group_by}")


def summarize_annotation_labels(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize label availability for an annotation pack."""

    task_counter = Counter(str(row.get("task_type") or "") for row in rows)
    labeled_counter = Counter(
        str(row.get("task_type") or "") for row in rows if parse_human_label(row.get("human_label")) is not None
    )
    return {
        "num_rows": len(rows),
        "num_labeled": sum(labeled_counter.values()),
        "task_counts": dict(task_counter),
        "task_labeled_counts": dict(labeled_counter),
    }
