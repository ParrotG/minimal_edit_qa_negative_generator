from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


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

