from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from dataio import optional_str, pick_first_non_empty_str


def normalize_list(value: Any) -> List[str]:
    """Normalize input into a list of non-empty strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def safe_int(value: Any, default: int) -> int:
    """Best-effort integer parsing with a fallback."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_dataset_split(data_path: str, split: str) -> Dataset:
    """Load a Dataset from save_to_disk path or JSON/JSONL file."""

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


def _extract_question_from_prompt(prompt: str) -> str:
    """Best-effort extraction of question text from a prompt."""

    text = (prompt or "").strip()
    if not text:
        return ""
    if text.startswith("Question:"):
        return text.split("Question:", 1)[1].strip()
    if "\nQuestion:" in text:
        _, right = text.rsplit("\nQuestion:", 1)
        return right.strip()
    return ""


def _extract_knowledge(row: Dict[str, Any]) -> str:
    knowledge = optional_str(row, "knowledge")
    if knowledge:
        return knowledge

    for key in ("eval_contexts", "contexts", "retrieval_context", "context"):
        values = normalize_list(row.get(key))
        if values:
            return "\n\n".join(values)
    return ""


def extract_generation_item(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """Extract one normalized QA item from a row."""

    question = pick_first_non_empty_str(row, ["question", "eval_question", "input"])
    if not question:
        question = _extract_question_from_prompt(optional_str(row, "prompt"))

    knowledge = _extract_knowledge(row)
    if not question or not knowledge:
        return None

    source_id = str(row.get("source_id") or row.get("id") or idx)
    reference_answer = pick_first_non_empty_str(
        row,
        ["reference_answer", "chosen", "reference", "right_answer"],
    )
    return {
        "sample_id": idx,
        "source_id": source_id,
        "question": question,
        "knowledge": knowledge,
        "reference_answer": reference_answer,
    }


def load_generation_items(
    *,
    data_path: str,
    split: str,
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Load normalized QA items for answer generation."""

    ds = load_dataset_split(data_path=data_path, split=split).shuffle(seed=seed)

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        item = extract_generation_item(row, idx)
        if item is None:
            continue
        out.append(item)
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def to_chat_prompt(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    """Convert plain prompt to chat prompt when tokenizer supports templates."""

    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def normalize_generated_row(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """Normalize one generated-answer row for downstream evaluators."""

    answer = pick_first_non_empty_str(row, ["answer", "actual_output"])
    if not answer:
        return None

    question = pick_first_non_empty_str(row, ["question", "eval_question", "input"])
    if not question:
        question = _extract_question_from_prompt(optional_str(row, "prompt"))

    knowledge = _extract_knowledge(row)
    if not question or not knowledge:
        return None

    model_tag = str(row.get("model_tag") or row.get("tag") or "model")
    model_step = safe_int(row.get("model_step", row.get("step")), 0)
    model_path = str(row.get("model_path") or row.get("model_name") or model_tag)

    sample_id = safe_int(row.get("sample_id"), idx)
    source_id = str(row.get("source_id") or row.get("id") or sample_id)
    reference_answer = pick_first_non_empty_str(row, ["reference_answer", "chosen"])

    return {
        "model_tag": model_tag,
        "model_step": model_step,
        "model_path": model_path,
        "sample_id": sample_id,
        "source_id": source_id,
        "question": question,
        "knowledge": knowledge,
        "reference_answer": reference_answer,
        "answer": answer,
    }


def load_generated_rows(
    *,
    generated_path: str,
    split: str,
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Load normalized generated-answer rows for evaluation."""

    ds = load_dataset_split(data_path=generated_path, split=split).shuffle(seed=seed)

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        rec = normalize_generated_row(row, idx)
        if rec is None:
            continue
        out.append(rec)
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def group_rows_by_model(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int, str], List[Dict[str, Any]]]:
    """Group rows by (model_tag, model_step, model_path)."""

    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model_tag"]), int(row["model_step"]), str(row["model_path"]))
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    """Write rows to CSV with a stable discovered header order."""

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
