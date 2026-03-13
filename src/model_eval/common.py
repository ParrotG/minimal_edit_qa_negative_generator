from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from project_config import PROJECT_SETTINGS
from qa_protocol import parse_structured_output

from dataio import optional_str, pick_first_non_empty_str, read_json, read_jsonl


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


def optional_stripped(value: Any) -> str:
    """Return a stripped string or an empty string for null-like inputs."""

    return str(value or "").strip()


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

    if os.path.isfile(data_path):
        lower = data_path.lower()
        if lower.endswith(".jsonl"):
            return Dataset.from_list(list(read_jsonl(data_path)))
        if lower.endswith(".json"):
            payload = read_json(data_path)
            if isinstance(payload, list):
                return Dataset.from_list([dict(item) for item in payload if isinstance(item, dict)])
            if isinstance(payload, dict):
                if split in payload and isinstance(payload[split], list):
                    return Dataset.from_list([dict(item) for item in payload[split] if isinstance(item, dict)])
                if "data" in payload and isinstance(payload["data"], list):
                    return Dataset.from_list([dict(item) for item in payload["data"] if isinstance(item, dict)])
                return Dataset.from_list([payload])
        return load_dataset("json", data_files=data_path, split="train")

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


def extract_knowledge_question_from_infer_prompt(prompt: str) -> Tuple[str, str]:
    """Best-effort extraction of knowledge and question from infer prompt text."""

    text = str(prompt or "")
    if not text.strip():
        return "", ""

    marker_knowledge = "\nKnowledge:\n"
    marker_question = "\n\nQuestion: "
    left = text.rfind(marker_knowledge)
    if left < 0:
        return "", _extract_question_from_prompt(text)

    tail = text[left + len(marker_knowledge) :]
    qpos = tail.rfind(marker_question.strip())
    if qpos < 0:
        qpos = tail.rfind(marker_question)
    if qpos < 0:
        return tail.strip(), _extract_question_from_prompt(text)

    if tail[qpos:].startswith("\n\nQuestion: "):
        question_text = tail[qpos + len("\n\nQuestion: ") :]
    elif tail[qpos:].startswith("Question: "):
        question_text = tail[qpos + len("Question: ") :]
    else:
        question_text = ""
    knowledge_text = tail[:qpos].strip()
    return knowledge_text, question_text.strip()


def _extract_answerability_label(row: Dict[str, Any]) -> str:
    label = optional_str(row, "answerability_label")
    if label:
        return label
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("answerability_label") or "").strip()
    return ""


def infer_eval_track(row: Dict[str, Any]) -> str:
    """Infer the evaluation track for a generated/evaluated row."""

    explicit = optional_stripped(row.get("eval_track"))
    if explicit:
        return explicit

    prompt_mode = optional_stripped(row.get("prompt_mode"))
    if prompt_mode in {"teacher", "teacher_fewshot"}:
        return "base_protocol"
    if "parsed_output" in row or prompt_mode == "infer":
        return "sft_structured"
    if "answer" in row:
        return "base_task"
    return ""


def infer_eval_variant(row: Dict[str, Any]) -> str:
    """Infer the evaluation variant for a generated/evaluated row."""

    explicit = optional_stripped(row.get("eval_variant"))
    if explicit:
        return explicit

    prompt_mode = optional_stripped(row.get("prompt_mode"))
    if prompt_mode in {"teacher", "teacher_fewshot"}:
        return "fewshot_retry"
    if prompt_mode == "infer" or "parsed_output" in row:
        return "checkpoint"
    if row.get("enable_thinking") is True:
        return "think"
    if row.get("enable_thinking") is False:
        return "no_think"
    thinking_hint = optional_stripped(row.get("thinking_mode")).lower()
    if thinking_hint in {"think", "enabled"}:
        return "think"
    if thinking_hint in {"no_think", "disabled"}:
        return "no_think"
    return ""


def _extract_reference_answer(row: Dict[str, Any]) -> str:
    reference = pick_first_non_empty_str(
        row,
        ["reference_answer", "answer", "chosen", "reference", "right_answer"],
    )
    if reference:
        return reference
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("reference_answer") or "").strip()
    return ""


def _extract_knowledge(row: Dict[str, Any]) -> str:
    knowledge = optional_str(row, "knowledge")
    if knowledge:
        return knowledge

    for key in ("eval_contexts", "contexts", "retrieval_context", "context"):
        values = normalize_list(row.get(key))
        if values:
            return "\n\n".join(values)
    prompt = optional_str(row, "prompt")
    if prompt:
        parsed_knowledge, _ = extract_knowledge_question_from_infer_prompt(prompt)
        return parsed_knowledge
    return ""


def extract_generation_item(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """Extract one normalized QA item from a row."""

    prompt = optional_str(row, "prompt")
    question = pick_first_non_empty_str(row, ["question", "eval_question", "input"])
    if not question:
        if prompt:
            _, parsed_question = extract_knowledge_question_from_infer_prompt(prompt)
            question = parsed_question or _extract_question_from_prompt(prompt)

    knowledge = _extract_knowledge(row)
    if not question or not knowledge:
        return None

    source_id = str(row.get("source_id") or row.get("id") or idx).strip()
    row_id = str(row.get("id") or source_id).strip()
    reference_answer = _extract_reference_answer(row)
    if not prompt:
        prompt = ""
    return {
        "sample_id": idx,
        "id": row_id,
        "source_id": source_id,
        "question": question,
        "knowledge": knowledge,
        "prompt": prompt,
        "reference_answer": reference_answer,
        "answerability_label": _extract_answerability_label(row),
        "data_split": str(row.get("data_split") or "").strip(),
        "eval_track": infer_eval_track(row),
        "eval_variant": infer_eval_variant(row),
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


def load_structured_generation_items(
    *,
    data_path: str,
    split: str,
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Load normalized QA items for structured generation."""

    return load_generation_items(
        data_path=data_path,
        split=split,
        max_samples=max_samples,
        seed=seed,
    )


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


def _extract_generated_answer(row: Dict[str, Any], answer_source: str) -> str:
    parsed_output = row.get("parsed_output")
    parsed_answer = ""
    parsed_rationale = ""
    if isinstance(parsed_output, dict):
        parsed_answer = str(parsed_output.get("answer") or "").strip()
        parsed_rationale = str(parsed_output.get("rationale") or "").strip()

    if answer_source == "rationale_plus_answer":
        if parsed_rationale and parsed_answer:
            return f"{parsed_rationale}\nTherefore the answer is {parsed_answer}"
        if parsed_answer:
            return parsed_answer

    if parsed_answer:
        return parsed_answer
    return pick_first_non_empty_str(row, ["answer", "actual_output", "raw_output"])


def extract_structured_output_text(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Extract rationale-plus-answer text from a structured output row."""

    parsed_output = row.get("parsed_output")
    payload: Dict[str, Any] = {}
    parse_ok = False
    if isinstance(parsed_output, dict):
        payload = dict(parsed_output)
        parse_ok = True
    else:
        raw_output = pick_first_non_empty_str(row, ["raw_output", "answer", "actual_output"])
        if raw_output:
            parse_result = parse_structured_output(raw_output)
            if parse_result.ok and parse_result.parsed is not None:
                payload = parse_result.parsed.model_dump(mode="json")
                parse_ok = True

    rationale = optional_stripped(payload.get("rationale"))
    answer = optional_stripped(payload.get("answer"))
    text = ""
    if rationale and answer:
        text = f"{rationale}\nTherefore the answer is {answer}"
    elif answer:
        text = answer

    meta = {
        "structured_parse_ok": parse_ok,
        "structured_answer_present": bool(answer),
        "structured_rationale_present": bool(rationale),
        "structured_extract_failed": not bool(text),
    }
    return text, meta


def extract_structured_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a structured payload dict from parsed_output or raw_output when available."""

    parsed_output = row.get("parsed_output")
    if isinstance(parsed_output, dict):
        return dict(parsed_output)

    raw_output = pick_first_non_empty_str(row, ["raw_output", "answer", "actual_output"])
    if not raw_output:
        return None
    parse_result = parse_structured_output(raw_output)
    if not parse_result.ok or parse_result.parsed is None:
        return None
    return parse_result.parsed.model_dump(mode="json")


def is_flat_answerable_for_content_eval(row: Dict[str, Any]) -> bool:
    """Return whether a flat-baseline row should enter correctness and semantic evaluation."""

    if not bool(row.get("extraction_parse_ok")):
        return False
    if row.get("refusal_detected") is not False:
        return False
    if not optional_stripped(row.get("extracted_answer")):
        return False
    pred_answerability = optional_stripped(row.get("pred_answerability"))
    if pred_answerability and pred_answerability != "answerable":
        return False
    return True


def flat_content_eval_skipped_reason(row: Dict[str, Any]) -> str:
    """Explain why a flat-baseline row does not enter correctness and semantic evaluation."""

    if not bool(row.get("extraction_parse_ok")):
        return "extraction_parse_failed"
    if row.get("refusal_detected") is True:
        return "predicted_refusal"
    if row.get("refusal_detected") is None:
        return "missing_refusal_signal"
    if not optional_stripped(row.get("extracted_answer")):
        return "empty_extracted_answer"
    pred_answerability = optional_stripped(row.get("pred_answerability"))
    if pred_answerability and pred_answerability != "answerable":
        return "predicted_unanswerable"
    return ""


def is_flat_matcher_applicable(row: Dict[str, Any]) -> bool:
    """Return whether a flat-baseline row is eligible for matcher review."""

    if not is_flat_answerable_for_content_eval(row):
        return False
    if not optional_stripped(row.get("reference_answer")):
        return False
    correctness_ok = row.get("correctness_ok")
    if correctness_ok is not None and bool(correctness_ok):
        return False
    return True


def is_structured_answerable_for_semantic_eval(row: Dict[str, Any]) -> bool:
    """Return whether a structured row should enter semantic evaluation."""

    payload = extract_structured_payload(row)
    if not payload:
        return False
    if optional_stripped(payload.get("answerability")) != "answerable":
        return False
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return False
    has_quote = any(optional_stripped(item.get("quote")) for item in evidence if isinstance(item, dict))
    if not has_quote:
        return False
    if not optional_stripped(payload.get("rationale")):
        return False
    if not optional_stripped(payload.get("answer")):
        return False
    return True


def is_structured_matcher_applicable(row: Dict[str, Any]) -> bool:
    """Return whether a structured row is eligible for matcher review."""

    if not is_structured_answerable_for_semantic_eval(row):
        return False
    if not _extract_reference_answer(row):
        return False
    correctness_ok = row.get("correctness_ok")
    if correctness_ok is not None and bool(correctness_ok):
        return False
    return True


def normalize_generated_row(
    row: Dict[str, Any],
    idx: int,
    *,
    answer_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize one generated-answer row for downstream evaluators."""

    resolved_answer_source = answer_source or PROJECT_SETTINGS.eval.deepeval.answer_source
    answer = _extract_generated_answer(row, answer_source=resolved_answer_source)
    if not answer:
        return None

    prompt = optional_str(row, "prompt")
    question = pick_first_non_empty_str(row, ["question", "eval_question", "input"])
    if not question:
        if prompt:
            _, parsed_question = extract_knowledge_question_from_infer_prompt(prompt)
            question = parsed_question or _extract_question_from_prompt(prompt)

    knowledge = _extract_knowledge(row)
    if not question or not knowledge:
        return None

    model_tag = str(row.get("model_tag") or row.get("tag") or "model")
    model_step = safe_int(row.get("model_step", row.get("step")), 0)
    model_path = str(row.get("model_path") or row.get("model_name") or model_tag)

    sample_id = safe_int(row.get("sample_id"), idx)
    source_id = str(row.get("source_id") or row.get("id") or sample_id)
    reference_answer = _extract_reference_answer(row)

    return {
        "model_tag": model_tag,
        "model_step": model_step,
        "model_path": model_path,
        "eval_track": infer_eval_track(row),
        "eval_variant": infer_eval_variant(row),
        "sample_id": sample_id,
        "source_id": source_id,
        "question": question,
        "knowledge": knowledge,
        "prompt": prompt,
        "reference_answer": reference_answer,
        "answerability_label": _extract_answerability_label(row),
        "data_split": str(row.get("data_split") or "").strip(),
        "answer": answer,
    }


def load_generated_rows(
    *,
    generated_path: str,
    split: str,
    max_samples: int,
    seed: int,
    answer_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load normalized generated-answer rows for evaluation."""

    ds = load_dataset_split(data_path=generated_path, split=split).shuffle(seed=seed)

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        rec = normalize_generated_row(row, idx, answer_source=answer_source)
        if rec is None:
            continue
        out.append(rec)
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def group_rows_by_model(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int, str, str, str], List[Dict[str, Any]]]:
    """Group rows by model identity plus evaluation track metadata."""

    grouped: Dict[Tuple[str, int, str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model_tag"]),
            int(row["model_step"]),
            str(row["model_path"]),
            str(row.get("eval_track") or ""),
            str(row.get("eval_variant") or ""),
        )
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
