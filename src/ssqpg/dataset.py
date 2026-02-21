from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

from dataio import pick_first_non_empty_str, read_jsonl
from .config import HaluEvalSourceConfig, SourceLengthConfig
from prompt import build_qa_answer_prefix


@dataclass(frozen=True)
class SourceRecord:
    """A canonical QA source record used by ssqpg."""

    record_id: str
    knowledge: str
    question: str
    reference_answer: Optional[str]


def _token_len(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def iter_halueval_source(
    cfg: HaluEvalSourceConfig,
    max_samples: Optional[int] = None,
    seed: int = 42,
    include_reference_answer: bool = False,
) -> Iterator[SourceRecord]:
    """Iterate over shuffled HaluEval QA source records."""

    ds = load_dataset(cfg.dataset_name, cfg.subset, split=cfg.split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if max_samples is not None and max_samples > 0:
        indices = indices[:max_samples]

    for idx in indices:
        row = ds[idx]
        knowledge = str(row[cfg.knowledge_field]).strip()
        question = str(row[cfg.question_field]).strip()
        reference_answer = None
        if include_reference_answer:
            reference_answer = str(row[cfg.answer_field]).strip()
        if not knowledge or not question:
            continue
        if include_reference_answer and not reference_answer:
            continue
        yield SourceRecord(
            record_id=str(idx),
            knowledge=knowledge,
            question=question,
            reference_answer=reference_answer,
        )


def iter_jsonl_source(
    path: str,
    knowledge_field: str,
    question_field: str,
    answer_field: str,
    id_field: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
    include_reference_answer: bool = False,
) -> Iterator[SourceRecord]:
    """Iterate over source records from JSONL using field mapping."""

    rows = list(read_jsonl(path))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    for i, row in enumerate(rows):
        rid = str(row.get(id_field) or f"jsonl:{i}")
        knowledge = pick_first_non_empty_str(row, [knowledge_field, "knowledge"])
        question = pick_first_non_empty_str(row, [question_field, "question"])

        reference_answer: Optional[str] = None
        if include_reference_answer:
            reference_answer = pick_first_non_empty_str(
                row,
                [answer_field, "reference_answer", "right_answer", "answer"],
            )

        if not knowledge or not question:
            continue
        if include_reference_answer and not reference_answer:
            continue

        yield SourceRecord(
            record_id=rid,
            knowledge=knowledge,
            question=question,
            reference_answer=reference_answer,
        )


def filter_source_by_length(
    records: Sequence[SourceRecord],
    cfg: SourceLengthConfig,
) -> Tuple[List[SourceRecord], Dict[str, int]]:
    """Filter source records by prompt/(optional) reference/total token lengths."""

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    kept: List[SourceRecord] = []
    num_prompt = 0
    num_answer = 0
    num_total = 0

    for rec in records:
        prompt = build_qa_answer_prefix(rec.knowledge, rec.question)
        p_len = _token_len(tok, prompt)

        a_len = 0
        if rec.reference_answer:
            a_len = _token_len(tok, rec.reference_answer)

        total = p_len + a_len

        if p_len > cfg.max_prompt_tokens:
            num_prompt += 1
            continue

        if rec.reference_answer and a_len > cfg.max_answer_tokens:
            num_answer += 1
            continue

        if total > cfg.max_total_tokens:
            num_total += 1
            continue

        kept.append(rec)

    metrics = {
        "num_input": len(records),
        "num_kept": len(kept),
        "num_drop_prompt": num_prompt,
        "num_drop_answer": num_answer,
        "num_drop_total": num_total,
    }
    return kept, metrics


def source_to_json(records: Iterable[SourceRecord], include_reference_answer: bool = False) -> List[Dict[str, str]]:
    """Convert source records into serializable dictionaries."""

    out: List[Dict[str, str]] = []
    for rec in records:
        row: Dict[str, str] = {
            "id": rec.record_id,
            "knowledge": rec.knowledge,
            "question": rec.question,
        }
        if include_reference_answer and rec.reference_answer is not None:
            row["reference_answer"] = rec.reference_answer
        out.append(row)
    return out
