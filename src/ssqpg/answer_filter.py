from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from transformers import AutoTokenizer

from .config import AnswerFilterConfig
from .text import normalize_whitespace


_PROMPT_LEAK_RE = re.compile(r"\b(?:question|knowledge|context|answer)\s*:", flags=re.IGNORECASE)
_OPTION_STYLE_RE = re.compile(r"\boptions?\s*:", flags=re.IGNORECASE)


def _count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def apply_answer_filters(rows: Sequence[Dict[str, Any]], cfg: AnswerFilterConfig) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Apply single-answer filters to judged rows before question-level pairing."""

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

    counts: Dict[str, int] = {
        "num_input": len(rows),
        "num_kept": 0,
        "drop_empty": 0,
        "drop_min_tokens": 0,
        "drop_max_tokens": 0,
        "drop_max_chars": 0,
        "drop_prompt_leak": 0,
        "drop_option_style": 0,
        "drop_qa_consistency": 0,
        "drop_reference_support": 0,
    }

    kept: List[Dict[str, Any]] = []
    for row in rows:
        answer = normalize_whitespace(str(row.get("answer") or ""))
        judge = row.get("judge") or {}
        reason = "ok"

        token_count = _count_tokens(tokenizer, answer) if answer else 0
        char_count = len(answer)

        if not answer:
            reason = "empty"
            counts["drop_empty"] += 1
        elif token_count < cfg.min_answer_tokens:
            reason = "min_tokens"
            counts["drop_min_tokens"] += 1
        elif token_count > cfg.max_answer_tokens:
            reason = "max_tokens"
            counts["drop_max_tokens"] += 1
        elif char_count > cfg.max_answer_chars:
            reason = "max_chars"
            counts["drop_max_chars"] += 1
        elif cfg.drop_prompt_leak and bool(_PROMPT_LEAK_RE.search(answer)):
            reason = "prompt_leak"
            counts["drop_prompt_leak"] += 1
        elif cfg.drop_option_style and bool(_OPTION_STYLE_RE.search(answer)):
            reason = "option_style"
            counts["drop_option_style"] += 1
        elif cfg.require_qa_consistent and bool(judge.get("qa_consistent") is False):
            reason = "qa_consistency"
            counts["drop_qa_consistency"] += 1
        elif cfg.require_reference_supported and bool(judge.get("has_reference")) and not bool(judge.get("reference_supported_primary", False)):
            reason = "reference_support"
            counts["drop_reference_support"] += 1

        out_row = dict(row)
        out_row["answer"] = answer
        out_row["answer_filter"] = {
            "kept": reason == "ok",
            "reason": reason,
            "answer_tokens": token_count,
            "answer_chars": char_count,
        }
        if reason == "ok":
            kept.append(out_row)

    counts["num_kept"] = len(kept)
    return kept, counts
