from __future__ import annotations

from functools import lru_cache
from typing import Sequence, List

from transformers import AutoTokenizer


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_name: str):
    """Load and cache one tokenizer for length budgeting."""

    return AutoTokenizer.from_pretrained(tokenizer_name)


def count_text_tokens(text: str, tokenizer_name: str) -> int:
    """Count tokens for one text string without adding special tokens."""

    tokenizer = _load_tokenizer(tokenizer_name)
    return int(len(tokenizer.encode(str(text or ""), add_special_tokens=False)))


def count_text_tokens_batch(
    texts: Sequence[str],
    tokenizer_name: str,
    batch_size: int = 128,
) -> List[int]:
    """Count tokens for many text strings while preserving input order."""

    if not texts:
        return []

    tokenizer = _load_tokenizer(tokenizer_name)
    out: List[int] = []
    effective_batch_size = max(1, int(batch_size))
    text_list = [str(text or "") for text in texts]
    for start in range(0, len(text_list), effective_batch_size):
        chunk = text_list[start : start + effective_batch_size]
        encoded = tokenizer(
            chunk,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        out.extend(len(ids) for ids in encoded["input_ids"])
    return [int(item) for item in out]
