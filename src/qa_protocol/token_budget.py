from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_name: str):
    """Load and cache one tokenizer for length budgeting."""

    return AutoTokenizer.from_pretrained(tokenizer_name)


def count_text_tokens(text: str, tokenizer_name: str) -> int:
    """Count tokens for one text string without adding special tokens."""

    tokenizer = _load_tokenizer(tokenizer_name)
    return int(len(tokenizer.encode(str(text or ""), add_special_tokens=False)))
