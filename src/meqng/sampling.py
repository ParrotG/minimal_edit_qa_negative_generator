from __future__ import annotations

import random
from dataclasses import asdict
from typing import Iterable, Iterator, Optional

from transformers import AutoTokenizer

from .halu import QASample
from .io import write_jsonl


def token_len(tokenizer, text: str) -> int:
    """Count tokens for a single text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def sample_halueval(
    samples: Iterable[QASample],
    out_path: str,
    max_samples: int,
    max_prompt_tokens: int,
    max_total_tokens: int,
    tokenizer_name: str = "gpt2",
    seed: int = 0,
) -> None:
    """Sample and filter QA examples by prompt/total token length."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    rng = random.Random(seed)

    buf = []
    for s in samples:
        prompt = f"{s.knowledge}\nQuestion: {s.question}"
        prompt_tokens = token_len(tok, prompt)
        total_tokens = prompt_tokens + token_len(tok, s.right_answer) + token_len(tok, s.hallucinated_answer)
        if prompt_tokens > max_prompt_tokens:
            continue
        if total_tokens > max_total_tokens:
            continue
        buf.append(s)

    rng.shuffle(buf)
    buf = buf[:max_samples]

    records = []
    for s in buf:
        rec = {
            "id": s.uid,
            "knowledge": s.knowledge,
            "question": s.question,
            "chosen": s.right_answer,
            "rejected_orig": s.hallucinated_answer,
        }
        records.append(rec)

    write_jsonl(out_path, records)
