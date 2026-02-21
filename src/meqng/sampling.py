from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Optional

from transformers import AutoTokenizer

from dataio import write_jsonl
from .halu import QASample
from .nli import NLIVerifier
from prompt import build_qa_premise


def token_len(tokenizer, text: str) -> int:
    """Count tokens for a single text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def sample_halueval(
    samples: Iterable[QASample],
    out_path: str,
    max_samples: int,
    max_prompt_tokens: int,
    max_total_tokens: int,
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    seed: int = 0,
    nli_verifier: Optional[NLIVerifier] = None,
    min_chosen_entail: Optional[float] = None,
) -> Dict[str, Any]:
    """Sample and filter QA examples by prompt/total token length."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    rng = random.Random(seed)

    num_input = 0
    length_passed = []
    for s in samples:
        num_input += 1
        prompt = build_qa_premise(s.knowledge, s.question)
        prompt_tokens = token_len(tok, prompt)
        total_tokens = prompt_tokens + token_len(tok, s.right_answer) + token_len(tok, s.hallucinated_answer)
        if prompt_tokens > max_prompt_tokens:
            continue
        if total_tokens > max_total_tokens:
            continue
        length_passed.append(s)

    nli_filtered = list(length_passed)
    nli_kept = None
    if min_chosen_entail is not None:
        if nli_verifier is None:
            raise ValueError("nli_verifier must be provided when min_chosen_entail is set.")
        premises = [build_qa_premise(s.knowledge, s.question) for s in length_passed]
        hypotheses = [s.right_answer for s in length_passed]
        scores = nli_verifier.score(premises, hypotheses)
        nli_filtered = [s for s, sc in zip(length_passed, scores) if sc.entail >= min_chosen_entail]
        nli_kept = len(nli_filtered)

    rng.shuffle(nli_filtered)
    final_samples = nli_filtered[:max_samples]

    records = []
    for s in final_samples:
        rec = {
            "id": s.uid,
            "knowledge": s.knowledge,
            "question": s.question,
            "chosen": s.right_answer,
            "rejected_orig": s.hallucinated_answer,
        }
        records.append(rec)

    write_jsonl(out_path, records)
    return {
        "num_input": num_input,
        "num_pass_length": len(length_passed),
        "num_pass_nli": nli_kept,
        "num_final": len(records),
        "min_chosen_entail": min_chosen_entail,
    }
