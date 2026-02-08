"""
Shared utilities for HaluEval -> DPO preprocessing and evaluation.

This module centralizes:
- Subset definitions
- Consistent prompt construction (aligned with the RAGAS evaluation script format)
- Consistent (question, contexts) construction for RAGAS faithfulness
- Deterministic sampling from a preprocessed Dataset split
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer


SUPPORTED_SUBSETS = {"qa", "dialogue", "summarization"}


def parse_subsets_arg(subsets: str) -> List[str]:
    """
    Parse a comma-separated subset list and validate it.
    """
    subset_list = [s.strip() for s in subsets.split(",") if s.strip()]
    unknown = [s for s in subset_list if s not in SUPPORTED_SUBSETS]
    if unknown:
        raise ValueError(f"Unsupported subsets: {unknown}. Supported: {sorted(SUPPORTED_SUBSETS)}")
    return subset_list


def build_eval_item_from_raw_row(subset: str, row: Dict[str, str]) -> Tuple[str, List[str]]:
    """
    Build (question, contexts) for RAGAS from a raw HaluEval row.

    Notes:
    - For dialogue, we include both knowledge and dialogue history as contexts,
      matching the original evaluation script behavior.
    - For summarization, we use a fixed question prompt as in the original script.
    """
    if subset == "qa":
        question = row["question"].strip()
        contexts = [row["knowledge"].strip()]
        return question, contexts

    if subset == "dialogue":
        dialogue_history = row["dialogue_history"].strip()
        knowledge = row["knowledge"].strip()
        question = dialogue_history
        contexts = [knowledge, dialogue_history]
        return question, contexts

    if subset == "summarization":
        question = "Summarize the given document faithfully without adding unsupported information."
        contexts = [row["document"].strip()]
        return question, contexts

    raise ValueError(f"Unsupported subset: {subset}")


def build_eval_item_from_preprocessed_row(row: Dict) -> Tuple[str, List[str]]:
    """
    Build (question, contexts) from a preprocessed row.

    The preprocessor is expected to store:
    - eval_question: str
    - eval_contexts: List[str]
    """
    question = (row.get("eval_question") or "").strip()
    contexts = row.get("eval_contexts") or []
    contexts = [str(x).strip() for x in contexts]
    return question, contexts


def build_generation_prompt(subset: str, question: str, contexts: List[str]) -> str:
    """
    Build a generation prompt that encourages grounding to the provided context.

    This function is the single source of truth for prompt formatting across:
    - preprocessing (prompt saved into DPO dataset)
    - pairwise log-prob evaluation (uses the saved prompt)
    - RAGAS faithfulness evaluation (uses the saved prompt)
    """
    if subset == "qa":
        ctx = "\n\n".join(contexts)
        return (
            "You are a helpful assistant. Answer using ONLY the provided knowledge. "
            "If the knowledge is insufficient, say you don't know.\n\n"
            "=== Knowledge ===\n"
            f"{ctx}\n\n"
            "=== Question ===\n"
            f"{question}\n\n"
            "Answer:"
        )

    if subset == "dialogue":
        knowledge = contexts[0] if len(contexts) > 0 else ""
        history = contexts[1] if len(contexts) > 1 else question
        return (
            "You are a helpful assistant. Use ONLY the provided knowledge and dialogue history to respond faithfully. "
            "Do not add information that is not supported by them.\n\n"
            "=== Knowledge ===\n"
            f"{knowledge}\n\n"
            "=== Dialogue History ===\n"
            f"{history}\n\n"
            "Write the next assistant response:"
        )

    if subset == "summarization":
        ctx = "\n\n".join(contexts)
        return (
            "You are a helpful assistant. Produce a faithful summary of the document. "
            "Do not add information that is not supported by the document.\n\n"
            "=== Document ===\n"
            f"{ctx}\n\n"
            "Summary:"
        )

    raise ValueError(f"Unsupported subset: {subset}")


def get_pair_from_raw_row(subset: str, row: Dict[str, str]) -> Tuple[str, str]:
    """
    Return (chosen, rejected) for a given raw HaluEval subset row.
    """
    if subset == "dialogue":
        return row["right_response"].strip(), row["hallucinated_response"].strip()
    if subset == "qa":
        return row["right_answer"].strip(), row["hallucinated_answer"].strip()
    if subset == "summarization":
        return row["right_summary"].strip(), row["hallucinated_summary"].strip()
    raise ValueError(f"Unsupported subset: {subset}")


def to_chat_prompt(tokenizer: AutoTokenizer, prompt: str, enable_thinking: bool) -> str:
    """
    Wrap a single user turn with the model's chat template and add generation prompt.

    The returned string ends with the assistant header so that a completion can be appended directly.
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def sample_preprocessed_split(
    ds: Dataset,
    subsets: Sequence[str],
    max_samples_per_subset: int,
    seed: int,
) -> Dataset:
    """
    Deterministically sample from a preprocessed split by task/subset.

    This is used to prevent train-test leakage in evaluation by ensuring:
    - evaluation draws ONLY from a chosen split (e.g., test)
    - sampling is subset-aware and reproducible
    """
    parts: List[Dataset] = []
    for s in subsets:
        sub_ds = ds.filter(lambda x, subset=s: x["task"] == subset)
        sub_ds = sub_ds.shuffle(seed=seed)

        if max_samples_per_subset > 0:
            sub_ds = sub_ds.select(range(min(max_samples_per_subset, len(sub_ds))))

        parts.append(sub_ds)

    if not parts:
        raise ValueError("No subsets selected.")

    merged = concatenate_datasets(parts).shuffle(seed=seed)
    return merged
