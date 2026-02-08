import argparse
import os
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from .halueval_common import (
    SUPPORTED_SUBSETS,
    build_eval_item_from_raw_row,
    build_generation_prompt,
    get_pair_from_raw_row,
    parse_subsets_arg,
    to_chat_prompt,
)


def token_len(tokenizer: AutoTokenizer, text: str) -> int:
    """
    Compute token length without truncation.
    """
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def convert_config_to_dpo(
    config_name: str,
    tokenizer: AutoTokenizer,
    max_samples: Optional[int],
    seed: int,
    max_prompt_length: int,
    max_length: int,
    enable_thinking: bool,
    keep_chat_prompt: bool,
    keep_length_metadata: bool,
) -> Dataset:
    """
    Convert a HaluEval subset with paired (right_*, hallucinated_*) into DPO format:
    {
        task, source_id,
        prompt,
        eval_question, eval_contexts,
        chosen, rejected,
        ...optional metadata...
    }

    Length filtering:
      - chat_prompt_tokens <= max_prompt_length
      - chat_prompt_tokens + chosen_tokens <= max_length
      - chat_prompt_tokens + rejected_tokens <= max_length

    Samples violating any constraint are dropped.

    IMPORTANT:
    - `prompt` is the *plain* generation prompt (not chat wrapped).
    - `eval_question` and `eval_contexts` are stored to support leakage-free RAGAS evaluation
      from the preprocessed split (no need to re-load raw HF datasets).
    """
    ds = load_dataset("pminervini/HaluEval", config_name, split="data").shuffle(seed=seed)

    def _map(row: Dict, idx: int) -> Dict:
        eval_question, eval_contexts = build_eval_item_from_raw_row(config_name, row)
        plain_prompt = build_generation_prompt(config_name, eval_question, eval_contexts)

        chosen, rejected = get_pair_from_raw_row(config_name, row)

        chat_prompt = to_chat_prompt(tokenizer, plain_prompt, enable_thinking=enable_thinking)

        # Compute lengths once so filtering and later analysis are consistent.
        p_len = token_len(tokenizer, chat_prompt)
        c_len = token_len(tokenizer, chosen) if chosen else 0
        r_len = token_len(tokenizer, rejected) if rejected else 0

        out = {
            "task": config_name,
            "source_id": f"{config_name}:{idx}",
            "prompt": plain_prompt,
            "eval_question": eval_question,
            "eval_contexts": eval_contexts,
            "chat_prompt": chat_prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

        if keep_length_metadata:
            out.update(
                {
                    "prompt_tokens": p_len,
                    "chosen_tokens": c_len,
                    "rejected_tokens": r_len,
                    "chosen_total_tokens": p_len + c_len,
                    "rejected_total_tokens": p_len + r_len,
                }
            )
        return out

    ds = ds.map(_map, with_indices=True, remove_columns=ds.column_names)

    def _length_ok(row: Dict) -> bool:
        # Drop empty completions early.
        if not row["chosen"] or not row["rejected"]:
            return False

        p_len = row["prompt_tokens"] if "prompt_tokens" in row else token_len(tokenizer, row["chat_prompt"])
        if p_len > max_prompt_length:
            return False

        c_len = row["chosen_tokens"] if "chosen_tokens" in row else token_len(tokenizer, row["chosen"])
        r_len = row["rejected_tokens"] if "rejected_tokens" in row else token_len(tokenizer, row["rejected"])

        if p_len + c_len > max_length:
            return False
        if p_len + r_len > max_length:
            return False
        return True

    ds = ds.filter(_length_ok)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Keep or drop chat_prompt according to flag.
    if not keep_chat_prompt:
        ds = ds.remove_columns(["chat_prompt"])

    return ds


def split_dataset(
    ds: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    """
    Split dataset into train/validation/test.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    ds = ds.shuffle(seed=seed)
    tmp = ds.train_test_split(test_size=test_ratio, seed=seed)
    train_val = tmp["train"]
    test = tmp["test"]

    val_size = val_ratio / (train_ratio + val_ratio)
    train_val_split = train_val.train_test_split(test_size=val_size, seed=seed)

    return DatasetDict(
        train=train_val_split["train"],
        validation=train_val_split["test"],
        test=test,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subsets",
        type=str,
        default="dialogue,qa,summarization",
        help="Comma-separated HaluEval subsets to include: dialogue,qa,summarization",
    )

    parser.add_argument(
        "--max_samples_per_subset",
        type=int,
        default=2000,
        help="Max samples per subset after length filtering. Use -1 for all.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer used to compute token lengths for filtering.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Whether to enable Qwen3 thinking mode in chat template during length calc.",
    )

    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)

    parser.add_argument(
        "--keep_chat_prompt",
        action="store_true",
        help="If set, keep chat_prompt in the saved dataset (useful for debugging).",
    )
    parser.add_argument(
        "--keep_length_metadata",
        action="store_true",
        help="If set, keep token length metadata columns for later analysis.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/halueval_dpo_pairs",
        help="Output directory for DatasetDict saved with save_to_disk().",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    subsets = parse_subsets_arg(args.subsets)

    max_n = None if args.max_samples_per_subset < 0 else args.max_samples_per_subset

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    parts: List[Dataset] = []
    for subset in subsets:
        part = convert_config_to_dpo(
            config_name=subset,
            tokenizer=tokenizer,
            max_samples=max_n,
            seed=args.seed,
            max_prompt_length=args.max_prompt_length,
            max_length=args.max_length,
            enable_thinking=args.enable_thinking,
            keep_chat_prompt=args.keep_chat_prompt,
            keep_length_metadata=args.keep_length_metadata,
        )
        parts.append(part)

    merged = concatenate_datasets(parts).shuffle(seed=args.seed)
    ds_dict = split_dataset(
        merged,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    ds_dict.save_to_disk(args.output_dir)
    print(f"Saved DatasetDict to: {args.output_dir}")
    print(ds_dict)


if __name__ == "__main__":
    main()
