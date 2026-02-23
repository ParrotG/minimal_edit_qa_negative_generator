from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig
except ImportError as exc:  # pragma: no cover
    raise ImportError("LoRA training requires `peft` to be installed.") from exc

try:
    from trl import DPOConfig, DPOTrainer
except ImportError as exc:  # pragma: no cover
    raise ImportError("DPO training requires `trl` to be installed.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path from DatasetDict.save_to_disk.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LoRA adapter checkpoints.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")

    parser.add_argument("--train_epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=768)

    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=100)
    parser.add_argument("--precompute_ref_log_probs", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def _build_dpo_config(args: argparse.Namespace, has_eval: bool) -> DPOConfig:
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.train_epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "per_device_train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "max_prompt_length": args.max_prompt_length,
        "max_length": args.max_length,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "label_smoothing": args.label_smoothing,
        "precompute_ref_log_probs": args.precompute_ref_log_probs,
        "remove_unused_columns": False,
    }
    if has_eval:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = args.save_steps

    return DPOConfig(**kwargs)


def _build_trainer(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    training_args: DPOConfig,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    peft_config: LoraConfig,
) -> DPOTrainer:
    kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }
    try:
        return DPOTrainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return DPOTrainer(tokenizer=tokenizer, **kwargs)


def _load_dataset(data_dir: str) -> tuple[Dataset, Optional[Dataset]]:
    ds_obj = load_from_disk(data_dir)
    if isinstance(ds_obj, DatasetDict):
        if "train" not in ds_obj:
            raise ValueError(f"'train' split is required, but missing in {data_dir}.")
        train_ds = ds_obj["train"]
        eval_ds = ds_obj.get("validation")
        return train_ds, eval_ds
    if isinstance(ds_obj, Dataset):
        return ds_obj, None
    raise ValueError(f"Unsupported dataset type from {data_dir}: {type(ds_obj)}")


def _validate_columns(ds: Dataset, split_name: str) -> None:
    required_columns = {"prompt", "chosen", "rejected"}
    missing = sorted(required_columns - set(ds.column_names))
    if missing:
        raise ValueError(f"Split '{split_name}' is missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, eval_ds = _load_dataset(args.data_dir)
    _validate_columns(train_ds, "train")
    if eval_ds is not None:
        _validate_columns(eval_ds, "validation")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token.")

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype="auto")
    model.config.use_cache = False

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules cannot be empty.")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    training_args = _build_dpo_config(args=args, has_eval=eval_ds is not None)
    trainer = _build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
