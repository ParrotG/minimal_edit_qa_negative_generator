from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from project_config import PROJECT_SETTINGS

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover
    raise ImportError("LoRA training requires `peft` to be installed.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path from DatasetDict.save_to_disk.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LoRA checkpoints.")
    parser.add_argument("--model_name_or_path", type=str, default=PROJECT_SETTINGS.model.target_training_llm)
    parser.add_argument("--train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def _load_dataset(data_dir: str) -> tuple[Dataset, Optional[Dataset]]:
    ds_obj = load_from_disk(data_dir)
    if isinstance(ds_obj, DatasetDict):
        if "train" not in ds_obj:
            raise ValueError(f"'train' split is required, but missing in {data_dir}.")
        return ds_obj["train"], ds_obj.get("validation")
    if isinstance(ds_obj, Dataset):
        return ds_obj, None
    raise ValueError(f"Unsupported dataset type from {data_dir}: {type(ds_obj)}")


def _validate_columns(ds: Dataset, split_name: str) -> None:
    required_columns = {"prompt", "completion"}
    missing = sorted(required_columns - set(ds.column_names))
    if missing:
        raise ValueError(f"Split '{split_name}' is missing required columns: {missing}")


def _tokenize_pair(tokenizer: Any, prompt: str, completion: str, max_length: int) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id

    if eos_id is not None:
        completion_ids = completion_ids + [eos_id]

    available_prompt = max_length - len(completion_ids)
    if available_prompt < 0:
        completion_ids = completion_ids[:max_length]
        available_prompt = 0

    prompt_ids = prompt_ids[-available_prompt:] if available_prompt > 0 else []
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _preprocess_dataset(ds: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    def _map_fn(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        input_ids: List[List[int]] = []
        labels: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            tokenized = _tokenize_pair(tokenizer, str(prompt), str(completion), max_length=max_length)
            input_ids.append(tokenized["input_ids"])
            labels.append(tokenized["labels"])
            attention_mask.append(tokenized["attention_mask"])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    remove_columns = list(ds.column_names)
    return ds.map(_map_fn, batched=True, remove_columns=remove_columns)


@dataclass
class SftDataCollator:
    """Pad already-tokenized prompt-completion pairs for causal-LM training."""

    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids: List[List[int]] = []
        labels: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for feature in features:
            input_pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * input_pad)
            labels.append(feature["labels"] + [-100] * input_pad)
            attention_mask.append(feature["attention_mask"] + [0] * input_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


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
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    tokenized_train = _preprocess_dataset(train_ds, tokenizer=tokenizer, max_length=args.max_length)
    tokenized_eval = None if eval_ds is None else _preprocess_dataset(eval_ds, tokenizer=tokenizer, max_length=args.max_length)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        eval_strategy="steps" if tokenized_eval is not None else "no",
        eval_steps=args.save_steps if tokenized_eval is not None else None,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=SftDataCollator(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA SFT checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
