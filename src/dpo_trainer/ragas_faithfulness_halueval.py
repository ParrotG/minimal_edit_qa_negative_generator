import argparse
import csv
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from openai import AsyncOpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import faithfulness

from .halueval_common import (
    SUPPORTED_SUBSETS,
    parse_subsets_arg,
    sample_preprocessed_split,
    to_chat_prompt,
    build_eval_item_from_preprocessed_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data (leakage-free): evaluate ONLY from a specified split of a preprocessed dataset.
    parser.add_argument("--data_dir", type=str, required=True, help="Path to DatasetDict saved by the preprocessor.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument(
        "--subsets",
        type=str,
        default="qa,dialogue,summarization",
        help="Comma-separated subsets to evaluate: qa,dialogue,summarization",
    )
    parser.add_argument("--max_samples_per_subset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # Models
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")

    # Single-model mode
    parser.add_argument("--lora_model", type=str, default=None, help="Path to LoRA adapter or merged model dir.")
    parser.add_argument(
        "--lora_is_merged",
        action="store_true",
        help="If set, lora_model is a standalone merged model directory (not a PEFT adapter).",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="If set and lora_is_merged is False, merge LoRA into base weights for faster inference.",
    )

    # Curve mode
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="If set, scan checkpoint-* under this directory and evaluate each for a learning curve.",
    )
    parser.add_argument(
        "--include_base",
        action="store_true",
        help="If set, also evaluate the base model (useful for plotting baseline).",
    )

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--enable_thinking", action="store_true")

    # RAGAS judge
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--ragas_batch_size", type=int, default=8)

    # Output
    parser.add_argument("--out_csv", type=str, default="ragas_faithfulness_curve.csv")
    parser.add_argument("--out_jsonl", type=str, default="ragas_faithfulness_samples.jsonl")

    return parser.parse_args()


def load_base_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base causal LM and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    model.generation_config.top_k = None
    model.eval()
    return model, tokenizer


def load_lora_model(
    base_model_name: str,
    lora_path_or_model_dir: str,
    lora_is_merged: bool,
    merge_lora: bool,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load LoRA model either as:
    - merged full model directory (lora_is_merged=True), or
    - PEFT adapter on top of base_model (lora_is_merged=False).
    """
    if lora_is_merged:
        tokenizer = AutoTokenizer.from_pretrained(lora_path_or_model_dir, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            lora_path_or_model_dir,
            dtype="auto",
            device_map="auto",
        )
        model.generation_config.top_k = None
        model.eval()
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, lora_path_or_model_dir)
    if merge_lora:
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """
    Generate answers for a batch of prompts.
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(next(model.parameters()).device)

    do_sample = temperature > 0.0
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )

    results: List[str] = []
    for i in range(len(prompts)):
        prompt_len = int(inputs["attention_mask"][i].sum().item())
        out_ids = gen_ids[i][prompt_len:]
        text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        results.append(text)
    return results


def ragas_faithfulness_eval(dataset: Dataset, judge_model: str, ragas_batch_size: int) -> pd.DataFrame:
    """
    Run RAGAS Faithfulness on a dataset with columns: question, answer, contexts.

    Efficiency:
    - This function is called ONCE for all models/checkpoints combined, enabling
      parallel judge requests across the full evaluation set.
    """
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")

    client = AsyncOpenAI()
    judge_llm = llm_factory(
        judge_model,
        client=client,
        temperature=0.0,
        max_tokens=4096,
        top_p=1.0,
    )
    faithfulness.llm = judge_llm

    result = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=judge_llm,
        batch_size=ragas_batch_size,
        raise_exceptions=False,
        show_progress=True,
    )
    return result.to_pandas()


def _discover_checkpoints(checkpoints_dir: str) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    if not os.path.isdir(checkpoints_dir):
        raise ValueError(f"checkpoints_dir does not exist: {checkpoints_dir}")

    for name in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, name)
        if not os.path.isdir(path):
            continue
        m = re.match(r"^checkpoint-(\d+)$", name)
        if m:
            items.append((int(m.group(1)), path))

    items.sort(key=lambda x: x[0])
    return items


def _load_eval_items(args: argparse.Namespace) -> List[Dict]:
    """
    Load evaluation items strictly from the specified preprocessed split to avoid leakage.
    """
    ds_dict = load_from_disk(args.data_dir)
    ds_split = ds_dict[args.split]

    subsets = parse_subsets_arg(args.subsets)
    ds_sampled = sample_preprocessed_split(
        ds=ds_split,
        subsets=subsets,
        max_samples_per_subset=args.max_samples_per_subset,
        seed=args.seed,
    )

    items: List[Dict] = []
    for i, row in enumerate(ds_sampled):
        subset = row["task"]
        question, contexts = build_eval_item_from_preprocessed_row(row)
        prompt = row["prompt"]
        items.append(
            {
                "sample_id": i,
                "source_id": row.get("source_id", str(i)),
                "subset": subset,
                "question": question,
                "contexts": contexts,
                "prompt": prompt,
            }
        )
    return items


def _generate_for_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    items: List[Dict],
    args: argparse.Namespace,
    model_meta: Dict[str, str],
) -> List[Dict]:
    """
    Generate answers in batches and return rows ready for RAGAS evaluation.
    """
    rows: List[Dict] = []
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        prompts = [to_chat_prompt(tokenizer, x["prompt"], enable_thinking=args.enable_thinking) for x in batch]
        answers = generate_answers(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        for x, ans in zip(batch, answers):
            rows.append(
                {
                    "tag": model_meta["tag"],
                    "step": int(model_meta["step"]),
                    "model_path": model_meta["model_path"],
                    "sample_id": int(x["sample_id"]),
                    "source_id": x["source_id"],
                    "subset": x["subset"],
                    "question": x["question"],
                    "contexts": x["contexts"],
                    "answer": ans,
                }
            )
    return rows


def _write_curve_csv(rows: List[Dict], out_csv: str, subsets: List[str]) -> None:
    """
    Aggregate per-model and per-subset faithfulness into a curve CSV.
    """
    df = pd.DataFrame(rows)
    if "faithfulness" not in df.columns:
        raise RuntimeError("No faithfulness column found in results; RAGAS evaluation may have failed.")

    curve_rows: List[Dict[str, str]] = []
    group_cols = ["tag", "step", "model_path"]
    for (tag, step, model_path), g in df.groupby(group_cols):
        row: Dict[str, str] = {
            "tag": str(tag),
            "step": str(int(step)),
            "model_path": str(model_path),
            "faithfulness": f"{float(g['faithfulness'].mean())}",
        }
        for s in subsets:
            sub_g = g[g["subset"] == s]
            if len(sub_g) > 0:
                row[f"faithfulness_subset/{s}"] = f"{float(sub_g['faithfulness'].mean())}"
        curve_rows.append(row)

    # Sort by step then tag for readability.
    curve_rows.sort(key=lambda r: (int(r["step"]), r["tag"]))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    keys: List[str] = []
    for r in curve_rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(curve_rows)

    print(f"Saved: {out_csv}")


def _write_samples_jsonl(rows: List[Dict], out_jsonl: str) -> None:
    """
    Write per-sample generations and scores to JSONL.
    """
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(pd.Series(r).to_json(force_ascii=False) + "\n")
    print(f"Saved: {out_jsonl}")


def main() -> None:
    args = parse_args()
    subsets = parse_subsets_arg(args.subsets)

    items = _load_eval_items(args)

    # Build list of models to evaluate.
    model_specs: List[Dict[str, str]] = []

    if args.include_base:
        model_specs.append({"tag": "base", "step": "0", "model_path": args.base_model, "kind": "base"})

    if args.checkpoints_dir:
        ckpts = _discover_checkpoints(args.checkpoints_dir)
        if not ckpts:
            raise RuntimeError(f"No checkpoint-* directories found under: {args.checkpoints_dir}")
        for step, path in ckpts:
            model_specs.append({"tag": "checkpoint", "step": str(step), "model_path": path, "kind": "checkpoint"})
    else:
        # Single model mode: evaluate one LoRA (optionally alongside base if include_base).
        if not args.lora_model:
            raise ValueError("Provide either --checkpoints_dir (curve mode) or --lora_model (single mode).")
        model_specs.append({"tag": "lora", "step": "0", "model_path": args.lora_model, "kind": "lora"})

    all_rows: List[Dict] = []

    # Phase 1: serial GPU generation for each model/checkpoint.
    for spec in model_specs:
        kind = spec["kind"]
        if kind == "base":
            model, tok = load_base_model(args.base_model)
        elif kind == "checkpoint":
            model, tok = load_lora_model(
                base_model_name=args.base_model,
                lora_path_or_model_dir=spec["model_path"],
                lora_is_merged=False,
                merge_lora=args.merge_lora,
            )
        elif kind == "lora":
            model, tok = load_lora_model(
                base_model_name=args.base_model,
                lora_path_or_model_dir=spec["model_path"],
                lora_is_merged=args.lora_is_merged,
                merge_lora=args.merge_lora,
            )
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        rows = _generate_for_model(model=model, tokenizer=tok, items=items, args=args, model_meta=spec)
        all_rows.extend(rows)

        del model
        torch.cuda.empty_cache()

    # Phase 2: one combined RAGAS evaluation (parallel judge requests).
    ds_eval = Dataset.from_list(all_rows)
    df_scores = ragas_faithfulness_eval(ds_eval, judge_model=args.judge_model, ragas_batch_size=args.ragas_batch_size)

    # RAGAS may drop metadata columns; rely on row order to reattach.
    if "faithfulness" in df_scores.columns:
        scores = df_scores["faithfulness"].tolist()
    else:
        scores = [float("nan")] * len(all_rows)

    for i, s in enumerate(scores):
        all_rows[i]["faithfulness"] = float(s) if s == s else float("nan")  # NaN-safe

    _write_samples_jsonl(all_rows, args.out_jsonl)
    _write_curve_csv(all_rows, args.out_csv, subsets=subsets)


if __name__ == "__main__":
    main()
