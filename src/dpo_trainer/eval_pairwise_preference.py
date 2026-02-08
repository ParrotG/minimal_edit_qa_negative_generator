import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_from_disk
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .halueval_common import to_chat_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true")

    # Ensure the log-prob computation uses the same truncation limit as training.
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Max total length for prompt+completion tokenization in log-prob computation.",
    )

    # Single model mode
    parser.add_argument("--model_name_or_path", type=str, default=None)

    # Curve mode
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="If set, scan checkpoint-* under this directory and evaluate each for a learning curve.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="pairwise_preference_curve.csv",
        help="Output CSV for curve mode (or single mode if provided).",
    )

    return parser.parse_args()


def _get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_model_and_tokenizer(model_path: str):
    """
    Load either a full model checkpoint or a PEFT adapter checkpoint.
    """
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base_model_id = peft_cfg.base_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype="auto",
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def sequence_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    completions: List[str],
    max_length: int,
) -> torch.Tensor:
    """
    Compute sum log P(completion | prompt) for each pair in the batch.
    Returns a tensor of shape (batch,).
    """
    assert len(prompts) == len(completions)

    full_texts = [p + c for p, c in zip(prompts, completions)]
    tok_full = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(_get_model_device(model))

    input_ids = tok_full["input_ids"]
    attention_mask = tok_full["attention_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    prompt_lens: List[int] = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
        prompt_lens.append(int(ids.shape[0]))

    scores: List[torch.Tensor] = []
    for i, pl in enumerate(prompt_lens):
        start = max(pl - 1, 0)
        mask_i = shift_mask[i].bool()
        mask_i[:start] = False
        scores.append((token_logp[i][mask_i]).sum())

    return torch.stack(scores, dim=0)


def _quantile(x: List[float], q: float) -> float:
    if not x:
        return float("nan")
    xs = sorted(x)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def evaluate_model(
    model_path: str,
    ds: Dataset,
    batch_size: int,
    enable_thinking: bool,
    max_length: int,
) -> Dict[str, float]:
    """
    Evaluate pairwise preference accuracy and margin stats.
    """
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    correct = 0
    total = 0
    margins: List[float] = []

    # Optional per-task breakdown if task column exists.
    per_task_correct: Dict[str, int] = {}
    per_task_total: Dict[str, int] = {}

    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))

        raw_prompts = [x["prompt"] for x in batch]
        prompts = [to_chat_prompt(tokenizer, p, enable_thinking=enable_thinking) for p in raw_prompts]
        chosen = [x["chosen"] for x in batch]
        rejected = [x["rejected"] for x in batch]
        tasks = [x.get("task", "__unknown__") for x in batch]

        chosen_lp = sequence_logprob(model, tokenizer, prompts, chosen, max_length=max_length)
        rejected_lp = sequence_logprob(model, tokenizer, prompts, rejected, max_length=max_length)

        chosen_lp_list = chosen_lp.detach().float().cpu().tolist()
        rejected_lp_list = rejected_lp.detach().float().cpu().tolist()

        for t, c_lp, r_lp in zip(tasks, chosen_lp_list, rejected_lp_list):
            ok = 1 if c_lp > r_lp else 0
            m = float(c_lp - r_lp)

            correct += ok
            total += 1
            margins.append(m)

            per_task_correct[t] = per_task_correct.get(t, 0) + ok
            per_task_total[t] = per_task_total.get(t, 0) + 1

    acc = correct / max(1, total)
    margin_mean = sum(margins) / max(1, len(margins))
    margin_std = (sum((m - margin_mean) ** 2 for m in margins) / max(1, len(margins))) ** 0.5

    out: Dict[str, float] = {
        "pairwise_acc": acc,
        "margin_mean": margin_mean,
        "margin_std": margin_std,
        "margin_p50": _quantile(margins, 0.50),
        "margin_p90": _quantile(margins, 0.90),
        "margin_p95": _quantile(margins, 0.95),
    }

    # Add per-task accuracies if available.
    for t in sorted(per_task_total.keys()):
        out[f"pairwise_acc_task/{t}"] = per_task_correct.get(t, 0) / max(1, per_task_total[t])

    return out


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


def main() -> None:
    args = parse_args()

    ds_dict = load_from_disk(args.data_dir)
    ds = ds_dict[args.split].shuffle(seed=args.seed)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    rows: List[Dict[str, str]] = []

    if args.checkpoints_dir:
        ckpts = _discover_checkpoints(args.checkpoints_dir)
        if not ckpts:
            raise RuntimeError(f"No checkpoint-* directories found under: {args.checkpoints_dir}")

        for step, path in ckpts:
            metrics = evaluate_model(
                model_path=path,
                ds=ds,
                batch_size=args.batch_size,
                enable_thinking=args.enable_thinking,
                max_length=args.max_length,
            )
            row = {"step": str(step), "model_path": path}
            row.update({k: f"{v}" for k, v in metrics.items()})
            rows.append(row)

    else:
        if not args.model_name_or_path:
            raise ValueError("Provide either --model_name_or_path (single mode) or --checkpoints_dir (curve mode).")
        metrics = evaluate_model(
            model_path=args.model_name_or_path,
            ds=ds,
            batch_size=args.batch_size,
            enable_thinking=args.enable_thinking,
            max_length=args.max_length,
        )
        row = {"step": "0", "model_path": args.model_name_or_path}
        row.update({k: f"{v}" for k, v in metrics.items()})
        rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
