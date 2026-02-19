from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec

from .common import to_chat_prompt


DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true")

    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Max total length for prompt+completion tokenization in log-prob computation.",
    )

    parser.add_argument("--base_model", type=str, default=None, help="Base model name/path for LoRA adapter checkpoints.")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="Single LoRA adapter checkpoint path.")
    parser.add_argument(
        "--lora_ckpt_list_path",
        type=str,
        default=None,
        help="Directory containing checkpoint-* subdirectories.",
    )
    parser.add_argument("--include_base", action=argparse.BooleanOptionalAction, default=False)

    # Backward-compatible aliases from previous script versions.
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Alias of --lora_ckpt_path.")
    parser.add_argument("--checkpoints_dir", type=str, default=None, help="Alias of --lora_ckpt_list_path.")

    parser.add_argument(
        "--out_csv",
        type=str,
        default="pairwise_preference_curve.csv",
        help="Output CSV for curve mode (or single mode if provided).",
    )

    return parser.parse_args()


def _discover_checkpoints(checkpoints_dir: str) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    if not os.path.isdir(checkpoints_dir):
        raise ValueError(f"checkpoints_dir does not exist: {checkpoints_dir}")

    for name in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, name)
        if not os.path.isdir(path):
            continue
        match = re.match(r"^checkpoint-(\d+)$", name)
        if match:
            items.append((int(match.group(1)), path))

    items.sort(key=lambda x: x[0])
    return items


def _infer_adapter_base_model(path: str) -> Optional[str]:
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return None

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_model = cfg.get("base_model_name_or_path")
    if isinstance(base_model, str) and base_model.strip():
        return base_model.strip()
    return None


def _is_adapter_checkpoint(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _resolve_model_specs(args: argparse.Namespace) -> Tuple[str, List[GeneratorModelSpec]]:
    lora_ckpt_path = args.lora_ckpt_path
    lora_ckpt_list_path = args.lora_ckpt_list_path
    include_base = bool(args.include_base)

    if args.model_name_or_path:
        if lora_ckpt_path or lora_ckpt_list_path or args.checkpoints_dir:
            raise ValueError(
                "Do not mix deprecated --model_name_or_path with --lora_ckpt_path/--lora_ckpt_list_path/--checkpoints_dir."
            )
        lora_ckpt_path = args.model_name_or_path

    if args.checkpoints_dir:
        if lora_ckpt_path or lora_ckpt_list_path:
            raise ValueError("Do not mix deprecated --checkpoints_dir with --lora_ckpt_path/--lora_ckpt_list_path.")
        lora_ckpt_list_path = args.checkpoints_dir

    if not include_base and not lora_ckpt_path and not lora_ckpt_list_path:
        raise ValueError(
            "No model selected. Provide --lora_ckpt_path/--lora_ckpt_list_path "
            "(or deprecated --model_name_or_path/--checkpoints_dir), or set --include_base."
        )

    if lora_ckpt_path and not _is_adapter_checkpoint(lora_ckpt_path):
        raise ValueError(f"Expected an unmerged LoRA adapter path, but adapter_config.json is missing: {lora_ckpt_path}")
    if lora_ckpt_list_path:
        for _, ckpt_path in _discover_checkpoints(lora_ckpt_list_path):
            if not _is_adapter_checkpoint(ckpt_path):
                raise ValueError(
                    f"Expected unmerged LoRA adapter checkpoints, but adapter_config.json is missing: {ckpt_path}"
                )

    base_model = (args.base_model or "").strip()
    if not base_model:
        if lora_ckpt_path:
            base_model = _infer_adapter_base_model(lora_ckpt_path) or DEFAULT_BASE_MODEL
        elif lora_ckpt_list_path:
            inferred: Optional[str] = None
            for _, ckpt_path in _discover_checkpoints(lora_ckpt_list_path):
                inferred = _infer_adapter_base_model(ckpt_path)
                if inferred:
                    break
            base_model = inferred or DEFAULT_BASE_MODEL
        else:
            base_model = DEFAULT_BASE_MODEL

    model_specs = build_generator_model_specs(
        base_model_name=base_model,
        lora_ckpt_path=lora_ckpt_path,
        lora_ckpt_list_path=lora_ckpt_list_path,
        include_base=include_base,
    )
    return base_model, model_specs


def _get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


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
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
        prompt_lens.append(int(ids.shape[0]))

    scores: List[torch.Tensor] = []
    for i, prompt_len in enumerate(prompt_lens):
        start = max(prompt_len - 1, 0)
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
    *,
    base_model_name: str,
    model_spec: GeneratorModelSpec,
    ds: Dataset,
    batch_size: int,
    enable_thinking: bool,
    max_length: int,
    seed: int,
) -> Dict[str, float]:
    """Evaluate pairwise preference accuracy and margin stats."""

    generator = load_generator_from_spec(
        spec=model_spec,
        base_model_name=base_model_name,
        batch_size=batch_size,
        use_chat_template=True,
        enable_thinking=enable_thinking,
        seed=seed,
    )
    model = generator.model
    tokenizer = generator.tokenizer

    correct = 0
    total = 0
    margins: List[float] = []

    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))

        raw_prompts = [x["prompt"] for x in batch]
        prompts = [to_chat_prompt(tokenizer, p, enable_thinking=enable_thinking) for p in raw_prompts]
        chosen = [x["chosen"] for x in batch]
        rejected = [x["rejected"] for x in batch]

        chosen_lp = sequence_logprob(model, tokenizer, prompts, chosen, max_length=max_length)
        rejected_lp = sequence_logprob(model, tokenizer, prompts, rejected, max_length=max_length)

        chosen_lp_list = chosen_lp.detach().float().cpu().tolist()
        rejected_lp_list = rejected_lp.detach().float().cpu().tolist()

        for c_lp, r_lp in zip(chosen_lp_list, rejected_lp_list):
            is_correct = 1 if c_lp > r_lp else 0
            margin = float(c_lp - r_lp)

            correct += is_correct
            total += 1
            margins.append(margin)

    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    return out


def main() -> None:
    args = parse_args()

    ds_obj = load_from_disk(args.data_dir)
    if isinstance(ds_obj, DatasetDict):
        ds = ds_obj[args.split]
    elif isinstance(ds_obj, Dataset):
        ds = ds_obj
    else:
        raise ValueError(f"Unsupported dataset object from {args.data_dir}: {type(ds_obj)}")

    ds = ds.shuffle(seed=args.seed)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    base_model_name, model_specs = _resolve_model_specs(args)
    rows: List[Dict[str, str]] = []

    for spec in model_specs:
        metrics = evaluate_model(
            base_model_name=base_model_name,
            model_spec=spec,
            ds=ds,
            batch_size=args.batch_size,
            enable_thinking=args.enable_thinking,
            max_length=args.max_length,
            seed=args.seed,
        )
        row = {
            "tag": spec.tag,
            "step": str(spec.step),
            "model_path": spec.display_name,
        }
        row.update({k: f"{v}" for k, v in metrics.items()})
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
