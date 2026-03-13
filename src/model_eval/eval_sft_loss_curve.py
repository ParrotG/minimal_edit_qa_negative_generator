from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from project_config.resolve import resolve_eval_args

try:
    from src.llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec

from dataio import write_jsonl

from .common import load_dataset_split, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path (save_to_disk dir or JSON/JSONL).")
    parser.add_argument("--split", type=str, default=None, help="Split name when data_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum sampled rows. -1 means all.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="Single LoRA adapter checkpoint path.")
    parser.add_argument(
        "--lora_ckpt_list_path",
        type=str,
        default=None,
        help="Directory containing checkpoint-* subdirectories, or a text file with checkpoint paths.",
    )
    parser.add_argument("--include_base", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--eval_track", type=str, default=None, help="Evaluation track label written into the curve rows.")
    parser.add_argument("--eval_variant", type=str, default=None, help="Evaluation variant label written into the curve rows.")
    parser.add_argument("--out_csv", type=str, required=True, help="Per-model loss curve CSV path.")
    parser.add_argument("--details_out", type=str, default=None, help="Optional per-sample loss details JSONL path.")
    return resolve_eval_args(parser.parse_args(), preset="loss_curve")


def _extract_eval_rows(data_path: str, split: str, max_samples: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset_split(data_path=data_path, split=split).shuffle(seed=seed)
    out: List[Dict[str, Any]] = []
    kept = 0
    for idx, row in enumerate(ds):
        prompt = str(row.get("prompt") or "").strip()
        completion = str(row.get("completion") or "").strip()
        if not prompt or not completion:
            out.append(
                {
                    "sample_id": int(row.get("sample_id") or idx),
                    "source_id": str(row.get("source_id") or row.get("id") or idx),
                    "prompt": prompt,
                    "completion": completion,
                    "skip_reason": "missing_prompt_or_completion",
                }
            )
            continue
        out.append(
            {
                "sample_id": int(row.get("sample_id") or idx),
                "source_id": str(row.get("source_id") or row.get("id") or idx),
                "prompt": prompt,
                "completion": completion,
                "skip_reason": "",
            }
        )
        kept += 1
        if max_samples > 0 and kept >= max_samples:
            break
    return out


def _tokenize_prompt_completion(tokenizer: Any, prompt: str, completion: str, max_length: int) -> Tuple[List[int], List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        completion_ids = completion_ids + [int(eos_id)]

    available_prompt = max_length - len(completion_ids)
    if available_prompt < 0:
        completion_ids = completion_ids[:max_length]
        available_prompt = 0

    prompt_ids = prompt_ids[-available_prompt:] if available_prompt > 0 else []
    input_ids = [int(x) for x in (prompt_ids + completion_ids)]
    labels = [-100] * len(prompt_ids) + [int(x) for x in completion_ids]
    return input_ids, labels


def _pad_batch(input_ids: Sequence[Sequence[int]], labels: Sequence[Sequence[int]], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in input_ids)
    batch_input: List[List[int]] = []
    batch_labels: List[List[int]] = []
    for ids, lbs in zip(input_ids, labels):
        pad = max_len - len(ids)
        batch_input.append(list(ids) + [pad_token_id] * pad)
        batch_labels.append(list(lbs) + [-100] * pad)
    return torch.tensor(batch_input, dtype=torch.long), torch.tensor(batch_labels, dtype=torch.long)


@torch.inference_mode()
def _compute_batch_losses(
    model: Any,
    batch_input_ids: torch.Tensor,
    batch_labels: torch.Tensor,
    *,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=batch_input_ids, attention_mask=batch_input_ids.ne(int(pad_token_id)).long())
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch_labels[:, 1:].contiguous()
    token_mask = shift_labels.ne(-100)

    loss_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    token_loss = loss_flat.view(shift_labels.shape[0], shift_labels.shape[1])
    row_nll = (token_loss * token_mask).sum(dim=1)
    row_tokens = token_mask.sum(dim=1)
    return row_nll, row_tokens


def _evaluate_one_model(
    *,
    model_spec: GeneratorModelSpec,
    base_model: str,
    rows: Sequence[Dict[str, Any]],
    batch_size: int,
    max_length: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    generator = load_generator_from_spec(
        spec=model_spec,
        base_model_name=base_model,
        batch_size=batch_size,
        seed=seed,
        use_chat_template=False,
    )
    model = generator.model
    tokenizer = generator.tokenizer
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token.")
    pad_token_id = int(tokenizer.pad_token_id)
    device = next(model.parameters()).device

    valid_rows = [row for row in rows if not row.get("skip_reason")]
    num_skipped_missing_completion = len(rows) - len(valid_rows)
    if not valid_rows:
        summary = {
            "model_tag": model_spec.tag,
            "model_step": int(model_spec.step),
            "model_path": model_spec.display_name,
            "num_rows": len(rows),
            "num_used_rows": 0,
            "num_skipped_missing_completion": num_skipped_missing_completion,
            "mean_loss": float("nan"),
            "mean_nll_per_token": float("nan"),
            "perplexity": float("nan"),
        }
        return summary, []

    total_nll = 0.0
    total_tokens = 0.0
    row_losses: List[float] = []
    details: List[Dict[str, Any]] = []

    for start in range(0, len(valid_rows), max(1, int(batch_size))):
        batch = valid_rows[start : start + max(1, int(batch_size))]
        tokenized = [
            _tokenize_prompt_completion(tokenizer, str(row["prompt"]), str(row["completion"]), max_length=max_length)
            for row in batch
        ]
        input_ids = [item[0] for item in tokenized]
        labels = [item[1] for item in tokenized]
        batch_input_ids, batch_labels = _pad_batch(input_ids=input_ids, labels=labels, pad_token_id=pad_token_id)
        batch_input_ids = batch_input_ids.to(device)
        batch_labels = batch_labels.to(device)

        row_nll, row_tokens = _compute_batch_losses(
            model,
            batch_input_ids,
            batch_labels,
            pad_token_id=pad_token_id,
        )
        row_nll_list = row_nll.detach().float().cpu().tolist()
        row_tokens_list = row_tokens.detach().float().cpu().tolist()
        for row, nll, tok_count in zip(batch, row_nll_list, row_tokens_list):
            token_count = int(tok_count)
            if token_count <= 0:
                continue
            mean_row_loss = float(nll / max(1, token_count))
            row_losses.append(mean_row_loss)
            total_nll += float(nll)
            total_tokens += float(token_count)
            details.append(
                {
                    "model_tag": model_spec.tag,
                    "model_step": int(model_spec.step),
                    "model_path": model_spec.display_name,
                    "eval_track": "sft_structured",
                    "eval_variant": "checkpoint",
                    "sample_id": row["sample_id"],
                    "source_id": row["source_id"],
                    "num_completion_tokens": token_count,
                    "mean_loss": mean_row_loss,
                }
            )

    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_loss = float(sum(row_losses) / len(row_losses)) if row_losses else float("nan")
    mean_nll_per_token = float(total_nll / total_tokens) if total_tokens > 0 else float("nan")
    perplexity = float(math.exp(mean_nll_per_token)) if mean_nll_per_token == mean_nll_per_token else float("nan")

    summary = {
        "model_tag": model_spec.tag,
        "model_step": int(model_spec.step),
        "model_path": model_spec.display_name,
        "eval_track": "sft_structured",
        "eval_variant": "checkpoint",
        "num_rows": len(rows),
        "num_used_rows": len(row_losses),
        "num_skipped_missing_completion": num_skipped_missing_completion,
        "mean_loss": mean_loss,
        "mean_nll_per_token": mean_nll_per_token,
        "perplexity": perplexity,
    }
    return summary, details


def run_sft_loss_curve(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute SFT loss summaries and optional per-sample detail rows."""

    if int(args.max_length) <= 0:
        raise ValueError("--max_length must be positive.")

    rows = _extract_eval_rows(
        data_path=args.data_path,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not rows:
        raise RuntimeError("No rows found for SFT loss evaluation.")

    model_specs = build_generator_model_specs(
        base_model_name=args.base_model,
        lora_ckpt_path=args.lora_ckpt_path,
        lora_ckpt_list_path=args.lora_ckpt_list_path,
        include_base=bool(args.include_base),
    )

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for spec in model_specs:
        summary, details = _evaluate_one_model(
            model_spec=spec,
            base_model=args.base_model,
            rows=rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=args.seed,
        )
        summary["eval_track"] = str(args.eval_track or "sft_structured")
        summary["eval_variant"] = str(args.eval_variant or "checkpoint")
        for row in details:
            row["eval_track"] = str(args.eval_track or "sft_structured")
            row["eval_variant"] = str(args.eval_variant or "checkpoint")
        summary_rows.append(summary)
        detail_rows.extend(details)

    return summary_rows, detail_rows


def main() -> None:
    args = parse_args()
    summary_rows, detail_rows = run_sft_loss_curve(args)

    write_csv(summary_rows, args.out_csv)
    print(f"Saved SFT loss curve to: {args.out_csv}")
    if args.details_out:
        write_jsonl(args.details_out, detail_rows)
        print(f"Saved SFT loss details to: {args.details_out}")


if __name__ == "__main__":
    main()
