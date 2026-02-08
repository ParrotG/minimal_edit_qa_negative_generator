import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from trl import DPOConfig, DPOTrainer

from .halueval_common import to_chat_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3_dpo_lora_pairs")

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--enable_thinking", action="store_true")

    # Tracking / logging
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Comma-separated list of trackers: "tensorboard", "wandb", "mlflow", "clearml", or "none".',
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name for the tracker. If not set, derived from output_dir.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="Optional logging directory (TensorBoard). If not set, uses output_dir/runs.",
    )

    # Diagnostic metrics
    parser.add_argument(
        "--diag_enable",
        action="store_true",
        help="If set, compute extra diagnostic metrics (pairwise logprob accuracy, margin stats) during evaluation.",
    )
    parser.add_argument(
        "--diag_max_samples",
        type=int,
        default=256,
        help="Max number of eval samples used for diagnostics at each evaluation.",
    )
    parser.add_argument(
        "--diag_batch_size",
        type=int,
        default=4,
        help="Batch size for diagnostic computations (logprob based).",
    )
    parser.add_argument(
        "--diag_eval_every",
        type=int,
        default=1,
        help="Run diagnostics every N evaluations. Use >1 to reduce overhead.",
    )

    parser.add_argument(
        "--precompute_ref_log_probs",
        action="store_true",
        help="If set, precompute reference model logprobs (supported by newer TRL versions).",
    )

    return parser.parse_args()


def _parse_report_to(report_to: str) -> List[str]:
    report_to = report_to.strip()
    if not report_to or report_to.lower() == "none":
        return []
    return [x.strip() for x in report_to.split(",") if x.strip()]


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get a reasonable device handle for models that may be sharded or wrapped.
    """
    return next(model.parameters()).device


@torch.no_grad()
def _sequence_logprob(
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

    # Compute prompt lengths individually to mask out prompt tokens.
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
    """
    Compute quantile with linear interpolation on a Python list.
    """
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


@dataclass
class DiagnosticConfig:
    max_samples: int
    batch_size: int
    eval_every: int
    seed: int


class PairwiseDiagnosticsCallback(TrainerCallback):
    """
    Log extra diagnostics during evaluation:
    - pairwise accuracy based on logP(chosen|prompt) > logP(rejected|prompt)
    - margin distribution statistics
    """

    def __init__(self, tokenizer: AutoTokenizer, eval_dataset: Dataset, cfg: DiagnosticConfig):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.cfg = cfg
        self._eval_count = 0
        self._fixed_indices: Optional[List[int]] = None

    def _build_fixed_indices(self) -> List[int]:
        n = len(self.eval_dataset)
        k = min(self.cfg.max_samples, n) if self.cfg.max_samples > 0 else n
        rng = random.Random(self.cfg.seed)
        indices = list(range(n))
        rng.shuffle(indices)
        return indices[:k]

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        model = kwargs.get("model", None)

        if trainer is None or model is None:
            return control
        if not trainer.is_world_process_zero():
            return control

        self._eval_count += 1
        if self.cfg.eval_every > 1 and (self._eval_count % self.cfg.eval_every != 0):
            return control

        if self._fixed_indices is None:
            self._fixed_indices = self._build_fixed_indices()

        ds = self.eval_dataset.select(self._fixed_indices)

        chosen_logps: List[float] = []
        rejected_logps: List[float] = []
        margins: List[float] = []
        correct = 0
        total = 0

        model.eval()

        for start in range(0, len(ds), self.cfg.batch_size):
            batch = ds.select(range(start, min(start + self.cfg.batch_size, len(ds))))

            prompts = [x["prompt"] for x in batch]
            chosen = [x["chosen"] for x in batch]
            rejected = [x["rejected"] for x in batch]

            c_lp = _sequence_logprob(model, self.tokenizer, prompts, chosen, args.max_length)
            r_lp = _sequence_logprob(model, self.tokenizer, prompts, rejected, args.max_length)

            c_lp_list = c_lp.detach().float().cpu().tolist()
            r_lp_list = r_lp.detach().float().cpu().tolist()

            for a, b in zip(c_lp_list, r_lp_list):
                chosen_logps.append(float(a))
                rejected_logps.append(float(b))
                m = float(a - b)
                margins.append(m)
                correct += 1 if a > b else 0
                total += 1

        acc = correct / max(1, total)
        margin_mean = sum(margins) / max(1, len(margins))
        margin_std = (sum((m - margin_mean) ** 2 for m in margins) / max(1, len(margins))) ** 0.5

        logs = {
            "diag/pairwise_acc": acc,
            "diag/margin_mean": margin_mean,
            "diag/margin_std": margin_std,
            "diag/margin_p50": _quantile(margins, 0.50),
            "diag/margin_p90": _quantile(margins, 0.90),
            "diag/margin_p95": _quantile(margins, 0.95),
            "diag/chosen_logp_mean": sum(chosen_logps) / max(1, len(chosen_logps)),
            "diag/rejected_logp_mean": sum(rejected_logps) / max(1, len(rejected_logps)),
            "diag/samples": int(total),
        }

        trainer.log(logs)
        return control


def _build_training_args(args: argparse.Namespace) -> DPOConfig:
    """
    Build DPOConfig with compatibility fallbacks across TRL/Transformers versions.
    """
    report_to_list = _parse_report_to(args.report_to)
    run_name = args.run_name if args.run_name else os.path.basename(os.path.abspath(args.output_dir))
    logging_dir = args.logging_dir if args.logging_dir else os.path.join(args.output_dir, "runs")

    base_kwargs = dict(
        output_dir=args.output_dir,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=report_to_list,
        run_name=run_name,
        logging_dir=logging_dir,
        load_best_model_at_end=True,
        metric_for_best_model="rewards/margins",
        greater_is_better=True,
        save_safetensors=True,
    )

    # Prefer the canonical TrainingArguments keys.
    kwargs = dict(base_kwargs)
    kwargs["evaluation_strategy"] = "steps"
    kwargs["save_strategy"] = "steps"

    if args.precompute_ref_log_probs:
        kwargs["precompute_ref_log_probs"] = True

    # Try multiple fallbacks for older/newer TRL versions.
    try:
        return DPOConfig(**kwargs)
    except TypeError:
        # Fallback: eval_strategy key name (used by some TRL versions)
        kwargs2 = dict(base_kwargs)
        kwargs2["eval_strategy"] = "steps"
        kwargs2["save_strategy"] = "steps"
        if args.precompute_ref_log_probs:
            kwargs2["precompute_ref_log_probs"] = True
        try:
            return DPOConfig(**kwargs2)
        except TypeError:
            # Fallback: precompute_ref_log_probs might be unsupported.
            if "precompute_ref_log_probs" in kwargs2:
                kwargs2.pop("precompute_ref_log_probs", None)
            return DPOConfig(**kwargs2)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_from_disk(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype="auto",
    )
    model.config.use_cache = False

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    def _map_chat(row: Dict) -> Dict:
        # Convert plain prompt to chat-wrapped prompt for training.
        row["prompt"] = to_chat_prompt(tokenizer, row["prompt"], enable_thinking=args.enable_thinking)
        return row

    ds = ds.map(_map_chat)

    training_args = _build_training_args(args)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if args.diag_enable:
        diag_cfg = DiagnosticConfig(
            max_samples=args.diag_max_samples,
            batch_size=args.diag_batch_size,
            eval_every=args.diag_eval_every,
            seed=args.seed,
        )
        trainer.add_callback(PairwiseDiagnosticsCallback(tokenizer=tokenizer, eval_dataset=ds["validation"], cfg=diag_cfg))

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
