from __future__ import annotations

import argparse
from typing import Any, Dict, List

import torch

try:
    from src.llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec

from dataio import write_jsonl
from .common import load_generation_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path (save_to_disk dir or JSON/JSONL).")
    parser.add_argument("--split", type=str, default="test", help="Split name when data_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=200, help="Maximum sampled rows. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)

    # Models
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="Single LoRA adapter checkpoint path.")
    parser.add_argument(
        "--lora_ckpt_list_path",
        type=str,
        default=None,
        help="Directory containing checkpoint-* subdirectories, or a text file with checkpoint paths.",
    )
    parser.add_argument(
        "--include_base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include the base model in generation.",
    )

    # Generation
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--strip_think_tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strip_role_markers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encourage_refusal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_track", type=str, default="base_task", help="Evaluation track label saved into generated rows.")
    parser.add_argument("--eval_variant", type=str, default="", help="Optional evaluation variant label. Defaults to think/no_think from generation mode.")

    # Output
    parser.add_argument("--out_jsonl", type=str, required=True, help="Generated answers JSONL output path.")
    return parser.parse_args()


def run_answer_generation(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Generate non-structured QA answers for one or more model specs."""

    items = load_generation_items(
        data_path=args.data_path,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not items:
        raise RuntimeError("No valid rows found for generation after loading/filtering.")

    model_specs: List[GeneratorModelSpec] = build_generator_model_specs(
        base_model_name=args.base_model,
        lora_ckpt_path=args.lora_ckpt_path,
        lora_ckpt_list_path=args.lora_ckpt_list_path,
        include_base=bool(args.include_base),
    )

    generated_rows: List[Dict[str, Any]] = []
    eval_variant = str(args.eval_variant or ("think" if bool(args.enable_thinking) else "no_think"))
    for spec in model_specs:
        generator = load_generator_from_spec(
            spec=spec,
            base_model_name=args.base_model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
            strip_think_tags=args.strip_think_tags,
            strip_role_markers=args.strip_role_markers,
            seed=args.seed,
        )

        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            qa_items = [
                {
                    "knowledge": str(item["knowledge"]),
                    "question": str(item["question"]),
                }
                for item in batch
            ]

            answers = generator.generate_many_from_qa(
                qa_items=qa_items,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                use_chat_template=args.use_chat_template,
                enable_thinking=args.enable_thinking,
                strip_think_tags=args.strip_think_tags,
                strip_role_markers=args.strip_role_markers,
                encourage_refusal=bool(args.encourage_refusal),
            )

            for sample, answer in zip(batch, answers):
                generated_rows.append(
                    {
                        "model_tag": spec.tag,
                        "model_step": int(spec.step),
                        "model_path": spec.display_name,
                        "id": str(sample.get("id") or sample["source_id"]),
                        "sample_id": int(sample["sample_id"]),
                        "source_id": sample["source_id"],
                        "question": sample["question"],
                        "knowledge": sample["knowledge"],
                        "reference_answer": sample["reference_answer"],
                        "answerability_label": sample.get("answerability_label") or "",
                        "data_split": sample.get("data_split") or args.split,
                        "eval_track": str(args.eval_track or "base_task"),
                        "eval_variant": eval_variant,
                        "enable_thinking": bool(args.enable_thinking),
                        "answer": answer,
                    }
                )

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generated_rows


def main() -> None:
    args = parse_args()
    generated_rows = run_answer_generation(args)
    write_jsonl(args.out_jsonl, generated_rows)
    print(f"Saved generated answers to: {args.out_jsonl}")


if __name__ == "__main__":
    main()
