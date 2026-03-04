from __future__ import annotations

import argparse
from typing import Any, Dict, List

import torch

try:
    from src.llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec

from dataio import pick_first_non_empty_str, write_jsonl
from qa_protocol import build_infer_prompt, parse_structured_output

from .common import load_dataset_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path (save_to_disk dir or JSON/JSONL).")
    parser.add_argument("--split", type=str, default="test", help="Split name when data_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=200, help="Maximum sampled rows. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="Single LoRA adapter checkpoint path.")
    parser.add_argument(
        "--lora_ckpt_list_path",
        type=str,
        default=None,
        help="Directory containing checkpoint-* subdirectories, or a text file with checkpoint paths.",
    )
    parser.add_argument("--include_base", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--out_jsonl", type=str, required=True, help="Generated outputs JSONL path.")
    return parser.parse_args()


def _load_eval_items(data_path: str, split: str, max_samples: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset_split(data_path=data_path, split=split).shuffle(seed=seed)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        question = pick_first_non_empty_str(row, ["question", "eval_question", "input"])
        knowledge = pick_first_non_empty_str(row, ["knowledge", "context"])
        if not question or not knowledge:
            continue
        out.append(
            {
                "sample_id": idx,
                "source_id": str(row.get("source_id") or row.get("id") or idx),
                "question": question,
                "knowledge": knowledge,
                "reference_answer": pick_first_non_empty_str(row, ["reference_answer", "answer"]),
                "answerability_label": str(row.get("answerability_label") or "").strip(),
                "data_split": row.get("data_split") or split,
            }
        )
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def main() -> None:
    args = parse_args()
    items = _load_eval_items(
        data_path=args.data_path,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not items:
        raise RuntimeError("No valid rows found for structured generation.")

    model_specs: List[GeneratorModelSpec] = build_generator_model_specs(
        base_model_name=args.base_model,
        lora_ckpt_path=args.lora_ckpt_path,
        lora_ckpt_list_path=args.lora_ckpt_list_path,
        include_base=bool(args.include_base),
    )

    generated_rows: List[Dict[str, Any]] = []
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
            prompts = [build_infer_prompt(knowledge=item["knowledge"], question=item["question"]) for item in batch]
            outputs = generator.generate_many(
                prompts,
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
            )

            for sample, prompt, raw_output in zip(batch, prompts, outputs):
                parse_result = parse_structured_output(raw_output)
                generated_rows.append(
                    {
                        "model_tag": spec.tag,
                        "model_step": int(spec.step),
                        "model_path": spec.display_name,
                        "sample_id": int(sample["sample_id"]),
                        "source_id": sample["source_id"],
                        "question": sample["question"],
                        "knowledge": sample["knowledge"],
                        "reference_answer": sample["reference_answer"],
                        "answerability_label": sample["answerability_label"],
                        "data_split": sample["data_split"],
                        "prompt_style": "infer_v1",
                        "prompt": prompt,
                        "raw_output": raw_output,
                        "answer": raw_output,
                        "parse_ok": bool(parse_result.ok),
                        "parsed_output": None if parse_result.parsed is None else parse_result.parsed.model_dump(mode="json"),
                        "parse_errors": list(parse_result.errors),
                    }
                )

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(args.out_jsonl, generated_rows)
    print(f"Saved structured generations to: {args.out_jsonl}")


if __name__ == "__main__":
    main()
