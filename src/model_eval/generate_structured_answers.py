from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import torch

try:
    from src.llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec

from dataio import write_jsonl
from project_config.resolve import resolve_structured_eval_args_by_mode
from qa_checks import check_protocol_constraints
from qa_protocol import build_infer_prompt, build_teacher_prompt, parse_structured_output

from .common import load_structured_generation_items


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
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--prompt_mode", type=str, default=None, choices=["infer", "teacher", "teacher_fewshot"])
    parser.add_argument("--fewshot_k", type=int, default=None, help="Number of few-shot examples for teacher_fewshot mode.")
    parser.add_argument(
        "--retry_on_protocol_fail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compatibility flag kept for older scripts. Structured evaluation now always uses a single pass.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help="Compatibility flag kept for older scripts. Structured evaluation now always uses a single pass.",
    )
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strip_think_tags", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strip_role_markers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--record_token_usage", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval_track", type=str, default=None, help="Optional evaluation track label saved into generated rows.")
    parser.add_argument("--eval_variant", type=str, default=None, help="Optional evaluation variant label saved into generated rows.")
    parser.add_argument("--out_jsonl", type=str, required=True, help="Generated outputs JSONL path.")
    args = resolve_structured_eval_args_by_mode(parser.parse_args())
    args.retry_on_protocol_fail = False
    args.max_attempts = 1
    return args


def _fewshot_examples() -> List[Dict[str, Any]]:
    return [
        {
            "knowledge": "DocA: The capital of Italy is Rome.",
            "question": "What is the capital of Italy?",
            "output": {
                "answerability": "answerable",
                "evidence": [{"quote": "The capital of Italy is Rome."}],
                "rationale": "The evidence directly states the capital of Italy.",
                "answer": "Rome",
                "confidence": "high",
            },
        },
        {
            "knowledge": "DocB: Marie Curie discovered polonium.\nDocC: The document does not mention her birthplace.",
            "question": "Where was Marie Curie born?",
            "output": {
                "answerability": "unanswerable",
                "evidence": [],
                "rationale": "The provided knowledge mentions discoveries but does not provide her birthplace.",
                "answer": "I don't know based on the provided knowledge.",
                "confidence": "low",
            },
        },
        {
            "knowledge": "DocD: The tallest mountain in Earth is Mount Everest with a height of 8,848 meters.",
            "question": "Which mountain is the tallest on Earth?",
            "output": {
                "answerability": "answerable",
                "evidence": [{"quote": "The tallest mountain in Earth is Mount Everest"}],
                "rationale": "The evidence explicitly identifies the tallest mountain.",
                "answer": "Mount Everest",
                "confidence": "high",
            },
        },
    ]


def _build_prompt(sample: Dict[str, Any], prompt_mode: str, fewshot_k: int) -> str:
    question = str(sample.get("question") or "").strip()
    knowledge = str(sample.get("knowledge") or "").strip()
    reference_answer = str(sample.get("reference_answer") or "").strip()

    if prompt_mode == "infer":
        return str(sample.get("prompt") or "").strip() or build_infer_prompt(
            knowledge=knowledge,
            question=question,
        )

    teacher_prompt = build_teacher_prompt(
        knowledge=knowledge,
        question=question,
        reference_answer=reference_answer or None,
    )
    if prompt_mode == "teacher":
        return teacher_prompt

    num_examples = max(0, min(int(fewshot_k), 3))
    if num_examples <= 0:
        return teacher_prompt
    examples = _fewshot_examples()[:num_examples]
    rendered: List[str] = [
        "Few-shot format examples (follow the same JSON schema and field order):"
    ]
    for idx, item in enumerate(examples, start=1):
        rendered.append(f"Example {idx}")
        rendered.append(f"Knowledge:\n{item['knowledge']}")
        rendered.append(f"Question: {item['question']}")
        rendered.append("Output:")
        rendered.append(json.dumps(item["output"], indent=2, ensure_ascii=False))
    rendered.append("Now solve the next case.")
    return "\n\n".join(rendered) + "\n\n" + teacher_prompt


def _parse_protocol(raw_output: str) -> tuple[bool, bool, Any]:
    parse_result = parse_structured_output(raw_output)
    parse_ok = bool(parse_result.ok and parse_result.parsed is not None)
    protocol_ok = False
    if parse_ok:
        protocol_ok = bool(check_protocol_constraints(parse_result.parsed).ok)
    return parse_ok, protocol_ok, parse_result


def _generate_batch_single_pass(
    *,
    generator: Any,
    batch: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows_state: List[Dict[str, Any]] = []
    for sample in batch:
        prompt_text = _build_prompt(sample, prompt_mode=args.prompt_mode, fewshot_k=args.fewshot_k)
        rows_state.append(
            {
                "sample": sample,
                "prompt": prompt_text,
                "raw_output": "",
                "parse_ok": False,
                "protocol_passed": False,
                "parse_result": None,
                "attempt_count": 0,
                "token_usage_totals": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "source": "",
                },
            }
        )

    prompts = [state["prompt"] for state in rows_state]
    outputs = generator.generate_many_results(
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

    for state, output in zip(rows_state, outputs):
        raw_output = str(output.text)
        parse_ok, protocol_ok, parse_result = _parse_protocol(raw_output)
        state["raw_output"] = raw_output
        state["parse_ok"] = bool(parse_ok)
        state["protocol_passed"] = bool(protocol_ok)
        state["parse_result"] = parse_result
        state["attempt_count"] = 1
        if output.token_usage is not None:
            token_usage_totals = state["token_usage_totals"]
            token_usage_totals["prompt_tokens"] += int(output.token_usage.prompt_tokens)
            token_usage_totals["completion_tokens"] += int(output.token_usage.completion_tokens)
            token_usage_totals["total_tokens"] += int(output.token_usage.total_tokens)
            token_usage_totals["source"] = str(output.token_usage.source)

    out_rows: List[Dict[str, Any]] = []
    for state in rows_state:
        sample = state["sample"]
        parse_result = state["parse_result"]
        parsed_payload = None
        parse_errors: List[str] = []
        if parse_result is not None:
            parsed_payload = None if parse_result.parsed is None else parse_result.parsed.model_dump(mode="json")
            parse_errors = list(parse_result.errors)
        out_rows.append(
            {
                "id": str(sample.get("id") or sample["source_id"]),
                "sample_id": int(sample["sample_id"]),
                "source_id": sample["source_id"],
                "question": sample["question"],
                "knowledge": sample["knowledge"],
                "reference_answer": sample["reference_answer"],
                "answerability_label": sample["answerability_label"],
                "data_split": sample.get("data_split") or args.split,
                "prompt_style": (
                    "infer_v1"
                    if args.prompt_mode == "infer"
                    else ("teacher_v1" if args.prompt_mode == "teacher" else "teacher_fewshot_v1")
                ),
                "prompt_mode": args.prompt_mode,
                "fewshot_k": int(max(0, min(int(args.fewshot_k), 3))) if args.prompt_mode == "teacher_fewshot" else 0,
                "prompt": state["prompt"],
                "raw_output": state["raw_output"],
                "answer": state["raw_output"],
                "parse_ok": bool(state["parse_ok"]),
                "parse_passed": bool(state["parse_ok"]),
                "protocol_passed": bool(state["protocol_passed"]),
                "generation_failed": not bool(state["protocol_passed"]),
                "attempt_count": int(state["attempt_count"]),
                "parsed_output": parsed_payload,
                "parse_errors": parse_errors,
                "prompt_tokens": (
                    int(state["token_usage_totals"]["prompt_tokens"])
                    if int(state["token_usage_totals"]["total_tokens"]) > 0
                    else None
                ),
                "completion_tokens": (
                    int(state["token_usage_totals"]["completion_tokens"])
                    if int(state["token_usage_totals"]["total_tokens"]) > 0
                    else None
                ),
                "total_tokens": (
                    int(state["token_usage_totals"]["total_tokens"])
                    if int(state["token_usage_totals"]["total_tokens"]) > 0
                    else None
                ),
                "token_usage_source": (
                    str(state["token_usage_totals"]["source"])
                    if int(state["token_usage_totals"]["total_tokens"]) > 0
                    else None
                ),
            }
        )
    return out_rows


def _resolve_eval_track(args: argparse.Namespace) -> str:
    if str(args.eval_track or "").strip():
        return str(args.eval_track).strip()
    if args.prompt_mode == "infer":
        return "sft_structured"
    return "base_protocol"


def _resolve_eval_variant(args: argparse.Namespace) -> str:
    if str(args.eval_variant or "").strip():
        return str(args.eval_variant).strip()
    if args.prompt_mode == "infer":
        return "checkpoint"
    return "fewshot"


def run_structured_generation(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Generate structured outputs for one or more model specs."""

    items = load_structured_generation_items(
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
    eval_track = _resolve_eval_track(args)
    eval_variant = _resolve_eval_variant(args)
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
            record_token_usage=bool(args.record_token_usage),
            seed=args.seed,
        )

        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            batch_rows = _generate_batch_single_pass(
                generator=generator,
                batch=batch,
                args=args,
            )
            for row in batch_rows:
                generated_rows.append(
                    {
                        "model_tag": spec.tag,
                        "model_step": int(spec.step),
                        "model_path": spec.display_name,
                        "eval_track": eval_track,
                        "eval_variant": eval_variant,
                        **row,
                    }
                )

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generated_rows


def main() -> None:
    args = parse_args()
    generated_rows = run_structured_generation(args)
    write_jsonl(args.out_jsonl, generated_rows)
    print(f"Saved structured generations to: {args.out_jsonl}")


if __name__ == "__main__":
    main()
