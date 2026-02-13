import argparse
import csv
import json
import os
from typing import Any, Dict, List

import torch
from datasets import load_from_disk

from .halueval_common import (
    build_eval_item_from_preprocessed_row,
    parse_subsets_arg,
    sample_preprocessed_split,
)

try:
    from src.llm_textgen import (
        GeneratorModelSpec,
        UnifiedTextGenerator,
        build_generator_model_specs,
        load_generator_from_spec,
    )
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from llm_textgen import (
        GeneratorModelSpec,
        UnifiedTextGenerator,
        build_generator_model_specs,
        load_generator_from_spec,
    )

try:
    from src.ssqpg.config import JudgeConfig, NLIConfig
    from src.ssqpg.judge import AnswerJudge
    from src.ssqpg.nli import NLIVerifier
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from ssqpg.config import JudgeConfig, NLIConfig
    from ssqpg.judge import AnswerJudge
    from ssqpg.nli import NLIVerifier

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data (leakage-free): evaluate from one split of a preprocessed dataset.
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
    parser.add_argument(
        "--strip_think_tags",
        action="store_true",
        help="If set, strip <think>...</think> blocks from generated answers before NLI evaluation.",
    )
    parser.add_argument(
        "--strip_role_markers",
        action="store_true",
        help="If set, strip leaked role markers (user/assistant) from generated answers before NLI evaluation.",
    )

    # NLI judgment (same logic family as ssqpg.AnswerJudge).
    parser.add_argument("--nli_model_name", type=str, default=NLIConfig.model_name)
    parser.add_argument("--secondary_nli_model_name", type=str, default=None)
    parser.add_argument("--nli_device", type=str, default=NLIConfig.device)
    parser.add_argument("--nli_batch_size", type=int, default=NLIConfig.batch_size)
    parser.add_argument("--nli_max_length", type=int, default=NLIConfig.max_length)
    parser.add_argument(
        "--nli_fp16",
        action=argparse.BooleanOptionalAction,
        default=NLIConfig.fp16,
        help="Whether to enable fp16 for NLI scoring on CUDA.",
    )
    parser.add_argument("--reference_entail_threshold", type=float, default=JudgeConfig.reference_entail_threshold)
    parser.add_argument("--candidate_entail_threshold", type=float, default=JudgeConfig.candidate_entail_threshold)
    parser.add_argument("--candidate_contradict_threshold", type=float, default=JudgeConfig.candidate_contradict_threshold)
    parser.add_argument("--vote_mode", type=str, default=JudgeConfig.vote_mode, choices=["primary", "and", "or"])
    parser.add_argument(
        "--qa_similarity_model_name",
        type=str,
        default="",
        help="Sentence-Transformer model name for optional QA consistency check. Empty string disables it.",
    )
    parser.add_argument("--qa_similarity_min", type=float, default=JudgeConfig.qa_similarity_min)
    parser.add_argument(
        "--contradict_rate_threshold",
        type=float,
        default=0.5,
        help="Threshold used only for reporting contradiction-rate statistics.",
    )

    # Output
    parser.add_argument("--out_csv", type=str, default="nli_faithfulness_curve.csv")
    parser.add_argument("--out_jsonl", type=str, default="nli_faithfulness_samples.jsonl")
    return parser.parse_args()


def _load_eval_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Load evaluation items strictly from the specified preprocessed split.
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

    items: List[Dict[str, Any]] = []
    for i, row in enumerate(ds_sampled):
        question, contexts = build_eval_item_from_preprocessed_row(row)
        prompt = (row.get("prompt") or "").strip()
        reference = (row.get("chosen") or "").strip()
        items.append(
            {
                "sample_id": i,
                "source_id": row.get("source_id", str(i)),
                "subset": row.get("task", "__unknown__"),
                "question": question,
                "contexts": contexts,
                "prompt": prompt,
                "reference_answer": reference,
            }
        )
    return items


def _generate_for_model(
    generator: UnifiedTextGenerator,
    items: List[Dict[str, Any]],
    args: argparse.Namespace,
    model_meta: GeneratorModelSpec,
) -> List[Dict[str, Any]]:
    """
    Generate answers in batches and return rows ready for NLI evaluation.
    """
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        qa_items = [
            {
                "knowledge": "\n\n".join(x["contexts"]),
                "question": str(x["question"]),
            }
            for x in batch
        ]
        answers = generator.generate_many_from_qa(
            qa_items=qa_items,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_chat_template=True,
            enable_thinking=args.enable_thinking,
            strip_think_tags=args.strip_think_tags,
            strip_role_markers=args.strip_role_markers,
        )

        for x, ans in zip(batch, answers):
            rows.append(
                {
                    "tag": model_meta.tag,
                    "step": int(model_meta.step),
                    "model_path": model_meta.display_name,
                    "sample_id": int(x["sample_id"]),
                    "source_id": x["source_id"],
                    "subset": x["subset"],
                    "question": x["question"],
                    "contexts": x["contexts"],
                    "knowledge": "\n\n".join(x["contexts"]),
                    "reference_answer": x["reference_answer"],
                    "answer": ans,
                }
            )
    return rows


def _build_answer_judge(args: argparse.Namespace) -> AnswerJudge:
    """
    Build ssqpg-compatible AnswerJudge for cheap NLI-based faithfulness evaluation.
    """
    primary = NLIVerifier(
        model_name=args.nli_model_name,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
        fp16=args.nli_fp16,
    )

    secondary = None
    if args.secondary_nli_model_name:
        secondary = NLIVerifier(
            model_name=args.secondary_nli_model_name,
            device=args.nli_device,
            batch_size=args.nli_batch_size,
            max_length=args.nli_max_length,
            fp16=args.nli_fp16,
        )

    qa_model_name = args.qa_similarity_model_name.strip() or None
    cfg = JudgeConfig(
        reference_entail_threshold=args.reference_entail_threshold,
        candidate_entail_threshold=args.candidate_entail_threshold,
        candidate_contradict_threshold=args.candidate_contradict_threshold,
        vote_mode=args.vote_mode,
        qa_similarity_model_name=qa_model_name,
        qa_similarity_min=args.qa_similarity_min,
    )
    return AnswerJudge(cfg=cfg, primary_verifier=primary, secondary_verifier=secondary)


def _attach_judgment(
    rows: List[Dict[str, Any]],
    judge: AnswerJudge,
) -> List[Dict[str, Any]]:
    """
    Run NLI judgment and attach judge payload to generated rows.
    """
    if not rows:
        return []

    judge_inputs: List[Dict[str, Any]] = []
    for row in rows:
        judge_inputs.append(
            {
                "knowledge": row["knowledge"],
                "question": row["question"],
                "reference_answer": row.get("reference_answer", ""),
                "answer": row["answer"],
            }
        )

    judged_core, _ = judge.judge(judge_inputs)
    out: List[Dict[str, Any]] = []
    for base, judged in zip(rows, judged_core):
        rec = dict(base)
        rec["judge"] = judged.get("judge", {})
        out.append(rec)
    return out


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _summarize_rows(rows: List[Dict[str, Any]], contradict_rate_threshold: float, subsets: List[str]) -> Dict[str, str]:
    """
    Build one metric row for the curve CSV.
    """
    out: Dict[str, str] = {}
    if not rows:
        out["num_samples"] = "0"
        return out

    tag = str(rows[0]["tag"])
    step = int(rows[0]["step"])
    model_path = str(rows[0]["model_path"])

    js = [r.get("judge", {}) for r in rows]
    n = len(js)

    is_correct = [1.0 if bool(j.get("is_correct", False)) else 0.0 for j in js]
    candidate_supported = [1.0 if bool(j.get("candidate_supported", False)) else 0.0 for j in js]
    reference_supported = [1.0 if bool(j.get("reference_supported_primary", False)) else 0.0 for j in js]
    entail = [float(j.get("candidate_entail_primary", 0.0)) for j in js]
    contradict = [float(j.get("candidate_contradict_primary", 0.0)) for j in js]
    contradict_rate = [1.0 if x >= contradict_rate_threshold else 0.0 for x in contradict]

    qa_valid_values: List[float] = []
    for j in js:
        if j.get("qa_similarity") is None:
            continue
        qa_valid_values.append(1.0 if bool(j.get("qa_consistent", False)) else 0.0)

    out.update(
        {
            "tag": tag,
            "step": str(step),
            "model_path": model_path,
            "num_samples": str(n),
            "nli_correct_rate": f"{_safe_mean(is_correct)}",
            "candidate_supported_rate": f"{_safe_mean(candidate_supported)}",
            "reference_supported_rate": f"{_safe_mean(reference_supported)}",
            "candidate_entail_mean": f"{_safe_mean(entail)}",
            "candidate_contradict_mean": f"{_safe_mean(contradict)}",
            "candidate_contradict_rate": f"{_safe_mean(contradict_rate)}",
            "candidate_contradict_rate_threshold": f"{contradict_rate_threshold}",
            "qa_consistent_rate": "" if not qa_valid_values else f"{_safe_mean(qa_valid_values)}",
        }
    )

    for subset in subsets:
        sub_rows = [r for r in rows if str(r.get("subset", "")) == subset]
        if not sub_rows:
            continue
        sub_j = [x.get("judge", {}) for x in sub_rows]
        sub_correct = [1.0 if bool(j.get("is_correct", False)) else 0.0 for j in sub_j]
        out[f"nli_correct_rate_subset/{subset}"] = f"{_safe_mean(sub_correct)}"

    return out


def _write_curve_csv(rows: List[Dict[str, str]], out_csv: str) -> None:
    """
    Write model-level metrics to CSV.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")


def _write_samples_jsonl(rows: List[Dict[str, Any]], out_jsonl: str) -> None:
    """
    Write per-sample generations and NLI judgment payloads to JSONL.
    """
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved: {out_jsonl}")


def main() -> None:
    args = parse_args()
    subsets = parse_subsets_arg(args.subsets)
    items = _load_eval_items(args)

    model_specs: List[GeneratorModelSpec]
    if args.checkpoints_dir:
        model_specs = build_generator_model_specs(
            base_model_name=args.base_model,
            lora_ckpt_list_path=args.checkpoints_dir,
            lora_list_are_merged=False,
            include_base=bool(args.include_base),
        )
    else:
        if not args.lora_model:
            raise ValueError("Provide either --checkpoints_dir (curve mode) or --lora_model (single mode).")
        model_specs = build_generator_model_specs(
            base_model_name=args.base_model,
            lora_ckpt_path=args.lora_model,
            lora_is_merged=bool(args.lora_is_merged),
            include_base=bool(args.include_base),
        )

    judge = _build_answer_judge(args)

    all_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, str]] = []

    for spec in model_specs:
        generator = load_generator_from_spec(
            spec=spec,
            base_model_name=args.base_model,
            merge_lora=args.merge_lora,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_chat_template=True,
            enable_thinking=args.enable_thinking,
            strip_think_tags=args.strip_think_tags,
            strip_role_markers=args.strip_role_markers,
            seed=args.seed,
        )

        generated_rows = _generate_for_model(generator=generator, items=items, args=args, model_meta=spec)
        judged_rows = _attach_judgment(generated_rows, judge=judge)
        all_rows.extend(judged_rows)

        curve_row = _summarize_rows(
            judged_rows,
            contradict_rate_threshold=args.contradict_rate_threshold,
            subsets=subsets,
        )
        curve_rows.append(curve_row)

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    curve_rows.sort(key=lambda x: (int(x.get("step", "0")), x.get("tag", "")))
    _write_curve_csv(curve_rows, args.out_csv)
    _write_samples_jsonl(all_rows, args.out_jsonl)


if __name__ == "__main__":
    main()
