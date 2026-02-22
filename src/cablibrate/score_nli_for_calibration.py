from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List

try:
    from src.dataio import write_jsonl
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from dataio import write_jsonl

try:
    from src.nli_judge.config import NLIConfig
    from src.nli_judge.nli import NLIVerifier
    from src.prompt import build_qa_premise
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from nli_judge.config import NLIConfig
    from nli_judge.nli import NLIVerifier
    from prompt import build_qa_premise

from .common import (
    group_rows_by_model,
    load_labeled_rows,
    mean,
    parse_value_set,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True, help="External labeled dataset path (JSONL or save_to_disk).")
    parser.add_argument("--split", type=str, default="train", help="Split name when data_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum rows to score. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--knowledge_field", type=str, default="knowledge")
    parser.add_argument("--context_field", type=str, default="context")
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--label_field", type=str, default="label")
    parser.add_argument("--positive_label_values", type=str, default="1,true,yes,supported,faithful,correct,entail")
    parser.add_argument("--negative_label_values", type=str, default="0,false,no,unsupported,unfaithful,incorrect,neutral,contradict")
    parser.add_argument("--allow_unlabeled", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--model_tag_field", type=str, default="model_tag")
    parser.add_argument("--model_step_field", type=str, default="model_step")
    parser.add_argument("--model_path_field", type=str, default="model_path")
    parser.add_argument("--sample_id_field", type=str, default="sample_id")
    parser.add_argument("--source_id_field", type=str, default="source_id")

    parser.add_argument("--nli_model_name", type=str, default=NLIConfig.model_name)
    parser.add_argument("--nli_device", type=str, default=NLIConfig.device)
    parser.add_argument("--nli_batch_size", type=int, default=NLIConfig.batch_size)
    parser.add_argument("--nli_max_length", type=int, default=NLIConfig.max_length)
    parser.add_argument(
        "--nli_fp16",
        action=argparse.BooleanOptionalAction,
        default=NLIConfig.fp16,
        help="Whether to enable fp16 for NLI scoring on CUDA.",
    )

    parser.add_argument("--out_jsonl", type=str, required=True, help="Per-sample NLI-scored output JSONL.")
    parser.add_argument("--out_csv", type=str, default="", help="Optional summary CSV output.")
    return parser.parse_args()


def _to_logit(prob: float, eps: float = 1e-12) -> float:
    q = min(max(float(prob), eps), 1.0 - eps)
    return math.log(q / (1.0 - q))


def _label_by_argmax(entail: float, neutral: float, contradict: float) -> str:
    if entail >= neutral and entail >= contradict:
        return "entail"
    if neutral >= entail and neutral >= contradict:
        return "neutral"
    return "contradict"


def main() -> None:
    args = parse_args()

    rows = load_labeled_rows(
        data_path=args.data_path,
        split=args.split,
        seed=args.seed,
        max_samples=args.max_samples,
        question_field=args.question_field,
        answer_field=args.answer_field,
        knowledge_field=args.knowledge_field,
        context_field=args.context_field,
        label_field=args.label_field,
        positive_values=parse_value_set(args.positive_label_values),
        negative_values=parse_value_set(args.negative_label_values),
        allow_unlabeled=bool(args.allow_unlabeled),
        model_tag_field=args.model_tag_field,
        model_step_field=args.model_step_field,
        model_path_field=args.model_path_field,
        sample_id_field=args.sample_id_field,
        source_id_field=args.source_id_field,
    )
    if not rows:
        raise RuntimeError("No valid rows found after loading external labeled dataset.")

    verifier = NLIVerifier(
        model_name=args.nli_model_name,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
        fp16=args.nli_fp16,
    )

    premises = [build_qa_premise(knowledge=row["knowledge"], question=row["question"]) for row in rows]
    hypotheses = [row["answer"] for row in rows]
    scores = verifier.score(premises, hypotheses)

    scored_rows: List[Dict[str, Any]] = []
    for row, score in zip(rows, scores):
        entail = float(score.entail)
        neutral = float(score.neutral)
        contradict = float(score.contradict)
        argmax_label = _label_by_argmax(entail, neutral, contradict)
        argmax_prob = max(entail, neutral, contradict)

        logit_entail = _to_logit(entail)
        logit_neutral = _to_logit(neutral)
        logit_contradict = _to_logit(contradict)
        margin = logit_entail - max(logit_neutral, logit_contradict)

        scored_rows.append(
            {
                **row,
                "nli_entail": entail,
                "nli_neutral": neutral,
                "nli_contradict": contradict,
                "nli_argmax_label": argmax_label,
                "nli_argmax_prob": argmax_prob,
                "nli_logit_entail": logit_entail,
                "nli_logit_neutral": logit_neutral,
                "nli_logit_contradict": logit_contradict,
                "nli_margin_logit_entail_vs_max_other": margin,
            }
        )

    write_jsonl(args.out_jsonl, scored_rows)
    print(f"Saved NLI-scored rows: {args.out_jsonl}")

    if args.out_csv.strip():
        grouped = group_rows_by_model(scored_rows)
        summary_rows: List[Dict[str, Any]] = []
        for key, model_rows in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
            labeled = [r for r in model_rows if r.get("gold_supported") is not None]
            summary_rows.append(
                {
                    "model_tag": key[0],
                    "model_step": key[1],
                    "model_path": key[2],
                    "num_rows": len(model_rows),
                    "num_labeled": len(labeled),
                    "gold_positive_rate": mean([1.0 if bool(r["gold_supported"]) else 0.0 for r in labeled]),
                    "nli_entail_mean": mean([float(r["nli_entail"]) for r in model_rows]),
                    "nli_neutral_mean": mean([float(r["nli_neutral"]) for r in model_rows]),
                    "nli_contradict_mean": mean([float(r["nli_contradict"]) for r in model_rows]),
                    "nli_argmax_entail_rate": mean([1.0 if r["nli_argmax_label"] == "entail" else 0.0 for r in model_rows]),
                    "nli_argmax_neutral_rate": mean([1.0 if r["nli_argmax_label"] == "neutral" else 0.0 for r in model_rows]),
                    "nli_argmax_contradict_rate": mean([1.0 if r["nli_argmax_label"] == "contradict" else 0.0 for r in model_rows]),
                    "nli_model_name": args.nli_model_name,
                }
            )
        write_csv(summary_rows, args.out_csv)
        print(f"Saved score summary: {args.out_csv}")


if __name__ == "__main__":
    main()
