from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import typer
from rich.console import Console

from dataio import optional_str, read_jsonl_list, require_non_empty_str, write_json, write_jsonl
from .config import NLIConfig
from .nli import NLIVerifier
from prompt import build_qa_premise


app = typer.Typer(add_completion=False)
console = Console()


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _score_supported(
    verifier: NLIVerifier,
    premises: Sequence[str],
    answers: Sequence[str],
    entail_threshold: float,
    contradict_threshold: float,
) -> List[Dict[str, Any]]:
    scores = verifier.score(list(premises), list(answers))
    out: List[Dict[str, Any]] = []
    for sc in scores:
        supported = bool(sc.entail >= entail_threshold and sc.contradict <= contradict_threshold)
        out.append(
            {
                "entail": float(sc.entail),
                "neutral": float(sc.neutral),
                "contradict": float(sc.contradict),
                "supported": supported,
            }
        )
    return out


@app.command("run")
def run(
    in_path: str = typer.Option(..., help="Input JSONL path."),
    out: Optional[str] = typer.Option(None, help="Optional output JSONL path with per-row NLI verification results."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional output JSON path for aggregate deviation metrics."),
    model_name: str = typer.Option(NLIConfig.model_name, help="NLI model name."),
    device: str = typer.Option(NLIConfig.device, help="NLI device."),
    batch_size: int = typer.Option(NLIConfig.batch_size, help="NLI batch size."),
    max_length: int = typer.Option(NLIConfig.max_length, help="NLI max length."),
    fp16: bool = typer.Option(NLIConfig.fp16, help="Whether to enable fp16 on CUDA."),
    support_entail_threshold: float = typer.Option(0.60, help="Entailment threshold for predicting supported."),
    support_contradict_threshold: float = typer.Option(
        0.90,
        help="Contradiction upper threshold for predicting supported.",
    ),
    knowledge_field: str = typer.Option("knowledge", help="Knowledge field name."),
    question_field: str = typer.Option("question", help="Question field name."),
    chosen_field: str = typer.Option("chosen", help="Chosen-answer field name."),
    rejected_field: str = typer.Option("rejected", help="Rejected-answer field name (optional)."),
    skip_empty_rejected: bool = typer.Option(True, help="Skip rejected verification when rejected field is empty."),
) -> None:
    """Verify NLI support for chosen/rejected and report deviation from field-implied labels."""

    rows = read_jsonl_list(in_path)
    if not rows:
        raise typer.BadParameter("Input JSONL has no rows.")

    verifier = NLIVerifier(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        fp16=fp16,
    )

    chosen_premises: List[str] = []
    chosen_answers: List[str] = []
    chosen_row_indices: List[int] = []
    rejected_premises: List[str] = []
    rejected_answers: List[str] = []
    rejected_row_indices: List[int] = []
    annotated_rows: List[Dict[str, Any]] = [dict(row) for row in rows]

    for idx, row in enumerate(rows, start=1):
        try:
            knowledge = require_non_empty_str(row, knowledge_field, row_index=idx)
            question = require_non_empty_str(row, question_field, row_index=idx)
            chosen = require_non_empty_str(row, chosen_field, row_index=idx)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        premise = build_qa_premise(knowledge=knowledge, question=question)
        chosen_premises.append(premise)
        chosen_answers.append(chosen)
        chosen_row_indices.append(idx - 1)

        if rejected_field not in row:
            continue

        rejected = optional_str(row, rejected_field)
        if not rejected:
            if skip_empty_rejected:
                continue
            raise typer.BadParameter(f"Row {idx} has empty {rejected_field} while skip_empty_rejected=false.")

        rejected_premises.append(premise)
        rejected_answers.append(rejected)
        rejected_row_indices.append(idx - 1)

    chosen_scored = _score_supported(
        verifier=verifier,
        premises=chosen_premises,
        answers=chosen_answers,
        entail_threshold=support_entail_threshold,
        contradict_threshold=support_contradict_threshold,
    )

    rejected_scored = _score_supported(
        verifier=verifier,
        premises=rejected_premises,
        answers=rejected_answers,
        entail_threshold=support_entail_threshold,
        contradict_threshold=support_contradict_threshold,
    )

    chosen_deviation = 0
    rejected_deviation = 0
    row_has_deviation: List[bool] = [False] * len(rows)

    for row_idx, scored in zip(chosen_row_indices, chosen_scored):
        # Chosen is a positive label by dataset convention.
        deviation = not bool(scored["supported"])
        chosen_deviation += int(deviation)
        row_has_deviation[row_idx] = bool(row_has_deviation[row_idx] or deviation)
        payload = dict(annotated_rows[row_idx].get("nli_verifier") or {})
        payload["chosen"] = {
            **scored,
            "expected_supported": True,
            "deviation": deviation,
        }
        annotated_rows[row_idx]["nli_verifier"] = payload

    for row_idx, scored in zip(rejected_row_indices, rejected_scored):
        # Rejected is a negative label by dataset convention.
        deviation = bool(scored["supported"])
        rejected_deviation += int(deviation)
        row_has_deviation[row_idx] = bool(row_has_deviation[row_idx] or deviation)
        payload = dict(annotated_rows[row_idx].get("nli_verifier") or {})
        payload["rejected"] = {
            **scored,
            "expected_supported": False,
            "deviation": deviation,
        }
        annotated_rows[row_idx]["nli_verifier"] = payload

    for idx, has_dev in enumerate(row_has_deviation):
        payload = dict(annotated_rows[idx].get("nli_verifier") or {})
        payload["row_has_deviation"] = bool(has_dev)
        annotated_rows[idx]["nli_verifier"] = payload

    total_labels = len(chosen_scored) + len(rejected_scored)
    total_deviation = chosen_deviation + rejected_deviation
    rows_with_deviation = sum(1 for x in row_has_deviation if x)
    summary: Dict[str, Any] = {
        "num_rows": len(rows),
        "num_rows_with_deviation": rows_with_deviation,
        "row_deviation_rate": _safe_rate(rows_with_deviation, len(rows)),
        "num_labels_checked": total_labels,
        "num_label_deviation": total_deviation,
        "label_deviation_rate": _safe_rate(total_deviation, total_labels),
        "chosen": {
            "num_checked": len(chosen_scored),
            "num_deviation": chosen_deviation,
            "deviation_rate": _safe_rate(chosen_deviation, len(chosen_scored)),
        },
        "rejected": {
            "num_checked": len(rejected_scored),
            "num_deviation": rejected_deviation,
            "deviation_rate": _safe_rate(rejected_deviation, len(rejected_scored)),
        },
        "thresholds": {
            "support_entail_threshold": support_entail_threshold,
            "support_contradict_threshold": support_contradict_threshold,
        },
        "fields": {
            "knowledge": knowledge_field,
            "question": question_field,
            "chosen": chosen_field,
            "rejected": rejected_field,
        },
    }

    if out:
        write_jsonl(out, annotated_rows)
    if metrics_out:
        write_json(metrics_out, summary)

    if out:
        console.print(f"Saved annotated rows to {out}")
    console.print_json(json.dumps(summary))


if __name__ == "__main__":
    app()
