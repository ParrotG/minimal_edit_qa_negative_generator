from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from qa_judge.judge import AnswerJudge
from qa_protocol import build_infer_prompt, count_text_tokens

from .unanswerable_prefilter import check_reference_answer_unsupported


@dataclass
class PromptBudgetReport:
    """Prompt-length filter result for one source example."""

    keep: bool
    prompt_tokens: int
    prompt: str
    issues: list[str] = field(default_factory=list)


def check_infer_prompt_budget(
    question: str,
    knowledge: str,
    tokenizer_name: str,
    max_prompt_tokens: int,
) -> PromptBudgetReport:
    """Check whether one infer prompt fits within the source-stage token budget."""

    prompt = build_infer_prompt(
        knowledge=str(knowledge or "").strip(),
        question=str(question or "").strip(),
    )
    prompt_tokens = count_text_tokens(prompt, tokenizer_name)
    keep = prompt_tokens <= max_prompt_tokens
    issues: list[str] = []
    if not keep:
        issues.append(f"Infer prompt exceeds max_prompt_tokens={max_prompt_tokens}.")
    return PromptBudgetReport(keep=keep, prompt_tokens=prompt_tokens, prompt=prompt, issues=issues)


def prefilter_mixed_examples(
    rows: Iterable[Dict[str, Any]],
    *,
    tokenizer_name: str,
    max_prompt_tokens: int,
    enable_unanswerable_nli: bool,
    judge: AnswerJudge | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run source-stage token and unanswerable NLI filtering on mixed examples."""

    kept_rows: List[Dict[str, Any]] = []
    num_input = 0
    num_drop_prompt_budget = 0
    num_unanswerable_nli_drop = 0
    num_unanswerable_checked = 0
    num_unanswerable_kept = 0
    num_unanswerable_yes_no_checked = 0

    for raw_row in rows:
        num_input += 1
        row = dict(raw_row)
        prompt_report = check_infer_prompt_budget(
            question=str(row.get("question") or ""),
            knowledge=str(row.get("knowledge") or ""),
            tokenizer_name=tokenizer_name,
            max_prompt_tokens=max_prompt_tokens,
        )

        metadata = dict(row.get("metadata") or {})
        metadata["source_prefilter"] = {
            "infer_prompt_tokens": int(prompt_report.prompt_tokens),
            "prompt_within_budget": bool(prompt_report.keep),
            "issues": list(prompt_report.issues),
        }
        row["metadata"] = metadata

        if not prompt_report.keep:
            num_drop_prompt_budget += 1
            continue

        if str(row.get("answerability_label") or "") != "unanswerable" or not enable_unanswerable_nli:
            kept_rows.append(row)
            continue

        if judge is None:
            raise ValueError("judge must be provided when enable_unanswerable_nli is True.")

        num_unanswerable_checked += 1
        report = check_reference_answer_unsupported(
            knowledge=str(row.get("knowledge") or "").strip(),
            question=str(row.get("question") or "").strip(),
            reference_answer=str(row.get("reference_answer") or "").strip(),
            judge=judge,
        )
        metadata = dict(row.get("metadata") or {})
        metadata["nli_prefilter"] = {
            "keep": bool(report.keep),
            "full_binary_decision": report.full_binary_decision,
            "margin": report.margin,
            "score": report.score,
            "used_flipped_check": bool(report.used_flipped_check),
            "flipped_full_binary_decision": report.flipped_full_binary_decision,
            "flipped_margin": report.flipped_margin,
            "issues": list(report.issues),
            "judge_payload": report.judge_payload,
        }
        row["metadata"] = metadata
        num_unanswerable_yes_no_checked += int(bool(report.used_flipped_check))
        if not report.keep:
            num_unanswerable_nli_drop += 1
            continue

        num_unanswerable_kept += 1
        kept_rows.append(row)

    metrics = {
        "num_input": num_input,
        "num_kept": len(kept_rows),
        "num_dropped": num_input - len(kept_rows),
        "num_drop_prompt_budget": num_drop_prompt_budget,
        "num_unanswerable_checked": num_unanswerable_checked,
        "num_unanswerable_kept": num_unanswerable_kept,
        "num_unanswerable_nli_drop": num_unanswerable_nli_drop,
        "num_unanswerable_yes_no_checked": num_unanswerable_yes_no_checked,
    }
    return kept_rows, metrics
