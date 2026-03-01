from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .report import CorrectnessCheckReport


_TOKEN_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class CorrectnessConfig:
    """Configuration for simple answer correctness checks."""

    semantic_match_f1_threshold: float = 0.85


def _normalize_answer(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(text or "").lower()))


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = sum((pred_counter & ref_counter).values())
    if common <= 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def check_answer_correctness(
    *,
    answer: str,
    reference_answer: str,
    cfg: CorrectnessConfig = CorrectnessConfig(),
) -> CorrectnessCheckReport:
    """Check answer correctness against the reference answer."""

    issues: list[str] = []
    exact_match = _normalize_answer(answer) == _normalize_answer(reference_answer)
    token_f1 = _token_f1(answer, reference_answer)
    semantic_match = bool(exact_match or token_f1 >= cfg.semantic_match_f1_threshold)

    if not semantic_match:
        issues.append("Answer does not match the reference answer strongly enough.")

    return CorrectnessCheckReport(
        ok=semantic_match,
        exact_match=exact_match,
        token_f1=float(token_f1),
        semantic_match=semantic_match,
        issues=issues,
    )
