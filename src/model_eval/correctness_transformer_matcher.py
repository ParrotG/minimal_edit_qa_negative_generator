from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

try:
    from qa_metrics.transformerMatcher import TransformerMatcher as QaMetricsTransformerMatcher
except ImportError:  # pragma: no cover
    QaMetricsTransformerMatcher = None


@dataclass(frozen=True)
class TransformerMatcherConfig:
    """Configuration for qa-metrics TransformerMatcher reviewer."""

    model_name: str = "zli12321/answer_equivalence_roberta-large"


@dataclass(frozen=True)
class TransformerMatcherReviewReport:
    """Result for one strict-failure sample reviewed by qa-metrics."""

    used: bool
    ok: Optional[bool]
    match_score: Optional[float] = None
    issues: List[str] = field(default_factory=list)


class AnswerEquivalenceTransformerMatcher:
    """Wrapper around qa_metrics.transformerMatcher.TransformerMatcher."""

    def __init__(self, cfg: TransformerMatcherConfig) -> None:
        if QaMetricsTransformerMatcher is None:
            raise ImportError(
                "qa-metrics is required for TransformerMatcher review. Install it with: pip install qa-metrics"
            )
        self.cfg = cfg
        self.matcher = QaMetricsTransformerMatcher(cfg.model_name)

    def review_batch(self, rows: Sequence[Dict[str, str]]) -> List[TransformerMatcherReviewReport]:
        """Return one review report for each row."""

        outputs: List[TransformerMatcherReviewReport] = []
        for row in rows:
            question = str(row.get("question") or "")
            reference_answer = str(row.get("reference_answer") or "")
            candidate_answer = str(row.get("answer") or "")
            score = float(self.matcher.get_score(reference_answer, candidate_answer, question))
            match = bool(self.matcher.transformer_match([reference_answer], candidate_answer, question))
            outputs.append(
                TransformerMatcherReviewReport(
                    used=True,
                    ok=match,
                    match_score=score,
                    issues=[],
                )
            )
        return outputs
