from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from project_config import PROJECT_SETTINGS

try:
    from qa_metrics.transformerMatcher import TransformerMatcher as QaMetricsTransformerMatcher
except ImportError:  # pragma: no cover
    QaMetricsTransformerMatcher = None


@dataclass(frozen=True)
class TransformerMatcherConfig:
    """Configuration for qa-metrics TransformerMatcher reviewer."""

    model_name: str = PROJECT_SETTINGS.calibration.matcher_model_name
    threshold: float = PROJECT_SETTINGS.calibration.matcher_runtime_threshold


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
        if not 0.0 <= float(cfg.threshold) <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {cfg.threshold!r}")
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
            match = bool(score >= float(self.cfg.threshold))
            outputs.append(
                TransformerMatcherReviewReport(
                    used=True,
                    ok=match,
                    match_score=score,
                    issues=[],
                )
            )
        return outputs
