from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class BemConfig:
    """Configuration for the answer-equivalence BEM reviewer."""

    model_name: str = "kortukov/answer-equivalence-bem"
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 512


@dataclass(frozen=True)
class BemReviewReport:
    """Result for one strict-failure sample reviewed by BEM."""

    used: bool
    ok: Optional[bool]
    label_index: Optional[int] = None
    equivalent_probability: Optional[float] = None
    issues: List[str] = field(default_factory=list)


class AnswerEquivalenceBemJudge:
    """Batch answer-equivalence reviewer using the BEM checkpoint."""

    def __init__(self, cfg: BemConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
        self.device = self._resolve_device(cfg.device)
        self.model.to(self.device)
        self.model.eval()
        self._positive_label = self._resolve_positive_label()

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested = (device or "cuda").strip().lower()
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested or "cpu"

    def _resolve_positive_label(self) -> int:
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if isinstance(id2label, dict):
            for idx, label in id2label.items():
                text = str(label or "").strip().lower()
                if text.startswith("not_") or text.startswith("non_") or text in {"negative", "contradiction"}:
                    continue
                if text in {"equivalent", "entailment", "yes", "positive"}:
                    return int(idx)
            for idx, label in id2label.items():
                text = str(label or "").strip().lower()
                if text.startswith("not_") or text.startswith("non_"):
                    continue
                if "equivalent" in text:
                    return int(idx)
        num_labels = int(getattr(getattr(self.model, "config", None), "num_labels", 2) or 2)
        return 1 if num_labels > 1 else 0

    def _tokenize_batch(
        self,
        *,
        questions: Sequence[str],
        references: Sequence[str],
        candidates: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        text = [f"[CLS] {candidate} [SEP]" for candidate in candidates]
        text_pair = [
            f"{reference} [SEP] {question} [SEP]"
            for question, reference in zip(questions, references)
        ]
        encoded = self.tokenizer(
            text=text,
            text_pair=text_pair,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=int(self.cfg.max_length),
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def review_batch(self, rows: Sequence[Dict[str, str]]) -> List[BemReviewReport]:
        """Return one review report for each row."""

        if not rows:
            return []

        outputs: List[BemReviewReport] = []
        batch_size = max(1, int(self.cfg.batch_size))
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            questions = [str(row.get("question") or "") for row in batch]
            references = [str(row.get("reference_answer") or "") for row in batch]
            candidates = [str(row.get("answer") or "") for row in batch]
            model_inputs = self._tokenize_batch(
                questions=questions,
                references=references,
                candidates=candidates,
            )
            with torch.no_grad():
                logits = self.model(**model_inputs).logits
                probs = torch.softmax(logits, dim=-1)
                pred = probs.argmax(dim=-1).tolist()
                pos_probs = probs[:, self._positive_label].tolist()
            for label_index, pos_prob in zip(pred, pos_probs):
                outputs.append(
                    BemReviewReport(
                        used=True,
                        ok=bool(int(label_index) == int(self._positive_label)),
                        label_index=int(label_index),
                        equivalent_probability=float(pos_prob),
                        issues=[],
                    )
                )
        return outputs
