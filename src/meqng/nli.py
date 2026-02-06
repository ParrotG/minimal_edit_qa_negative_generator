from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class NLIScores:
    """Container for NLI probabilities."""
    entail: float
    neutral: float
    contradict: float


class NLIVerifier:
    """Lightweight NLI verifier wrapper with label-robust mapping."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
        fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.fp16 = fp16 and device.startswith("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.entail_idx = self._find_label_index(id2label, "entail")
        self.neutral_idx = self._find_label_index(id2label, "neutral")
        self.contra_idx = self._find_label_index(id2label, "contrad")

    @staticmethod
    def _find_label_index(id2label: Dict[int, str], key_substr: str) -> int:
        """Find a label index whose name contains `key_substr` (case-insensitive)."""
        key_substr = key_substr.lower()
        for idx, name in id2label.items():
            if key_substr in name.lower():
                return idx
        raise ValueError(f"Cannot find label containing '{key_substr}' in {id2label}")

    @torch.inference_mode()
    def score(self, premises: List[str], hypotheses: List[str]) -> List[NLIScores]:
        """Compute NLI probabilities for (premise, hypothesis) pairs."""
        assert len(premises) == len(hypotheses), "Premises and hypotheses must align."
        results: List[NLIScores] = []
        dtype_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.fp16 else nullcontext()

        for i in range(0, len(premises), self.batch_size):
            p_batch = premises[i : i + self.batch_size]
            h_batch = hypotheses[i : i + self.batch_size]
            enc = self.tokenizer(
                p_batch,
                h_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**enc).logits
            else:
                logits = self.model(**enc).logits

            probs = torch.softmax(logits, dim=-1).detach().cpu()
            for row in probs:
                results.append(
                    NLIScores(
                        entail=float(row[self.entail_idx]),
                        neutral=float(row[self.neutral_idx]),
                        contradict=float(row[self.contra_idx]),
                    )
                )
        return results


# Python 3.11 compatible local context manager for optional autocast
from contextlib import contextmanager

@contextmanager
def nullcontext():
    yield
