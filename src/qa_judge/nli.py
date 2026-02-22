from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import NLIConfig


@dataclass(frozen=True)
class NLIScores:
    """NLI probabilities for a single premise-hypothesis pair."""

    entail: float
    neutral: float
    contradict: float


class NLIVerifier:
    """Batched NLI scorer with label mapping robust to model-specific id2label names."""

    def __init__(
        self,
        model_name: str = NLIConfig.model_name,
        device: str = NLIConfig.device,
        batch_size: int = NLIConfig.batch_size,
        max_length: int = NLIConfig.max_length,
        fp16: bool = NLIConfig.fp16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.fp16 = fp16 and device.startswith("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.entail_idx = self._find_label_index(id2label, "entail", required=True)
        self.neutral_idx = self._find_label_index(id2label, "neutral", required=False)
        self.contra_idx = self._find_label_index(id2label, "contrad", required=False)

    @staticmethod
    def _find_label_index(id2label: Dict[int, str], key_substr: str, required: bool) -> Optional[int]:
        """Find label index by substring match."""

        key = key_substr.lower()
        for idx, label in id2label.items():
            if key in label.lower():
                return idx
        if required:
            raise ValueError(f"Cannot find label containing '{key_substr}' in {id2label}")
        return None

    @torch.inference_mode()
    def score(self, premises: List[str], hypotheses: List[str]) -> List[NLIScores]:
        """Compute NLI probabilities for aligned (premise, hypothesis) lists."""

        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have equal length.")

        out: List[NLIScores] = []
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
                entail = float(row[self.entail_idx])
                neutral = float(row[self.neutral_idx]) if self.neutral_idx is not None else 0.0
                contradict = float(row[self.contra_idx]) if self.contra_idx is not None else 0.0
                out.append(NLIScores(entail=entail, neutral=neutral, contradict=contradict))
        return out
