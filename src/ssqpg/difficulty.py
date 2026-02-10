from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DifficultyConfig
from .prompt import build_qa_answer_prefix


@dataclass(frozen=True)
class DifficultyRecord:
    """Difficulty score for one (prompt, chosen, rejected) tuple."""

    chosen_avg_logprob: float
    rejected_avg_logprob: float
    delta: float
    bucket: str


class DifficultyScorer:
    """Compute policy log-probability gaps and assign difficulty buckets."""

    def __init__(self, cfg: DifficultyConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                raise ValueError("Tokenizer has no pad/eos/unk token.")
        self.pad_token_id = int(self.tokenizer.pad_token_id)

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        self.model.to(cfg.device)
        self.model.eval()

        self.fp16 = bool(cfg.fp16 and cfg.device.startswith("cuda"))

    def _encode_pair(self, prompt_prefix: str, answer: str) -> Tuple[List[int], List[int]]:
        prefix_ids = self.tokenizer.encode(prompt_prefix, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if not answer_ids:
            answer_ids = [self.pad_token_id]

        total_allowed = self.cfg.max_length
        if len(answer_ids) >= total_allowed:
            answer_ids = answer_ids[-(total_allowed - 1) :]
            prefix_ids = []
        else:
            keep_prefix = total_allowed - len(answer_ids)
            prefix_ids = prefix_ids[-keep_prefix:]

        input_ids = prefix_ids + answer_ids
        labels = [-100] * len(prefix_ids) + answer_ids
        if len(input_ids) < 2:
            input_ids = [self.pad_token_id] + input_ids
            labels = [-100] + labels
        return input_ids, labels

    @torch.inference_mode()
    def _batch_avg_logprob(self, prefixes: Sequence[str], answers: Sequence[str]) -> List[float]:
        rows = [self._encode_pair(p, a) for p, a in zip(prefixes, answers)]
        out: List[float] = []

        for i in range(0, len(rows), self.cfg.batch_size):
            batch = rows[i : i + self.cfg.batch_size]
            max_len = max(len(ids) for ids, _ in batch)

            bsz = len(batch)
            input_ids = torch.full((bsz, max_len), fill_value=self.pad_token_id, device=self.cfg.device, dtype=torch.long)
            labels = torch.full((bsz, max_len), fill_value=-100, device=self.cfg.device, dtype=torch.long)
            attention_mask = torch.zeros((bsz, max_len), device=self.cfg.device, dtype=torch.long)

            for j, (ids, lbs) in enumerate(batch):
                seq_len = len(ids)
                input_ids[j, :seq_len] = torch.tensor(ids, device=self.cfg.device, dtype=torch.long)
                labels[j, :seq_len] = torch.tensor(lbs, device=self.cfg.device, dtype=torch.long)
                attention_mask[j, :seq_len] = 1

            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            shift_mask = (shift_labels != -100).float()
            safe_labels = shift_labels.clamp(min=0)

            token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            token_logprobs = token_logprobs * shift_mask

            sums = token_logprobs.sum(dim=-1)
            counts = shift_mask.sum(dim=-1).clamp(min=1.0)
            avgs = (sums / counts).detach().cpu().tolist()
            out.extend(float(x) for x in avgs)

        return out

    @staticmethod
    def bucket_from_delta(delta: float, hard_max_delta: float, medium_max_delta: float) -> str:
        if delta <= hard_max_delta:
            return "hard"
        if delta <= medium_max_delta:
            return "medium"
        return "easy"

    def score_rows(self, rows: Sequence[Dict[str, Any]]) -> List[DifficultyRecord]:
        prefixes = [build_qa_answer_prefix(r["knowledge"], r["question"]) for r in rows]
        chosens = [str(r["chosen"]) for r in rows]
        rejecteds = [str(r["rejected"]) for r in rows]

        c_scores = self._batch_avg_logprob(prefixes, chosens)
        r_scores = self._batch_avg_logprob(prefixes, rejecteds)

        out: List[DifficultyRecord] = []
        for c, r in zip(c_scores, r_scores):
            delta = c - r
            out.append(
                DifficultyRecord(
                    chosen_avg_logprob=float(c),
                    rejected_avg_logprob=float(r),
                    delta=float(delta),
                    bucket=self.bucket_from_delta(delta, self.cfg.hard_max_delta, self.cfg.medium_max_delta),
                )
            )
        return out


def assign_difficulty_buckets(rows: Sequence[Dict[str, Any]], cfg: DifficultyConfig) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Assign difficulty buckets for final pairs using model log-prob deltas."""

    rows = [dict(r) for r in rows]
    if not rows:
        return rows, {"easy": 0, "medium": 0, "hard": 0}

    scorer = DifficultyScorer(cfg)
    records = scorer.score_rows(rows)

    counts = {"easy": 0, "medium": 0, "hard": 0}
    for row, rec in zip(rows, records):
        row["difficulty"] = {
            "chosen_avg_logprob": rec.chosen_avg_logprob,
            "rejected_avg_logprob": rec.rejected_avg_logprob,
            "delta": rec.delta,
        }
        row["difficulty_bucket"] = rec.bucket
        counts[rec.bucket] = counts.get(rec.bucket, 0) + 1

    return rows, counts
