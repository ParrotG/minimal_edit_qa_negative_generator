from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RepairConfig
from nli_judge.nli import NLIVerifier
from .prompt import build_qa_premise, build_repair_prompt
from .text import length_ratio, normalized_edit_distance, normalize_whitespace


def _extract_repaired_answer(text: str) -> str:
    """Extract clean repaired answer text from model output."""

    out = (text or "").strip()
    if not out:
        return ""

    marker = "revised answer:"
    lower = out.lower()
    if marker in lower:
        idx = lower.rfind(marker)
        out = out[idx + len(marker) :].strip()

    if "\n" in out:
        # Keep the first non-empty line to reduce instruction leakage.
        for line in out.splitlines():
            line = line.strip()
            if line:
                return line
    return out


class MinimalEditRepairer:
    """Repair very hard negatives into supported chosens with minimal edits."""

    def __init__(self, cfg: RepairConfig, verifier: NLIVerifier) -> None:
        self.cfg = cfg
        self.verifier = verifier

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("Tokenizer has no pad/eos token; cannot run repair generation.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype="auto",
            device_map="auto" if cfg.device.startswith("cuda") else None,
        )
        self.model.eval()

    def _generate_attempts(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

        jobs: List[Dict[str, Any]] = []
        for row_idx, row in enumerate(rows):
            if row.get("status") != "needs_repair":
                continue
            reference_answer = str(row.get("reference_answer") or "").strip() or None
            for attempt in range(self.cfg.attempts_per_record):
                prompt = build_repair_prompt(
                    knowledge=row["knowledge"],
                    question=row["question"],
                    rejected_answer=row["rejected"],
                    reference_answer=reference_answer,
                )
                jobs.append(
                    {
                        "row_idx": row_idx,
                        "attempt": attempt,
                        "prompt": prompt,
                    }
                )

        if not jobs:
            return []

        do_sample = self.cfg.temperature > 0.0
        attempts: List[Dict[str, Any]] = []

        num_batches = int(math.ceil(len(jobs) / self.cfg.batch_size))
        for bidx in range(num_batches):
            batch = jobs[bidx * self.cfg.batch_size : (bidx + 1) * self.cfg.batch_size]
            prompts = [x["prompt"] for x in batch]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(next(self.model.parameters()).device)

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.cfg.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "repetition_penalty": self.cfg.repetition_penalty,
            }
            if do_sample:
                gen_kwargs["temperature"] = self.cfg.temperature
                gen_kwargs["top_p"] = self.cfg.top_p
                gen_kwargs["top_k"] = self.cfg.top_k

            with torch.inference_mode():
                gen_ids = self.model.generate(**inputs, **gen_kwargs)

            prompt_padded_len = int(inputs["input_ids"].shape[1])
            for meta, ids in zip(batch, gen_ids):
                text = self.tokenizer.decode(ids[prompt_padded_len:], skip_special_tokens=True)
                repaired = _extract_repaired_answer(text)
                attempts.append(
                    {
                        "row_idx": meta["row_idx"],
                        "attempt": meta["attempt"],
                        "candidate": repaired,
                    }
                )

        return attempts

    def repair(self, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Apply minimal-edit repair to rows with status=needs_repair."""

        rows = [dict(r) for r in rows]
        attempts = self._generate_attempts(rows)

        if not attempts:
            return rows, {"num_attempts": 0, "num_repaired": 0, "num_fallback": 0, "num_failed": 0}

        premises: List[str] = []
        hypos: List[str] = []
        valid_attempts: List[Dict[str, Any]] = []

        for item in attempts:
            row = rows[item["row_idx"]]
            candidate = normalize_whitespace(item["candidate"])
            if not candidate:
                continue
            if candidate == normalize_whitespace(row["rejected"]):
                continue

            norm_edit = normalized_edit_distance(row["rejected"], candidate)
            ratio = length_ratio(row["rejected"], candidate)
            if norm_edit < self.cfg.min_norm_edit or norm_edit > self.cfg.max_norm_edit:
                continue
            if ratio < self.cfg.min_length_ratio or ratio > self.cfg.max_length_ratio:
                continue

            premises.append(build_qa_premise(row["knowledge"], row["question"]))
            hypos.append(candidate)
            valid_attempts.append(
                {
                    **item,
                    "candidate": candidate,
                    "norm_edit": norm_edit,
                    "length_ratio": ratio,
                }
            )

        if not valid_attempts:
            metrics = {"num_attempts": len(attempts), "num_repaired": 0, "num_fallback": 0, "num_failed": 0}
            return rows, metrics

        nli_scores = self.verifier.score(premises, hypos)
        by_row: Dict[int, List[Dict[str, Any]]] = {}
        for item, sc in zip(valid_attempts, nli_scores):
            if sc.entail < self.cfg.entail_threshold:
                continue
            score = float(sc.entail - 0.25 * item["norm_edit"] - 0.05 * abs(1.0 - item["length_ratio"]))
            keep = {
                **item,
                "entail": sc.entail,
                "neutral": sc.neutral,
                "contradict": sc.contradict,
                "repair_score": score,
            }
            by_row.setdefault(item["row_idx"], []).append(keep)

        num_repaired = 0
        num_fallback = 0
        num_failed = 0

        for idx, row in enumerate(rows):
            if row.get("status") != "needs_repair":
                continue

            cands = by_row.get(idx, [])
            if cands:
                best = max(cands, key=lambda x: x["repair_score"])
                row["chosen"] = best["candidate"]
                row["chosen_origin"] = "repair_model"
                row["status"] = "ready"
                pair_meta = dict(row.get("pair_meta") or {})
                pair_meta["repair_meta"] = {
                    "model_name": self.cfg.model_name,
                    "attempt": int(best["attempt"]),
                    "entail": best["entail"],
                    "neutral": best["neutral"],
                    "contradict": best["contradict"],
                    "norm_edit": best["norm_edit"],
                    "length_ratio": best["length_ratio"],
                    "repair_score": best["repair_score"],
                }
                row["pair_meta"] = pair_meta
                num_repaired += 1
                continue

            reference_answer: Optional[str] = str(row.get("reference_answer") or "").strip() or None
            if self.cfg.fallback_to_reference and reference_answer:
                row["chosen"] = reference_answer
                row["chosen_origin"] = "reference_fallback"
                row["status"] = "ready"
                num_fallback += 1
            else:
                row["status"] = "repair_failed"
                num_failed += 1

        metrics = {
            "num_attempts": len(attempts),
            "num_repaired": num_repaired,
            "num_fallback": num_fallback,
            "num_failed": num_failed,
        }
        return rows, metrics
