from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..ner import extract_numbers
from .base import PerturbCandidate


def _format_like(original: str, new_value: float | int) -> str:
    """Format a number to mimic the original surface form."""
    has_percent = original.endswith("%")
    core = original[:-1] if has_percent else original

    # Detect commas and decimals.
    has_commas = "," in core
    if "." in core:
        decimals = len(core.split(".")[1])
        fmt = f"{{:.{decimals}f}}"
        s = fmt.format(float(new_value))
    else:
        s = str(int(round(float(new_value))))

    if has_commas:
        # Re-insert commas for integers only.
        if "." not in s:
            s = f"{int(s):,}"
    if has_percent:
        s = s + "%"
    return s


@dataclass
class NumericPerturbator:
    """Perturb one numeric substring with a small arithmetic delta."""
    name: str = "numeric_perturb"

    def generate(
        self,
        knowledge: str,
        question: str,
        answer: str,
        entity_bank: Optional[Dict[str, Any]],
        max_candidates: int = 3,
        seed: int = 0,
    ) -> List[PerturbCandidate]:
        rng = random.Random(seed)
        nums = extract_numbers(answer)
        rng.shuffle(nums)

        out: List[PerturbCandidate] = []
        for raw, start, end in nums:
            raw_clean = raw.replace(",", "").replace("%", "")
            try:
                val = float(raw_clean)
            except ValueError:
                continue

            # Heuristic: treat 4-digit integers as years.
            is_year = raw_clean.isdigit() and len(raw_clean) == 4 and 1000 <= int(raw_clean) <= 2100
            if is_year:
                delta = rng.choice([-3, -2, -1, 1, 2, 3])
            else:
                # Smaller relative delta for non-year numbers.
                delta = rng.choice([-5, -2, -1, 1, 2, 5])
                if abs(val) > 100:
                    delta = delta * 2

            new_val = val + delta
            new_raw = _format_like(raw, new_val)
            if new_raw == raw:
                continue
            new_text = answer[:start] + new_raw + answer[end:]
            out.append(
                PerturbCandidate(
                    text=new_text,
                    perturbator=self.name,
                    meta={"orig": raw, "repl": new_raw, "span": [start, end], "delta": delta},
                )
            )
            if len(out) >= max_candidates:
                break
        return out
