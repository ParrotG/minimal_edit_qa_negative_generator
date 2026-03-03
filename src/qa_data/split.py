from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from .records import QaExample


@dataclass(frozen=True)
class SplitConfig:
    """Split configuration for grounded-QA raw pools."""

    validation_ratio: float = 0.05
    test_ratio: float = 0.05
    train_sft_ratio: float = 0.45
    train_dpo_ratio: float = 0.45


def assign_split_name(source_id: str, cfg: SplitConfig) -> str:
    """Assign one split name using a stable hash of the source id."""

    total = cfg.validation_ratio + cfg.test_ratio + cfg.train_sft_ratio + cfg.train_dpo_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive number.")

    value = int(hashlib.sha1(str(source_id).encode("utf-8")).hexdigest()[:8], 16) / float(16**8 - 1)
    boundary_validation = cfg.validation_ratio / total
    boundary_test = boundary_validation + cfg.test_ratio / total
    boundary_train_sft = boundary_test + cfg.train_sft_ratio / total

    if value < boundary_validation:
        return "validation"
    if value < boundary_test:
        return "test"
    if value < boundary_train_sft:
        return "train_sft_raw"
    return "train_dpo_raw"


def assign_split(example: QaExample, cfg: SplitConfig) -> QaExample:
    """Return a copy of one example with a stable split label."""

    return replace(example, split=assign_split_name(example.source_id, cfg))
