from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from datasets import load_dataset


@dataclass(frozen=True)
class HotpotSourceConfig:
    """Configuration for loading HotpotQA source rows for SFT-only tagging."""

    dataset_name: str = "hotpotqa/hotpot_qa"
    train_split: str = "train"
    validation_split: str = "validation"
    max_train_samples: int = -1
    max_validation_samples: int = -1
    seed: int = 42


def iter_hotpot_rows(
    cfg: HotpotSourceConfig,
    *,
    split_name: str,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, object]]:
    """Iterate over shuffled HotpotQA rows from one named split."""

    ds = load_dataset(cfg.dataset_name, "distractor", split=split_name)
    indices = list(range(len(ds)))
    rng = random.Random(cfg.seed)
    rng.shuffle(indices)

    effective_max_samples = cfg.max_train_samples if max_samples is None else max_samples
    if effective_max_samples > 0:
        indices = indices[:effective_max_samples]

    for idx in indices:
        row = dict(ds[idx])
        row.setdefault("_source_index", idx)
        yield row
