from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from datasets import load_dataset
from project_config import PROJECT_SETTINGS


@dataclass(frozen=True)
class HotpotSourceConfig:
    """Configuration for loading HotpotQA source rows for SFT-only tagging."""

    dataset_name: str = PROJECT_SETTINGS.source.dataset_name
    train_split: str = PROJECT_SETTINGS.source.train_split
    validation_split: str = PROJECT_SETTINGS.source.validation_split
    max_train_samples: int = PROJECT_SETTINGS.source.max_train_samples
    max_validation_samples: int = PROJECT_SETTINGS.source.max_validation_samples
    seed: int = PROJECT_SETTINGS.source.seed


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
