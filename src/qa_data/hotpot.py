from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from datasets import load_dataset


@dataclass(frozen=True)
class HotpotSourceConfig:
    """Configuration for loading HotpotQA source rows."""

    dataset_name: str = "hotpotqa/hotpot_qa"
    split: str = "train"
    max_samples: int = -1
    seed: int = 42


def iter_hotpot_rows(cfg: HotpotSourceConfig) -> Iterator[Dict[str, object]]:
    """Iterate over shuffled HotpotQA rows."""

    ds = load_dataset(cfg.dataset_name, "distractor", split=cfg.split)
    indices = list(range(len(ds)))
    rng = random.Random(cfg.seed)
    rng.shuffle(indices)

    if cfg.max_samples > 0:
        indices = indices[: cfg.max_samples]

    for idx in indices:
        row = dict(ds[idx])
        row.setdefault("_source_index", idx)
        yield row
