from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterator

from .hotpot import HotpotSourceConfig, iter_hotpot_rows


@dataclass(frozen=True)
class DataSplitConfig:
    """Stable split configuration for downstream training and evaluation stages."""

    validation_ratio: float = 0.05
    test_ratio: float = 0.05
    train_sft_ratio: float = 0.45
    train_dpo_ratio: float = 0.45
    hash_salt: str = "data_split"


@dataclass(frozen=True)
class AnswerabilitySplitConfig:
    """Stable split configuration for answerability-oriented source construction."""

    answerable_ratio: float = 0.7
    unanswerable_ratio: float = 0.2
    both_ratio: float = 0.1
    hash_salt: str = "answerability_split"


def _stable_ratio(source_id: str, salt: str) -> float:
    digest = hashlib.sha1(f"{salt}:{source_id}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / float(16**8 - 1)


def assign_data_split_name(source_id: str, cfg: DataSplitConfig) -> str:
    """Assign one stable downstream split using a salted hash."""

    total = cfg.validation_ratio + cfg.test_ratio + cfg.train_sft_ratio + cfg.train_dpo_ratio
    if total <= 0:
        raise ValueError("Data split ratios must sum to a positive number.")

    value = _stable_ratio(source_id, cfg.hash_salt)
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


def assign_answerability_split_name(source_id: str, cfg: AnswerabilitySplitConfig) -> str:
    """Assign one stable answerability-construction split using a salted hash."""

    total = cfg.answerable_ratio + cfg.unanswerable_ratio + cfg.both_ratio
    if total <= 0:
        raise ValueError("Answerability split ratios must sum to a positive number.")

    value = _stable_ratio(source_id, cfg.hash_salt)
    boundary_answerable = cfg.answerable_ratio / total
    boundary_unanswerable = boundary_answerable + cfg.unanswerable_ratio / total

    if value < boundary_answerable:
        return "answerable"
    if value < boundary_unanswerable:
        return "unanswerable"
    return "both"


def tag_hotpot_row(
    row: Dict[str, object],
    data_cfg: DataSplitConfig,
    answerability_cfg: AnswerabilitySplitConfig,
) -> Dict[str, object]:
    """Return one raw Hotpot row annotated with independent split tags."""

    source_id = str(row.get("_id") or row.get("id") or row.get("_source_index") or "").strip()
    if not source_id:
        raise ValueError("Hotpot row is missing a stable source identifier.")

    payload = dict(row)
    payload["data_split"] = assign_data_split_name(source_id, data_cfg)
    payload["answerability_split"] = assign_answerability_split_name(source_id, answerability_cfg)
    return payload


def iter_tagged_hotpot_rows(
    source_cfg: HotpotSourceConfig,
    data_cfg: DataSplitConfig,
    answerability_cfg: AnswerabilitySplitConfig,
) -> Iterator[Dict[str, object]]:
    """Iterate over Hotpot rows annotated with independent split tags."""

    for row in iter_hotpot_rows(source_cfg):
        yield tag_hotpot_row(dict(row), data_cfg, answerability_cfg)
