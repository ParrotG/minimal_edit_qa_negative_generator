from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterator

from .hotpot import HotpotSourceConfig, iter_hotpot_rows


@dataclass(frozen=True)
class DataSplitConfig:
    """Stable split configuration for downstream SFT training and evaluation stages."""

    validation_ratio: float = 0.5
    test_ratio: float = 0.5
    hash_salt: str = "data_split"


@dataclass(frozen=True)
class AnswerabilitySplitConfig:
    """Stable split configuration for answerability-oriented source construction."""

    answerable_ratio: float = 0.8
    unanswerable_ratio: float = 0.1
    both_ratio: float = 0.1
    hash_salt: str = "answerability_split"


def _stable_ratio(source_id: str, salt: str) -> float:
    digest = hashlib.sha1(f"{salt}:{source_id}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / float(16**8 - 1)


def assign_data_split_name(source_id: str, hotpot_source_split: str, cfg: DataSplitConfig) -> str:
    """Assign one stable downstream split using source provenance plus salted hash."""

    normalized_split = str(hotpot_source_split or "").strip().lower()
    if normalized_split == "train":
        return "train_sft_raw"
    if normalized_split != "validation":
        raise ValueError(f"Unsupported Hotpot source split: {hotpot_source_split!r}")

    total = cfg.validation_ratio + cfg.test_ratio
    if total <= 0:
        raise ValueError("Validation/test split ratios must sum to a positive number.")

    value = _stable_ratio(source_id, cfg.hash_salt)
    boundary_validation = cfg.validation_ratio / total
    if value < boundary_validation:
        return "validation"
    return "test"


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
    hotpot_source_split = str(row.get("hotpot_source_split") or "").strip().lower()
    if hotpot_source_split not in {"train", "validation"}:
        raise ValueError("Hotpot row is missing a valid hotpot_source_split tag.")

    payload = dict(row)
    payload["hotpot_source_split"] = hotpot_source_split
    payload["data_split"] = assign_data_split_name(source_id, hotpot_source_split, data_cfg)
    payload["answerability_split"] = assign_answerability_split_name(source_id, answerability_cfg)
    return payload


def iter_tagged_hotpot_rows(
    source_cfg: HotpotSourceConfig,
    data_cfg: DataSplitConfig,
    answerability_cfg: AnswerabilitySplitConfig,
) -> Iterator[Dict[str, object]]:
    """Iterate over Hotpot train/validation rows annotated with independent split tags."""

    split_specs = (
        ("train", source_cfg.train_split, source_cfg.max_train_samples),
        ("validation", source_cfg.validation_split, source_cfg.max_validation_samples),
    )
    for hotpot_source_split, raw_split_name, max_samples in split_specs:
        for row in iter_hotpot_rows(source_cfg, split_name=raw_split_name, max_samples=max_samples):
            payload = dict(row)
            payload["hotpot_source_split"] = hotpot_source_split
            yield tag_hotpot_row(payload, data_cfg, answerability_cfg)
