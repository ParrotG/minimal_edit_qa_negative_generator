from __future__ import annotations

from typing import Dict, Iterator

from .hotpot import HotpotSourceConfig, iter_hotpot_rows
from .split import SplitConfig, assign_split_name


def assign_hotpot_split(row: Dict[str, object], split_cfg: SplitConfig) -> Dict[str, object]:
    """Return a copy of one raw Hotpot row with a stable split assignment."""

    source_id = str(row.get("_id") or row.get("id") or row.get("_source_index") or "").strip()
    if not source_id:
        raise ValueError("Hotpot row is missing a stable source identifier.")

    payload = dict(row)
    payload["split"] = assign_split_name(source_id, split_cfg)
    return payload


def iter_split_hotpot_rows(cfg: HotpotSourceConfig, split_cfg: SplitConfig) -> Iterator[Dict[str, object]]:
    """Iterate over raw Hotpot rows with stable split assignments."""

    for row in iter_hotpot_rows(cfg):
        yield assign_hotpot_split(dict(row), split_cfg)
