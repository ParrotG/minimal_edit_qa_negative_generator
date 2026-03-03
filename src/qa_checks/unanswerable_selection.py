from __future__ import annotations

from typing import Any, Dict, List


def select_first_valid_unanswerable_candidate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select the first hard-pass unanswerable candidate per example id."""

    grouped: dict[str, list[tuple[int, Dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        if not bool((row.get("validation_report") or {}).get("hard_pass")):
            continue
        grouped.setdefault(str(row.get("id") or ""), []).append((index, row))

    selected: List[Dict[str, Any]] = []
    for _, items in grouped.items():
        best = min(
            items,
            key=lambda item: (
                int((item[1].get("candidate_id") or 0)),
                item[0],
            ),
        )[1]
        selected.append(best)
    return selected
