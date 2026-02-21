from __future__ import annotations

from typing import Any, Mapping, Sequence


def optional_str(row: Mapping[str, Any], field: str) -> str:
    """Return a trimmed string for one field; missing values become an empty string."""

    return str(row.get(field) or "").strip()


def require_non_empty_str(
    row: Mapping[str, Any],
    field: str,
    row_index: int | None = None,
) -> str:
    """Return a required non-empty string field or raise ValueError."""

    if field not in row:
        if row_index is None:
            raise ValueError(f"Missing required field: {field}")
        raise ValueError(f"Row {row_index} is missing required field: {field}")

    value = optional_str(row, field)
    if value:
        return value

    if row_index is None:
        raise ValueError(f"Required field is empty: {field}")
    raise ValueError(f"Row {row_index} has empty required field: {field}")


def pick_first_non_empty_str(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Pick the first non-empty string from an ordered list of field names."""

    for field in fields:
        value = optional_str(row, field)
        if value:
            return value
    return ""
