from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

import typer
from rich.console import Console

from .io import read_jsonl, write_json, write_jsonl


console = Console()

_REQUIRED_FIELDS: Tuple[str, ...] = ("knowledge", "question", "chosen", "rejected")


def _validate_and_select(row: Dict[str, Any], row_index: int, required_fields: Sequence[str]) -> Dict[str, Any]:
    """Validate required fields and keep only DPO training columns."""

    missing = [name for name in required_fields if name not in row]
    if missing:
        miss = ", ".join(missing)
        raise ValueError(f"Row {row_index} is missing required fields: {miss}")
    return {name: row[name] for name in required_fields}


def main(
    in_path: str = typer.Option(..., help="Input JSONL path from any ssqpg stage."),
    out: str = typer.Option(..., help="Output JSONL path containing only DPO core fields."),
    metrics_out: str | None = typer.Option(None, help="Optional JSON metrics output path."),
) -> None:
    """Export strict DPO JSONL with knowledge/question/chosen/rejected only."""

    rows = list(read_jsonl(in_path))
    cleaned: List[Dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        try:
            cleaned.append(_validate_and_select(row=row, row_index=row_index, required_fields=_REQUIRED_FIELDS))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    write_jsonl(out, cleaned)

    summary = {
        "num_input": len(rows),
        "num_output": len(cleaned),
        "required_fields": list(_REQUIRED_FIELDS),
    }
    if metrics_out:
        write_json(metrics_out, summary)

    console.print(f"Saved {len(cleaned)} strict DPO rows to {out}")
    console.print_json(json.dumps(summary))


if __name__ == "__main__":
    typer.run(main)
