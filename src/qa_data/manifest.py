from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class BuildManifest:
    """Manifest metadata for one grounded-QA build artifact."""

    builder_name: str
    builder_version: str
    protocol_version: str
    source_name: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the manifest to a JSON-serializable dictionary."""

        return asdict(self)
