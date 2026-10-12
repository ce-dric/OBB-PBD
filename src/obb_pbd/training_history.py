from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_training_history(
    save_dir: str | Path,
    training_payload: Mapping[str, Any],
    partition_provenance: Mapping[str, Any],
) -> None:
    """Persist one training history payload with its partition provenance."""
    path = Path(save_dir) / "history.json"
    path.write_text(
        json.dumps(
            {**training_payload, **partition_provenance}, indent=2
        ),
        encoding="utf-8",
    )
