from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _epoch_from(name: str, checkpoint: str) -> int:
    for value in (name, checkpoint):
        match = re.search(r"(?:epoch_|checkpoint_epoch_?)(\d+)", value)
        if match:
            return int(match.group(1))
    raise ValueError(f"cannot determine epoch from {name!r} / {checkpoint!r}")


def select_checkpoint(
    summary_paths: Sequence[str | Path],
    *,
    expected_epochs: int,
    expected_samples: int,
) -> dict[str, Any]:
    rows = []
    for raw_path in summary_paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results", {})
        if len(results) != 1:
            raise ValueError(f"{path} must contain exactly one validation result")
        name, metrics = next(iter(results.items()))
        if int(metrics.get("samples", -1)) != expected_samples:
            raise ValueError(f"{path} validation samples do not equal {expected_samples}")
        checkpoint = str(metrics.get("checkpoint", ""))
        epoch = _epoch_from(name, checkpoint)
        rows.append({
            "epoch": epoch,
            "name": name,
            "checkpoint": checkpoint,
            "validation_metrics": metrics,
            "summary_path": str(path),
        })

    expected = set(range(1, expected_epochs + 1))
    observed = {row["epoch"] for row in rows}
    if observed != expected or len(rows) != expected_epochs:
        raise ValueError(f"validation summaries must cover epochs {sorted(expected)}, got {sorted(observed)}")
    selected = max(rows, key=lambda row: (float(row["validation_metrics"]["mIoU"]),
                                          -int(row["epoch"])))
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": "AVVG official-train-derived validation partition",
        "selection_metric": "validation_mean_rotIoU",
        "tie_break": "earlier epoch on exact stored-value tie",
        "expected_validation_samples": expected_samples,
        "selected_epoch": selected["epoch"],
        "selected_name": selected["name"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_validation_metrics": selected["validation_metrics"],
        "candidates": sorted(rows, key=lambda row: row["epoch"]),
    }


def _cluster_ci(values: np.ndarray, images: np.ndarray, n_boot: int,
                seed: int) -> dict[str, float]:
    unique = np.unique(images)
    grouped = {image: values[images == image] for image in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draw = rng.choice(unique, size=len(unique), replace=True)
        means[index] = np.concatenate([grouped[image] for image in draw]).mean()
    return {
        "mean": float(values.mean()),
        "ci95_lo": float(np.quantile(means, 0.025)),
        "ci95_hi": float(np.quantile(means, 0.975)),
    }


def summarise_prediction_records(
    records: Sequence[dict[str, Any]],
    *,
    expected_records: int,
    expected_images: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    if len(records) != expected_records:
        raise ValueError(f"expected {expected_records} prediction records, got {len(records)}")
    indices = [int(record["sample_index"]) for record in records]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate sample indices in predictions")
    expected_indices = set(range(expected_records))
    if set(indices) != expected_indices:
        raise ValueError("missing or unexpected sample indices in predictions")
    images = np.asarray([str(record["image_id"]) for record in records])
    if np.unique(images).size != expected_images:
        raise ValueError(
            f"expected {expected_images} prediction images, got {np.unique(images).size}"
        )
    rotated = np.asarray([float(record["rotated_iou"]) for record in records])
    aabb = np.asarray([float(record["aabb_iou"]) for record in records])
    decoded = np.asarray([bool(record["decoded"]) for record in records])
    angles = [float(record["angle_error_deg"]) for record in records
              if record.get("angle_error_deg") is not None]
    return {
        "n_records": len(records),
        "n_images": int(np.unique(images).size),
        "n_boot": n_boot,
        "seed": seed,
        "rotated_iou": _cluster_ci(rotated, images, n_boot, seed),
        "aabb_iou": _cluster_ci(aabb, images, n_boot, seed),
        "acc50": float((rotated >= 0.5).mean()),
        "acc75": float((rotated >= 0.75).mean()),
        "decoded_pct": float(100.0 * decoded.mean()),
        "angle_mae": float(np.mean(angles)) if angles else None,
    }
