from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from obb_pbd.metrics import aabb_iou, angle_error_deg, rotated_iou


class DatasetIndexView:
    """A dataset view with stable access to the source dataset indices."""

    def __init__(self, dataset: Any, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.samples = [dataset.samples[index] for index in self.indices]
        self.split = getattr(dataset, "split", None)
        self.subset = getattr(dataset, "subset", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def source_index(self, index: int) -> int:
        return self.indices[index]


def box_to_record(box: Any | None) -> dict[str, float] | None:
    if box is None:
        return None
    return {
        "cx": float(box.cx),
        "cy": float(box.cy),
        "w": float(box.w),
        "h": float(box.h),
        "theta_deg": float(box.theta_deg),
    }


def make_prediction_record(
    *,
    sample_index: int,
    sample: dict[str, Any],
    target: Any,
    prediction: Any | None,
    checkpoint: str,
    bins: Sequence[int],
) -> dict[str, Any]:
    decoded = prediction is not None
    return {
        "sample_index": int(sample_index),
        "record_id": sample.get("record_id"),
        "image_id": sample.get("image_id"),
        "image_path": sample.get("img_path"),
        "checkpoint": checkpoint,
        "target": box_to_record(target),
        "prediction": box_to_record(prediction),
        "bins": [int(value) for value in bins],
        "decoded": decoded,
        "rotated_iou": float(rotated_iou(prediction, target)) if decoded else 0.0,
        "aabb_iou": float(aabb_iou(prediction, target)) if decoded else 0.0,
        "angle_error_deg": (
            float(angle_error_deg(prediction.theta_deg, target.theta_deg))
            if decoded else None
        ),
    }


def write_prediction_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
