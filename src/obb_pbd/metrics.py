from __future__ import annotations

from typing import Sequence

import numpy as np
from shapely.geometry import Polygon

from .geometry import OrientedBox, obb_to_polygon


def rotated_iou(prediction: OrientedBox, target: OrientedBox) -> float:
    pred = Polygon(obb_to_polygon(prediction))
    truth = Polygon(obb_to_polygon(target))
    union = pred.union(truth).area
    return 0.0 if union <= 0 else float(pred.intersection(truth).area / union)


def obb_to_aabb(box: OrientedBox) -> OrientedBox:
    """Smallest axis-aligned box containing ``box``, in canonical OBB form.

    This is what a horizontal-box detector is structurally limited to emitting for
    a rotated object, so it lets us state the HBB ceiling in the same units as the
    rotated-IoU numbers the paper reports. ``theta`` is 0 when the enclosing box is
    wider than tall and 90 otherwise, because ``OrientedBox`` requires ``w >= h``.
    """
    points = obb_to_polygon(box)
    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    centre = points.mean(axis=0)
    if width >= height:
        return OrientedBox(float(centre[0]), float(centre[1]), width, height, 0.0)
    return OrientedBox(float(centre[0]), float(centre[1]), height, width, 90.0)


def aabb_iou(prediction: OrientedBox, target: OrientedBox) -> float:
    """IoU between the axis-aligned envelopes of two oriented boxes.

    The AVVG zero-shot baseline was scored this way because the untrained model can
    only emit four coordinates, so reporting it alongside rotated IoU is the only way
    to compare the trained OBB model against that baseline on one metric.
    """
    pred = obb_to_polygon(prediction)
    truth = obb_to_polygon(target)
    lo_x = max(pred[:, 0].min(), truth[:, 0].min())
    lo_y = max(pred[:, 1].min(), truth[:, 1].min())
    hi_x = min(pred[:, 0].max(), truth[:, 0].max())
    hi_y = min(pred[:, 1].max(), truth[:, 1].max())
    overlap = max(0.0, hi_x - lo_x) * max(0.0, hi_y - lo_y)

    pred_area = ((pred[:, 0].max() - pred[:, 0].min())
                 * (pred[:, 1].max() - pred[:, 1].min()))
    truth_area = ((truth[:, 0].max() - truth[:, 0].min())
                  * (truth[:, 1].max() - truth[:, 1].min()))
    union = pred_area + truth_area - overlap
    return 0.0 if union <= 0 else float(overlap / union)


def angle_error_deg(prediction: float, target: float) -> float:
    difference = abs((prediction - target) % 180.0)
    return float(min(difference, 180.0 - difference))


def accuracy_at_iou(values: Sequence[float], threshold: float) -> float:
    if len(values) == 0:
        raise ValueError("values cannot be empty")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    return float(np.mean(np.asarray(values) >= threshold))


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap inputs are invalid")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(array.mean()), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))
