"""Pure aggregation helpers for the image-dependence diagnostic."""
from __future__ import annotations

from collections import defaultdict
import random
import statistics
from typing import Sequence

import numpy as np


def seeded_derangement(size: int, *, seed: int = 0) -> list[int]:
    """Return the legacy deterministic shuffle with every fixed point swapped."""
    if size < 2:
        raise ValueError("derangement requires at least two records")
    permutation = list(range(size))
    random.Random(seed).shuffle(permutation)
    for index in range(size):
        if permutation[index] == index:
            other = (index + 1) % size
            permutation[index], permutation[other] = (
                permutation[other], permutation[index])
    return permutation


def intervention_variants(raw: dict, shuffled: dict) -> dict[str, dict]:
    """Construct matched and single-modality interventions without joint shuffle."""
    return {
        "matched": raw,
        "image_shuffled": {**raw, "image": shuffled["image"]},
        "prompt_shuffled": {**raw, "prompt": shuffled["prompt"]},
    }


def image_cluster_bootstrap_ci(
    values: Sequence[float],
    image_ids: Sequence[str],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Case-bootstrap target images and pool their correlated records."""
    if len(values) == 0 or len(values) != len(image_ids):
        raise ValueError("values and image_ids must be non-empty and aligned")
    if n_boot <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap inputs are invalid")

    by_image: dict[str, list[float]] = defaultdict(list)
    for value, image_id in zip(values, image_ids):
        by_image[str(image_id)].append(float(value))
    images = np.asarray(sorted(by_image), dtype=object)
    arrays = {key: np.asarray(group, dtype=np.float64)
              for key, group in by_image.items()}

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for draw_index in range(n_boot):
        drawn = rng.choice(images, size=len(images), replace=True)
        pooled = np.concatenate([arrays[str(image_id)] for image_id in drawn])
        boot_means[draw_index] = pooled.mean()

    alpha = (1.0 - confidence) / 2.0
    values_array = np.asarray(values, dtype=np.float64)
    return (
        float(values_array.mean()),
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1.0 - alpha)),
    )


def summarise(
    ious: Sequence[float],
    angles: Sequence[float],
    bin_rows: Sequence[Sequence[int]],
    decoded: int,
    total: int,
    *,
    image_ids: Sequence[str],
    n_boot: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Aggregate one intervention, clustering rotIoU uncertainty by image."""
    if total <= 0 or len(ious) != total or len(image_ids) != total:
        raise ValueError("ious, image_ids, and total must be aligned and non-empty")
    mean, low, high = image_cluster_bootstrap_ci(
        ious, image_ids, n_boot=n_boot, seed=bootstrap_seed)
    bin_std = []
    if len(bin_rows) > 1:
        for slot in range(len(bin_rows[0])):
            bin_std.append(statistics.pstdev([row[slot] for row in bin_rows]))
    return {
        "samples": total,
        "n_images": len(set(image_ids)),
        "decoded_pct": 100.0 * decoded / total,
        "mIoU": mean,
        "mIoU_ci95": [low, high],
        "median_IoU": statistics.median(ious),
        "acc25": sum(1 for value in ious if value >= 0.25) / total,
        "acc50": sum(1 for value in ious if value >= 0.5) / total,
        "acc75": sum(1 for value in ious if value >= 0.75) / total,
        "angle_mae": statistics.mean(angles) if angles else float("nan"),
        "bin_mean": ([statistics.mean([row[slot] for row in bin_rows])
                      for slot in range(len(bin_rows[0]))] if bin_rows else []),
        "bin_std": bin_std,
    }
