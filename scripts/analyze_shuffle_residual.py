#!/usr/bin/env python3
"""
where does the image-shuffled residual come from?
===============================================================================================

Table 4 reports rotIoU 0.0063 for the selected OBB-PBD checkpoint (seed 1) when each record's image is
replaced by another record's image (seed-0 derangement over records). The question is
whether that residual reflects prior knowledge of text prompts or target locations in the
dataset. This script answers from existing artifacts only (no GPU):

  1. Same-image leakage: the derangement is over records, and each test image carries ~51
     records, so a "shuffled" image can be the target's own image. Those cases are not an
     image intervention at all.
  2. Location prior: a model-free baseline that predicts, for every record, the mean target
     box of the test records from *other* images (leave-one-image-out), scored with the same
     rotIoU. If the residual is about this size, a generic location/size prior explains it.
  3. Prompt-driven location: under image shuffling the prompt is intact. If the shuffled
     prediction lands closer to the model's own matched prediction for that prompt than to a
     random record's matched prediction, the text alone carries location information
     (e.g. spatial words in the referring expression).

Inputs: OBB_FULL_TEST_IMAGE_DEPENDENCE.predictions.jsonl (per-record conditions) and
OBB_FULL_TEST.predictions.jsonl (targets + matched predictions, same record order).

    python scripts/analyze_shuffle_residual.py \\
        --probe path/OBB_FULL_TEST_IMAGE_DEPENDENCE.predictions.jsonl \\
        --full-test experiments/obb_pbd/OBB_FULL_TEST.predictions.jsonl \\
        --out experiments/analysis/SHUFFLE_RESIDUAL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from obb_pbd.geometry import OrientedBox  # noqa: E402
from obb_pbd.metrics import rotated_iou  # noqa: E402

# AVVG images are 4000x2250, resized to 1024x576 before tokenization; the targets in the
# full-test predictions are already in that frame.
FRAME_W, FRAME_H = 1024.0, 576.0


def decode(bins) -> OrientedBox:
    cx, cy, w, h, s, c = (b / 1000.0 for b in bins)
    s, c = 2 * s - 1, 2 * c - 1
    theta = (np.degrees(np.arctan2(s, c)) / 2.0) % 180.0
    w, h = w * FRAME_W, h * FRAME_H
    if w < h:
        w, h, theta = h, w, (theta + 90.0) % 180.0
    return OrientedBox(cx * FRAME_W, cy * FRAME_H, max(w, 1e-6), max(h, 1e-6), theta)


def box(record) -> OrientedBox:
    t = record
    return OrientedBox(t["cx"], t["cy"], t["w"], t["h"], t["theta_deg"])


def cluster_ci(values, images, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(images))
    groups = {u: [] for u in uniq}
    for v, im in zip(values, images):
        groups[im].append(v)
    sums = np.array([np.sum(groups[u]) for u in uniq])
    counts = np.array([len(groups[u]) for u in uniq])
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--full-test", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    probe = [json.loads(line) for line in open(args.probe, encoding="utf-8")]
    full = [json.loads(line) for line in open(args.full_test, encoding="utf-8")]
    assert len(probe) == len(full) == 6139
    for p, f in zip(probe, full):
        assert p["record_id"] == f["record_id"] and p["image_id"] == f["image_id"]
    # the probe's matched condition must reproduce the full-test evaluation exactly
    assert all(p["conditions"]["matched"]["predicted_bins"] == f["bins"] for p, f in zip(probe, full))

    images = [p["image_id"] for p in probe]
    targets = [box(f["target"]) for f in full]
    shuffled_iou = np.array([p["conditions"]["image_shuffled"]["rotated_iou"] for p in probe])
    same_image = np.array([p["shuffle_image_id"] == p["image_id"] for p in probe])

    # 2. leave-one-image-out mean box (location + size prior, orientation via double angle)
    arr = np.array([[t.cx, t.cy, t.w, t.h, np.sin(np.radians(2 * t.theta_deg)), np.cos(np.radians(2 * t.theta_deg))]
                    for t in targets])
    img_arr = np.array(images)
    total, n = arr.sum(axis=0), len(arr)
    per_image = {u: (arr[img_arr == u].sum(axis=0), int((img_arr == u).sum())) for u in set(images)}
    prior_iou = []
    for i, t in enumerate(targets):
        s, k = per_image[images[i]]
        m = (total - s) / (n - k)
        theta = (np.degrees(np.arctan2(m[4], m[5])) / 2.0) % 180.0
        w, h = m[2], m[3]
        if w < h:
            w, h, theta = h, w, (theta + 90.0) % 180.0
        prior_iou.append(rotated_iou(OrientedBox(m[0], m[1], w, h, theta), t))
    prior_iou = np.array(prior_iou)

    # 3. prompt-driven location: shuffled prediction vs own matched prediction vs random
    matched_c = np.array([[f["prediction"]["cx"], f["prediction"]["cy"]] if f["decoded"] else [np.nan, np.nan]
                          for f in full])
    shuf_c = np.array([[b.cx, b.cy] for b in (decode(p["conditions"]["image_shuffled"]["predicted_bins"])
                                              for p in probe)])
    perm = np.random.default_rng(0).permutation(len(probe))
    d_own = np.linalg.norm(shuf_c - matched_c, axis=1)
    d_rand = np.linalg.norm(shuf_c - matched_c[perm], axis=1)
    ok = ~np.isnan(d_own) & ~np.isnan(d_rand)
    d_to_shuffle_img_pred = np.linalg.norm(shuf_c - matched_c[[p["shuffle_index"] for p in probe]], axis=1)

    result = {
        "records": len(probe),
        "images": len(set(images)),
        "image_shuffled_rotIoU": cluster_ci(shuffled_iou, images),
        "same_image_fraction": float(same_image.mean()),
        "image_shuffled_rotIoU_same_image": float(shuffled_iou[same_image].mean()) if same_image.any() else None,
        "image_shuffled_rotIoU_different_image": cluster_ci(shuffled_iou[~same_image],
                                                            [im for im, s in zip(images, same_image) if not s]),
        "share_of_residual_from_same_image": float(shuffled_iou[same_image].sum() / shuffled_iou.sum()),
        "image_shuffled_nonzero_fraction": float((shuffled_iou > 0).mean()),
        "loio_mean_box_prior_rotIoU": cluster_ci(prior_iou, images),
        "center_px_shuffled_to_own_matched_median": float(np.median(d_own[ok])),
        "center_px_shuffled_to_random_matched_median": float(np.median(d_rand[ok])),
        "center_px_shuffled_to_shuffle_source_matched_median": float(np.nanmedian(d_to_shuffle_img_pred)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k, v in result.items():
        print(f"  {k:52s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
