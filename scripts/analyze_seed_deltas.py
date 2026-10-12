#!/usr/bin/env python3
"""
per-seed paired effects with image-cluster CIs
==========================================================================================

For each training seed, compares two trained arms on the complete AVVG test split record by
record (same sample_index, same image) and reports the paired difference in rotIoU and
aabbIoU with the image-cluster bootstrap used throughout the paper (resample the 120 test
images with replacement, pool every record of each drawn image, 10,000 draws, seed 0 --
identical to ``obb_pbd.reporting._cluster_ci``). Seed 1 must reproduce the submitted
Table 3 orientation-learning interval, which is asserted when --check-seed1 is given.

    python scripts/analyze_seed_deltas.py --pair OBB ENVELOPE \\
        --seed s1 experiments/obb_pbd/OBB_FULL_TEST.predictions.jsonl \\
                  experiments/hbb_controls/HBB_ENVELOPE/HBB_ENVELOPE_FULL_TEST.predictions.jsonl \\
        --seed s2 ... --seed s3 ... --out experiments/analysis/SEED_DELTAS_OBB_vs_ENVELOPE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cluster_ci(values: np.ndarray, images: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    """Same procedure as obb_pbd.reporting._cluster_ci (kept identical on purpose)."""
    unique = np.unique(images)
    grouped = {image: values[images == image] for image in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draw = rng.choice(unique, size=len(unique), replace=True)
        means[index] = np.concatenate([grouped[image] for image in draw]).mean()
    return {"mean": float(values.mean()), "ci95_lo": float(np.quantile(means, 0.025)),
            "ci95_hi": float(np.quantile(means, 0.975))}


def load(path):
    rows = sorted((json.loads(line) for line in open(path, encoding="utf-8")),
                  key=lambda r: int(r["sample_index"]))
    assert len(rows) == 6139 and [int(r["sample_index"]) for r in rows] == list(range(6139)), path
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"), required=True,
                        help="names of the two arms; the effect is A minus B")
    parser.add_argument("--seed", nargs=3, action="append", metavar=("TAG", "A_JSONL", "B_JSONL"),
                        required=True)
    parser.add_argument("--check-seed1", nargs=2, type=float, metavar=("LO", "HI"),
                        help="assert the s1 rotIoU delta CI matches the published interval")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    a_name, b_name = args.pair
    report = {"effect": f"{a_name} - {b_name}", "n_boot": 10_000, "bootstrap_seed": 0, "seeds": {}}
    for tag, a_path, b_path in args.seed:
        a, b = load(a_path), load(b_path)
        for ra, rb in zip(a, b):
            assert ra["image_id"] == rb["image_id"], (tag, ra["sample_index"])
        images = np.asarray([r["image_id"] for r in a])
        assert np.unique(images).size == 120
        entry = {}
        for metric in ("rotated_iou", "aabb_iou"):
            va = np.asarray([float(r[metric]) for r in a])
            vb = np.asarray([float(r[metric]) for r in b])
            entry[metric] = {a_name: float(va.mean()), b_name: float(vb.mean()),
                             "delta": cluster_ci(va - vb, images)}
        d = entry["rotated_iou"]["delta"]
        entry["rotIoU_delta_ci_excludes_zero"] = d["ci95_lo"] > 0 or d["ci95_hi"] < 0
        report["seeds"][tag] = entry
        print(f"{tag}: rotIoU {a_name} {entry['rotated_iou'][a_name]:.4f}  {b_name} "
              f"{entry['rotated_iou'][b_name]:.4f}  delta {d['mean']:+.4f} "
              f"[{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}]   aabb delta "
              f"{entry['aabb_iou']['delta']['mean']:+.4f} "
              f"[{entry['aabb_iou']['delta']['ci95_lo']:+.4f}, {entry['aabb_iou']['delta']['ci95_hi']:+.4f}]")

    deltas = np.array([e["rotated_iou"]["delta"]["mean"] for e in report["seeds"].values()])
    report["across_seeds_rotIoU_delta"] = {"mean": float(deltas.mean()),
                                           "sd": float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
                                           "min": float(deltas.min()), "max": float(deltas.max())}
    print("across seeds:", {k: round(v, 4) if v is not None else None
                            for k, v in report["across_seeds_rotIoU_delta"].items()})

    if args.check_seed1:
        lo, hi = args.check_seed1
        d = report["seeds"]["s1"]["rotated_iou"]["delta"]
        assert round(d["ci95_lo"], 4) == lo and round(d["ci95_hi"], 4) == hi, d
        print("seed 1 reproduces the published interval")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
